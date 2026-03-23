import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.cloud import bigquery, storage
from transformers import BertModel, BertTokenizerFast

PROJECT_ID    = "reddit-search-relevance-485717"
REGION        = "us-central1"
MODELS_BUCKET = "reddit-search-relevance-models"
CHECKPOINT    = f"gs://{MODELS_BUCKET}/colbert/epoch_5/model.pt"
OUTPUT_TABLE  = f"{PROJECT_ID}.esci_search.colbert_scores"
COLBERT_DIM   = 128
QUERY_MAXLEN  = 32
DOC_MAXLEN    = 128
DOC_BATCH     = 128
CHECKPOINT_EVERY = 5_000
LOCAL_CKPT    = "/tmp/colbert_scores_ckpt.parquet"
GCS_CKPT      = f"gs://reddit-search-relevance-data/esci/colbert_scores/checkpoint.parquet"


class ColBERT(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", dim: int = 128):
        super().__init__()
        self.bert   = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)
        self.dim    = dim

    def encode_query(self, input_ids, attention_mask):
        out       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = self.linear(out.last_hidden_state)
        return F.normalize(token_emb, p=2, dim=-1)

    def encode_document(self, input_ids, attention_mask):
        out       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = self.linear(out.last_hidden_state)
        token_emb = F.normalize(token_emb, p=2, dim=-1)
        mask      = attention_mask.unsqueeze(-1).float()
        return token_emb * mask


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _download(gcs_uri: str, local_path: str) -> None:
    client = storage.Client()
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    client.bucket(bucket).blob(blob).download_to_filename(local_path)
    print(f"Downloaded {gcs_uri} -> {local_path}")


def load_model(device: torch.device, tokenizer: BertTokenizerFast) -> ColBERT:
    local = "/tmp/colbert_epoch5.pt"
    _download(CHECKPOINT, local)
    model = ColBERT(dim=COLBERT_DIM)
    # Match the vocab size the training script used: BERT base (30522) + [Q] + [D] = 30524
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})
    model.bert.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(local, map_location=device))
    model.to(device).eval()
    print(f"ColBERT loaded on {device}. Vocab size: {len(tokenizer)}")
    return model


def score_group(
    query_text: str,
    titles: list,
    tokenizer: BertTokenizerFast,
    model: ColBERT,
    device: torch.device,
) -> np.ndarray:
    # Prepend [Q] / [D] markers to match training tokenization exactly
    q_enc = tokenizer(
        [f"[Q] {query_text}"], padding=True, truncation=True,
        max_length=QUERY_MAXLEN, return_tensors="pt",
    )
    with torch.no_grad():
        q_emb = model.encode_query(
            q_enc["input_ids"].to(device),
            q_enc["attention_mask"].to(device),
        )[0]  # (Lq, dim)

    scores = []
    for start in range(0, len(titles), DOC_BATCH):
        batch = titles[start : start + DOC_BATCH]
        d_enc = tokenizer(
            [f"[D] {t}" for t in batch], padding=True, truncation=True,
            max_length=DOC_MAXLEN, return_tensors="pt",
        )
        with torch.no_grad():
            d_embs = model.encode_document(
                d_enc["input_ids"].to(device),
                d_enc["attention_mask"].to(device),
            )  # (B, Ld, dim)
        sim = torch.einsum("qd,bld->bql", q_emb, d_embs)
        scores.append(sim.max(dim=2).values.sum(dim=1).cpu().numpy())

    return np.concatenate(scores)


def run(args) -> None:
    device    = _best_device()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model     = load_model(device, tokenizer)

    bq = bigquery.Client(project=PROJECT_ID)
    split_filter = f"WHERE e.split = '{args.split}'" if args.split != "all" else ""
    sql = f"""
        SELECT e.query_id, e.query, e.product_id,
               COALESCE(p.product_title, '') AS product_title
        FROM `{PROJECT_ID}.esci_search.examples` e
        JOIN `{PROJECT_ID}.esci_search.products` p USING (product_id)
        {split_filter}
        ORDER BY e.query_id
    """
    print("Loading pairs from BigQuery...")
    df = bq.query(sql).to_dataframe()
    if args.smoke:
        sample_qids = df["query_id"].unique()[:20]
        df = df[df["query_id"].isin(sample_qids)]
        print(f"  [SMOKE] limited to {len(sample_qids)} queries")
    total_queries = df["query_id"].nunique()
    print(f"  {len(df):,} pairs, {total_queries:,} queries")

    # Resume from GCS checkpoint if it exists, otherwise local
    done_ids: set = set()
    existing: list = []
    gcs_client = storage.Client()
    ckpt_bucket, ckpt_blob = GCS_CKPT.replace("gs://", "").split("/", 1)
    gcs_blob = gcs_client.bucket(ckpt_bucket).blob(ckpt_blob)
    if gcs_blob.exists():
        gcs_blob.download_to_filename(LOCAL_CKPT)
        print(f"Resuming from GCS checkpoint: {GCS_CKPT}")
        prev     = pd.read_parquet(LOCAL_CKPT)
        done_ids = set(prev["query_id"].unique())
        existing = prev.to_dict("records")
        print(f"  {len(done_ids):,} queries already done, skipping.")
    elif os.path.exists(LOCAL_CKPT):
        prev     = pd.read_parquet(LOCAL_CKPT)
        done_ids = set(prev["query_id"].unique())
        existing = prev.to_dict("records")
        print(f"Resuming from local checkpoint — {len(done_ids):,} queries already done.")

    records = list(existing)
    pending = [(qid, g) for qid, g in df.groupby("query_id", sort=False)
               if qid not in done_ids]

    for i, (query_id, group) in enumerate(pending):
        if i % 1000 == 0:
            done_so_far = len(done_ids) + i
            pct = 100 * done_so_far / total_queries
            print(f"  [{done_so_far:,}/{total_queries:,}] ({pct:.1f}%)")

        group_scores = score_group(
            group["query"].iloc[0],
            group["product_title"].tolist(),
            tokenizer, model, device,
        )
        for pid, score in zip(group["product_id"].tolist(), group_scores):
            records.append({
                "query_id":      query_id,
                "product_id":    pid,
                "colbert_score": float(score),
            })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt_df = pd.DataFrame(records)
            ckpt_df["query_id"]   = ckpt_df["query_id"].astype(str)
            ckpt_df["product_id"] = ckpt_df["product_id"].astype(str)
            ckpt_df.to_parquet(LOCAL_CKPT, index=False)
            gcs_blob.upload_from_filename(LOCAL_CKPT)
            print(f"  Checkpoint: {len(records):,} records saved to GCS.")

    scores_df = pd.DataFrame(records)
    scores_df["query_id"]   = scores_df["query_id"].astype(str)
    scores_df["product_id"] = scores_df["product_id"].astype(str)
    scores_df.to_parquet(LOCAL_CKPT, index=False)
    print(f"\nScoring done: {len(scores_df):,} records. Uploading to BigQuery...")

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("query_id",      "STRING"),
            bigquery.SchemaField("product_id",    "STRING"),
            bigquery.SchemaField("colbert_score", "FLOAT64"),
        ],
    )
    bq.load_table_from_dataframe(scores_df, OUTPUT_TABLE, job_config=job_config).result()
    print(f"Uploaded -> {OUTPUT_TABLE}")
    print("Next step: python scripts/05_train_lambdamart.py")


def submit_vertex_job(args) -> None:
    from google.cloud import aiplatform

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{MODELS_BUCKET}",
    )
    job = aiplatform.CustomTrainingJob(
        display_name="colbert-score-generation",
        script_path="scripts/08_generate_colbert_scores.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=[
            "transformers==4.36.0",
            "google-cloud-bigquery",
            "google-cloud-storage",
            "db-dtypes",
            "pandas",
            "pyarrow",
        ],
    )
    print("Submitting Vertex AI job...")
    job.run(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        replica_count=1,
        args=["--split", args.split],
        sync=False,   # don't block — job runs in the background
    )
    print("Job submitted. Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  choices=["train", "test", "all"], default="all")
    parser.add_argument("--submit", action="store_true", help="Submit to Vertex AI T4 GPU instead of running locally")
    parser.add_argument("--smoke",  action="store_true", help="Smoke test: run on 20 queries only")
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job(args)
    else:
        run(args)
