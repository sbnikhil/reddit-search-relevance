"""
Script 08 — Generate ColBERT scores for all query-product pairs.

Scores every (query, product_title) pair in the ESCI dataset using the trained
ColBERT model and writes results to BigQuery (colbert_scores table). All
downstream scripts (10-14) read from this table rather than re-running inference.

This script is intentionally self-contained so it can run on Vertex AI without
needing the full project package — config and model code are inlined.

Runtime: ~2.5h on a single T4 GPU.

Usage:
    python scripts/08_generate_colbert_scores.py --submit   # Vertex AI
    python scripts/08_generate_colbert_scores.py --smoke    # 20 queries locally
    python scripts/08_generate_colbert_scores.py            # full local run
"""

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

# ── Config — reads from env vars (set by Vertex AI) or falls back to defaults ─
PROJECT_ID              = os.environ.get("GCP_PROJECT_ID",       "search-relevance-500100")
BQ_DATASET              = os.environ.get("BQ_DATASET",           "esci_search")
MODELS_BUCKET           = os.environ.get("GCP_MODELS_BUCKET",    "search-models-nikhil")
DATA_BUCKET             = os.environ.get("GCP_DATA_BUCKET",      "search-data-nikhil")
REGION                  = os.environ.get("GCP_REGION",           "us-central1")
COLBERT_MODEL_NAME      = os.environ.get("COLBERT_MODEL_NAME",   "bert-base-uncased")
COLBERT_DIM             = int(os.environ.get("COLBERT_DIM",      "128"))
COLBERT_MAXLEN_QUERY    = int(os.environ.get("COLBERT_MAXLEN_QUERY", "32"))
COLBERT_MAXLEN_DOC      = int(os.environ.get("COLBERT_MAXLEN_DOC",  "128"))
COLBERT_CKPT_GCS        = os.environ.get(
    "COLBERT_CHECKPOINT",
    f"gs://{MODELS_BUCKET}/colbert/epoch_5/model.pt",
)
DOC_BATCH               = int(os.environ.get("SCORE_GEN_DOC_BATCH",        "128"))
CHECKPOINT_EVERY        = int(os.environ.get("SCORE_GEN_CHECKPOINT_EVERY", "5000"))

OUTPUT_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.colbert_scores"
LOCAL_CKPT   = "/tmp/colbert_scores_ckpt.parquet"
GCS_CKPT     = f"gs://{DATA_BUCKET}/esci/colbert_scores/checkpoint.parquet"


# ── Inlined ColBERT model (avoids needing the models/ package on Vertex AI) ───

class ColBERT(nn.Module):
    def __init__(self, model_name: str = COLBERT_MODEL_NAME, dim: int = COLBERT_DIM):
        super().__init__()
        self.bert   = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out  = self.linear(out)
        return F.normalize(out, p=2, dim=-1)

    def encode_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out  = self.linear(out)
        embs = F.normalize(out, p=2, dim=-1)
        return embs * attention_mask.unsqueeze(-1).float()


# ── Inlined GCS download ───────────────────────────────────────────────────────

def _gcs_download(gcs_uri: str, local_path: str) -> None:
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)


# ── Core logic ────────────────────────────────────────────────────────────────

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device) -> tuple:
    local = "/tmp/colbert_epoch5.pt"
    print(f"Downloading ColBERT checkpoint from {COLBERT_CKPT_GCS}...")
    _gcs_download(COLBERT_CKPT_GCS, local)

    tokenizer = BertTokenizerFast.from_pretrained(COLBERT_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    model = ColBERT(model_name=COLBERT_MODEL_NAME, dim=COLBERT_DIM)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(local, map_location=device))
    model.to(device).eval()
    print(f"ColBERT loaded on {device}. Vocab: {len(tokenizer)}")
    return model, tokenizer


def score_group(
    query_text: str,
    titles: list,
    tokenizer: BertTokenizerFast,
    model: ColBERT,
    device: torch.device,
) -> np.ndarray:
    q_enc = tokenizer(
        [f"[Q] {query_text}"], padding=True, truncation=True,
        max_length=COLBERT_MAXLEN_QUERY, return_tensors="pt",
    )
    with torch.no_grad():
        q_emb = model.encode_query(
            q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device),
        )[0]

    scores = []
    for start in range(0, len(titles), DOC_BATCH):
        batch = [f"[D] {t}" for t in titles[start : start + DOC_BATCH]]
        d_enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=COLBERT_MAXLEN_DOC, return_tensors="pt",
        )
        with torch.no_grad():
            d_embs = model.encode_document(
                d_enc["input_ids"].to(device), d_enc["attention_mask"].to(device),
            )
        sim = torch.einsum("qd,bld->bql", q_emb, d_embs)
        scores.append(sim.max(dim=2).values.sum(dim=1).cpu().numpy())

    return np.concatenate(scores)


def run(args) -> None:
    device = _best_device()
    model, tokenizer = load_model(device)

    bq = bigquery.Client(project=PROJECT_ID)
    split_filter = f"WHERE e.split = '{args.split}'" if args.split != "all" else ""
    sql = f"""
        SELECT e.query_id, e.query, e.product_id,
               COALESCE(p.product_title, '') AS product_title
        FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
        JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
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

    # Resume from checkpoint if available
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
        print(f"  {len(done_ids):,} queries already done.")

    records = list(existing)
    pending = [(qid, g) for qid, g in df.groupby("query_id", sort=False)
               if qid not in done_ids]

    for i, (query_id, group) in enumerate(pending):
        if i % 1000 == 0:
            pct = 100 * (len(done_ids) + i) / total_queries
            print(f"  [{len(done_ids)+i:,}/{total_queries:,}] ({pct:.1f}%)")

        group_scores = score_group(
            group["query"].iloc[0],
            group["product_title"].tolist(),
            tokenizer, model, device,
        )
        for pid, score in zip(group["product_id"].tolist(), group_scores):
            records.append({
                "query_id":      str(query_id),
                "product_id":    str(pid),
                "colbert_score": float(score),
            })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt_df = pd.DataFrame(records)
            ckpt_df.to_parquet(LOCAL_CKPT, index=False)
            gcs_blob.upload_from_filename(LOCAL_CKPT)
            print(f"  Checkpoint: {len(records):,} records saved.")

    scores_df = pd.DataFrame(records)
    scores_df["query_id"]   = scores_df["query_id"].astype(str)
    scores_df["product_id"] = scores_df["product_id"].astype(str)
    print(f"\nScoring done: {len(scores_df):,} records. Writing to BigQuery...")

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
    print("Next step: python scripts/10_ltr_regression_analysis.py")


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
        environment_variables={
            "GCP_PROJECT_ID":            PROJECT_ID,
            "BQ_DATASET":                BQ_DATASET,
            "GCP_MODELS_BUCKET":         MODELS_BUCKET,
            "GCP_DATA_BUCKET":           DATA_BUCKET,
            "GCP_REGION":                REGION,
            "COLBERT_MODEL_NAME":        COLBERT_MODEL_NAME,
            "COLBERT_DIM":               str(COLBERT_DIM),
            "COLBERT_MAXLEN_QUERY":      str(COLBERT_MAXLEN_QUERY),
            "COLBERT_MAXLEN_DOC":        str(COLBERT_MAXLEN_DOC),
            "COLBERT_CHECKPOINT":        COLBERT_CKPT_GCS,
            "SCORE_GEN_DOC_BATCH":       str(DOC_BATCH),
            "SCORE_GEN_CHECKPOINT_EVERY": str(CHECKPOINT_EVERY),
        },
        sync=False,
    )
    print(f"Job submitted. Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


if __name__ == "__main__":
    # When running locally, load config from the project package
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import (
            PROJECT_ID as _PID, BQ_DATASET as _BQ, MODELS_BUCKET as _MB,
            DATA_BUCKET as _DB, REGION as _R, COLBERT_MODEL_NAME as _CMN,
            COLBERT_DIM as _CD, COLBERT_MAXLEN_QUERY as _CMQ,
            COLBERT_MAXLEN_DOC as _CMD, COLBERT_CKPT_GCS as _CKPT,
            SCORE_GEN_DOC_BATCH as _BATCH, SCORE_GEN_CHECKPOINT_EVERY as _CPE,
        )
        PROJECT_ID = _PID; BQ_DATASET = _BQ; MODELS_BUCKET = _MB
        DATA_BUCKET = _DB; REGION = _R; COLBERT_MODEL_NAME = _CMN
        COLBERT_DIM = _CD; COLBERT_MAXLEN_QUERY = _CMQ; COLBERT_MAXLEN_DOC = _CMD
        COLBERT_CKPT_GCS = _CKPT; DOC_BATCH = _BATCH; CHECKPOINT_EVERY = _CPE
        OUTPUT_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.colbert_scores"
        GCS_CKPT     = f"gs://{DATA_BUCKET}/esci/colbert_scores/checkpoint.parquet"
    except ImportError:
        pass  # running on Vertex AI — env vars already set above

    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  choices=["train", "test", "all"], default="all")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--smoke",  action="store_true", help="Run on 20 queries only (for testing)")
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job(args)
    else:
        run(args)
