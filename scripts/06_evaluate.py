import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.cloud import bigquery, storage, aiplatform
from rank_bm25 import BM25Okapi
from transformers import BertModel, BertTokenizerFast

PROJECT_ID    = "reddit-search-relevance-485717"
MODELS_BUCKET = "reddit-search-relevance-models"
DATA_BUCKET   = "reddit-search-relevance-data"
COLBERT_CKPT  = f"gs://{MODELS_BUCKET}/colbert/epoch_5/model.pt"
LTR_MODEL_GCS = f"gs://{MODELS_BUCKET}/ltr/lambdamart.txt"
COLBERT_DIM   = 128
QUERY_MAXLEN  = 32
DOC_MAXLEN    = 180
SAMPLE_QUERIES = 1000
DOC_BATCH      = 64
RESULTS_DIR    = "results"

FEATURE_NAMES = [
    "bm25_score", "colbert_score", "title_query_overlap",
    "desc_query_overlap", "title_bigram_overlap", "brand_match",
    "title_length", "desc_length", "has_description", "query_length",
]


class ColBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)

    def encode_query(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.linear(out.last_hidden_state), p=2, dim=-1)

    def encode_document(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embs = F.normalize(self.linear(out.last_hidden_state), p=2, dim=-1)
        return embs * attention_mask.unsqueeze(-1).float()


def _download(gcs_uri: str, local_path: str) -> None:
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket).blob(blob).download_to_filename(local_path)


def _upload(local_path: str, gcs_uri: str) -> None:
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)


def _dcg(gains: list, k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(pred: list, ideal: list, k: int) -> float:
    idcg = _dcg(sorted(ideal, reverse=True), k)
    return _dcg(pred, k) / idcg if idcg > 0 else 0.0


def mrr_at_k(pred: list, k: int) -> float:
    for i, g in enumerate(pred[:k]):
        if g >= 1.0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(pred: list, k: int) -> float:
    return sum(1 for g in pred[:k] if g > 0) / k


def evaluate_stage(df: pd.DataFrame, score_col: str) -> tuple:
    ndcgs, mrrs, precs = [], [], []
    for _, group in df.groupby("query_id"):
        ranked = group.sort_values(score_col, ascending=False)
        pred  = ranked["gain"].tolist()
        ideal = group["gain"].tolist()
        ndcgs.append(ndcg_at_k(pred, ideal, 10))
        mrrs.append(mrr_at_k(pred, 10))
        precs.append(precision_at_k(pred, 10))
    return float(np.mean(ndcgs)), float(np.mean(mrrs)), float(np.mean(precs))


def _bigrams(tokens: list) -> set:
    return set(zip(tokens, tokens[1:]))


def build_features(query, title, description, brand, bm25_score=0.0, colbert_score=0.0):
    q_tok = query.lower().split()
    t_tok = (title or "").lower().split()
    d_tok = (description or "").lower().split()
    brand = (brand or "").lower()
    q_set = set(q_tok)
    q_bg, t_bg = _bigrams(q_tok), _bigrams(t_tok)
    return [
        bm25_score,
        colbert_score,
        len(q_set & set(t_tok)) / max(len(q_set), 1),
        len(q_set & set(d_tok)) / max(len(q_set), 1),
        len(q_bg & t_bg) / max(len(q_bg), 1) if q_bg else 0.0,
        float(bool(brand) and brand in query.lower()),
        np.log1p(len(t_tok)),
        np.log1p(len(d_tok)),
        float(bool(d_tok)),
        float(len(q_tok)),
    ]


def add_bm25_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_id, group in df.groupby("query_id"):
        query  = group["query"].iloc[0]
        corpus = (group["product_title"].fillna("") + " " + group["product_description"].fillna("")).tolist()
        bm25   = BM25Okapi([t.lower().split() for t in corpus])
        raw    = bm25.get_scores(query.lower().split())
        norm   = raw / (raw.max() or 1.0)
        for pid, s in zip(group["product_id"], norm):
            rows.append({"product_id": pid, "query_id": query_id, "bm25_score": s})
    return df.merge(pd.DataFrame(rows), on=["query_id", "product_id"])


def load_colbert(device: torch.device) -> tuple:
    local = "/tmp/colbert_eval.pt"
    print(f"Downloading ColBERT checkpoint...")
    _download(COLBERT_CKPT, local)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    model = ColBERT(dim=COLBERT_DIM)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(local, map_location=device))
    model.to(device).eval()
    print(f"ColBERT loaded ({device}).")
    return model, tokenizer


def add_colbert_scores(df: pd.DataFrame, model: ColBERT, tokenizer, device) -> pd.DataFrame:
    rows = []
    for query_id, group in df.groupby("query_id"):
        query_text = group["query"].iloc[0]
        titles     = group["product_title"].fillna("").tolist()
        pids       = group["product_id"].tolist()

        q_enc = tokenizer(
            [f"[Q] {query_text}"], padding=True, truncation=True,
            max_length=QUERY_MAXLEN, return_tensors="pt",
        )
        with torch.no_grad():
            q_emb = model.encode_query(
                q_enc["input_ids"].to(device),
                q_enc["attention_mask"].to(device),
            )[0]

        scores = []
        for start in range(0, len(titles), DOC_BATCH):
            batch = [f"[D] {t}" for t in titles[start : start + DOC_BATCH]]
            d_enc = tokenizer(batch, padding=True, truncation=True,
                              max_length=DOC_MAXLEN, return_tensors="pt")
            with torch.no_grad():
                d_embs = model.encode_document(
                    d_enc["input_ids"].to(device),
                    d_enc["attention_mask"].to(device),
                )
            sim = torch.einsum("qd,bld->bql", q_emb, d_embs)
            scores.extend(sim.max(dim=2).values.sum(dim=1).cpu().numpy().tolist())

        arr = np.array(scores)
        for pid, s in zip(pids, arr.tolist()):
            rows.append({"product_id": pid, "query_id": query_id, "colbert_score": s})

    return df.merge(pd.DataFrame(rows), on=["query_id", "product_id"])


def add_ltr_scores(df: pd.DataFrame) -> pd.DataFrame:
    import lightgbm as lgb
    local = "/tmp/lambdamart_eval.txt"
    print("Downloading LambdaMART model...")
    _download(LTR_MODEL_GCS, local)
    booster = lgb.Booster(model_file=local)
    print("LambdaMART loaded.")

    features = np.array([
        build_features(
            row["query"], row["product_title"],
            row["product_description"], row["product_brand"],
            bm25_score=row["bm25_score"],
            colbert_score=row["colbert_score"],
        )
        for _, row in df.iterrows()
    ], dtype=np.float32)

    df = df.copy()
    df["ltr_score"] = booster.predict(features)
    return df


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bq = bigquery.Client(project=PROJECT_ID)
    sql = f"""
        WITH sampled AS (
            SELECT DISTINCT query_id
            FROM `{PROJECT_ID}.esci_search.examples`
            WHERE split = 'test'
            LIMIT {SAMPLE_QUERIES}
        )
        SELECT e.query_id, e.query, e.product_id, e.gain,
               COALESCE(p.product_title, '')       AS product_title,
               COALESCE(p.product_description, '') AS product_description,
               COALESCE(p.product_brand, '')       AS product_brand
        FROM `{PROJECT_ID}.esci_search.examples` e
        JOIN `{PROJECT_ID}.esci_search.products` p USING (product_id)
        JOIN sampled USING (query_id)
    """
    print(f"Loading {SAMPLE_QUERIES} test queries from BigQuery...")
    df = bq.query(sql).to_dataframe()
    print(f"  {len(df):,} query-product pairs loaded.")

    print("\nStage 1: BM25...")
    df = add_bm25_scores(df)
    bm25_ndcg, bm25_mrr, bm25_prec = evaluate_stage(df, "bm25_score")
    print(f"  NDCG@10={bm25_ndcg:.4f}  MRR@10={bm25_mrr:.4f}  P@10={bm25_prec:.4f}")

    print("\nStage 2: ColBERT re-ranking...")
    colbert, tokenizer = load_colbert(device)
    df = add_colbert_scores(df, colbert, tokenizer, device)
    cb_ndcg, cb_mrr, cb_prec = evaluate_stage(df, "colbert_score")
    print(f"  NDCG@10={cb_ndcg:.4f}  MRR@10={cb_mrr:.4f}  P@10={cb_prec:.4f}")

    print("\nStage 3: LambdaMART fusion...")
    df = add_ltr_scores(df)
    ltr_ndcg, ltr_mrr, ltr_prec = evaluate_stage(df, "ltr_score")
    print(f"  NDCG@10={ltr_ndcg:.4f}  MRR@10={ltr_mrr:.4f}  P@10={ltr_prec:.4f}")

    header = f"{'Stage':<25} {'NDCG@10':>8} {'MRR@10':>8} {'P@10':>8}"
    sep    = "-" * len(header)
    print(f"\n{header}\n{sep}")
    print(f"{'BM25 Baseline':<25} {bm25_ndcg:>8.4f} {bm25_mrr:>8.4f} {bm25_prec:>8.4f}")
    print(f"{'+ ColBERT Re-ranking':<25} {cb_ndcg:>8.4f} {cb_mrr:>8.4f} {cb_prec:>8.4f}")
    print(f"{'+ LambdaMART Fusion':<25} {ltr_ndcg:>8.4f} {ltr_mrr:>8.4f} {ltr_prec:>8.4f}")

    out = {
        "bm25":       {"ndcg_10": bm25_ndcg, "mrr_10": bm25_mrr, "precision_10": bm25_prec},
        "colbert":    {"ndcg_10": cb_ndcg,   "mrr_10": cb_mrr,   "precision_10": cb_prec},
        "lambdamart": {"ndcg_10": ltr_ndcg,  "mrr_10": ltr_mrr,  "precision_10": ltr_prec},
        "sample_queries": SAMPLE_QUERIES,
    }

    local_out = "/tmp/final_evaluation.json"
    with open(local_out, "w") as f:
        json.dump(out, f, indent=2)

    gcs_out = f"gs://{DATA_BUCKET}/esci/results/final_evaluation.json"
    _upload(local_out, gcs_out)
    print(f"\nSaved -> {gcs_out}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    local_results = os.path.join(RESULTS_DIR, "final_evaluation.json")
    with open(local_results, "w") as f:
        json.dump(out, f, indent=2)


def submit_vertex_job() -> None:
    aiplatform.init(
        project=PROJECT_ID,
        location="us-central1",
        staging_bucket=f"gs://{MODELS_BUCKET}",
    )
    job = aiplatform.CustomTrainingJob(
        display_name="esci-evaluate-pipeline",
        script_path="scripts/06_evaluate.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
        requirements=[
            "rank-bm25>=0.2.2",
            "transformers>=4.35.0",
            "lightgbm>=4.0.0",
            "google-cloud-bigquery",
            "google-cloud-storage",
            "google-cloud-aiplatform",
            "pandas",
            "pyarrow",
            "db-dtypes",
        ],
    )
    job.run(
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        replica_count=1,
    )
    print("Vertex AI evaluation job submitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Submit to Vertex AI T4 GPU")
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job()
    else:
        run()
