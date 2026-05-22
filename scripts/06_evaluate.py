"""
Full pipeline evaluation — two protocols.

Protocol A (editorial_pool):
  Re-ranks pre-selected ESCI candidates (~20 per query).
  Comparable to published ESCI literature (KDD Cup 2022, COLING 2025, etc.)
  Runs locally.

Protocol B (hybrid_retrieval):
  Retrieves from the full ~200K ESCI-labeled product catalog using BM25+FAISS
  hybrid with Reciprocal Rank Fusion. Honest end-to-end retrieval evaluation.
  Requires Vertex AI (faiss+torch OpenMP conflict on macOS).

Usage:
  python scripts/06_evaluate.py                                 # Protocol A (local)
  python scripts/06_evaluate.py --conditioned_ltr               # Protocol A + Contrib 1
  python scripts/06_evaluate.py --protocol hybrid_retrieval --submit   # Protocol B on Vertex
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from google.cloud import bigquery, aiplatform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROJECT_ID, BQ_DATASET, MODELS_BUCKET, DATA_BUCKET, REGION,
    LAMBDAMART_CKPT_GCS,
    CONTRIB1_RHO_LOW, CONTRIB1_RHO_HIGH,
    CONTRIB1_MODEL_LOW_GCS, CONTRIB1_MODEL_MED_GCS, CONTRIB1_MODEL_HIGH_GCS,
    EVAL_SAMPLE_QUERIES,
    FAISS_INDEX_GCS, FAISS_IDS_GCS, FAISS_NPROBE,
    COLBERT_CKPT_GCS, COLBERT_MODEL_NAME, COLBERT_DIM,
    COLBERT_MAXLEN_QUERY, COLBERT_MAXLEN_DOC,
)
from models.ltr.lambdamart import build_features, bm25_scores_for_group
from utils.gcs import download, upload
from utils.metrics import ndcg_at_k, mrr_at_k, precision_at_k, gain_to_label

LTR_MODEL_GCS = LAMBDAMART_CKPT_GCS
RESULTS_DIR   = "results"
RRF_K         = 60
RETRIEVAL_TOP_K = 50   # BM25 top-k and FAISS top-k before RRF


# ─────────────────────────────────────────────────────────────────────────────
# Protocol A — Editorial Pool (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(sample_queries: int) -> pd.DataFrame:
    bq = bigquery.Client(project=PROJECT_ID)
    sql = f"""
    WITH sampled AS (
        SELECT DISTINCT query_id AS sid
        FROM `{PROJECT_ID}.{BQ_DATASET}.examples`
        WHERE split = 'test'
        LIMIT {sample_queries}
    )
    SELECT
        e.query_id, e.query, e.product_id, e.gain, e.esci_label,
        COALESCE(p.product_title, '')       AS product_title,
        COALESCE(p.product_description, '') AS product_description,
        COALESCE(p.product_brand, '')       AS product_brand,
        COALESCE(c.colbert_score, 0.0)      AS colbert_score
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p ON e.product_id = p.product_id
    LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.colbert_scores` c
        ON CAST(e.query_id AS STRING) = c.query_id
        AND CAST(e.product_id AS STRING) = c.product_id
    JOIN sampled s ON e.query_id = s.sid
    WHERE e.split = 'test'
    ORDER BY e.query_id
    """
    print(f"Loading {sample_queries} test queries from BigQuery...")
    df = bq.query(sql).to_dataframe()
    print(f"  {df['query_id'].nunique():,} queries, {len(df):,} pairs")
    return df.fillna({"product_title": "", "product_description": "", "product_brand": ""})


def load_booster(gcs_path: str, local_name: str = "lambdamart.txt"):
    import lightgbm as lgb
    local = f"/tmp/{local_name}"
    download(gcs_path, local)
    return lgb.Booster(model_file=local)


def add_bm25_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_id, group in df.groupby("query_id", sort=False):
        query    = group["query"].iloc[0]
        bm25_raw = bm25_scores_for_group(
            query, group["product_title"].tolist(), group["product_description"].tolist()
        )
        bm25_norm = bm25_raw / (bm25_raw.max() or 1.0)
        for pid, s in zip(group["product_id"], bm25_norm):
            rows.append({"product_id": pid, "query_id": query_id, "bm25_score": float(s)})
    return df.merge(pd.DataFrame(rows), on=["query_id", "product_id"])


def add_ltr_scores(df: pd.DataFrame, booster, score_col: str = "ltr_score") -> pd.DataFrame:
    features = np.array([
        build_features(
            row["query"], row["product_title"], row["product_description"],
            row["product_brand"], row["bm25_score"], row["colbert_score"],
        )
        for _, row in df.iterrows()
    ], dtype=np.float32)
    df = df.copy()
    df[score_col] = booster.predict(features)
    return df


def add_conditioned_ltr_scores(df: pd.DataFrame) -> pd.DataFrame:
    boosters = {
        "low":    load_booster(CONTRIB1_MODEL_LOW_GCS,  "lambdamart_low.txt"),
        "medium": load_booster(CONTRIB1_MODEL_MED_GCS,  "lambdamart_med.txt"),
        "high":   load_booster(CONTRIB1_MODEL_HIGH_GCS, "lambdamart_high.txt"),
    }
    print("Conditioned LTR models loaded.")

    rows = []
    for query_id, group in df.groupby("query_id", sort=False):
        if len(group) < 2:
            continue
        rho_result = spearmanr(group["bm25_score"].values, group["colbert_score"].values)
        rho = float(rho_result.statistic) if len(group) > 2 else 0.0
        if np.isnan(rho):
            rho = 0.0

        booster = (boosters["low"]    if rho < CONTRIB1_RHO_LOW  else
                   boosters["medium"] if rho <= CONTRIB1_RHO_HIGH else
                   boosters["high"])

        feats = np.array([
            build_features(
                row["query"], row["product_title"], row["product_description"],
                row["product_brand"], row["bm25_score"], row["colbert_score"],
            )
            for _, row in group.iterrows()
        ], dtype=np.float32)
        for pid, s in zip(group["product_id"], booster.predict(feats)):
            rows.append({"product_id": pid, "query_id": query_id,
                         "conditioned_ltr_score": float(s), "rho": float(rho)})

    return df.merge(pd.DataFrame(rows), on=["query_id", "product_id"])


def _length_group(n: int) -> str:
    if n <= 2: return "short"
    if n <= 4: return "standard"
    return "long"


def _rho_bin(rho: float) -> str:
    if rho < CONTRIB1_RHO_LOW:  return "low"
    if rho <= CONTRIB1_RHO_HIGH: return "medium"
    return "high"


def evaluate_stage(df: pd.DataFrame, score_col: str,
                   group_cols: list[str] | None = None) -> dict:
    per_query = []
    for query_id, group in df.groupby("query_id", sort=False):
        ranked        = group.sort_values(score_col, ascending=False)
        ranked_labels = [gain_to_label(g) for g in ranked["gain"].tolist()]
        q_len         = len(group["query"].iloc[0].split())
        rho           = float(group["rho"].iloc[0]) if "rho" in group.columns else 0.0
        per_query.append({
            "ndcg":         ndcg_at_k(ranked_labels),
            "mrr":          mrr_at_k(ranked_labels),
            "precision":    precision_at_k(ranked_labels),
            "length_group": _length_group(q_len),
            "rho_bin":      _rho_bin(rho),
        })

    pq = pd.DataFrame(per_query)
    result = {
        "ndcg_10":      float(pq["ndcg"].mean()),
        "mrr_10":       float(pq["mrr"].mean()),
        "precision_10": float(pq["precision"].mean()),
    }
    if group_cols:
        for col in group_cols:
            result[f"by_{col}"] = {}
            for val, grp in pq.groupby(col):
                result[f"by_{col}"][str(val)] = {
                    "ndcg_10":      float(grp["ndcg"].mean()),
                    "mrr_10":       float(grp["mrr"].mean()),
                    "precision_10": float(grp["precision"].mean()),
                    "n":            int(len(grp)),
                }
    return result


def run_editorial_pool(args) -> dict:
    df = load_test_data(args.sample_queries)

    print("\nStage 1: BM25...")
    df = add_bm25_scores(df)
    bm25_metrics = evaluate_stage(df, "bm25_score", ["length_group"])
    print(f"  NDCG@10={bm25_metrics['ndcg_10']:.4f}  MRR@10={bm25_metrics['mrr_10']:.4f}")

    print("\nStage 2: ColBERT (pre-computed from BigQuery)...")
    cb_metrics = evaluate_stage(df, "colbert_score", ["length_group"])
    print(f"  NDCG@10={cb_metrics['ndcg_10']:.4f}  MRR@10={cb_metrics['mrr_10']:.4f}")

    print("\nStage 3: LambdaMART (base)...")
    base_booster = load_booster(LTR_MODEL_GCS, "lambdamart_base.txt")
    df = add_ltr_scores(df, base_booster, "ltr_score")
    ltr_metrics = evaluate_stage(df, "ltr_score", ["length_group", "rho_bin"])
    print(f"  NDCG@10={ltr_metrics['ndcg_10']:.4f}  MRR@10={ltr_metrics['mrr_10']:.4f}")

    out = {
        "protocol":       "editorial_pool",
        "sample_queries": args.sample_queries,
        "baseline": {
            "bm25":       bm25_metrics,
            "colbert":    cb_metrics,
            "lambdamart": ltr_metrics,
        },
    }

    if args.conditioned_ltr:
        print("\nContribution 1: ρ-conditioned LTR...")
        try:
            df = add_conditioned_ltr_scores(df)
            cond_metrics = evaluate_stage(df, "conditioned_ltr_score", ["rho_bin"])
            delta = cond_metrics["ndcg_10"] - ltr_metrics["ndcg_10"]
            print(f"  NDCG@10={cond_metrics['ndcg_10']:.4f}  delta vs base LTR: {delta:+.4f}")
            out["conditioned_ltr"] = cond_metrics
        except Exception as e:
            print(f"  WARNING: Conditioned LTR failed: {e}")

    _print_protocol_a_table(out, args)
    return out


def _print_protocol_a_table(out: dict, args) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print("  Protocol A — ESCI Editorial Pool")
    print(sep)
    print(f"  {'Stage':<36} {'NDCG@10':>8} {'MRR@10':>8}")
    print("  " + "─" * 62)

    def _row(label, m):
        print(f"  {label:<36} {m['ndcg_10']:>8.4f} {m['mrr_10']:>8.4f}")

    _row("BM25 Baseline",               out["baseline"]["bm25"])
    _row("+ ColBERT Re-ranking",         out["baseline"]["colbert"])
    _row("+ LambdaMART (base)",          out["baseline"]["lambdamart"])
    if "conditioned_ltr" in out:
        _row("+ ρ-Conditioned LTR",      out["conditioned_ltr"])

    if args.query_router:
        print()
        by_len = out["baseline"]["lambdamart"].get("by_length_group", {})
        for grp in ["short", "standard", "long"]:
            g = by_len.get(grp, {})
            if g:
                print(f"  {grp:<14} N={g['n']:>5,}  NDCG@10={g['ndcg_10']:.4f}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Protocol B — Hybrid Retrieval (BM25 + FAISS + RRF)
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_merge(lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for lst in lists:
        for rank, pid in enumerate(lst):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda p: scores[p], reverse=True)


def _load_hybrid_components() -> dict:
    import faiss
    from rank_bm25 import BM25Okapi

    print("Downloading FAISS index...")
    local_index = "/tmp/eval_product_index.faiss"
    local_idmap = "/tmp/eval_product_id_map.json"
    download(FAISS_INDEX_GCS, local_index)
    download(FAISS_IDS_GCS,   local_idmap)

    index = faiss.read_index(local_index)
    index.nprobe = FAISS_NPROBE
    with open(local_idmap) as f:
        faiss_ids = json.load(f)
    print(f"  FAISS index: {index.ntotal:,} vectors")

    bq = bigquery.Client(project=PROJECT_ID)
    sql = f"""
    SELECT DISTINCT p.product_id,
        COALESCE(p.product_title, '')       AS product_title,
        COALESCE(p.product_description, '') AS product_description
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
    ORDER BY product_id
    """
    print("Loading product catalog from BigQuery...")
    cat = bq.query(sql).to_dataframe()
    print(f"  {len(cat):,} products")

    print("Building BM25 index...")
    corpus = (cat["product_title"] + " " + cat["product_description"]).str.lower().str.split().tolist()
    bm25     = BM25Okapi(corpus)
    bm25_ids = cat["product_id"].tolist()

    return {
        "faiss_index": index,
        "faiss_ids":   faiss_ids,
        "bm25":        bm25,
        "bm25_ids":    bm25_ids,
        "n_products":  len(cat),
    }


def _load_colbert_query_encoder(device) -> tuple:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertModel, BertTokenizerFast

    class ColBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert   = BertModel.from_pretrained(COLBERT_MODEL_NAME)
            self.linear = nn.Linear(self.bert.config.hidden_size, COLBERT_DIM)

        def encode_query(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return F.normalize(self.linear(out), p=2, dim=-1)

    local_ckpt = "/tmp/colbert_eval.pt"
    download(COLBERT_CKPT_GCS, local_ckpt)

    tokenizer = BertTokenizerFast.from_pretrained(COLBERT_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    model = ColBERT()
    model.bert.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(local_ckpt, map_location="cpu", weights_only=False))
    model.to(device).eval()
    print(f"  ColBERT loaded on {device}")
    return model, tokenizer


def _load_labels_for_queries(query_ids: list, bq) -> dict:
    """Return {query_id: {product_id: esci_label}} for all test queries in one BQ call."""
    qids_str = ", ".join(f"'{qid}'" for qid in query_ids)
    sql = f"""
    SELECT CAST(query_id AS STRING) AS query_id,
           CAST(product_id AS STRING) AS product_id,
           esci_label
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples`
    WHERE CAST(query_id AS STRING) IN ({qids_str})
    """
    df = bq.query(sql).to_dataframe()
    labels: dict = {}
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        pid = str(row["product_id"])
        labels.setdefault(qid, {})[pid] = row["esci_label"]
    return labels


def run_hybrid_retrieval(args) -> dict:
    import torch
    import faiss

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    comp              = _load_hybrid_components()
    model, tokenizer  = _load_colbert_query_encoder(device)

    bq = bigquery.Client(project=PROJECT_ID)
    queries_df = bq.query(f"""
        SELECT DISTINCT CAST(query_id AS STRING) AS query_id, query
        FROM `{PROJECT_ID}.{BQ_DATASET}.examples`
        WHERE split = 'test'
        ORDER BY query_id
        LIMIT {args.sample_queries}
    """).to_dataframe()

    query_ids = queries_df["query_id"].tolist()
    queries   = queries_df["query"].tolist()
    print(f"\nLoaded {len(queries):,} test queries")

    print("Loading ESCI labels...")
    label_map = _load_labels_for_queries(query_ids, bq)

    print("Encoding queries...")
    q_embs_list = []
    with torch.no_grad():
        for start in range(0, len(queries), 32):
            batch = queries[start : start + 32]
            enc   = tokenizer(
                [f"[Q] {q}" for q in batch],
                max_length=COLBERT_MAXLEN_QUERY, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            out = model.encode_query(enc["input_ids"].to(device),
                                     enc["attention_mask"].to(device))
            cls = out[:, 0, :].cpu().numpy().astype("float32")
            faiss.normalize_L2(cls)
            q_embs_list.append(cls)
    q_embs = np.vstack(q_embs_list)
    print(f"  {q_embs.shape[0]} queries encoded")

    bm25_scores_list   = []
    faiss_scores_list  = []
    hybrid_scores_list = []

    for i, (qid, query, q_emb) in enumerate(zip(query_ids, queries, q_embs)):
        # BM25 retrieval
        raw_bm25 = comp["bm25"].get_scores(query.lower().split())
        k = min(RETRIEVAL_TOP_K, len(raw_bm25))
        bm25_top_idx = np.argpartition(raw_bm25, -k)[-k:]
        bm25_top_idx = bm25_top_idx[np.argsort(raw_bm25[bm25_top_idx])[::-1]]
        bm25_retrieved = [comp["bm25_ids"][j] for j in bm25_top_idx]

        # FAISS retrieval
        D, I = comp["faiss_index"].search(q_emb.reshape(1, -1), RETRIEVAL_TOP_K)
        faiss_retrieved = [comp["faiss_ids"][j] for j in I[0] if j >= 0]

        # RRF merge
        hybrid_retrieved = _rrf_merge([bm25_retrieved, faiss_retrieved])

        # Lookup labels
        q_labels = label_map.get(str(qid), {})
        def ranked_labels(pid_list):
            return [q_labels.get(str(pid), "I") for pid in pid_list[:100]]

        bm25_scores_list.append({
            "ndcg": ndcg_at_k(ranked_labels(bm25_retrieved)),
            "mrr":  mrr_at_k(ranked_labels(bm25_retrieved)),
        })
        faiss_scores_list.append({
            "ndcg": ndcg_at_k(ranked_labels(faiss_retrieved)),
            "mrr":  mrr_at_k(ranked_labels(faiss_retrieved)),
        })
        hybrid_scores_list.append({
            "ndcg": ndcg_at_k(ranked_labels(hybrid_retrieved)),
            "mrr":  mrr_at_k(ranked_labels(hybrid_retrieved)),
        })

        if i % 100 == 0:
            print(f"  {i}/{len(queries)} queries done")

    def agg(lst):
        return {
            "ndcg_10": float(np.mean([m["ndcg"] for m in lst])),
            "mrr_10":  float(np.mean([m["mrr"]  for m in lst])),
        }

    bm25_agg   = agg(bm25_scores_list)
    faiss_agg  = agg(faiss_scores_list)
    hybrid_agg = agg(hybrid_scores_list)

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  Protocol B — End-to-End Retrieval ({comp['n_products']:,} products)")
    print(sep)
    print(f"  {'Stage':<36} {'NDCG@10':>8} {'MRR@10':>8}")
    print("  " + "─" * 62)
    print(f"  {'BM25 retrieval':<36} {bm25_agg['ndcg_10']:>8.4f} {bm25_agg['mrr_10']:>8.4f}")
    print(f"  {'FAISS retrieval (ColBERT [CLS])':<36} {faiss_agg['ndcg_10']:>8.4f} {faiss_agg['mrr_10']:>8.4f}")
    print(f"  {'BM25 + FAISS (RRF k=60)':<36} {hybrid_agg['ndcg_10']:>8.4f} {hybrid_agg['mrr_10']:>8.4f}")
    print(sep)

    out = {
        "protocol":       "hybrid_retrieval",
        "n_products":     comp["n_products"],
        "n_queries":      len(queries),
        "bm25":           bm25_agg,
        "faiss":          faiss_agg,
        "hybrid":         hybrid_agg,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    local_out = os.path.join(RESULTS_DIR, "hybrid_eval.json")
    with open(local_out, "w") as f:
        json.dump(out, f, indent=2)
    gcs_out = f"gs://{DATA_BUCKET}/esci/results/hybrid_eval.json"
    upload(local_out, gcs_out)
    print(f"Saved -> {gcs_out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — save Protocol A results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(out: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    local_out = os.path.join(RESULTS_DIR, "final_evaluation.json")
    with open(local_out, "w") as f:
        json.dump(out, f, indent=2)
    gcs_out = f"gs://{DATA_BUCKET}/esci/results/final_evaluation.json"
    upload(local_out, gcs_out)
    print(f"\nSaved -> {gcs_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI submission
# ─────────────────────────────────────────────────────────────────────────────

def submit_vertex(args) -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION,
                    staging_bucket=f"gs://{MODELS_BUCKET}")

    extra = []
    if args.conditioned_ltr:
        extra.append("--conditioned_ltr")
    if args.query_router:
        extra.append("--query_router")

    protocol = args.protocol

    # Protocol B needs faiss — use GPU container (Linux, no OpenMP conflict)
    container = (
        "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"
        if protocol == "hybrid_retrieval"
        else "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"
    )
    reqs = [
        "rank-bm25>=0.2.2", "transformers==4.36.0", "lightgbm>=4.0.0",
        "scipy>=1.11.0", "scikit-learn>=1.3.0",
        "google-cloud-bigquery", "google-cloud-storage", "google-cloud-aiplatform",
        "pandas", "pyarrow", "db-dtypes",
    ]
    if protocol == "hybrid_retrieval":
        reqs.append("faiss-cpu==1.7.4")

    job = aiplatform.CustomTrainingJob(
        display_name=f"esci-evaluate-{protocol}",
        script_path="scripts/06_evaluate.py",
        container_uri=container,
        requirements=reqs,
    )
    job.run(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        replica_count=1,
        args=[
            "--protocol", protocol,
            "--sample_queries", str(args.sample_queries),
        ] + extra,
        sync=False,
    )
    print(f"Vertex job submitted. Protocol: {protocol}")
    print(f"Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit",         action="store_true")
    parser.add_argument("--protocol",       choices=["editorial_pool", "hybrid_retrieval"],
                        default="editorial_pool",
                        help="editorial_pool: re-rank pre-selected ESCI candidates (local). "
                             "hybrid_retrieval: BM25+FAISS from full catalog (Vertex).")
    parser.add_argument("--sample_queries", type=int, default=EVAL_SAMPLE_QUERIES)
    parser.add_argument("--conditioned_ltr", action="store_true",
                        help="Protocol A only: evaluate ρ-conditioned LTR (Contribution 1)")
    parser.add_argument("--query_router",   action="store_true",
                        help="Protocol A only: show NDCG breakdown by query length")
    args = parser.parse_args()

    if args.submit:
        submit_vertex(args)
    elif args.protocol == "hybrid_retrieval":
        run_hybrid_retrieval(args)
    else:
        out = run_editorial_pool(args)
        save_results(out)
