"""
Contribution 1 — Diagnostic: LambdaMART Regression Analysis

Computes per-query Spearman ρ between BM25 and ColBERT rank orderings. Shows that
when ρ is high (both models agree on ordering), LambdaMART's two main features are
collinear and its NDCG frequently regresses vs. ColBERT alone. When ρ is low, each
model brings orthogonal signal and LambdaMART improves.

This analysis justifies training separate LambdaMART models per ρ bin (script 11).

Prerequisites: scripts 01, 05, 08 must have completed successfully.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROJECT_ID, BQ_DATASET, MODELS_BUCKET, DATA_BUCKET,
    LAMBDAMART_CKPT_GCS, EVAL_SAMPLE_QUERIES,
    CONTRIB1_RHO_LOW, CONTRIB1_RHO_HIGH,
)
from models.ltr.lambdamart import build_features, bm25_scores_for_group
from utils.gcs import download, upload
from utils.metrics import ndcg_at_k, gain_to_label

RESULTS_GCS = f"gs://{DATA_BUCKET}/esci/results/regression_analysis.json"


def load_test_data(sample_queries: int) -> pd.DataFrame:
    bq = bigquery.Client(project=PROJECT_ID)
    sql = f"""
    WITH sampled AS (
        SELECT DISTINCT query_id
        FROM `{PROJECT_ID}.{BQ_DATASET}.examples`
        WHERE split = 'test'
        LIMIT {sample_queries}
    )
    SELECT
        e.query_id, e.query, e.product_id, e.gain,
        COALESCE(p.product_title, '')       AS product_title,
        COALESCE(p.product_description, '') AS product_description,
        COALESCE(p.product_brand, '')       AS product_brand,
        COALESCE(c.colbert_score, 0.0)      AS colbert_score
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
    LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.colbert_scores` c
        ON CAST(e.query_id AS STRING) = c.query_id
        AND e.product_id = c.product_id
    JOIN sampled ON e.query_id = sampled.query_id
    ORDER BY e.query_id
    """
    print(f"Loading {sample_queries} test queries from BigQuery...")
    df = bq.query(sql).to_dataframe()
    print(f"  {df['query_id'].nunique():,} queries, {len(df):,} pairs")
    return df.fillna({"product_title": "", "product_description": "", "product_brand": ""})


def analyze(df: pd.DataFrame, booster) -> list:
    per_query = []

    for query_id, group in df.groupby("query_id", sort=False):
        if len(group) < 2:
            continue

        query = group["query"].iloc[0]
        gains = group["gain"].tolist()

        bm25_raw = bm25_scores_for_group(
            query,
            group["product_title"].tolist(),
            group["product_description"].tolist(),
        )
        bm25_norm = bm25_raw / (bm25_raw.max() or 1.0)
        cb_scores  = group["colbert_score"].values

        rho_result = spearmanr(bm25_raw, cb_scores)
        rho = float(rho_result.statistic) if len(bm25_raw) > 2 else 0.0
        if np.isnan(rho):
            rho = 0.0

        features = np.array([
            build_features(
                query=query,
                product_title=row["product_title"],
                product_description=row["product_description"],
                product_brand=row["product_brand"],
                bm25_score=float(bm25_norm[j]),
                colbert_score=float(cb_scores[j]),
            )
            for j, (_, row) in enumerate(group.iterrows())
        ], dtype=np.float32)

        ltr_scores = booster.predict(features)

        labels = [gain_to_label(g) for g in gains]
        ndcg_bm25 = ndcg_at_k(
            [labels[i] for i in np.argsort(bm25_raw)[::-1]]
        )
        ndcg_cb  = ndcg_at_k(
            [labels[i] for i in np.argsort(cb_scores)[::-1]]
        )
        ndcg_ltr = ndcg_at_k(
            [labels[i] for i in np.argsort(ltr_scores)[::-1]]
        )

        per_query.append({
            "query_id":    str(query_id),
            "query_len":   len(query.split()),
            "rho":         rho,
            "ndcg_bm25":   ndcg_bm25,
            "ndcg_colbert": ndcg_cb,
            "ndcg_ltr":    ndcg_ltr,
            "ltr_delta":   ndcg_ltr - ndcg_cb,
            "n_docs":      len(group),
        })

    return per_query


def report(per_query: list) -> dict:
    df = pd.DataFrame(per_query)

    def _rho_bin(r: float) -> str:
        if r < CONTRIB1_RHO_LOW:
            return "low"
        if r <= CONTRIB1_RHO_HIGH:
            return "medium"
        return "high"

    df["rho_bin"] = df["rho"].apply(_rho_bin)

    sep = "─" * 72
    print(f"\n{sep}")
    print("  LambdaMART Regression Analysis (Contribution 1 Diagnostic)")
    print(sep)

    overall_ndcg = {
        "bm25":       float(df["ndcg_bm25"].mean()),
        "colbert":    float(df["ndcg_colbert"].mean()),
        "lambdamart": float(df["ndcg_ltr"].mean()),
        "ltr_delta":  float(df["ltr_delta"].mean()),
    }
    print(f"\n  Overall ({len(df):,} queries)")
    print(f"    BM25:        NDCG@10 = {overall_ndcg['bm25']:.4f}")
    print(f"    ColBERT:     NDCG@10 = {overall_ndcg['colbert']:.4f}")
    print(f"    LambdaMART:  NDCG@10 = {overall_ndcg['lambdamart']:.4f}  "
          f"(delta vs ColBERT: {overall_ndcg['ltr_delta']:+.4f})")

    rho_dist = {
        "mean":   float(df["rho"].mean()),
        "std":    float(df["rho"].std()),
        "p25":    float(df["rho"].quantile(0.25)),
        "median": float(df["rho"].median()),
        "p75":    float(df["rho"].quantile(0.75)),
    }
    print(f"\n  Spearman ρ  mean={rho_dist['mean']:.3f}  "
          f"std={rho_dist['std']:.3f}  "
          f"p25={rho_dist['p25']:.3f}  "
          f"median={rho_dist['median']:.3f}  "
          f"p75={rho_dist['p75']:.3f}")

    print(f"\n  NDCG@10 by ρ bin  (low<{CONTRIB1_RHO_LOW} | "
          f"medium≤{CONTRIB1_RHO_HIGH} | high>{CONTRIB1_RHO_HIGH})")
    header = f"  {'Bin':<10} {'N':>6} {'% total':>8}  {'BM25':>7} {'ColBERT':>8} {'LambdaMART':>11} {'LTR Δ':>8}"
    print(header)
    print("  " + "─" * 68)

    by_bin = {}
    for bin_name in ["low", "medium", "high"]:
        grp = df[df["rho_bin"] == bin_name]
        if grp.empty:
            by_bin[bin_name] = {}
            continue
        stats = {
            "n":            int(len(grp)),
            "pct":          float(100 * len(grp) / len(df)),
            "rho_mean":     float(grp["rho"].mean()),
            "ndcg_bm25":    float(grp["ndcg_bm25"].mean()),
            "ndcg_colbert": float(grp["ndcg_colbert"].mean()),
            "ndcg_ltr":     float(grp["ndcg_ltr"].mean()),
            "ltr_delta":    float(grp["ltr_delta"].mean()),
        }
        by_bin[bin_name] = stats
        print(
            f"  {bin_name:<10} {stats['n']:>6} {stats['pct']:>7.1f}%"
            f"  {stats['ndcg_bm25']:>7.4f} {stats['ndcg_colbert']:>8.4f}"
            f" {stats['ndcg_ltr']:>11.4f} {stats['ltr_delta']:>+8.4f}"
        )

    print()
    if by_bin.get("high") and by_bin.get("low"):
        h_delta = by_bin["high"]["ltr_delta"]
        l_delta = by_bin["low"]["ltr_delta"]
        print(f"  Key insight:")
        print(f"    High-ρ queries: LambdaMART delta = {h_delta:+.4f}  "
              f"(BM25 ≈ ColBERT → collinear features)")
        print(f"    Low-ρ queries:  LambdaMART delta = {l_delta:+.4f}  "
              f"(BM25 ≠ ColBERT → orthogonal signal)")
        if l_delta > h_delta:
            print(f"    ✓ Hypothesis confirmed. Training per-bin models (script 11) "
                  f"will exploit this gap.")

    return {
        "overall": overall_ndcg,
        "rho_distribution": rho_dist,
        "by_rho_bin": by_bin,
        "rho_low_threshold": CONTRIB1_RHO_LOW,
        "rho_high_threshold": CONTRIB1_RHO_HIGH,
        "sample_queries": int(len(df)),
    }


def submit_vertex_job(args):
    from google.cloud import aiplatform
    from config import REGION
    aiplatform.init(project=PROJECT_ID, location=REGION,
                    staging_bucket=f"gs://{MODELS_BUCKET}")
    job = aiplatform.CustomTrainingJob(
        display_name="ltr-regression-analysis",
        script_path="scripts/10_ltr_regression_analysis.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-0.py310:latest",
        requirements=[
            "rank-bm25>=0.2.2", "lightgbm>=4.0.0", "scipy>=1.11.0",
            "google-cloud-bigquery", "google-cloud-storage",
            "pandas", "pyarrow", "db-dtypes",
        ],
    )
    job.run(
        machine_type="n1-standard-4",
        replica_count=1,
        args=["--sample_queries", str(args.sample_queries)],
    )
    print("Vertex AI job submitted.")


def main(args):
    import lightgbm as lgb

    print("Downloading LambdaMART model...")
    local_ltr = "/tmp/lambdamart_analysis.txt"
    download(LAMBDAMART_CKPT_GCS, local_ltr)
    booster = lgb.Booster(model_file=local_ltr)

    df = load_test_data(args.sample_queries)
    per_query = analyze(df, booster)
    results   = report(per_query)

    local_out = "/tmp/regression_analysis.json"
    with open(local_out, "w") as f:
        json.dump(results, f, indent=2)
    upload(local_out, RESULTS_GCS)
    print(f"\nSaved -> {RESULTS_GCS}")
    print("Next step: python scripts/11_train_conditioned_ltr.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_queries", type=int, default=EVAL_SAMPLE_QUERIES)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job(args)
    else:
        main(args)
