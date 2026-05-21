"""
Contribution 1 — Training: Query-conditioned LambdaMART

Bins training queries by Spearman ρ between BM25 and ColBERT rank orderings, then
trains a separate LambdaMART model for each bin. At inference time the serving layer
computes ρ for the live query's candidates and routes to the appropriate model.

Bin definitions (from settings.yaml contribution1.*):
  low    (ρ < 0.30) — models mostly disagree; both features contribute orthogonally
  medium (ρ ≤ 0.60) — moderate agreement
  high   (ρ > 0.60) — models agree; collinear features; model focuses on lexical features

Prerequisites: scripts 01, 05, 08 must have completed (colbert_scores table populated).
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import bigquery, aiplatform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROJECT_ID, BQ_DATASET, MODELS_BUCKET, DATA_BUCKET, REGION,
    LAMBDAMART_PARAMS, LTR_NUM_BOOST_ROUND, LTR_EARLY_STOPPING_ROUNDS,
    CONTRIB1_RHO_LOW, CONTRIB1_RHO_HIGH,
    CONTRIB1_MODEL_LOW_GCS, CONTRIB1_MODEL_MED_GCS, CONTRIB1_MODEL_HIGH_GCS,
)
from models.ltr.lambdamart import build_features, bm25_scores_for_group, gain_to_label, FEATURE_NAMES
from utils.gcs import upload
from utils.metrics import ndcg_at_k, gain_to_label

BIN_MODELS = {
    "low":    CONTRIB1_MODEL_LOW_GCS,
    "medium": CONTRIB1_MODEL_MED_GCS,
    "high":   CONTRIB1_MODEL_HIGH_GCS,
}


def load_split(client: bigquery.Client, split: str) -> pd.DataFrame:
    sql = f"""
    SELECT
        e.query_id, e.query, e.gain,
        p.product_id, p.product_title, p.product_description, p.product_brand,
        COALESCE(c.colbert_score, 0.0) AS colbert_score
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
    LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.colbert_scores` c
        ON CAST(e.query_id AS STRING) = c.query_id
        AND e.product_id = c.product_id
    WHERE e.split = '{split}'
    ORDER BY e.query_id
    """
    print(f"  Querying BigQuery ({split})...")
    return client.query(sql).to_dataframe().fillna(
        {"product_title": "", "product_description": "", "product_brand": ""}
    )


def compute_rho(group: pd.DataFrame, query: str) -> float:
    bm25_raw = bm25_scores_for_group(
        query, group["product_title"].tolist(), group["product_description"].tolist()
    )
    cb_scores = group["colbert_score"].values
    if len(bm25_raw) < 3:
        return 0.0
    rho = float(spearmanr(bm25_raw, cb_scores).statistic)
    return 0.0 if np.isnan(rho) else rho


def assign_bin(rho: float) -> str:
    if rho < CONTRIB1_RHO_LOW:
        return "low"
    if rho <= CONTRIB1_RHO_HIGH:
        return "medium"
    return "high"


def build_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    feats, labels, groups = [], [], []
    for query_id, group in df.groupby("query_id", sort=False):
        if len(group) < 2:
            continue
        query = group["query"].iloc[0]
        bm25_raw = bm25_scores_for_group(
            query, group["product_title"].tolist(), group["product_description"].tolist()
        )
        bm25_norm = bm25_raw / (bm25_raw.max() or 1.0)
        for j, (_, row) in enumerate(group.iterrows()):
            feats.append(build_features(
                query=query,
                product_title=row["product_title"],
                product_description=row["product_description"],
                product_brand=row["product_brand"],
                bm25_score=float(bm25_norm[j]),
                colbert_score=float(row["colbert_score"]),
            ))
            labels.append(gain_to_label(float(row["gain"])))
        groups.append(len(group))
    return (
        np.array(feats, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        groups,
    )


def split_by_rho(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bin_assignments: dict[str, list] = {"low": [], "medium": [], "high": []}
    for query_id, group in df.groupby("query_id", sort=False):
        if len(group) < 2:
            continue
        rho = compute_rho(group, group["query"].iloc[0])
        b   = assign_bin(rho)
        bin_assignments[b].append(query_id)
    return {
        b: df[df["query_id"].isin(qids)]
        for b, qids in bin_assignments.items()
    }


def train_bin(
    bin_name: str,
    train_bin_df: pd.DataFrame,
    val_bin_df: pd.DataFrame,
    args,
) -> lgb.Booster:
    print(f"\n  ── {bin_name.upper()} ρ bin ──────────────────────────────")
    X_tr, y_tr, g_tr = build_dataset(train_bin_df)
    X_vl, y_vl, g_vl = build_dataset(val_bin_df)
    print(f"    Train: {len(g_tr):,} queries, {len(X_tr):,} examples")
    print(f"    Val:   {len(g_vl):,} queries, {len(X_vl):,} examples")

    train_set = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=FEATURE_NAMES)
    val_set   = lgb.Dataset(X_vl, label=y_vl, group=g_vl,
                            reference=train_set, feature_name=FEATURE_NAMES)

    booster = lgb.train(
        LAMBDAMART_PARAMS,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
        ],
    )
    print(f"    Best iteration: {booster.best_iteration}")
    return booster


def evaluate_bin(booster: lgb.Booster, val_df: pd.DataFrame) -> float:
    ndcgs = []
    for query_id, group in val_df.groupby("query_id", sort=False):
        if len(group) < 2:
            continue
        query = group["query"].iloc[0]
        gains = group["gain"].tolist()
        bm25_raw  = bm25_scores_for_group(
            query, group["product_title"].tolist(), group["product_description"].tolist()
        )
        bm25_norm = bm25_raw / (bm25_raw.max() or 1.0)
        feats = np.array([
            build_features(
                query=query,
                product_title=row["product_title"],
                product_description=row["product_description"],
                product_brand=row["product_brand"],
                bm25_score=float(bm25_norm[j]),
                colbert_score=float(row["colbert_score"]),
            )
            for j, (_, row) in enumerate(group.iterrows())
        ], dtype=np.float32)
        scores = booster.predict(feats)
        labels = [gain_to_label(g) for g in gains]
        ndcg   = ndcg_at_k([labels[i] for i in np.argsort(scores)[::-1]])
        ndcgs.append(ndcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def train(args):
    client = bigquery.Client(project=PROJECT_ID)

    print("Loading training data...")
    train_df = load_split(client, "train")
    print(f"  {train_df['query_id'].nunique():,} queries | {len(train_df):,} examples")

    print("\nLoading validation data (5 000-query cap)...")
    val_df = load_split(client, "test")
    val_qids = val_df["query_id"].unique()[:5000]
    val_df = val_df[val_df["query_id"].isin(val_qids)]
    print(f"  {val_df['query_id'].nunique():,} queries | {len(val_df):,} examples")

    print("\nAssigning training queries to ρ bins...")
    train_bins = split_by_rho(train_df)
    val_bins   = split_by_rho(val_df)

    for b, bdf in train_bins.items():
        pct = 100 * bdf["query_id"].nunique() / train_df["query_id"].nunique()
        print(f"  {b:<8} {bdf['query_id'].nunique():>6,} queries ({pct:.1f}%)")

    results = {}
    for bin_name in ["low", "medium", "high"]:
        tr_bin = train_bins[bin_name]
        vl_bin = val_bins.get(bin_name, val_df.head(0))
        if tr_bin.empty:
            print(f"\n  WARNING: {bin_name} bin has no training data — skipping.")
            continue

        booster   = train_bin(bin_name, tr_bin, vl_bin if not vl_bin.empty else tr_bin.head(100), args)
        bin_ndcg  = evaluate_bin(booster, vl_bin) if not vl_bin.empty else 0.0

        local_path = f"/tmp/lambdamart_{bin_name}.txt"
        booster.save_model(local_path)
        upload(local_path, BIN_MODELS[bin_name])
        print(f"    Saved  -> {BIN_MODELS[bin_name]}")
        print(f"    Val NDCG@10 (within-bin): {bin_ndcg:.4f}")
        results[bin_name] = {"ndcg_10": bin_ndcg, "n_queries": tr_bin["query_id"].nunique()}

        try:
            aiplatform.init(project=PROJECT_ID, location=REGION,
                            experiment="conditioned-ltr-esci")
            with aiplatform.start_run(run=f"ltr-{bin_name}-{int(time.time())}"):
                aiplatform.log_metrics({
                    "bin":      bin_name,
                    "ndcg_10":  bin_ndcg,
                    "n_queries": tr_bin["query_id"].nunique(),
                })
        except Exception as e:
            print(f"    Metric logging skipped: {e}")

    sep = "─" * 50
    print(f"\n{sep}")
    print("  Conditioned LTR — Summary")
    print(sep)
    for b, r in results.items():
        print(f"  {b:<8}  NDCG@10={r['ndcg_10']:.4f}  queries={r['n_queries']:,}")
    print(f"\nNext step: python scripts/06_evaluate.py --conditioned_ltr")


def submit_vertex_job(args):
    aiplatform.init(project=PROJECT_ID, location=REGION,
                    staging_bucket=f"gs://{MODELS_BUCKET}")
    job = aiplatform.CustomTrainingJob(
        display_name="conditioned-ltr-training",
        script_path="scripts/11_train_conditioned_ltr.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-0.py310:latest",
        requirements=[
            "lightgbm>=4.0.0", "rank-bm25>=0.2.2", "scipy>=1.11.0",
            "google-cloud-bigquery", "google-cloud-storage", "google-cloud-aiplatform",
            "pandas", "pyarrow", "db-dtypes",
        ],
    )
    job.run(
        machine_type="n1-standard-8",
        replica_count=1,
        args=[
            "--num_boost_round", str(args.num_boost_round),
            "--early_stopping",  str(args.early_stopping),
        ],
    )
    print("Vertex AI job submitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit",          action="store_true")
    parser.add_argument("--num_boost_round", type=int, default=LTR_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping",  type=int, default=LTR_EARLY_STOPPING_ROUNDS)
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job(args)
    else:
        train(args)
