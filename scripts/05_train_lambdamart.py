import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from google.cloud import bigquery, aiplatform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROJECT_ID, BQ_DATASET, MODELS_BUCKET, REGION,
    LAMBDAMART_CKPT_GCS, LTR_NUM_BOOST_ROUND, LTR_EARLY_STOPPING_ROUNDS,
)
from models.ltr.lambdamart import (
    FEATURE_NAMES, PARAMS, gain_to_label, build_features, bm25_scores_for_group,
)
from utils.gcs import upload

MODEL_GCS_PATH = LAMBDAMART_CKPT_GCS
FI_GCS_PATH    = f"gs://{MODELS_BUCKET}/ltr/feature_importance.json"


def load_split(client: bigquery.Client, split: str, sample_queries: int | None = None) -> pd.DataFrame:
    sample_clause = ""
    if sample_queries:
        sample_clause = f"""
        AND e.query_id IN (
            SELECT DISTINCT query_id
            FROM `{PROJECT_ID}.{BQ_DATASET}.examples`
            WHERE split = '{split}'
            LIMIT {sample_queries}
        )"""

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
    WHERE e.split = '{split}'{sample_clause}
    ORDER BY e.query_id
    """
    print(f"  Querying BigQuery ({split})...")
    return client.query(sql).to_dataframe()


def build_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    df = df.fillna({"product_title": "", "product_description": "", "product_brand": ""})
    features_list, labels_list, groups = [], [], []

    for query_id, group in df.groupby("query_id", sort=False):
        if len(group) < 2:
            continue
        query = group["query"].iloc[0]
        bm25 = bm25_scores_for_group(
            query,
            group["product_title"].tolist(),
            group["product_description"].tolist(),
        )
        for (_, row), score in zip(group.iterrows(), bm25):
            features_list.append(build_features(
                query=query,
                product_title=row["product_title"],
                product_description=row["product_description"],
                product_brand=row["product_brand"],
                bm25_score=float(score),
                colbert_score=float(row["colbert_score"]),
            ))
            labels_list.append(gain_to_label(float(row["gain"])))
        groups.append(len(group))

    return (
        np.array(features_list, dtype=np.float32),
        np.array(labels_list, dtype=np.int32),
        groups,
    )


def train(args) -> None:
    import lightgbm as lgb

    client = bigquery.Client(project=PROJECT_ID)

    print("Loading training data...")
    train_df = load_split(client, "train")
    print(f"  {train_df['query_id'].nunique():,} queries | {len(train_df):,} examples")

    print("Loading validation data (5 000-query cap)...")
    val_df = load_split(client, "test", sample_queries=5000)
    print(f"  {val_df['query_id'].nunique():,} queries | {len(val_df):,} examples")

    print("\nBuilding train dataset...")
    X_train, y_train, g_train = build_dataset(train_df)
    print(f"  {X_train.shape[0]:,} examples across {len(g_train):,} queries")

    print("Building val dataset...")
    X_val, y_val, g_val = build_dataset(val_df)
    print(f"  {X_val.shape[0]:,} examples across {len(g_val):,} queries")

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train, feature_name=FEATURE_NAMES)
    val_set   = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_set, feature_name=FEATURE_NAMES)

    print(f"\nTraining LambdaMART (max {args.num_boost_round} rounds, "
          f"early stopping after {args.early_stopping} no-improve)...")
    model = lgb.train(
        PARAMS, train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
        ],
    )

    local_model = "/tmp/lambdamart.txt"
    model.save_model(local_model)
    upload(local_model, MODEL_GCS_PATH)
    print(f"\nModel saved -> {MODEL_GCS_PATH}")
    print(f"Best iteration: {model.best_iteration}")

    fi = dict(sorted(
        zip(FEATURE_NAMES, model.feature_importance("gain").tolist()),
        key=lambda x: -x[1],
    ))
    for name, score in fi.items():
        bar = "█" * int(score / max(fi.values()) * 30)
        print(f"  {name:<28} {bar} {score:.1f}")

    local_fi = "/tmp/feature_importance.json"
    with open(local_fi, "w") as f:
        json.dump(fi, f, indent=2)
    upload(local_fi, FI_GCS_PATH)

    aiplatform.init(project=PROJECT_ID, location=REGION, experiment="lambdamart-esci")
    try:
        with aiplatform.start_run(run=f"lambdamart-{int(time.time())}"):
            aiplatform.log_metrics({
                "n_train_queries": len(g_train),
                "n_train_examples": int(X_train.shape[0]),
                "best_iteration": model.best_iteration,
            })
    except Exception as e:
        print(f"Metric logging skipped: {e}")

    print("\nLambdaMART training complete.")


def submit_vertex_job(args) -> None:
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{MODELS_BUCKET}",
    )
    job = aiplatform.CustomTrainingJob(
        display_name="lambdamart-esci-training",
        script_path="scripts/05_train_lambdamart.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-0.py310:latest",
        requirements=[
            "lightgbm>=4.0.0", "rank-bm25>=0.2.2", "google-cloud-bigquery",
            "google-cloud-storage", "google-cloud-aiplatform", "pandas", "pyarrow", "db-dtypes",
        ],
    )
    job.run(
        machine_type="n1-standard-8",
        replica_count=1,
        args=["--num_boost_round", str(args.num_boost_round), "--early_stopping", str(args.early_stopping)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--num_boost_round", type=int, default=LTR_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping",  type=int, default=LTR_EARLY_STOPPING_ROUNDS)
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job(args)
    else:
        train(args)
