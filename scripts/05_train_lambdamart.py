import argparse
import json
import time
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage, aiplatform

PROJECT_ID = "reddit-search-relevance-485717"
BQ_DATASET = "esci_search"
MODEL_GCS_PATH = "gs://reddit-search-relevance-models/ltr/lambdamart.txt"
FI_GCS_PATH = "gs://reddit-search-relevance-models/ltr/feature_importance.json"

FEATURE_NAMES = [
    "bm25_score",
    "colbert_score",
    "title_query_overlap",
    "desc_query_overlap",
    "title_bigram_overlap",
    "brand_match",
    "title_length",
    "desc_length",
    "has_description",
    "query_length",
]

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 5, 10],
    "label_gain": [0, 1, 3, 7],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}


def gain_to_label(gain: float) -> int:
    if gain >= 1.0:
        return 3
    if gain >= 0.1:
        return 2
    if gain >= 0.01:
        return 1
    return 0


def _bigrams(tokens: list) -> set:
    return set(zip(tokens, tokens[1:]))


def build_features(query, product_title, product_description, product_brand,
                   bm25_score=0.0, colbert_score=0.0):
    q_tokens = query.lower().split()
    t_tokens = (product_title or "").lower().split()
    d_tokens = (product_description or "").lower().split()
    brand = (product_brand or "").lower()

    q_set = set(q_tokens)
    q_bg = _bigrams(q_tokens)
    t_bg = _bigrams(t_tokens)

    return [
        bm25_score,
        colbert_score,
        len(q_set & set(t_tokens)) / max(len(q_set), 1),
        len(q_set & set(d_tokens)) / max(len(q_set), 1),
        len(q_bg & t_bg) / max(len(q_bg), 1) if q_bg else 0.0,
        float(bool(brand) and brand in query.lower()),
        np.log1p(len(t_tokens)),
        np.log1p(len(d_tokens)),
        float(bool(d_tokens)),
        float(len(q_tokens)),
    ]


def bm25_scores_for_group(query: str, titles: list, descriptions: list) -> np.ndarray:
    from rank_bm25 import BM25Okapi
    corpus = [
        ((t or "") + " " + (d or "")).lower().split() or [""]
        for t, d in zip(titles, descriptions)
    ]
    scores = BM25Okapi(corpus).get_scores(query.lower().split())
    max_s = scores.max()
    return scores / max_s if max_s > 0 else scores


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
        e.query_id,
        e.query,
        e.gain,
        p.product_id,
        p.product_title,
        p.product_description,
        p.product_brand,
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


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    bucket = gcs_path.replace("gs://", "").split("/")[0]
    blob = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)


def train(args) -> None:
    import lightgbm as lgb

    client = bigquery.Client(project=PROJECT_ID)

    print("Loading training data...")
    train_df = load_split(client, "train")
    print(f"  {train_df['query_id'].nunique():,} queries | {len(train_df):,} examples")

    print("Loading validation data (5 000-query cap)...")
    val_df = load_split(client, "test", sample_queries=5000)
    print(f"  {val_df['query_id'].nunique():,} queries | {len(val_df):,} examples")

    print("\nBuilding train dataset (BM25 per query)...")
    X_train, y_train, g_train = build_dataset(train_df)
    print(f"  {X_train.shape[0]:,} examples across {len(g_train):,} queries")

    print("Building val dataset...")
    X_val, y_val, g_val = build_dataset(val_df)
    print(f"  {X_val.shape[0]:,} examples across {len(g_val):,} queries")

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(
        X_val, label=y_val, group=g_val, reference=train_set, feature_name=FEATURE_NAMES
    )

    print(f"\nTraining LambdaMART (max {args.num_boost_round} rounds, "
          f"early stopping after {args.early_stopping} no-improve)...")
    model = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
        ],
    )

    local_model = "/tmp/lambdamart.txt"
    model.save_model(local_model)
    upload_to_gcs(local_model, MODEL_GCS_PATH)
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
    upload_to_gcs(local_fi, FI_GCS_PATH)
    print(f"Feature importance saved -> {FI_GCS_PATH}")

    aiplatform.init(project=PROJECT_ID, location="us-central1", experiment="lambdamart-esci")
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
        location="us-central1",
        staging_bucket="gs://reddit-search-relevance-models",
    )
    job = aiplatform.CustomTrainingJob(
        display_name="lambdamart-esci-training",
        script_path="scripts/05_train_lambdamart.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-0.py310:latest",
        requirements=[
            "lightgbm>=4.0.0",
            "rank-bm25>=0.2.2",
            "google-cloud-bigquery",
            "google-cloud-storage",
            "google-cloud-aiplatform",
            "pandas",
            "pyarrow",
            "db-dtypes",
        ],
    )
    job.run(
        machine_type="n1-standard-8",
        replica_count=1,
        args=[
            "--num_boost_round", str(args.num_boost_round),
            "--early_stopping", str(args.early_stopping),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Submit to Vertex AI")
    parser.add_argument("--num_boost_round", type=int, default=500)
    parser.add_argument("--early_stopping", type=int, default=50)
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job(args)
    else:
        train(args)
