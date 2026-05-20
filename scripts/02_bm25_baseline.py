import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from google.cloud import bigquery
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ID, BQ_DATASET, DATA_BUCKET
from utils.gcs import upload
from utils.metrics import ndcg_at_k, mrr_at_k, precision_at_k

RESULTS_DIR     = "results"
RESULTS_GCS_PATH = f"gs://{DATA_BUCKET}/esci/results/baseline_bm25.json"


def load_test_data(client, sample_queries=5000):
    query = f"""
    SELECT e.query_id, e.query, e.product_id, e.gain,
           p.product_title, p.product_description
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
    WHERE e.split = 'test'
    """
    df = client.query(query).to_dataframe()
    unique_queries = df["query_id"].unique()
    if len(unique_queries) > sample_queries:
        sampled = np.random.choice(unique_queries, sample_queries, replace=False)
        df = df[df["query_id"].isin(sampled)]
    return df


def evaluate_bm25(df):
    ndcgs, mrrs, precs = [], [], []
    for query_id, group in df.groupby("query_id"):
        query = group["query"].iloc[0]
        corpus_texts = (
            group["product_title"].fillna("") + " " + group["product_description"].fillna("")
        ).tolist()
        product_ids  = group["product_id"].tolist()
        gains        = dict(zip(product_ids, group["gain"].tolist()))
        ideal_gains  = list(gains.values())

        tokenized  = [t.lower().split() for t in corpus_texts]
        bm25       = BM25Okapi(tokenized)
        scores     = bm25.get_scores(query.lower().split())
        ranked_ids = [product_ids[i] for i in np.argsort(scores)[::-1]]
        ranked_gains = [gains.get(pid, 0.0) for pid in ranked_ids]

        ndcgs.append(ndcg_at_k(ranked_gains, ideal_gains, 10))
        mrrs.append(mrr_at_k(ranked_gains, 10))
        precs.append(precision_at_k(ranked_gains, 10))

    return np.mean(ndcgs), np.mean(mrrs), np.mean(precs), len(ndcgs)


def main(sample=5000):
    client = bigquery.Client(project=PROJECT_ID)
    print(f"Loading test data (sample={sample} queries)...")
    df = load_test_data(client, sample)
    print(f"Evaluating BM25 on {df['query_id'].nunique()} queries...")
    ndcg, mrr, prec, n_queries = evaluate_bm25(df)

    print(f"BM25 Baseline")
    print(f"NDCG@10:      {ndcg:.4f}")
    print(f"MRR@10:       {mrr:.4f}")
    print(f"Precision@10: {prec:.4f}")
    print(f"Queries eval: {n_queries:,}")

    results = {"ndcg_10": ndcg, "mrr_10": mrr, "precision_10": prec, "n_queries": n_queries}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    local_path = f"{RESULTS_DIR}/baseline_bm25.json"
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2)

    upload(local_path, RESULTS_GCS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()
    main(args.sample)
