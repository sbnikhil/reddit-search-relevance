import argparse
import json
import os
import math
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from rank_bm25 import BM25Okapi

PROJECT_ID = "reddit-search-relevance-485717"
RESULTS_BUCKET = "reddit-search-relevance-data"
RESULTS_GCS_PATH = "esci/results/baseline_bm25.json"
RESULTS_DIR = "results"


def dcg_at_k(scores, k):
    scores = scores[:k]
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(pred_gains, ideal_gains, k):
    ideal = sorted(ideal_gains, reverse=True)
    dcg = dcg_at_k(pred_gains, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(pred_gains, k):
    for i, g in enumerate(pred_gains[:k]):
        if g >= 1.0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(pred_gains, k):
    return sum(1 for g in pred_gains[:k] if g > 0) / k


def load_test_data(client, sample_queries=5000):
    query = f"""
    SELECT e.query_id, e.query, e.product_id, e.gain,
           p.product_title, p.product_description
    FROM `{PROJECT_ID}.esci_search.examples` e
    JOIN `{PROJECT_ID}.esci_search.products` p USING (product_id)
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
        product_ids = group["product_id"].tolist()
        gains = dict(zip(product_ids, group["gain"].tolist()))
        ideal_gains = list(gains.values())

        tokenized = [t.lower().split() for t in corpus_texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
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

    print("BM25 Baseline")
    print(f"NDCG@10:      {ndcg:.4f}")
    print(f"MRR@10:       {mrr:.4f}")
    print(f"Precision@10: {prec:.4f}")
    print(f"Queries eval: {n_queries:,}")

    results = {"ndcg_10": ndcg, "mrr_10": mrr, "precision_10": prec, "n_queries": n_queries}

    # Save locally if possible, always save to GCS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    local_path = f"{RESULTS_DIR}/baseline_bm25.json"
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2)

    gcs_client = storage.Client(project=PROJECT_ID)
    gcs_client.bucket(RESULTS_BUCKET).blob(RESULTS_GCS_PATH).upload_from_filename(local_path)
    print(f"Saved to gs://{RESULTS_BUCKET}/{RESULTS_GCS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()
    main(args.sample)
