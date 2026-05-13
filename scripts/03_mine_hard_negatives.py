import os
import sys

import numpy as np
import pandas as pd
from google.cloud import bigquery
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROJECT_ID, BQ_DATASET, DATA_BUCKET,
    HARD_NEG_TOP_K, HARD_NEG_RANK_THRESHOLD,
)
from utils.gcs import upload

OUTPUT_PATH = f"gs://{DATA_BUCKET}/esci/triplets/train_triplets.parquet"


def load_train_data(client):
    query = f"""
    SELECT e.query_id, e.query, e.product_id, e.esci_label, e.gain,
           p.product_title, p.product_description
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
    WHERE e.split = 'train'
    """
    return client.query(query).to_dataframe()


def mine_triplets(df):
    triplets = []
    for query_id, group in df.groupby("query_id"):
        query     = group["query"].iloc[0]
        positives = group[group["esci_label"] == "E"]
        if positives.empty:
            continue

        corpus_texts = (
            group["product_title"].fillna("") + " " + group["product_description"].fillna("")
        ).tolist()
        product_ids = group["product_id"].tolist()

        tokenized = [t.lower().split() for t in corpus_texts]
        bm25      = BM25Okapi(tokenized)
        scores    = bm25.get_scores(query.lower().split())
        ranked_indices = np.argsort(scores)[::-1][:HARD_NEG_TOP_K]

        hard_negatives = []
        for rank, idx in enumerate(ranked_indices[:HARD_NEG_RANK_THRESHOLD]):
            pid = product_ids[idx]
            row = group[group["product_id"] == pid]
            if not row.empty and row["esci_label"].iloc[0] in ("I", "C"):
                hard_negatives.append(row.iloc[0])

        if not hard_negatives:
            continue

        for _, pos_row in positives.iterrows():
            for neg_row in hard_negatives:
                triplets.append({
                    "query":                pos_row["query"],
                    "positive_product_id":  pos_row["product_id"],
                    "negative_product_id":  neg_row["product_id"],
                    "positive_title":       pos_row["product_title"],
                    "negative_title":       neg_row["product_title"],
                    "positive_description": pos_row["product_description"],
                    "negative_description": neg_row["product_description"],
                })

    return pd.DataFrame(triplets)


def main():
    bq_client = bigquery.Client(project=PROJECT_ID)
    print("Loading training data...")
    df = load_train_data(bq_client)
    print(f"Mining hard negatives from {df['query_id'].nunique()} queries...")
    triplets_df = mine_triplets(df)
    print(f"Mined {len(triplets_df)} triplets")

    local_path = "/tmp/train_triplets.parquet"
    triplets_df.to_parquet(local_path, index=False)
    upload(local_path, OUTPUT_PATH)


if __name__ == "__main__":
    main()
