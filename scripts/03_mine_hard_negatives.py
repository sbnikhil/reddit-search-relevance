import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from rank_bm25 import BM25Okapi

PROJECT_ID = "reddit-search-relevance-485717"
OUTPUT_PATH = "gs://reddit-search-relevance-data/esci/triplets/train_triplets.parquet"
TOP_K_BM25 = 50
HARD_NEG_RANK_THRESHOLD = 20


def load_train_data(client):
    query = f"""
    SELECT e.query_id, e.query, e.product_id, e.esci_label, e.gain,
           p.product_title, p.product_description
    FROM `{PROJECT_ID}.esci_search.examples` e
    JOIN `{PROJECT_ID}.esci_search.products` p USING (product_id)
    WHERE e.split = 'train'
    """
    return client.query(query).to_dataframe()


def mine_triplets(df):
    triplets = []

    for query_id, group in df.groupby("query_id"):
        query = group["query"].iloc[0]
        positives = group[group["esci_label"] == "E"]
        if positives.empty:
            continue

        corpus_texts = (
            group["product_title"].fillna("") + " " + group["product_description"].fillna("")
        ).tolist()
        product_ids = group["product_id"].tolist()

        tokenized = [t.lower().split() for t in corpus_texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        ranked_indices = np.argsort(scores)[::-1][:TOP_K_BM25]

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
                    "query": query,
                    "positive_product_id": pos_row["product_id"],
                    "negative_product_id": neg_row["product_id"],
                    "positive_title": pos_row["product_title"],
                    "negative_title": neg_row["product_title"],
                    "positive_description": pos_row["product_description"],
                    "negative_description": neg_row["product_description"],
                })

    return pd.DataFrame(triplets)


def upload_to_gcs(df, gcs_path):
    local_path = "/tmp/train_triplets.parquet"
    df.to_parquet(local_path, index=False)

    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    blob_path = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {len(df)} triplets to {gcs_path}")


def main():
    bq_client = bigquery.Client(project=PROJECT_ID)
    print("Loading training data...")
    df = load_train_data(bq_client)
    print(f"Mining hard negatives from {df['query_id'].nunique()} queries...")
    triplets_df = mine_triplets(df)
    print(f"Mined {len(triplets_df)} triplets")
    upload_to_gcs(triplets_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
