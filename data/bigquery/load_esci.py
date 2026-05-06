import os
import sys

from google.cloud import bigquery

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROJECT_ID, BQ_DATASET, DATA_BUCKET

GCS_PREFIX = f"gs://{DATA_BUCKET}/esci/raw"

PRODUCTS_SCHEMA = [
    bigquery.SchemaField("product_id",          "STRING"),
    bigquery.SchemaField("product_title",        "STRING"),
    bigquery.SchemaField("product_description",  "STRING"),
    bigquery.SchemaField("product_bullet_point", "STRING"),
    bigquery.SchemaField("product_brand",        "STRING"),
    bigquery.SchemaField("product_color",        "STRING"),
    bigquery.SchemaField("product_locale",       "STRING"),
]

EXAMPLES_SCHEMA = [
    bigquery.SchemaField("query_id",    "STRING"),
    bigquery.SchemaField("query",       "STRING"),
    bigquery.SchemaField("product_id",  "STRING"),
    bigquery.SchemaField("esci_label",  "STRING"),
    bigquery.SchemaField("split",       "STRING"),
    bigquery.SchemaField("gain",        "FLOAT64"),
]

SPLITS_SCHEMA = [
    bigquery.SchemaField("query_id",       "STRING"),
    bigquery.SchemaField("split",          "STRING"),
    bigquery.SchemaField("query",          "STRING"),
    bigquery.SchemaField("product_locale", "STRING"),
]


def load_table(client, table_id, gcs_uri, schema=None, write_disposition="WRITE_TRUNCATE"):
    table_ref  = f"{PROJECT_ID}.{BQ_DATASET}.{table_id}"
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    if gcs_uri.endswith(".parquet"):
        job_config.source_format = bigquery.SourceFormat.PARQUET
        job_config.autodetect    = True
    else:
        job_config.source_format    = bigquery.SourceFormat.CSV
        job_config.skip_leading_rows = 1
        job_config.autodetect       = True
    if schema:
        job_config.schema    = schema
        job_config.autodetect = False
    load_job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
    load_job.result()
    print(f"Loaded {table_id}: {client.get_table(table_ref).num_rows} rows")


def add_gain_column(client):
    query = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{BQ_DATASET}.examples` AS
    SELECT
      example_id, query, query_id, product_id, product_locale,
      esci_label, small_version, large_version, split,
      CASE esci_label
        WHEN 'E' THEN 1.0
        WHEN 'S' THEN 0.1
        WHEN 'C' THEN 0.01
        WHEN 'I' THEN 0.0
        ELSE 0.0
      END AS gain
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples`
    """
    client.query(query).result()
    print("Added gain column to examples table")


def validate(client):
    queries = [
        f"SELECT COUNT(*) as cnt FROM `{PROJECT_ID}.{BQ_DATASET}.products`",
        f"SELECT COUNT(*) as cnt FROM `{PROJECT_ID}.{BQ_DATASET}.examples`",
        f"SELECT esci_label, COUNT(*) as cnt FROM `{PROJECT_ID}.{BQ_DATASET}.examples` GROUP BY 1",
        f"SELECT split, COUNT(*) as cnt FROM `{PROJECT_ID}.{BQ_DATASET}.examples` GROUP BY 1",
    ]
    for q in queries:
        print(f"\n{q}")
        for row in client.query(q).result():
            print(dict(row))


def main():
    client  = bigquery.Client(project=PROJECT_ID)
    dataset = bigquery.Dataset(f"{PROJECT_ID}.{BQ_DATASET}")
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)

    load_table(client, "products", f"{GCS_PREFIX}/shopping_queries_dataset_products.parquet")
    load_table(client, "examples", f"{GCS_PREFIX}/shopping_queries_dataset_examples.parquet")
    add_gain_column(client)
    load_table(client, "splits",   f"{GCS_PREFIX}/shopping_queries_dataset_sources.csv")
    validate(client)


if __name__ == "__main__":
    main()
