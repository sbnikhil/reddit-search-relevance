import yaml
from google.cloud import bigquery
from pathlib import Path


def load_config(config_path="config/settings.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_expertise_query():
    config = load_config()
    client = bigquery.Client(project=config["database"]["project_id"])

    sql_path = Path("data/schemas/expertise_schema.sql")
    query_text = sql_path.read_text()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_upvote_threshold", "INT64", config["features"]["min_upvote_threshold"]),
            bigquery.ScalarQueryParameter("training_cutoff_timestamp", "TIMESTAMP", "2025-01-01 00:00:00"),
        ]
    )

    final_query = query_text.replace("@source_table", config["database"]["source_table"])

    print(f"launching BigQuery job for {config['database']['dataset']}...")

    query_job = client.query(final_query, job_config=job_config)
    results = query_job.result()

    print(f"success! processed {query_job.total_bytes_processed} bytes.")

    for row in results:
        print(f"User: {row.author} | Subreddit: {row.subreddit} | Focus: {row.topical_focus_score:.2f}")
        break


if __name__ == "__main__":
    run_expertise_query()
