from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.process_topical_reputation import run_expertise_query
from data.pipelines.utility_extractor import run as run_utility_pipeline
from scripts.ingest_to_solr import RedditIndexer
from scripts.train_ranker import train as start_training

default_args = {
    'owner': 'mle_intern',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'reddit_relevance_engine_v1',
    default_args=default_args,
    description='End-to-end Reddit Search Relevance Pipeline',
    schedule_interval=timedelta(days=1), 
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['reddit', 'relevance', 'mlops'],
) as dag:

    task_reputation = PythonOperator(
        task_id='process_topical_reputation',
        python_callable=run_expertise_query,
    )

    task_utility = PythonOperator(
        task_id='extract_utility_signals',
        python_callable=run_utility_pipeline,
    )

    task_ingestion = PythonOperator(
        task_id='ingest_to_solr',
        python_callable=lambda: print("Ingestion task - implement batch_index call"),
    )

    task_training = PythonOperator(
        task_id='train_relevance_model',
        python_callable=start_training,
    )

    [task_reputation, task_utility] >> task_ingestion >> task_training