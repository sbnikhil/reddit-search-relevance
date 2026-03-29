from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "esci_ml_pipeline",
    default_args=default_args,
    schedule_interval="@weekly",
    catchup=False,
    tags=["esci", "search", "ml"],
) as dag:

    load_bigquery = BashOperator(
        task_id="load_esci_bigquery",
        bash_command="python scripts/01_load_bigquery.py",
    )

    mine_hard_negatives = BashOperator(
        task_id="mine_hard_negatives",
        bash_command="python scripts/03_mine_hard_negatives.py",
    )

    train_colbert = BashOperator(
        task_id="train_colbert",
        bash_command="python scripts/04_train_colbert.py --submit",
    )

    train_lambdamart = BashOperator(
        task_id="train_lambdamart",
        bash_command="python scripts/05_train_lambdamart.py --submit",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="python scripts/06_evaluate.py",
    )

    deploy = BashOperator(
        task_id="deploy",
        bash_command="python scripts/07_deploy_vertex.py",
    )

    load_bigquery >> mine_hard_negatives >> train_colbert >> train_lambdamart >> evaluate >> deploy
