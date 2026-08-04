from __future__ import annotations

from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.bash import BashOperator

from airflow import DAG

DEFAULT_ARGS = {
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="dataengine_full_pipeline",
    description="Orquestra o pipeline completo do DataEngine sob execução manual.",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["data-engineering", "aws", "bigquery", "dbt", "ml"],
) as dag:
    validate_environment = BashOperator(
        task_id="validate_environment",
        bash_command="python -m pipelines.orchestration.validate_environment",
        retries=0,
        execution_timeout=timedelta(minutes=10),
        do_xcom_push=False,
    )

    run_data_pipeline = BashOperator(
        task_id="run_data_pipeline",
        bash_command=(
            "cd /opt/airflow/project && "
            "python -m pipelines.run_pipeline"
        ),
        retries=1,
        execution_timeout=timedelta(hours=2),
        do_xcom_push=False,
    )

    dbt_debug = BashOperator(
        task_id="dbt_debug",
        bash_command="cd /opt/airflow/project/dataengine_dbt && dbt debug",
        retries=0,
        execution_timeout=timedelta(minutes=15),
        do_xcom_push=False,
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd /opt/airflow/project/dataengine_dbt && dbt run",
        retries=1,
        execution_timeout=timedelta(hours=1),
        do_xcom_push=False,
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /opt/airflow/project/dataengine_dbt && dbt test",
        retries=0,
        execution_timeout=timedelta(hours=1),
        do_xcom_push=False,
    )

    run_ml_pipeline = BashOperator(
        task_id="run_ml_pipeline",
        bash_command=(
            "cd /opt/airflow/project && "
            "python -m pipelines.machine_learning.run_ml_pipeline "
            "--prepare-publication"
        ),
        retries=0,
        execution_timeout=timedelta(hours=4),
        do_xcom_push=False,
    )

    publish_ml_results = BashOperator(
        task_id="publish_ml_results",
        bash_command=(
            "cd /opt/airflow/project && "
            "python -m pipelines.machine_learning.publish_ml_results"
        ),
        retries=1,
        execution_timeout=timedelta(hours=1),
        do_xcom_push=False,
    )

    validate_final_outputs = BashOperator(
        task_id="validate_final_outputs",
        bash_command="python -m pipelines.orchestration.validate_final_outputs",
        retries=0,
        execution_timeout=timedelta(minutes=30),
        do_xcom_push=False,
    )

    (
        validate_environment
        >> run_data_pipeline
        >> dbt_debug
        >> dbt_run
        >> dbt_test
        >> run_ml_pipeline
        >> publish_ml_results
        >> validate_final_outputs
    )
