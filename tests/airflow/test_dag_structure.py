from __future__ import annotations

import ast

from tests.airflow import dag_call, dag_tree, keyword_node, keyword_value, task_calls

EXPECTED_TASKS = {
    "validate_environment",
    "run_data_pipeline",
    "dbt_debug",
    "dbt_run",
    "dbt_test",
    "run_ml_pipeline",
    "publish_ml_results",
    "validate_final_outputs",
}
EXPECTED_RETRIES = {
    "validate_environment": 0,
    "run_data_pipeline": 1,
    "dbt_debug": 0,
    "dbt_run": 1,
    "dbt_test": 0,
    "run_ml_pipeline": 0,
    "publish_ml_results": 1,
    "validate_final_outputs": 0,
}


def test_dag_has_manual_safe_configuration() -> None:
    call = dag_call(dag_tree())
    assert keyword_value(call, "dag_id") == "dataengine_full_pipeline"
    assert keyword_value(call, "schedule") is None
    assert keyword_value(call, "catchup") is False
    assert keyword_value(call, "max_active_runs") == 1
    assert keyword_value(call, "tags") == [
        "data-engineering",
        "aws",
        "bigquery",
        "dbt",
        "ml",
    ]
    start_date = keyword_node(call, "start_date")
    assert isinstance(start_date, ast.Call)
    assert ast.unparse(start_date) == "datetime(2026, 1, 1, tzinfo=timezone.utc)"


def test_dag_has_exact_task_ids_and_retries() -> None:
    tasks = task_calls(dag_tree())
    assert set(tasks) == EXPECTED_TASKS
    assert {
        task_name: keyword_value(call, "task_id") for task_name, call in tasks.items()
    } == {task_name: task_name for task_name in EXPECTED_TASKS}
    assert {
        task_name: keyword_value(call, "retries") for task_name, call in tasks.items()
    } == EXPECTED_RETRIES


def test_every_task_has_timeout_and_disables_xcom() -> None:
    for call in task_calls(dag_tree()).values():
        timeout = keyword_node(call, "execution_timeout")
        assert isinstance(timeout, ast.Call)
        assert keyword_value(call, "do_xcom_push") is False

