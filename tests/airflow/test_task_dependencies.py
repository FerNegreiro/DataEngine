from tests.airflow import dag_tree, dependency_chain


def test_task_dependencies_are_exactly_linear() -> None:
    assert dependency_chain(dag_tree()) == [
        "validate_environment",
        "run_data_pipeline",
        "dbt_debug",
        "dbt_run",
        "dbt_test",
        "run_ml_pipeline",
        "publish_ml_results",
        "validate_final_outputs",
    ]

