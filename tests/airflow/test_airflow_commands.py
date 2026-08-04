from __future__ import annotations

import re

from tests.airflow import PROJECT_ROOT, dag_tree, keyword_value, task_calls

EXPECTED_COMMANDS = {
    "validate_environment": "python -m pipelines.orchestration.validate_environment",
    "run_data_pipeline": (
        "cd /opt/airflow/project && python -m pipelines.run_pipeline"
    ),
    "dbt_debug": "cd /opt/airflow/project/dataengine_dbt && dbt debug",
    "dbt_run": "cd /opt/airflow/project/dataengine_dbt && dbt run",
    "dbt_test": "cd /opt/airflow/project/dataengine_dbt && dbt test",
    "run_ml_pipeline": (
        "cd /opt/airflow/project && "
        "python -m pipelines.machine_learning.run_ml_pipeline --prepare-publication"
    ),
    "publish_ml_results": (
        "cd /opt/airflow/project && "
        "python -m pipelines.machine_learning.publish_ml_results"
    ),
    "validate_final_outputs": (
        "python -m pipelines.orchestration.validate_final_outputs"
    ),
}


def _commands() -> dict[str, str]:
    return {
        task_name: str(keyword_value(call, "bash_command"))
        for task_name, call in task_calls(dag_tree()).items()
    }


def test_commands_are_exact_and_reference_real_modules() -> None:
    commands = _commands()
    assert commands == EXPECTED_COMMANDS
    for command in commands.values():
        for module_name in re.findall(r"python -m ([A-Za-z0-9_.]+)", command):
            module_path = PROJECT_ROOT / (module_name.replace(".", "/") + ".py")
            package_main = PROJECT_ROOT / module_name.replace(".", "/") / "__main__.py"
            assert module_path.is_file() or package_main.is_file()
    assert (PROJECT_ROOT / "dataengine_dbt" / "dbt_project.yml").is_file()


def test_publication_occurs_once_and_ml_generation_does_not_publish() -> None:
    commands = _commands()
    publisher = EXPECTED_COMMANDS["publish_ml_results"]
    assert list(commands.values()).count(publisher) == 1
    assert "--prepare-publication" in commands["run_ml_pipeline"]
    assert "--publish-bigquery" not in commands["run_ml_pipeline"]


def test_dag_has_no_hardcoded_credentials_or_large_xcom_payloads() -> None:
    source = (PROJECT_ROOT / "airflow" / "dags" / "dataengine_full_pipeline.py").read_text(
        encoding="utf-8"
    )
    upper_source = source.upper()
    assert "AKIA" not in upper_source
    assert "SECRET_ACCESS_KEY" not in upper_source
    assert "PRIVATE_KEY" not in upper_source
    assert ".XCOM_PUSH(" not in upper_source
    assert all(
        keyword_value(call, "do_xcom_push") is False
        for call in task_calls(dag_tree()).values()
    )
