from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipelines.orchestration import validate_environment as environment_module
from pipelines.orchestration import validate_final_outputs as outputs_module


class _FakeAwsClient:
    def __init__(self, service_name: str, calls: list[str]) -> None:
        self.service_name = service_name
        self.calls = calls

    def get_caller_identity(self) -> dict[str, str]:
        self.calls.append("sts")
        return {"Account": "masked"}

    def head_bucket(self, *, Bucket: str) -> None:
        assert Bucket == environment_module.BUCKET_NAME
        self.calls.append("s3")


class _FakeAwsSession:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def client(self, service_name: str) -> _FakeAwsClient:
        return _FakeAwsClient(service_name, self.calls)


class _FakeQueryJob:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def result(self) -> list[dict[str, Any]]:
        return self.rows


class _FakeBigQueryClient:
    def __init__(self, *, invalid_counts: bool = False) -> None:
        self.invalid_counts = invalid_counts
        self.queries: list[str] = []

    def list_datasets(self, **_: Any) -> list[object]:
        return [object()]

    def get_dataset(self, _: str) -> SimpleNamespace:
        return SimpleNamespace(location=outputs_module.BIGQUERY_LOCATION)

    def get_table(self, table_id: str) -> SimpleNamespace:
        if self.invalid_counts and table_id.endswith(".fct_sales"):
            raise RuntimeError("missing simulated relation")
        return SimpleNamespace(num_rows=10)

    def query(self, query: str, **_: Any) -> _FakeQueryJob:
        self.queries.append(query)
        if "ORDER BY COALESCE" in query:
            return _FakeQueryJob(
                [
                    {
                        "run_id": "run-latest",
                        "products_processed": 2,
                        "forecast_rows": 102,
                        "risk_rows": 2,
                        "champion_model": "moving_average_28",
                        "champion_version": "1.0.0",
                    }
                ]
            )
        counts = {
            "forecast_rows": 102,
            "forecast_products": 2,
            "forecast_horizons": [7, 14, 30],
            "forecast_duplicate_rows": 0,
            "incomplete_forecast_groups": 0,
            "invalid_forecast_model_rows": 0,
            "risk_rows": 2,
            "risk_duplicate_rows": 0,
            "metric_rows": 30,
            "metric_duplicate_rows": 0,
            "active_champion_rows": 1,
            "official_champion_rows": 1,
            "registry_duplicate_rows": 0,
            "pipeline_run_rows": 1,
        }
        if self.invalid_counts:
            counts["forecast_duplicate_rows"] = 1
            counts["forecast_horizons"] = [7, 14]
        return _FakeQueryJob([counts])


def _create_required_paths(root: Path, profiles_path: Path) -> None:
    for relative_path in environment_module.REQUIRED_PROJECT_DIRECTORIES:
        (root / relative_path).mkdir(parents=True, exist_ok=True)
    (root / "dataengine_dbt" / "dbt_project.yml").write_text(
        "name: dataengine_dbt\n",
        encoding="utf-8",
    )
    profiles_path.parent.mkdir(parents=True)
    profiles_path.write_text("dataengine_dbt: {}\n", encoding="utf-8")


def test_environment_validator_succeeds_with_simulated_clients(tmp_path: Path) -> None:
    profiles_path = tmp_path / "home" / ".dbt" / "profiles.yml"
    _create_required_paths(tmp_path, profiles_path)
    aws_calls: list[str] = []
    bigquery_client = _FakeBigQueryClient()

    report = environment_module.validate_environment(
        project_root=tmp_path,
        dbt_profiles_path=profiles_path,
        import_module_fn=lambda _: object(),
        aws_session_factory=lambda: _FakeAwsSession(aws_calls),
        google_auth_loader=lambda **_: (object(), None),
        bigquery_client_factory=lambda **_: bigquery_client,
    )

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert aws_calls == ["sts", "s3"]
    assert all(report["google_cloud"]["datasets"].values())


def test_environment_validator_reports_simulated_failures(tmp_path: Path) -> None:
    def fail_import(module_name: str) -> object:
        if module_name == "airflow":
            raise ImportError("simulated")
        return object()

    def fail_google(**_: Any) -> tuple[Any, str | None]:
        raise RuntimeError("simulated ADC failure")

    report = environment_module.validate_environment(
        project_root=tmp_path,
        dbt_profiles_path=tmp_path / "missing" / "profiles.yml",
        import_module_fn=fail_import,
        aws_session_factory=lambda: (_ for _ in ()).throw(
            RuntimeError("simulated AWS failure")
        ),
        google_auth_loader=fail_google,
        bigquery_client_factory=lambda **_: _FakeBigQueryClient(),
    )

    assert report["is_valid"] is False
    assert report["imports"]["airflow"] is False
    assert report["aws"]["sts"] is False
    assert report["google_cloud"]["adc"] is False
    assert report["errors"]


def test_final_output_validator_succeeds_with_read_only_simulation() -> None:
    client = _FakeBigQueryClient()
    report = outputs_module.validate_final_outputs(bigquery_client=client)

    assert report["is_valid"] is True
    assert report["latest_successful_run"]["run_id"] == "run-latest"
    assert report["counts"]["forecast_horizons"] == [7, 14, 30]
    assert len(client.queries) == 2


def test_final_output_validator_reports_relations_and_count_failures() -> None:
    report = outputs_module.validate_final_outputs(
        bigquery_client=_FakeBigQueryClient(invalid_counts=True)
    )

    assert report["is_valid"] is False
    assert any("fct_sales" in error for error in report["errors"])
    assert any("forecast_duplicate_rows" in error for error in report["errors"])
    assert any("Horizontes oficiais" in error for error in report["errors"])

