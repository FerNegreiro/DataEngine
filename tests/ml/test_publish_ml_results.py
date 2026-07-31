from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipelines.machine_learning import publish_ml_results as publisher
from src.ml.artifacts import save_production_artifacts
from src.ml.production import ProductionBundle


def _save_bundle(bundle: ProductionBundle, directory: Path) -> None:
    save_production_artifacts(
        manifest=bundle.manifest,
        forecasts=bundle.forecasts,
        inventory_risk=bundle.inventory_risk,
        model_metrics=bundle.model_metrics,
        model_registry=bundle.model_registry,
        pipeline_run=bundle.pipeline_run,
        artifacts_dir=directory,
    )


def _mock_successful_bigquery(
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[str],
) -> None:
    monkeypatch.setattr(
        publisher,
        "ensure_ml_dataset",
        lambda *args, **kwargs: {
            "full_dataset_id": "dataengine-fernando-2026.dataengine_ml",
            "location": "southamerica-east1",
            "created": False,
        },
    )
    monkeypatch.setattr(
        publisher,
        "ensure_ml_tables",
        lambda *args, **kwargs: {"sales_forecast": {"created": False}},
    )

    def capture_run(_: Any, pipeline_run: dict[str, Any], **__: Any) -> dict[str, Any]:
        statuses.append(pipeline_run["status"])
        return {"full_table_id": "dataengine_ml.pipeline_runs"}

    monkeypatch.setattr(publisher, "upsert_pipeline_run", capture_run)
    monkeypatch.setattr(
        publisher,
        "publish_ml_dataframes",
        lambda *args, **kwargs: {"sales_forecast": {"input_rows": 51}},
    )
    monkeypatch.setattr(
        publisher,
        "validate_ml_bigquery_load",
        lambda *args, **kwargs: {
            "is_valid": True,
            "errors": [],
            "tables": {"sales_forecast": "project.dataset.sales_forecast"},
            "counts": {},
        },
    )


def test_independent_publisher_rejects_incomplete_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="incompleto"):
        publisher.load_production_bundle(tmp_path)


def test_independent_publisher_loads_complete_artifact(
    production_bundle: ProductionBundle,
    tmp_path: Path,
) -> None:
    _save_bundle(production_bundle, tmp_path)
    loaded = publisher.load_production_bundle(tmp_path)
    assert loaded.manifest["run_id"] == production_bundle.manifest["run_id"]
    assert len(loaded.forecasts) == len(production_bundle.forecasts)


def test_publication_records_running_and_success_and_is_retry_safe(
    production_bundle: ProductionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses: list[str] = []
    _mock_successful_bigquery(monkeypatch, statuses)
    client = object()
    first = publisher.publish_ml_results(production_bundle, bigquery_client=client)
    second = publisher.publish_ml_results(production_bundle, bigquery_client=client)
    assert first["run_id"] == second["run_id"]
    assert first["success"] is True
    assert statuses == ["running", "success", "running", "success"]


def test_publication_attempts_to_record_failed_without_hiding_original_error(
    production_bundle: ProductionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses: list[str] = []
    _mock_successful_bigquery(monkeypatch, statuses)

    def fail_publish(*_: Any, **__: Any) -> None:
        raise RuntimeError("original write error")

    monkeypatch.setattr(publisher, "publish_ml_dataframes", fail_publish)
    with pytest.raises(RuntimeError, match="original write error"):
        publisher.publish_ml_results(production_bundle, bigquery_client=object())
    assert statuses == ["running", "failed"]


def test_initial_pipeline_run_failure_still_attempts_failed_status(
    production_bundle: ProductionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses: list[str] = []
    monkeypatch.setattr(
        publisher,
        "ensure_ml_dataset",
        lambda *args, **kwargs: {"full_dataset_id": "project.dataengine_ml"},
    )
    monkeypatch.setattr(
        publisher,
        "ensure_ml_tables",
        lambda *args, **kwargs: {"pipeline_runs": {"created": False}},
    )

    def fail_running(_: Any, pipeline_run: dict[str, Any], **__: Any) -> None:
        statuses.append(pipeline_run["status"])
        if pipeline_run["status"] == "running":
            raise RuntimeError("pipeline_runs unavailable")

    monkeypatch.setattr(publisher, "upsert_pipeline_run", fail_running)
    with pytest.raises(RuntimeError, match="pipeline_runs unavailable"):
        publisher.publish_ml_results(production_bundle, bigquery_client=object())
    assert statuses == ["running", "failed"]
