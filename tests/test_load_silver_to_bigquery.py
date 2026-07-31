from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO
from unittest.mock import Mock

import pandas as pd
import pytest
from google.api_core.exceptions import BadRequest, Forbidden, NotFound
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from pyarrow.parquet import ParquetFile

import pipelines.loading.load_silver_to_bigquery as load_module
from pipelines.loading.load_silver_to_bigquery import (
    BIGQUERY_DATASET_ID,
    BIGQUERY_LOCATION,
    GCP_PROJECT_ID,
    build_full_table_id,
    ensure_bigquery_dataset,
    load_silver_to_bigquery,
    validate_bigquery_configuration,
)
from src.validation.validate_bigquery_load import (
    BigQueryInputValidationError,
    BigQueryLoadValidationError,
)

PARTITION = "year=2026/month=02/day=03"


class FakeJob:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.result_called = False

    def result(self) -> FakeJob:
        self.result_called = True
        return self


class FakeBigQueryClient:
    def __init__(
        self,
        *,
        dataset_exists: bool = True,
        dataset_location: str = BIGQUERY_LOCATION,
    ) -> None:
        self.dataset_exists = dataset_exists
        self.dataset_location = dataset_location
        self.created_datasets: list[object] = []
        self.load_calls: list[dict[str, object]] = []
        self.jobs: list[FakeJob] = []
        self.loaded_rows: dict[str, int] = {}
        self.row_overrides: dict[str, int] = {}
        self.load_error: Exception | None = None

    def get_dataset(self, full_dataset_id: str) -> object:
        if not self.dataset_exists:
            raise NotFound("dataset ausente")
        return SimpleNamespace(
            full_dataset_id=full_dataset_id,
            location=self.dataset_location,
        )

    def create_dataset(self, dataset: object) -> object:
        self.dataset_exists = True
        self.dataset_location = str(dataset.location)
        self.created_datasets.append(dataset)
        return dataset

    def load_table_from_file(
        self,
        source_file: BinaryIO,
        full_table_id: str,
        *,
        job_config: bigquery.LoadJobConfig,
        location: str,
    ) -> FakeJob:
        if self.load_error is not None:
            raise self.load_error

        row_count = ParquetFile(source_file.name).metadata.num_rows
        self.loaded_rows[full_table_id] = row_count
        job = FakeJob(f"job-{len(self.jobs) + 1}")
        self.jobs.append(job)
        self.load_calls.append(
            {
                "full_table_id": full_table_id,
                "source_format": job_config.source_format,
                "write_disposition": job_config.write_disposition,
                "location": location,
                "file_closed": source_file.closed,
            }
        )
        return job

    def get_table(self, full_table_id: str) -> object:
        loaded_rows = self.row_overrides.get(
            full_table_id,
            self.loaded_rows[full_table_id],
        )
        return SimpleNamespace(
            num_rows=loaded_rows,
            schema=[object(), object()],
        )


def _write_silver_files(
    directory: Path,
    datasets: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for dataset_name, dataframe in datasets.items():
        path = directory / f"{dataset_name}.parquet"
        dataframe.to_parquet(path, index=False, compression="snappy")
        files[dataset_name] = path
    return files


def _silver_report(files: dict[str, Path]) -> dict[str, object]:
    return {
        "partition": PARTITION,
        "validation": {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        },
        "files": [
            {
                "dataset": dataset_name,
                "local_path": str(path),
                "object_key": (
                    f"silver/{dataset_name}/{PARTITION}/{dataset_name}.parquet"
                ),
            }
            for dataset_name, path in files.items()
        ],
    }


@pytest.fixture
def silver_files(
    tmp_path: Path,
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _write_silver_files(tmp_path / "silver", valid_silver_dataframes)


def test_builds_exact_full_table_ids() -> None:
    assert build_full_table_id("customers") == (
        "dataengine-fernando-2026.dataengine.customers"
    )
    assert build_full_table_id("orders") == (
        "dataengine-fernando-2026.dataengine.orders"
    )
    assert build_full_table_id("order_items") == (
        "dataengine-fernando-2026.dataengine.order_items"
    )
    assert build_full_table_id("products") == (
        "dataengine-fernando-2026.dataengine.products"
    )


def test_configuration_constants_and_validation() -> None:
    assert GCP_PROJECT_ID == "dataengine-fernando-2026"
    assert BIGQUERY_DATASET_ID == "dataengine"
    assert BIGQUERY_LOCATION == "southamerica-east1"
    assert validate_bigquery_configuration(
        GCP_PROJECT_ID,
        BIGQUERY_DATASET_ID,
        BIGQUERY_LOCATION,
    )["is_valid"]
    assert not validate_bigquery_configuration("", "", "")["is_valid"]


def test_accepts_existing_dataset_in_correct_location() -> None:
    client = FakeBigQueryClient()

    report = ensure_bigquery_dataset(client)

    assert report["created"] is False
    assert report["location"] == BIGQUERY_LOCATION
    assert client.created_datasets == []


def test_creates_dataset_when_missing() -> None:
    client = FakeBigQueryClient(dataset_exists=False)

    report = ensure_bigquery_dataset(client)

    assert report["created"] is True
    assert report["location"] == BIGQUERY_LOCATION
    assert len(client.created_datasets) == 1


def test_rejects_dataset_in_wrong_location() -> None:
    client = FakeBigQueryClient(dataset_location="US")

    with pytest.raises(ValueError, match="localização incompatível"):
        ensure_bigquery_dataset(client)

    assert client.created_datasets == []


def test_loads_four_parquet_files_with_full_refresh_and_complete_report(
    silver_files: dict[str, Path],
) -> None:
    client = FakeBigQueryClient()

    report = load_silver_to_bigquery(
        silver_files=silver_files,
        silver_report=_silver_report(silver_files),
        execution_date=pd.Timestamp("2026-02-03", tz="UTC").to_pydatetime(),
        bigquery_client=client,
    )

    assert report["success"] is True
    assert report["partition"] == PARTITION
    assert report["project_id"] == GCP_PROJECT_ID
    assert report["dataset_id"] == BIGQUERY_DATASET_ID
    assert report["location"] == BIGQUERY_LOCATION
    assert report["write_disposition"] == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert report["loaded_count"] == 4
    assert report["input_rows"] == report["output_rows"]
    assert report["validation"]["is_valid"] is True
    assert report["duration_seconds"] >= 0
    assert len(client.load_calls) == 4
    assert all(
        call["write_disposition"] == bigquery.WriteDisposition.WRITE_TRUNCATE
        and call["source_format"] == bigquery.SourceFormat.PARQUET
        and call["location"] == BIGQUERY_LOCATION
        for call in client.load_calls
    )
    assert all(job.result_called for job in client.jobs)
    assert [table["job_id"] for table in report["tables"].values()] == [
        "job-1",
        "job-2",
        "job-3",
        "job-4",
    ]
    assert all(
        table["schema_field_count"] == 2
        and table["write_disposition"]
        == bigquery.WriteDisposition.WRITE_TRUNCATE
        and table["success"]
        for table in report["tables"].values()
    )


def test_row_count_divergence_fails_post_load_validation(
    silver_files: dict[str, Path],
) -> None:
    client = FakeBigQueryClient()
    client.row_overrides[
        "dataengine-fernando-2026.dataengine.orders"
    ] = 1

    with pytest.raises(BigQueryLoadValidationError, match="divergência de linhas"):
        load_silver_to_bigquery(
            silver_files=silver_files,
            silver_report=_silver_report(silver_files),
            execution_date=pd.Timestamp("2026-02-03", tz="UTC").to_pydatetime(),
            bigquery_client=client,
        )


def test_invalid_input_starts_no_bigquery_job(
    silver_files: dict[str, Path],
) -> None:
    client = FakeBigQueryClient()
    silver_files["products"].write_bytes(b"")

    with pytest.raises(BigQueryInputValidationError, match="arquivo Silver está vazio"):
        load_silver_to_bigquery(
            silver_files=silver_files,
            silver_report=_silver_report(silver_files),
            execution_date=pd.Timestamp("2026-02-03", tz="UTC").to_pydatetime(),
            bigquery_client=client,
        )

    assert client.load_calls == []


def test_authentication_failure_has_context(
    silver_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_client(**options: object) -> object:
        raise DefaultCredentialsError("credenciais ausentes")

    monkeypatch.setattr(load_module.bigquery, "Client", fail_client)

    with pytest.raises(RuntimeError, match="Application Default Credentials"):
        load_silver_to_bigquery(
            silver_files=silver_files,
            silver_report=_silver_report(silver_files),
            execution_date=pd.Timestamp("2026-02-03", tz="UTC").to_pydatetime(),
        )


@pytest.mark.parametrize(
    "load_error",
    [Forbidden("acesso negado"), BadRequest("Parquet inválido")],
)
def test_bigquery_api_failures_include_table_and_file_context(
    silver_files: dict[str, Path],
    load_error: Exception,
) -> None:
    client = FakeBigQueryClient()
    client.load_error = load_error

    with pytest.raises(RuntimeError) as error:
        load_silver_to_bigquery(
            silver_files=silver_files,
            silver_report=_silver_report(silver_files),
            execution_date=pd.Timestamp("2026-02-03", tz="UTC").to_pydatetime(),
            bigquery_client=client,
        )

    message = str(error.value)
    assert "projeto=dataengine-fernando-2026" in message
    assert "dataset=dataengine" in message
    assert "tabela=customers" in message
    assert "customers.parquet" in message
    assert "etapa=job de carga" in message


def test_downloads_controlled_silver_partition_when_local_files_are_absent(
    silver_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = Mock(return_value=_silver_report(silver_files))
    monkeypatch.setattr(load_module, "download_silver_files", download)
    client = FakeBigQueryClient()

    report = load_silver_to_bigquery(
        execution_date=pd.Timestamp("2026-02-03", tz="UTC").to_pydatetime(),
        bigquery_client=client,
    )

    assert report["loaded_count"] == 4
    download.assert_called_once()


def test_independent_main_returns_zero_on_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received_dates = []

    def fake_load(**options: object) -> dict[str, object]:
        received_dates.append(options["execution_date"])
        return {
            "project_id": GCP_PROJECT_ID,
            "dataset_id": BIGQUERY_DATASET_ID,
            "partition": PARTITION,
            "output_rows": {
                "customers": 2,
                "orders": 2,
                "order_items": 2,
                "products": 2,
            },
        }

    monkeypatch.setattr(load_module, "load_silver_to_bigquery", fake_load)

    result = load_module.main(
        ["--execution-date", "2026-02-03T10:30:00+00:00"]
    )

    assert result == 0
    assert received_dates[0].isoformat() == "2026-02-03T10:30:00+00:00"
    assert "Carga da camada Silver no BigQuery concluída" in capsys.readouterr().out


def test_independent_main_returns_one_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(**options: object) -> dict[str, object]:
        raise RuntimeError("falha BigQuery simulada")

    monkeypatch.setattr(load_module, "load_silver_to_bigquery", fail_load)

    assert (
        load_module.main(
            ["--execution-date", "2026-02-03T10:30:00+00:00"]
        )
        == 1
    )
