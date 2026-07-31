from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import pipelines.run_pipeline as pipeline_module
from pipelines.run_pipeline import run_pipeline


@pytest.fixture(scope="module", autouse=True)
def mock_s3_upload():
    def fake_upload(
        processed_dir: Path | str,
        execution_date: datetime,
    ) -> dict[str, object]:
        partition = (
            f"year={execution_date:%Y}/"
            f"month={execution_date:%m}/"
            f"day={execution_date:%d}"
        )
        filenames = (
            "customers.parquet",
            "orders.parquet",
            "order_items.parquet",
            "products.parquet",
        )
        files = []
        for filename in filenames:
            object_key = f"bronze/{Path(filename).stem}/{partition}/{filename}"
            files.append(
                {
                    "local_path": str(Path(processed_dir) / filename),
                    "bucket": "test-bucket",
                    "object_key": object_key,
                    "s3_uri": f"s3://test-bucket/{object_key}",
                }
            )
        return {
            "execution_date": execution_date.isoformat(),
            "partition": partition,
            "uploaded_count": len(filenames),
            "bucket": "test-bucket",
            "files": files,
        }

    def fake_silver_processing(
        execution_date: datetime,
        bucket_name: str,
        staging_dir: Path | str,
    ) -> dict[str, object]:
        partition = (
            f"year={execution_date:%Y}/"
            f"month={execution_date:%m}/"
            f"day={execution_date:%d}"
        )
        return {
            "success": True,
            "execution_date": execution_date.isoformat(),
            "partition": partition,
            "bucket": bucket_name,
            "source_layer": "bronze",
            "destination_layer": "silver",
            "downloaded_count": 4,
            "transformed_count": 4,
            "uploaded_count": 4,
            "input_rows": {},
            "output_rows": {},
            "validation": {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "datasets": {},
            },
            "files": [],
            "duration_seconds": 0.01,
        }

    def fake_bigquery_load(
        silver_files: dict[str, Path | str],
        silver_report: dict[str, object],
        execution_date: datetime,
    ) -> dict[str, object]:
        return {
            "success": True,
            "execution_date": execution_date.isoformat(),
            "partition": silver_report["partition"],
            "project_id": "dataengine-fernando-2026",
            "dataset_id": "dataengine",
            "location": "southamerica-east1",
            "write_disposition": "WRITE_TRUNCATE",
            "loaded_count": 4,
            "input_rows": {},
            "output_rows": {},
            "validation": {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "tables": {},
            },
            "tables": {},
            "duration_seconds": 0.01,
        }

    with (
        patch.object(pipeline_module, "upload_processed_files", fake_upload),
        patch.object(
            pipeline_module,
            "process_bronze_to_silver",
            fake_silver_processing,
        ),
        patch.object(
            pipeline_module,
            "load_silver_to_bigquery",
            fake_bigquery_load,
        ),
    ):
        yield


@pytest.fixture(scope="module")
def successful_pipeline(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("pipeline")
    return run_pipeline(
        products_quantity=12,
        customers_quantity=20,
        orders_quantity=30,
        seed=50,
        raw_dir=directory / "raw",
        processed_dir=directory / "processed",
        silver_staging_dir=directory / "silver_staging",
    )


def _run_in_directory(
    directory: Path,
    seed: int = 42,
) -> dict[str, object]:
    return run_pipeline(
        products_quantity=10,
        customers_quantity=15,
        orders_quantity=25,
        seed=seed,
        raw_dir=directory / "raw",
        processed_dir=directory / "processed",
        silver_staging_dir=directory / "silver_staging",
    )


def test_pipeline_runs_successfully(successful_pipeline: dict[str, object]) -> None:
    assert successful_pipeline["success"] is True
    assert successful_pipeline["validation"]["is_valid"] is True


def test_pipeline_creates_four_csv_files(successful_pipeline: dict[str, object]) -> None:
    assert list(successful_pipeline["raw_files"]) == [
        "products",
        "customers",
        "orders",
        "order_items",
    ]
    assert all(Path(path).is_file() for path in successful_pipeline["raw_files"].values())


def test_pipeline_creates_four_parquet_files(
    successful_pipeline: dict[str, object],
) -> None:
    assert list(successful_pipeline["processed_files"]) == [
        "products",
        "customers",
        "orders",
        "order_items",
    ]
    assert all(
        Path(path).is_file() for path in successful_pipeline["processed_files"].values()
    )


def test_pipeline_uses_requested_quantities(
    successful_pipeline: dict[str, object],
) -> None:
    assert successful_pipeline["rows"]["products"] == 12
    assert successful_pipeline["rows"]["customers"] == 20
    assert successful_pipeline["rows"]["orders"] == 30
    assert successful_pipeline["rows"]["order_items"] >= 30


def test_validation_runs_before_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    original_validation = pipeline_module.validate_raw_data
    original_processing = pipeline_module.process_raw_to_parquet
    original_upload = pipeline_module.upload_processed_files
    original_silver_processing = pipeline_module.process_bronze_to_silver
    original_bigquery_load = pipeline_module.load_silver_to_bigquery

    def tracked_validation(**paths: Path | str) -> dict[str, object]:
        call_order.append("validation")
        return original_validation(**paths)

    def tracked_processing(**paths: Path | str) -> dict[str, object]:
        call_order.append("processing")
        return original_processing(**paths)

    def tracked_upload(**options: object) -> dict[str, object]:
        call_order.append("upload")
        return original_upload(**options)

    def tracked_silver_processing(**options: object) -> dict[str, object]:
        call_order.append("silver")
        return original_silver_processing(**options)

    def tracked_bigquery_load(**options: object) -> dict[str, object]:
        call_order.append("bigquery")
        return original_bigquery_load(**options)

    monkeypatch.setattr(pipeline_module, "validate_raw_data", tracked_validation)
    monkeypatch.setattr(pipeline_module, "process_raw_to_parquet", tracked_processing)
    monkeypatch.setattr(pipeline_module, "upload_processed_files", tracked_upload)
    monkeypatch.setattr(
        pipeline_module,
        "process_bronze_to_silver",
        tracked_silver_processing,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_silver_to_bigquery",
        tracked_bigquery_load,
    )
    _run_in_directory(tmp_path)

    assert call_order == [
        "validation",
        "processing",
        "upload",
        "silver",
        "bigquery",
    ]


def test_pipeline_report_has_expected_structure(
    successful_pipeline: dict[str, object],
) -> None:
    assert list(successful_pipeline) == [
        "success",
        "raw_files",
        "processed_files",
        "rows",
        "validation",
        "s3",
        "silver",
        "bigquery",
    ]
    assert list(successful_pipeline["rows"]) == [
        "products",
        "customers",
        "orders",
        "order_items",
    ]
    assert list(successful_pipeline["validation"]) == [
        "is_valid",
        "errors",
        "warnings",
    ]
    assert successful_pipeline["s3"]["bucket"] == "test-bucket"
    assert successful_pipeline["s3"]["uploaded_count"] == 4
    assert successful_pipeline["s3"]["partition"] == successful_pipeline["silver"][
        "partition"
    ]
    assert successful_pipeline["silver"]["uploaded_count"] == 4
    assert successful_pipeline["bigquery"]["loaded_count"] == 4


def test_bronze_and_silver_receive_same_fixed_execution_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_date = datetime(2026, 2, 3, 10, 30, tzinfo=timezone.utc)
    received_dates: list[datetime] = []
    partition = "year=2026/month=02/day=03"

    def fake_upload(
        processed_dir: Path | str,
        execution_date: datetime,
    ) -> dict[str, object]:
        received_dates.append(execution_date)
        return {
            "execution_date": execution_date.isoformat(),
            "partition": partition,
            "uploaded_count": 4,
            "bucket": "test-bucket",
            "files": [],
        }

    def fake_silver(
        execution_date: datetime,
        bucket_name: str,
        staging_dir: Path | str,
    ) -> dict[str, object]:
        received_dates.append(execution_date)
        return {
            "success": True,
            "execution_date": execution_date.isoformat(),
            "partition": partition,
            "bucket": bucket_name,
            "uploaded_count": 4,
            "files": [],
        }

    def fake_bigquery(
        silver_files: dict[str, Path | str],
        silver_report: dict[str, object],
        execution_date: datetime,
    ) -> dict[str, object]:
        received_dates.append(execution_date)
        return {
            "success": True,
            "execution_date": execution_date.isoformat(),
            "partition": partition,
            "project_id": "dataengine-fernando-2026",
            "dataset_id": "dataengine",
            "loaded_count": 4,
        }

    monkeypatch.setattr(pipeline_module, "upload_processed_files", fake_upload)
    monkeypatch.setattr(pipeline_module, "process_bronze_to_silver", fake_silver)
    monkeypatch.setattr(pipeline_module, "load_silver_to_bigquery", fake_bigquery)

    report = run_pipeline(
        products_quantity=10,
        customers_quantity=15,
        orders_quantity=25,
        seed=42,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        silver_staging_dir=tmp_path / "silver_staging",
        execution_date=fixed_date,
    )

    assert received_dates == [fixed_date, fixed_date, fixed_date]
    assert report["s3"]["partition"] == report["silver"]["partition"]
    assert report["silver"]["partition"] == report["bigquery"]["partition"]


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    first = _run_in_directory(tmp_path / "first", seed=100)
    second = _run_in_directory(tmp_path / "second", seed=100)

    for name in ("products", "customers", "orders", "order_items"):
        assert Path(first["raw_files"][name]).read_bytes() == Path(
            second["raw_files"][name]
        ).read_bytes()
        assert Path(first["processed_files"][name]).read_bytes() == Path(
            second["processed_files"][name]
        ).read_bytes()
        assert_frame_equal(
            pd.read_parquet(first["processed_files"][name]),
            pd.read_parquet(second["processed_files"][name]),
        )


def test_pipeline_changes_with_different_seeds(tmp_path: Path) -> None:
    first = _run_in_directory(tmp_path / "first", seed=100)
    second = _run_in_directory(tmp_path / "second", seed=200)

    assert Path(first["raw_files"]["products"]).read_bytes() != Path(
        second["raw_files"]["products"]
    ).read_bytes()
    assert Path(first["raw_files"]["customers"]).read_bytes() != Path(
        second["raw_files"]["customers"]
    ).read_bytes()
    assert Path(first["raw_files"]["orders"]).read_bytes() != Path(
        second["raw_files"]["orders"]
    ).read_bytes()


def test_pipeline_supports_temporary_directories(tmp_path: Path) -> None:
    report = _run_in_directory(tmp_path)

    for section in ("raw_files", "processed_files"):
        for path in report[section].values():
            assert Path(path).is_relative_to(tmp_path)


def test_pipeline_rejects_invalid_products_quantity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="products_quantity"):
        run_pipeline(products_quantity=0, raw_dir=tmp_path / "raw")


def test_pipeline_rejects_invalid_customers_quantity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="customers_quantity"):
        run_pipeline(customers_quantity=0, raw_dir=tmp_path / "raw")


def test_pipeline_rejects_invalid_orders_quantity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="orders_quantity"):
        run_pipeline(orders_quantity=0, raw_dir=tmp_path / "raw")


def test_pipeline_stops_when_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processing_called = False

    def invalid_validation(**paths: Path | str) -> dict[str, object]:
        return {
            "is_valid": False,
            "errors": ["erro simulado"],
            "warnings": [],
            "summary": {},
        }

    def unexpected_processing(**paths: Path | str) -> dict[str, object]:
        nonlocal processing_called
        processing_called = True
        return {}

    monkeypatch.setattr(pipeline_module, "validate_raw_data", invalid_validation)
    monkeypatch.setattr(
        pipeline_module,
        "process_raw_to_parquet",
        unexpected_processing,
    )

    with pytest.raises(RuntimeError, match="Falha na etapa validação"):
        _run_in_directory(tmp_path)

    assert processing_called is False


def test_pipeline_returns_context_when_silver_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_silver(**options: object) -> dict[str, object]:
        raise ValueError("falha Silver simulada")

    monkeypatch.setattr(
        pipeline_module,
        "process_bronze_to_silver",
        fail_silver,
    )

    with pytest.raises(
        RuntimeError,
        match="Falha na etapa processamento da camada Silver",
    ):
        _run_in_directory(tmp_path)


def test_pipeline_returns_context_when_bigquery_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_bigquery(**options: object) -> dict[str, object]:
        raise ValueError("falha BigQuery simulada")

    monkeypatch.setattr(
        pipeline_module,
        "load_silver_to_bigquery",
        fail_bigquery,
    )

    with pytest.raises(
        RuntimeError,
        match="Falha na etapa carga da camada Silver no BigQuery",
    ):
        _run_in_directory(tmp_path)


def test_direct_execution_returns_zero_on_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "success": True,
        "raw_files": {},
        "processed_files": {},
        "rows": {"products": 1, "customers": 1, "orders": 1, "order_items": 1},
        "validation": {"is_valid": True, "errors": [], "warnings": []},
        "s3": {
            "execution_date": "2026-01-02T03:04:05+00:00",
            "partition": "year=2026/month=01/day=02",
            "uploaded_count": 4,
            "bucket": "test-bucket",
            "files": [],
        },
        "silver": {
            "execution_date": "2026-01-02T03:04:05+00:00",
            "partition": "year=2026/month=01/day=02",
            "uploaded_count": 4,
            "bucket": "test-bucket",
            "files": [],
        },
        "bigquery": {
            "execution_date": "2026-01-02T03:04:05+00:00",
            "partition": "year=2026/month=01/day=02",
            "project_id": "dataengine-fernando-2026",
            "dataset_id": "dataengine",
            "loaded_count": 4,
        },
    }
    monkeypatch.setattr(pipeline_module, "run_pipeline", lambda: report)

    assert pipeline_module.main() == 0
    output = capsys.readouterr().out
    assert "Pipeline concluído com sucesso" in output
    assert "Bronze: 4 arquivo(s) enviado(s) para o bucket test-bucket" in output
    assert "Silver: 4 arquivo(s) enviado(s) para o bucket test-bucket" in output
    assert "BigQuery: 4 tabela(s) carregada(s)" in output
    assert "Projeto: dataengine-fernando-2026" in output
    assert "Dataset: dataengine" in output
    assert "Partição: year=2026/month=01/day=02" in output


def test_direct_execution_returns_one_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_pipeline() -> dict[str, object]:
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(pipeline_module, "run_pipeline", fail_pipeline)

    assert pipeline_module.main() == 1
    assert "falha simulada" in capsys.readouterr().out
