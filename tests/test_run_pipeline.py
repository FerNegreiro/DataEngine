from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import pipelines.run_pipeline as pipeline_module
from pipelines.run_pipeline import run_pipeline


@pytest.fixture(scope="module", autouse=True)
def mock_s3_upload():
    def fake_upload(processed_dir: Path | str) -> dict[str, object]:
        partition = "year=2026/month=01/day=02"
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
            "execution_date": "2026-01-02T03:04:05+00:00",
            "partition": partition,
            "uploaded_count": len(filenames),
            "bucket": "test-bucket",
            "files": files,
        }

    with patch.object(pipeline_module, "upload_processed_files", fake_upload):
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

    def tracked_validation(**paths: Path | str) -> dict[str, object]:
        call_order.append("validation")
        return original_validation(**paths)

    def tracked_processing(**paths: Path | str) -> dict[str, object]:
        call_order.append("processing")
        return original_processing(**paths)

    def tracked_upload(**options: Path | str) -> dict[str, object]:
        call_order.append("upload")
        return original_upload(**options)

    monkeypatch.setattr(pipeline_module, "validate_raw_data", tracked_validation)
    monkeypatch.setattr(pipeline_module, "process_raw_to_parquet", tracked_processing)
    monkeypatch.setattr(pipeline_module, "upload_processed_files", tracked_upload)
    _run_in_directory(tmp_path)

    assert call_order == ["validation", "processing", "upload"]


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
    assert successful_pipeline["s3"]["partition"] == "year=2026/month=01/day=02"


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
    }
    monkeypatch.setattr(pipeline_module, "run_pipeline", lambda: report)

    assert pipeline_module.main() == 0
    output = capsys.readouterr().out
    assert "Pipeline concluído com sucesso" in output
    assert "S3: 4 arquivo(s) enviado(s) para o bucket test-bucket" in output


def test_direct_execution_returns_one_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_pipeline() -> dict[str, object]:
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(pipeline_module, "run_pipeline", fail_pipeline)

    assert pipeline_module.main() == 1
    assert "falha simulada" in capsys.readouterr().out
