from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

import pipelines.loading.upload_to_s3 as upload_module
from pipelines.loading.upload_to_s3 import (
    FILES_TO_UPLOAD,
    build_partitioned_key,
    upload_processed_files,
)


@pytest.fixture
def processed_directory(tmp_path: Path) -> Path:
    for filename in FILES_TO_UPLOAD.values():
        (tmp_path / filename).write_bytes(b"parquet")
    return tmp_path


@pytest.fixture
def mocked_s3(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    s3_client = Mock()
    client_factory = Mock(return_value=s3_client)
    monkeypatch.setattr(upload_module.boto3, "client", client_factory)
    return client_factory, s3_client


def test_build_partitioned_key_uses_fixed_utc_date() -> None:
    execution_date = datetime(
        2026,
        7,
        30,
        20,
        15,
        tzinfo=timezone.utc,
    )

    key = build_partitioned_key(
        dataset_name="orders",
        filename="orders.parquet",
        execution_date=execution_date,
    )

    assert key == "bronze/orders/year=2026/month=07/day=30/orders.parquet"


def test_build_partitioned_key_zero_pads_month_and_day() -> None:
    execution_date = datetime(2026, 2, 3, tzinfo=timezone.utc)

    key = build_partitioned_key(
        dataset_name="customers",
        filename="customers.parquet",
        execution_date=execution_date,
    )

    assert key == "bronze/customers/year=2026/month=02/day=03/customers.parquet"


def test_build_partitioned_key_converts_execution_date_to_utc() -> None:
    brazil_timezone = timezone(timedelta(hours=-3))
    execution_date = datetime(2026, 7, 30, 22, 30, tzinfo=brazil_timezone)

    key = build_partitioned_key(
        dataset_name="products",
        filename="products.parquet",
        execution_date=execution_date,
    )

    assert key == "bronze/products/year=2026/month=07/day=31/products.parquet"


def test_build_partitioned_key_uses_current_utc_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_date = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            assert tz is timezone.utc
            return fixed_date

    monkeypatch.setattr(upload_module, "datetime", FixedDatetime)

    key = build_partitioned_key(
        dataset_name="order_items",
        filename="order_items.parquet",
    )

    assert key == (
        "bronze/order_items/year=2026/month=01/day=02/order_items.parquet"
    )


def test_upload_processed_files_uses_one_partition_and_returns_full_report(
    processed_directory: Path,
    mocked_s3: tuple[Mock, Mock],
) -> None:
    client_factory, s3_client = mocked_s3
    execution_date = datetime(2026, 4, 5, 12, 30, 45, tzinfo=timezone.utc)
    partition = "year=2026/month=04/day=05"

    report = upload_processed_files(
        processed_dir=processed_directory,
        bucket_name="test-bucket",
        execution_date=execution_date,
    )

    expected_files = []
    for dataset_name, filename in FILES_TO_UPLOAD.items():
        object_key = f"bronze/{dataset_name}/{partition}/{filename}"
        expected_files.append(
            {
                "local_path": str(processed_directory / filename),
                "bucket": "test-bucket",
                "object_key": object_key,
                "s3_uri": f"s3://test-bucket/{object_key}",
            }
        )

    assert report == {
        "execution_date": "2026-04-05T12:30:45+00:00",
        "partition": partition,
        "uploaded_count": 4,
        "bucket": "test-bucket",
        "files": expected_files,
    }
    client_factory.assert_called_once_with("s3")
    assert s3_client.upload_file.call_count == 4

    uploaded_keys = [
        call.kwargs["Key"] for call in s3_client.upload_file.call_args_list
    ]
    assert all(f"/{partition}/" in key for key in uploaded_keys)
    assert uploaded_keys == [file_report["object_key"] for file_report in expected_files]
