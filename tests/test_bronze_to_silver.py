from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from pipelines.loading.read_from_s3 import download_bronze_files
from pipelines.loading.upload_silver_to_s3 import upload_silver_files
from pipelines.loading.upload_to_s3 import (
    FILES_TO_UPLOAD,
    build_partitioned_key,
)
from pipelines.processing.process_bronze_to_silver import (
    SilverValidationError,
    process_bronze_to_silver,
)

EXECUTION_DATE = datetime(2026, 2, 3, 10, 30, tzinfo=timezone.utc)
PARTITION = "year=2026/month=02/day=03"
BUCKET = "test-bucket"


class InMemoryS3:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects.copy()
        self.downloaded_keys: list[str] = []
        self.uploaded_keys: list[str] = []

    def download_file(self, *, Bucket: str, Key: str, Filename: str) -> None:
        object_id = (Bucket, Key)
        if object_id not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                "GetObject",
            )
        Path(Filename).write_bytes(self.objects[object_id])
        self.downloaded_keys.append(Key)

    def upload_file(self, *, Filename: str, Bucket: str, Key: str) -> None:
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()
        self.uploaded_keys.append(Key)


def _parquet_objects(
    directory: Path,
    datasets: dict[str, pd.DataFrame],
) -> dict[tuple[str, str], bytes]:
    objects: dict[tuple[str, str], bytes] = {}
    directory.mkdir(parents=True, exist_ok=True)
    for dataset_name, filename in FILES_TO_UPLOAD.items():
        path = directory / filename
        datasets[dataset_name].to_parquet(path, index=False, compression="snappy")
        key = build_partitioned_key(
            dataset_name=dataset_name,
            filename=filename,
            execution_date=EXECUTION_DATE,
            layer="bronze",
        )
        objects[(BUCKET, key)] = path.read_bytes()
    return objects


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


def test_builds_correct_silver_key() -> None:
    key = build_partitioned_key(
        dataset_name="customers",
        filename="customers.parquet",
        execution_date=EXECUTION_DATE,
        layer="silver",
    )

    assert key == "silver/customers/year=2026/month=02/day=03/customers.parquet"


def test_downloads_four_bronze_files_from_same_partition(tmp_path: Path) -> None:
    s3_client = Mock()

    def download_file(**arguments: Any) -> None:
        Path(arguments["Filename"]).write_bytes(b"parquet")

    s3_client.download_file.side_effect = download_file

    report = download_bronze_files(
        staging_dir=tmp_path,
        bucket_name=BUCKET,
        execution_date=EXECUTION_DATE,
        s3_client=s3_client,
    )

    assert report["downloaded_count"] == 4
    assert report["partition"] == PARTITION
    assert s3_client.download_file.call_count == 4
    assert all(Path(item["local_path"]).is_file() for item in report["files"])
    assert all(f"/{PARTITION}/" in item["object_key"] for item in report["files"])


def test_missing_bronze_object_has_clear_context(tmp_path: Path) -> None:
    s3_client = Mock()
    s3_client.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
        "GetObject",
    )

    with pytest.raises(FileNotFoundError) as error:
        download_bronze_files(
            staging_dir=tmp_path,
            bucket_name=BUCKET,
            execution_date=EXECUTION_DATE,
            s3_client=s3_client,
        )

    message = str(error.value)
    assert "leitura da camada Bronze" in message
    assert "dataset=customers" in message
    assert f"bucket={BUCKET}" in message
    assert "bronze/customers/year=2026/month=02/day=03/customers.parquet" in message


def test_uploads_four_silver_files_with_complete_report(
    tmp_path: Path,
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    silver_files = _write_silver_files(
        tmp_path / "silver",
        valid_silver_dataframes,
    )
    row_counts = {
        name: len(dataframe)
        for name, dataframe in valid_silver_dataframes.items()
    }
    s3_client = Mock()

    report = upload_silver_files(
        silver_files=silver_files,
        row_counts=row_counts,
        bucket_name=BUCKET,
        execution_date=EXECUTION_DATE,
        s3_client=s3_client,
    )

    assert report["uploaded_count"] == 4
    assert report["partition"] == PARTITION
    assert s3_client.upload_file.call_count == 4
    assert all(f"/{PARTITION}/" in item["object_key"] for item in report["files"])
    assert all(item["object_key"].startswith("silver/") for item in report["files"])
    assert all(
        {
            "dataset",
            "local_path",
            "bucket",
            "object_key",
            "s3_uri",
            "row_count",
            "file_size_bytes",
        }.issubset(item)
        for item in report["files"]
    )


def test_processes_bronze_to_silver_with_full_report_and_same_partition(
    tmp_path: Path,
    bronze_dataframes: dict[str, pd.DataFrame],
) -> None:
    s3_client = InMemoryS3(
        _parquet_objects(tmp_path / "sources", bronze_dataframes)
    )

    report = process_bronze_to_silver(
        execution_date=EXECUTION_DATE,
        bucket_name=BUCKET,
        staging_dir=tmp_path / "staging",
        s3_client=s3_client,
    )

    assert report["success"] is True
    assert report["partition"] == PARTITION
    assert report["source_layer"] == "bronze"
    assert report["destination_layer"] == "silver"
    assert report["downloaded_count"] == 4
    assert report["transformed_count"] == 4
    assert report["uploaded_count"] == 4
    assert report["input_rows"] == report["output_rows"]
    assert report["validation"]["is_valid"] is True
    assert report["duration_seconds"] >= 0
    assert len(s3_client.downloaded_keys) == 4
    assert len(s3_client.uploaded_keys) == 4
    assert all(f"/{PARTITION}/" in key for key in s3_client.uploaded_keys)
    assert all(key.startswith("silver/") for key in s3_client.uploaded_keys)


def test_validation_failure_stops_silver_upload(
    tmp_path: Path,
    bronze_dataframes: dict[str, pd.DataFrame],
) -> None:
    invalid_dataframes = {
        name: dataframe.copy() for name, dataframe in bronze_dataframes.items()
    }
    invalid_dataframes["customers"] = pd.concat(
        [
            invalid_dataframes["customers"],
            invalid_dataframes["customers"].iloc[[0]],
        ],
        ignore_index=True,
    )
    s3_client = InMemoryS3(
        _parquet_objects(tmp_path / "sources", invalid_dataframes)
    )

    with pytest.raises(SilverValidationError, match="chave primária duplicada"):
        process_bronze_to_silver(
            execution_date=EXECUTION_DATE,
            bucket_name=BUCKET,
            staging_dir=tmp_path / "staging",
            s3_client=s3_client,
        )

    assert s3_client.uploaded_keys == []
