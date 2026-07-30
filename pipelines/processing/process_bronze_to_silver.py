from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import boto3
import pandas as pd
from pyarrow.lib import ArrowException

from pipelines.loading.read_from_s3 import (
    DEFAULT_SILVER_STAGING_DIR,
    download_bronze_files,
)
from pipelines.loading.upload_silver_to_s3 import upload_silver_files
from pipelines.loading.upload_to_s3 import (
    BUCKET_NAME,
    build_partition,
    normalize_execution_date,
)
from src.transformation.transform_silver_data import transform_silver_data
from src.validation.validate_silver_data import validate_silver_data

logger = logging.getLogger(__name__)


class SilverValidationError(ValueError):
    def __init__(self, validation_report: dict[str, object]) -> None:
        self.validation_report = validation_report
        errors = validation_report["errors"]
        details = "; ".join(errors) if errors else "erro desconhecido"
        super().__init__(f"Validação da camada Silver falhou: {details}")


def _read_bronze_parquets(
    download_report: dict[str, object],
) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for file_report in download_report["files"]:
        dataset_name = str(file_report["dataset"])
        local_path = Path(str(file_report["local_path"]))
        try:
            datasets[dataset_name] = pd.read_parquet(local_path)
        except (OSError, ValueError, ArrowException) as error:
            raise ValueError(
                "Falha na leitura do Parquet da camada Bronze: "
                f"dataset={dataset_name}, caminho={local_path}: {error}"
            ) from error
    return datasets


def _write_silver_parquets(
    datasets: dict[str, pd.DataFrame],
    staging_dir: Path | str,
) -> dict[str, Path]:
    destination_directory = Path(staging_dir) / "silver"
    destination_directory.mkdir(parents=True, exist_ok=True)
    destinations: dict[str, Path] = {}

    for dataset_name, dataframe in datasets.items():
        destination = destination_directory / f"{dataset_name}.parquet"
        try:
            dataframe.to_parquet(
                destination,
                index=False,
                compression="snappy",
            )
        except (OSError, ValueError, ArrowException) as error:
            raise ValueError(
                "Falha na gravação temporária da camada Silver: "
                f"dataset={dataset_name}, caminho={destination}: {error}"
            ) from error
        destinations[dataset_name] = destination

    return destinations


def process_bronze_to_silver(
    execution_date: datetime | None = None,
    bucket_name: str = BUCKET_NAME,
    staging_dir: Path | str = DEFAULT_SILVER_STAGING_DIR,
    s3_client: Any | None = None,
) -> dict[str, object]:
    started_at = perf_counter()
    utc_execution_date = normalize_execution_date(execution_date)
    partition = build_partition(utc_execution_date)
    client = s3_client if s3_client is not None else boto3.client("s3")

    download_report = download_bronze_files(
        staging_dir=staging_dir,
        bucket_name=bucket_name,
        execution_date=utc_execution_date,
        s3_client=client,
    )
    bronze_datasets = _read_bronze_parquets(download_report)
    input_rows = {
        dataset_name: len(dataframe)
        for dataset_name, dataframe in bronze_datasets.items()
    }

    silver_datasets = transform_silver_data(bronze_datasets)
    output_rows = {
        dataset_name: len(dataframe)
        for dataset_name, dataframe in silver_datasets.items()
    }
    silver_files = _write_silver_parquets(silver_datasets, staging_dir)
    validation_report = validate_silver_data(
        silver_files,
        input_rows=input_rows,
    )
    if not validation_report["is_valid"]:
        raise SilverValidationError(validation_report)

    upload_report = upload_silver_files(
        silver_files=silver_files,
        row_counts=output_rows,
        bucket_name=bucket_name,
        execution_date=utc_execution_date,
        s3_client=client,
    )
    if upload_report["partition"] != partition:
        raise RuntimeError(
            "A partição Silver divergiu da data UTC definida para a execução"
        )

    report = {
        "success": True,
        "execution_date": utc_execution_date.isoformat(),
        "partition": partition,
        "bucket": bucket_name,
        "source_layer": "bronze",
        "destination_layer": "silver",
        "downloaded_count": download_report["downloaded_count"],
        "transformed_count": len(silver_datasets),
        "uploaded_count": upload_report["uploaded_count"],
        "input_rows": input_rows,
        "output_rows": output_rows,
        "validation": validation_report,
        "files": upload_report["files"],
        "duration_seconds": round(perf_counter() - started_at, 6),
    }
    logger.info(
        "Processamento Bronze para Silver concluído em %.3f segundo(s).",
        report["duration_seconds"],
    )
    return report
