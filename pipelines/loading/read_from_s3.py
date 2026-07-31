from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from pipelines.loading.upload_to_s3 import (
    BUCKET_NAME,
    FILES_TO_UPLOAD,
    build_partition,
    build_partitioned_key,
    normalize_execution_date,
)

DEFAULT_SILVER_STAGING_DIR = Path("data/silver_staging")

logger = logging.getLogger(__name__)


def _missing_layer_message(
    layer_name: str,
    dataset_name: str,
    bucket_name: str,
    object_key: str,
) -> str:
    layer_label = layer_name.capitalize()
    return (
        f"Falha na leitura da camada {layer_label}: "
        f"objeto ausente para dataset={dataset_name}, "
        f"bucket={bucket_name}, chave={object_key}"
    )


def download_layer_files(
    layer_name: str,
    staging_dir: Path | str = DEFAULT_SILVER_STAGING_DIR,
    bucket_name: str = BUCKET_NAME,
    execution_date: datetime | None = None,
    s3_client: Any | None = None,
) -> dict[str, object]:
    utc_execution_date = normalize_execution_date(execution_date)
    partition = build_partition(utc_execution_date)
    client = s3_client if s3_client is not None else boto3.client("s3")
    destination_directory = Path(staging_dir) / layer_name
    destination_directory.mkdir(parents=True, exist_ok=True)
    layer_label = layer_name.capitalize()

    downloaded_files: list[dict[str, object]] = []
    for dataset_name, filename in FILES_TO_UPLOAD.items():
        object_key = build_partitioned_key(
            dataset_name=dataset_name,
            filename=filename,
            execution_date=utc_execution_date,
            layer=layer_name,
        )
        local_path = destination_directory / filename
        logger.info(
            "Baixando s3://%s/%s para %s",
            bucket_name,
            object_key,
            local_path,
        )

        try:
            client.download_file(
                Bucket=bucket_name,
                Key=object_key,
                Filename=str(local_path),
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                _missing_layer_message(
                    layer_name,
                    dataset_name,
                    bucket_name,
                    object_key,
                )
            ) from error
        except ClientError as error:
            error_code = str(error.response.get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    _missing_layer_message(
                        layer_name,
                        dataset_name,
                        bucket_name,
                        object_key,
                    )
                ) from error
            raise RuntimeError(
                f"Falha na leitura da camada {layer_label}: "
                f"dataset={dataset_name}, bucket={bucket_name}, "
                f"chave={object_key}: {error}"
            ) from error
        except BotoCoreError as error:
            raise RuntimeError(
                f"Falha na leitura da camada {layer_label}: "
                f"dataset={dataset_name}, bucket={bucket_name}, "
                f"chave={object_key}: {error}"
            ) from error

        downloaded_files.append(
            {
                "dataset": dataset_name,
                "local_path": str(local_path),
                "bucket": bucket_name,
                "object_key": object_key,
                "s3_uri": f"s3://{bucket_name}/{object_key}",
                "file_size_bytes": local_path.stat().st_size,
            }
        )

    return {
        "execution_date": utc_execution_date.isoformat(),
        "partition": partition,
        "bucket": bucket_name,
        "source_layer": layer_name,
        "downloaded_count": len(downloaded_files),
        "files": downloaded_files,
    }


def download_bronze_files(
    staging_dir: Path | str = DEFAULT_SILVER_STAGING_DIR,
    bucket_name: str = BUCKET_NAME,
    execution_date: datetime | None = None,
    s3_client: Any | None = None,
) -> dict[str, object]:
    return download_layer_files(
        layer_name="bronze",
        staging_dir=staging_dir,
        bucket_name=bucket_name,
        execution_date=execution_date,
        s3_client=s3_client,
    )


def download_silver_files(
    staging_dir: Path | str = DEFAULT_SILVER_STAGING_DIR,
    bucket_name: str = BUCKET_NAME,
    execution_date: datetime | None = None,
    s3_client: Any | None = None,
) -> dict[str, object]:
    return download_layer_files(
        layer_name="silver",
        staging_dir=staging_dir,
        bucket_name=bucket_name,
        execution_date=execution_date,
        s3_client=s3_client,
    )
