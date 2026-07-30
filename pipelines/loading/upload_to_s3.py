from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

BUCKET_NAME = "dataengine-fernando-2026"
DEFAULT_PROCESSED_DIR = Path("data/processed")

FILES_TO_UPLOAD = {
    "customers": "customers.parquet",
    "orders": "orders.parquet",
    "order_items": "order_items.parquet",
    "products": "products.parquet",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def normalize_execution_date(execution_date: datetime | None = None) -> datetime:
    if execution_date is None:
        return datetime.now(timezone.utc)
    if execution_date.tzinfo is None:
        return execution_date.replace(tzinfo=timezone.utc)
    return execution_date.astimezone(timezone.utc)


def build_partition(execution_date: datetime | None = None) -> str:
    utc_execution_date = normalize_execution_date(execution_date)
    return (
        f"year={utc_execution_date:%Y}/"
        f"month={utc_execution_date:%m}/"
        f"day={utc_execution_date:%d}"
    )


def build_partitioned_key(
    dataset_name: str,
    filename: str,
    execution_date: datetime | None = None,
    layer: str = "bronze",
) -> str:
    partition = build_partition(execution_date)
    return f"{layer}/{dataset_name}/{partition}/{filename}"


def upload_file(
    s3_client: Any,
    local_path: Path,
    bucket_name: str,
    object_key: str,
) -> dict[str, str]:
    if not local_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {local_path}")

    logger.info(
        "Enviando %s para s3://%s/%s",
        local_path,
        bucket_name,
        object_key,
    )

    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=bucket_name,
        Key=object_key,
    )

    s3_uri = f"s3://{bucket_name}/{object_key}"

    logger.info("Upload concluído: %s", s3_uri)

    return {
        "local_path": str(local_path),
        "bucket": bucket_name,
        "object_key": object_key,
        "s3_uri": s3_uri,
    }


def upload_processed_files(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    bucket_name: str = BUCKET_NAME,
    execution_date: datetime | None = None,
) -> dict[str, object]:
    processed_directory = Path(processed_dir)
    utc_execution_date = normalize_execution_date(execution_date)
    partition = build_partition(utc_execution_date)
    s3_client = boto3.client("s3")

    uploaded_files: list[dict[str, str]] = []

    for dataset_name, filename in FILES_TO_UPLOAD.items():
        local_path = processed_directory / filename
        object_key = build_partitioned_key(
            dataset_name=dataset_name,
            filename=filename,
            execution_date=utc_execution_date,
        )

        upload_report = upload_file(
            s3_client=s3_client,
            local_path=local_path,
            bucket_name=bucket_name,
            object_key=object_key,
        )

        uploaded_files.append(upload_report)

    logger.info(
        "Processo finalizado. %s arquivo(s) enviado(s).",
        len(uploaded_files),
    )

    return {
        "execution_date": utc_execution_date.isoformat(),
        "partition": partition,
        "uploaded_count": len(uploaded_files),
        "bucket": bucket_name,
        "files": uploaded_files,
    }


def main() -> int:
    try:
        upload_processed_files()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1
    except (BotoCoreError, ClientError) as error:
        logger.error("Erro ao acessar a AWS S3: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
