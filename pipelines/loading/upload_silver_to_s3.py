from __future__ import annotations

import logging
from collections.abc import Mapping
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
    upload_file,
)

logger = logging.getLogger(__name__)


def upload_silver_files(
    silver_files: Mapping[str, Path | str],
    row_counts: Mapping[str, int],
    bucket_name: str = BUCKET_NAME,
    execution_date: datetime | None = None,
    s3_client: Any | None = None,
) -> dict[str, object]:
    utc_execution_date = normalize_execution_date(execution_date)
    partition = build_partition(utc_execution_date)
    missing_datasets = sorted(set(FILES_TO_UPLOAD) - set(silver_files))
    if missing_datasets:
        raise FileNotFoundError(
            "Arquivos Silver não informados para: "
            f"{', '.join(missing_datasets)}"
        )

    client = s3_client if s3_client is not None else boto3.client("s3")
    uploaded_files: list[dict[str, object]] = []

    for dataset_name, filename in FILES_TO_UPLOAD.items():
        local_path = Path(silver_files[dataset_name])
        object_key = build_partitioned_key(
            dataset_name=dataset_name,
            filename=filename,
            execution_date=utc_execution_date,
            layer="silver",
        )
        try:
            upload_report = upload_file(
                s3_client=client,
                local_path=local_path,
                bucket_name=bucket_name,
                object_key=object_key,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Falha no upload da camada Silver: "
                f"arquivo ausente para dataset={dataset_name}, caminho={local_path}"
            ) from error
        except (BotoCoreError, ClientError) as error:
            raise RuntimeError(
                "Falha no upload da camada Silver: "
                f"dataset={dataset_name}, bucket={bucket_name}, "
                f"chave={object_key}: {error}"
            ) from error

        uploaded_files.append(
            {
                "dataset": dataset_name,
                **upload_report,
                "row_count": row_counts[dataset_name],
                "file_size_bytes": local_path.stat().st_size,
            }
        )

    logger.info(
        "Camada Silver finalizada. %s arquivo(s) enviado(s).",
        len(uploaded_files),
    )
    return {
        "execution_date": utc_execution_date.isoformat(),
        "partition": partition,
        "bucket": bucket_name,
        "uploaded_count": len(uploaded_files),
        "files": uploaded_files,
    }
