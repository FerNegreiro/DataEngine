from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.config import CHAMPION_MODEL, CHAMPION_MODEL_VERSION

RUN_ID_NAMESPACE = uuid.UUID("1cfa5145-52bb-4da7-b291-f9b3eaa61db3")
CODE_VERSION_ENVIRONMENT_VARIABLES = (
    "GIT_COMMIT_SHA",
    "COMMIT_SHA",
    "GITHUB_SHA",
)


def utc_timestamp(value: pd.Timestamp | str | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def generate_run_id(
    generated_at: pd.Timestamp | str,
    data_max_date: date | str,
    model_version: str = CHAMPION_MODEL_VERSION,
) -> str:
    timestamp = utc_timestamp(generated_at).isoformat()
    maximum_date = pd.Timestamp(data_max_date).date().isoformat()
    if not model_version.strip():
        raise ValueError("model_version não pode ser vazio")
    identity = f"{CHAMPION_MODEL}|{model_version}|{maximum_date}|{timestamp}"
    return str(uuid.uuid5(RUN_ID_NAMESPACE, identity))


def resolve_code_version(repository: Path | str | None = None) -> str | None:
    for variable in CODE_VERSION_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    working_directory = Path(repository) if repository is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    version = result.stdout.strip()
    return version or None


def build_pipeline_run(
    *,
    run_id: str,
    started_at: pd.Timestamp | str,
    source_data_min_date: date | str,
    source_data_max_date: date | str,
    products_processed: int,
    forecast_rows: int,
    risk_rows: int,
    champion_model: str = CHAMPION_MODEL,
    champion_version: str = CHAMPION_MODEL_VERSION,
) -> dict[str, Any]:
    if not run_id.strip():
        raise ValueError("run_id não pode ser vazio")
    for name, value in {
        "products_processed": products_processed,
        "forecast_rows": forecast_rows,
        "risk_rows": risk_rows,
    }.items():
        if value < 0:
            raise ValueError(f"{name} não pode ser negativo")
    return {
        "run_id": run_id,
        "started_at": utc_timestamp(started_at),
        "finished_at": None,
        "status": "running",
        "source_data_min_date": pd.Timestamp(source_data_min_date).date(),
        "source_data_max_date": pd.Timestamp(source_data_max_date).date(),
        "products_processed": int(products_processed),
        "forecast_rows": int(forecast_rows),
        "risk_rows": int(risk_rows),
        "champion_model": champion_model,
        "champion_version": champion_version,
        "error_message": None,
        "duration_seconds": None,
    }


def finish_pipeline_run(
    pipeline_run: Mapping[str, Any],
    *,
    status: str,
    finished_at: pd.Timestamp | str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    if status not in {"success", "failed"}:
        raise ValueError("status final deve ser success ou failed")
    started_at = utc_timestamp(pipeline_run["started_at"])
    ending = utc_timestamp(finished_at)
    if ending < started_at:
        raise ValueError("finished_at não pode ser anterior a started_at")
    result = dict(pipeline_run)
    result.update(
        {
            "started_at": started_at,
            "finished_at": ending,
            "status": status,
            "error_message": error_message if status == "failed" else None,
            "duration_seconds": float((ending - started_at).total_seconds()),
        }
    )
    return result
