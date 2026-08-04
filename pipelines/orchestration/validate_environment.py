from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import boto3
import google.auth
from google.cloud import bigquery

PROJECT_ID = "dataengine-fernando-2026"
BUCKET_NAME = "dataengine-fernando-2026"
BIGQUERY_LOCATION = "southamerica-east1"
REQUIRED_DATASETS = ("dataengine", "dataengine_dbt", "dataengine_ml")
REQUIRED_IMPORTS = (
    "airflow",
    "boto3",
    "dbt",
    "google.auth",
    "google.cloud.bigquery",
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
)
REQUIRED_PROJECT_DIRECTORIES = (
    "data",
    "dataengine_dbt",
    "pipelines",
    "src",
    "artifacts/ml",
    "artifacts/ml/experiments/iteration_02",
    "artifacts/ml/production",
)


def _failure(label: str, error: Exception) -> str:
    return f"{label}: {type(error).__name__}"


def validate_environment(
    *,
    project_root: Path | str | None = None,
    dbt_profiles_path: Path | str | None = None,
    import_module_fn: Callable[[str], object] | None = None,
    aws_session_factory: Callable[[], Any] | None = None,
    google_auth_loader: Callable[..., tuple[Any, str | None]] | None = None,
    bigquery_client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Valida dependências, caminhos e acessos externos sem alterar dados."""
    root = Path(
        project_root
        or os.environ.get("DATAENGINE_PROJECT_ROOT", "/opt/airflow/project")
    )
    profiles_path = Path(
        dbt_profiles_path
        or os.environ.get("DBT_PROFILES_PATH", "/home/airflow/.dbt/profiles.yml")
    )
    importer = import_module_fn or importlib.import_module
    session_factory = aws_session_factory or boto3.Session
    auth_loader = google_auth_loader or google.auth.default
    client_factory = bigquery_client_factory or bigquery.Client

    errors: list[str] = []
    imports: dict[str, bool] = {}
    for module_name in REQUIRED_IMPORTS:
        try:
            if module_name == "airflow" and import_module_fn is None:
                metadata.version("apache-airflow")
            else:
                importer(module_name)
        except Exception as error:
            imports[module_name] = False
            errors.append(_failure(f"Importação de {module_name} falhou", error))
        else:
            imports[module_name] = True

    required_files = (
        root / "dataengine_dbt" / "dbt_project.yml",
        profiles_path,
    )
    directory_status = {
        str(root / relative_path): (root / relative_path).is_dir()
        for relative_path in REQUIRED_PROJECT_DIRECTORIES
    }
    directory_status[str(root)] = root.is_dir()
    file_status = {str(path): path.is_file() for path in required_files}
    for path, exists in directory_status.items():
        if not exists:
            errors.append(f"Diretório obrigatório ausente: {path}")
    for path, exists in file_status.items():
        if not exists:
            errors.append(f"Arquivo obrigatório ausente: {path}")

    aws_report = {"sts": False, "bucket": False, "bucket_name": BUCKET_NAME}
    try:
        session = session_factory()
        session.client("sts").get_caller_identity()
        aws_report["sts"] = True
    except Exception as error:
        errors.append(_failure("Validação AWS STS falhou", error))
    try:
        session = session_factory()
        session.client("s3").head_bucket(Bucket=BUCKET_NAME)
        aws_report["bucket"] = True
    except Exception as error:
        errors.append(_failure(f"Acesso ao bucket {BUCKET_NAME} falhou", error))

    google_report: dict[str, Any] = {
        "adc": False,
        "project_access": False,
        "project_id": PROJECT_ID,
        "datasets": {dataset_id: False for dataset_id in REQUIRED_DATASETS},
    }
    try:
        credentials, _ = auth_loader(
            scopes=("https://www.googleapis.com/auth/cloud-platform",)
        )
        google_report["adc"] = True
        client = client_factory(
            project=PROJECT_ID,
            credentials=credentials,
            location=BIGQUERY_LOCATION,
        )
        next(iter(client.list_datasets(project=PROJECT_ID, max_results=1)), None)
        google_report["project_access"] = True
        for dataset_id in REQUIRED_DATASETS:
            try:
                client.get_dataset(f"{PROJECT_ID}.{dataset_id}")
            except Exception as error:
                errors.append(
                    _failure(f"Dataset obrigatório {dataset_id} indisponível", error)
                )
            else:
                google_report["datasets"][dataset_id] = True
    except Exception as error:
        errors.append(_failure("Validação do Google ADC ou projeto falhou", error))

    return {
        "is_valid": not errors,
        "errors": errors,
        "imports": imports,
        "paths": {"directories": directory_status, "files": file_status},
        "aws": aws_report,
        "google_cloud": google_report,
    }


def main() -> int:
    report = validate_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
