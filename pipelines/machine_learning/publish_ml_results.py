from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.bigquery_writer import (
    create_bigquery_client,
    ensure_ml_dataset,
    ensure_ml_tables,
    publish_ml_dataframes,
    upsert_pipeline_run,
)
from src.ml.config import (
    BIGQUERY_LOCATION,
    GCP_PROJECT_ID,
    ML_DATASET_ID,
    PRODUCTION_ARTIFACTS_DIR,
)
from src.ml.production import ProductionBundle
from src.ml.run_metadata import finish_pipeline_run, utc_timestamp
from src.validation.validate_ml_bigquery_load import (
    MLBigQueryLoadValidationError,
    MLPublicationValidationError,
    validate_ml_bigquery_load,
    validate_ml_publication,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Objeto não serializável: {type(value).__name__}")


def load_production_bundle(
    artifacts_dir: Path | str = PRODUCTION_ARTIFACTS_DIR,
) -> ProductionBundle:
    directory = Path(artifacts_dir)
    paths = {
        "manifest": directory / "publication_manifest.json",
        "forecasts": directory / "sales_forecast.parquet",
        "inventory_risk": directory / "inventory_risk.parquet",
        "model_metrics": directory / "model_metrics.parquet",
        "model_registry": directory / "model_registry.parquet",
        "pipeline_run": directory / "pipeline_run.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pacote produtivo ML incompleto: " + ", ".join(missing)
        )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    pipeline_run = json.loads(paths["pipeline_run"].read_text(encoding="utf-8"))
    pipeline_run["started_at"] = utc_timestamp(pipeline_run["started_at"])
    if pipeline_run.get("finished_at"):
        pipeline_run["finished_at"] = utc_timestamp(pipeline_run["finished_at"])
    for field in ("source_data_min_date", "source_data_max_date"):
        pipeline_run[field] = pd.Timestamp(pipeline_run[field]).date()
    return ProductionBundle(
        manifest=manifest,
        forecasts=pd.read_parquet(paths["forecasts"]),
        inventory_risk=pd.read_parquet(paths["inventory_risk"]),
        model_metrics=pd.read_parquet(paths["model_metrics"]),
        model_registry=pd.read_parquet(paths["model_registry"]),
        pipeline_run=pipeline_run,
    )


def publish_ml_results(
    bundle: ProductionBundle | None = None,
    *,
    artifacts_dir: Path | str = PRODUCTION_ARTIFACTS_DIR,
    bigquery_client: Any | None = None,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = ML_DATASET_ID,
    location: str = BIGQUERY_LOCATION,
) -> dict[str, Any]:
    publication = bundle or load_production_bundle(artifacts_dir)
    pre_load = validate_ml_publication(
        manifest=publication.manifest,
        forecasts=publication.forecasts,
        inventory_risk=publication.inventory_risk,
        model_metrics=publication.model_metrics,
        model_registry=publication.model_registry,
        pipeline_run=publication.pipeline_run,
    )
    if not pre_load["is_valid"]:
        raise MLPublicationValidationError(pre_load)

    client = bigquery_client or create_bigquery_client(project_id, location)
    dataset_report = ensure_ml_dataset(
        client,
        project_id=project_id,
        dataset_id=dataset_id,
        location=location,
    )
    table_definitions = ensure_ml_tables(
        client,
        project_id=project_id,
        dataset_id=dataset_id,
    )
    running = dict(publication.pipeline_run)
    try:
        upsert_pipeline_run(
            client,
            running,
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
        )
        table_writes = publish_ml_dataframes(
            client,
            forecasts=publication.forecasts,
            inventory_risk=publication.inventory_risk,
            model_metrics=publication.model_metrics,
            model_registry=publication.model_registry,
            run_id=str(publication.manifest["run_id"]),
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
        )
        successful = finish_pipeline_run(running, status="success")
        pipeline_write = upsert_pipeline_run(
            client,
            successful,
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
        )
        post_load = validate_ml_bigquery_load(
            client,
            run_id=str(publication.manifest["run_id"]),
            expected_forecast_rows=len(publication.forecasts),
            expected_risk_rows=len(publication.inventory_risk),
            expected_metric_rows=len(publication.model_metrics),
            expected_products=int(publication.manifest["products_processed"]),
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
        )
        if not post_load["is_valid"]:
            raise MLBigQueryLoadValidationError(post_load)
    except Exception as original_error:
        failed = finish_pipeline_run(
            running,
            status="failed",
            error_message=str(original_error),
        )
        try:
            upsert_pipeline_run(
                client,
                failed,
                project_id=project_id,
                dataset_id=dataset_id,
                location=location,
            )
        except Exception:
            pass
        raise

    return {
        "success": True,
        "run_id": publication.manifest["run_id"],
        "champion_model": publication.manifest["champion_model"],
        "champion_version": publication.manifest["champion_version"],
        "dataset": dataset_report,
        "table_definitions": table_definitions,
        "table_writes": {**table_writes, "pipeline_runs": pipeline_write},
        "pre_load_validation": pre_load,
        "post_load_validation": post_load,
        "pipeline_run": successful,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publica um pacote produtivo ML previamente validado no BigQuery."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PRODUCTION_ARTIFACTS_DIR,
        help="Diretório com o pacote produtivo completo e compatível.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        report = publish_ml_results(artifacts_dir=arguments.artifacts_dir)
    except Exception as error:
        print(f"Publicação ML falhou: {error}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
