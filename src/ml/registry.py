from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd

from src.ml.config import (
    CHAMPION_MODEL,
    CHAMPION_MODEL_VERSION,
    HURDLE_VERSION,
    INTERMITTENT_MODEL_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    PRIMARY_FORECAST_HORIZON,
)

REGISTERED_MODEL_VERSIONS = {
    CHAMPION_MODEL: CHAMPION_MODEL_VERSION,
    "croston_sba": INTERMITTENT_MODEL_VERSION,
    MODEL_NAME: MODEL_VERSION,
    "hurdle_poisson": HURDLE_VERSION,
    "hurdle_squared_error": HURDLE_VERSION,
}
REGISTERED_MODELS = tuple(REGISTERED_MODEL_VERSIONS)


def model_version_for(model_name: str) -> str:
    try:
        return REGISTERED_MODEL_VERSIONS[model_name]
    except KeyError as error:
        raise ValueError(f"Modelo sem versão produtiva registrada: {model_name}") from error


def _final_primary_metrics(aggregate_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "model_name", "horizon", "wape", "bias"}
    missing = required.difference(aggregate_metrics.columns)
    if missing:
        raise ValueError(
            "Métricas sem colunas obrigatórias para o registry: "
            + ", ".join(sorted(missing))
        )
    selected = aggregate_metrics.loc[
        aggregate_metrics["split"].eq("final_test")
        & aggregate_metrics["horizon"].eq(PRIMARY_FORECAST_HORIZON)
        & aggregate_metrics["model_name"].isin(REGISTERED_MODELS)
    ].copy()
    if selected["model_name"].duplicated().any():
        raise ValueError("Métricas finais duplicadas por modelo no horizonte principal")
    missing_models = set(REGISTERED_MODELS).difference(selected["model_name"])
    if missing_models:
        raise ValueError(
            "Métricas finais ausentes para o registry: "
            + ", ".join(sorted(missing_models))
        )
    return selected.set_index("model_name")


def validate_official_promotion_decision(
    promotion_decision: dict[str, Any],
    model_comparison: dict[str, Any],
) -> None:
    if promotion_decision.get("decision") != "rejected":
        raise ValueError("A decisão oficial esperada para a iteration_02 é rejected")
    if promotion_decision.get("final_champion") != CHAMPION_MODEL:
        raise ValueError(f"O champion oficial deve permanecer {CHAMPION_MODEL}")
    if model_comparison.get("final_champion") != CHAMPION_MODEL:
        raise ValueError("model_comparison diverge do champion oficial")
    if promotion_decision.get("challenger") != "croston_sba":
        raise ValueError("O challenger oficial avaliado deve ser croston_sba")


def _rejection_reason(
    model_name: str,
    promotion_decision: dict[str, Any],
) -> str | None:
    if model_name == CHAMPION_MODEL:
        return None
    if model_name == "croston_sba":
        return str(promotion_decision.get("reason") or "Viés agregado excessivo")
    if model_name == MODEL_NAME:
        return "Rejeitado na iteration_01: não superou o baseline nos folds de validação."
    return "Rejeitado na iteration_02: desempenho final inferior ao champion aprovado."


def build_model_registry(
    aggregate_metrics: pd.DataFrame,
    promotion_decision: dict[str, Any],
    model_comparison: dict[str, Any],
    *,
    registered_at: pd.Timestamp,
    training_data_min_date: date,
    training_data_max_date: date,
    code_version: str | None,
) -> pd.DataFrame:
    validate_official_promotion_decision(promotion_decision, model_comparison)
    final_metrics = _final_primary_metrics(aggregate_metrics)
    records: list[dict[str, Any]] = []
    for model_name in REGISTERED_MODELS:
        is_champion = model_name == CHAMPION_MODEL
        metric = final_metrics.loc[model_name]
        metadata = {
            "evaluation_period": "final_test",
            "forecast_horizon": PRIMARY_FORECAST_HORIZON,
            "official_champion": CHAMPION_MODEL,
            "iteration_02_decision": promotion_decision["decision"],
            "selected_challenger": promotion_decision["challenger"],
        }
        records.append(
            {
                "model_name": model_name,
                "model_version": model_version_for(model_name),
                "registered_at": registered_at,
                "status": "champion" if is_champion else "rejected",
                "is_champion": is_champion,
                "promotion_decision": "retained" if is_champion else "rejected",
                "rejection_reason": _rejection_reason(model_name, promotion_decision),
                "primary_metric": "wape",
                "primary_metric_value": float(metric["wape"]),
                "bias": float(metric["bias"]),
                "training_data_min_date": training_data_min_date,
                "training_data_max_date": training_data_max_date,
                "code_version": code_version,
                "metadata_json": json.dumps(
                    metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    registry = pd.DataFrame.from_records(records)
    if int(registry["is_champion"].sum()) != 1:
        raise ValueError("O model_registry deve possuir exatamente um champion")
    return registry
