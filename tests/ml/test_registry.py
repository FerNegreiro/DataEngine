from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ml.config import CHAMPION_MODEL
from src.ml.production import ProductionBundle
from src.ml.registry import REGISTERED_MODELS, validate_official_promotion_decision


def test_registry_contains_one_champion_and_all_required_rejections(
    production_bundle: ProductionBundle,
) -> None:
    registry = production_bundle.model_registry
    assert set(registry["model_name"]) == set(REGISTERED_MODELS)
    champion = registry.loc[registry["is_champion"]]
    assert champion["model_name"].tolist() == [CHAMPION_MODEL]
    assert champion["status"].tolist() == ["champion"]
    rejected = registry.loc[~registry["is_champion"]]
    assert rejected["status"].eq("rejected").all()
    assert rejected["rejection_reason"].notna().all()
    assert all(json.loads(value) for value in registry["metadata_json"])


def test_registry_rejects_an_unapproved_champion() -> None:
    promotion = {
        "decision": "promoted",
        "challenger": "croston_sba",
        "final_champion": "croston_sba",
    }
    with pytest.raises(ValueError, match="rejected"):
        validate_official_promotion_decision(
            promotion, {"final_champion": "croston_sba"}
        )


def test_production_metrics_identify_champion_and_rejected_challengers(
    production_bundle: ProductionBundle,
) -> None:
    metrics = production_bundle.model_metrics
    assert set(metrics["forecast_horizon"]) == {7, 14, 30}
    champion = metrics.loc[metrics["model_name"].eq(CHAMPION_MODEL)]
    rejected = metrics.loc[metrics["model_name"].ne(CHAMPION_MODEL)]
    assert champion["champion_status"].eq("champion").all()
    assert rejected["champion_status"].eq("challenger_rejected").all()
    assert pd.to_numeric(metrics["metric_value"]).notna().all()
