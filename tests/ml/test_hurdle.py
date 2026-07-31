from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.config import HURDLE_FEATURE_COLUMNS
from src.ml.feature_engineering import add_temporal_features
from src.ml.hurdle import (
    recursive_hurdle_forecast,
    select_occurrence_threshold,
    train_hurdle_candidates,
)

FAST_CLASSIFIER = {"max_iter": 4, "min_samples_leaf": 2}
FAST_REGRESSOR = {"max_iter": 4, "min_samples_leaf": 2}


def test_hurdle_trains_binary_and_positive_quantity_stages(ml_grid: pd.DataFrame) -> None:
    featured = add_temporal_features(ml_grid)
    candidates = train_hurdle_candidates(
        featured,
        classifier_parameters=FAST_CLASSIFIER,
        regressor_parameters=FAST_REGRESSOR,
        threshold=0.2,
    )
    assert set(candidates) == {"hurdle_poisson", "hurdle_squared_error"}

    sample = featured.loc[:, list(HURDLE_FEATURE_COLUMNS)].tail(4)
    for model in candidates.values():
        components = model.predict_components(sample)
        assert np.allclose(
            components["predicted_quantity"],
            components["sale_probability"] * components["conditional_quantity"],
        )
        assert (components["predicted_quantity"] >= 0).all()
        assert model.occurrence_model.named_steps["classifier"].class_weight == "balanced"


def test_threshold_is_selected_from_supplied_validation_probabilities_only() -> None:
    selected = select_occurrence_threshold(
        [0, 0, 1, 1],
        [0.05, 0.15, 0.25, 0.90],
        thresholds=[0.1, 0.2, 0.3, 0.5],
    )
    assert selected["threshold"] == 0.2
    assert selected["validation_f1"] == 1.0


def test_hurdle_recursive_forecast_uses_its_own_expected_predictions(
    ml_grid: pd.DataFrame,
) -> None:
    featured = add_temporal_features(ml_grid)
    model = train_hurdle_candidates(
        featured,
        conditional_losses=("poisson",),
        classifier_parameters=FAST_CLASSIFIER,
        regressor_parameters=FAST_REGRESSOR,
        threshold=0.2,
    )["hurdle_poisson"]
    forecast = recursive_hurdle_forecast(
        ml_grid,
        model,
        3,
        active_only=False,
        generated_at=pd.Timestamp("2026-07-31", tz="UTC"),
    )
    assert len(forecast) == 6
    assert forecast["predicted_quantity"].ge(0).all()
    assert forecast["sale_probability"].between(0, 1).all()
    assert forecast["occurrence_threshold"].eq(0.2).all()
