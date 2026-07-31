from __future__ import annotations

import pandas as pd
import pytest

from src.ml.config import MODEL_FEATURE_COLUMNS
from src.ml.feature_engineering import add_temporal_features
from src.ml.train import build_forecasting_pipeline, train_forecasting_model

FAST_PARAMETERS = {"max_iter": 5, "min_samples_leaf": 2}


def test_training_pipeline_predicts_nonnegative_values_with_unknown_category(
    ml_grid: pd.DataFrame,
) -> None:
    featured = add_temporal_features(ml_grid)
    model = train_forecasting_model(featured, FAST_PARAMETERS)
    future_like = featured.loc[:, list(MODEL_FEATURE_COLUMNS)].tail(2).copy()
    future_like["brand"] = "unseen-brand"

    predictions = model.predict(future_like)

    assert len(predictions) == 2
    assert (predictions >= 0).all()
    assert model.named_steps["regressor"].loss == "poisson"
    assert model.named_steps["regressor"].early_stopping is False


def test_hidden_random_early_stopping_is_rejected() -> None:
    with pytest.raises(ValueError, match="split aleatório"):
        build_forecasting_pipeline({"early_stopping": True})
