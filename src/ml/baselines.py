from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

BaselinePredictor = Callable[[pd.DataFrame], np.ndarray]


def _nonnegative_column(features: pd.DataFrame, column: str) -> np.ndarray:
    if column not in features:
        raise ValueError(f"Feature obrigatória ausente para baseline: {column}")
    values = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return np.clip(values.to_numpy(dtype=float), a_min=0.0, a_max=None)


def predict_last_observation(features: pd.DataFrame) -> np.ndarray:
    """Prevê a última quantidade observada (lag de um dia)."""
    return _nonnegative_column(features, "lag_1")


def predict_seasonal_lag_7(features: pd.DataFrame) -> np.ndarray:
    """Prevê a quantidade observada no mesmo dia da semana anterior."""
    return _nonnegative_column(features, "lag_7")


def predict_moving_average_28(features: pd.DataFrame) -> np.ndarray:
    """Prevê a média móvel causal dos 28 dias anteriores."""
    return _nonnegative_column(features, "rolling_mean_28")


BASELINE_PREDICTORS: dict[str, BaselinePredictor] = {
    "last_observation": predict_last_observation,
    "seasonal_lag_7": predict_seasonal_lag_7,
    "moving_average_28": predict_moving_average_28,
}
