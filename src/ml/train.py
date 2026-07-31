from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.ml.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    DEFAULT_MODEL_PARAMETERS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def build_forecasting_pipeline(
    model_parameters: Mapping[str, Any] | None = None,
) -> Pipeline:
    parameters = {**DEFAULT_MODEL_PARAMETERS, **(model_parameters or {})}
    if parameters.get("early_stopping") is not False:
        raise ValueError(
            "early_stopping deve permanecer False para não criar split aleatório oculto"
        )

    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=float,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_encoder,
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
            ("numeric", "passthrough", list(NUMERIC_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    regressor = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL_FEATURE_COLUMNS))),
        **parameters,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )


def train_forecasting_model(
    featured_grid: pd.DataFrame,
    model_parameters: Mapping[str, Any] | None = None,
) -> Pipeline:
    required = {*MODEL_FEATURE_COLUMNS, TARGET_COLUMN}
    missing = required.difference(featured_grid.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para treino: {', '.join(sorted(missing))}")
    if featured_grid.empty:
        raise ValueError("O conjunto de treino não pode ser vazio")

    target = pd.to_numeric(featured_grid[TARGET_COLUMN], errors="raise")
    if target.isna().any() or (target < 0).any():
        raise ValueError("O target quantity_sold deve ser não nulo e não negativo")
    if target.sum() <= 0:
        raise ValueError("A loss Poisson requer ao menos uma venda positiva no treino")

    model = build_forecasting_pipeline(model_parameters)
    model.fit(featured_grid.loc[:, list(MODEL_FEATURE_COLUMNS)], target)
    return model
