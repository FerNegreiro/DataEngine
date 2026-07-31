from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.ml.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    DEFAULT_HURDLE_CLASSIFIER_PARAMETERS,
    DEFAULT_HURDLE_REGRESSOR_PARAMETERS,
    HURDLE_CONDITIONAL_LOSSES,
    HURDLE_FEATURE_COLUMNS,
    HURDLE_NAME_PREFIX,
    HURDLE_VERSION,
    OCCURRENCE_THRESHOLD_GRID,
    TARGET_COLUMN,
)
from src.ml.predict import _build_step_features, _validate_history_grid


@dataclass
class HurdleModel:
    occurrence_model: Pipeline
    quantity_model: Pipeline
    conditional_loss: str
    threshold: float = 0.5

    @property
    def model_name(self) -> str:
        return f"{HURDLE_NAME_PREFIX}_{self.conditional_loss}"

    def predict_components(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        selected = features.loc[:, list(HURDLE_FEATURE_COLUMNS)]
        probabilities = self.occurrence_model.predict_proba(selected)[:, 1]
        conditional = np.clip(
            self.quantity_model.predict(selected), a_min=0.0, a_max=None
        )
        expected = np.clip(probabilities * conditional, a_min=0.0, a_max=None)
        return {
            "sale_probability": np.asarray(probabilities, dtype=float),
            "conditional_quantity": np.asarray(conditional, dtype=float),
            "predicted_occurrence": (probabilities >= self.threshold).astype("int8"),
            "predicted_quantity": np.asarray(expected, dtype=float),
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_components(features)["predicted_quantity"]


def _build_preprocessor() -> ColumnTransformer:
    numeric_columns = [
        column
        for column in HURDLE_FEATURE_COLUMNS
        if column not in CATEGORICAL_FEATURE_COLUMNS
    ]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=float,
                ),
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _reject_hidden_random_split(parameters: Mapping[str, Any]) -> None:
    if parameters.get("early_stopping") is not False:
        raise ValueError(
            "early_stopping deve permanecer False para não criar split aleatório oculto"
        )


def train_hurdle_candidates(
    featured_grid: pd.DataFrame,
    *,
    conditional_losses: Iterable[str] = HURDLE_CONDITIONAL_LOSSES,
    classifier_parameters: Mapping[str, Any] | None = None,
    regressor_parameters: Mapping[str, Any] | None = None,
    threshold: float = 0.5,
) -> dict[str, HurdleModel]:
    required = {*HURDLE_FEATURE_COLUMNS, TARGET_COLUMN}
    missing = required.difference(featured_grid.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para hurdle: {', '.join(sorted(missing))}")
    if featured_grid.empty:
        raise ValueError("O conjunto de treino hurdle não pode ser vazio")
    if not 0 < threshold < 1:
        raise ValueError("threshold deve estar entre zero e um")

    target = pd.to_numeric(featured_grid[TARGET_COLUMN], errors="raise")
    occurrence_target = target.gt(0).astype("int8")
    if occurrence_target.nunique() < 2:
        raise ValueError("A etapa de ocorrência requer exemplos positivos e negativos")
    positive_mask = target.gt(0)
    features = featured_grid.loc[:, list(HURDLE_FEATURE_COLUMNS)]

    classifier_config = {
        **DEFAULT_HURDLE_CLASSIFIER_PARAMETERS,
        **(classifier_parameters or {}),
    }
    _reject_hidden_random_split(classifier_config)
    occurrence_model = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            (
                "classifier",
                HistGradientBoostingClassifier(**classifier_config),
            ),
        ]
    )
    occurrence_model.fit(features, occurrence_target)

    regressor_config = {
        **DEFAULT_HURDLE_REGRESSOR_PARAMETERS,
        **(regressor_parameters or {}),
    }
    _reject_hidden_random_split(regressor_config)
    candidates: dict[str, HurdleModel] = {}
    for loss in tuple(conditional_losses):
        if loss not in HURDLE_CONDITIONAL_LOSSES:
            raise ValueError(f"Loss condicional não suportada: {loss}")
        quantity_model = Pipeline(
            steps=[
                ("preprocess", _build_preprocessor()),
                (
                    "regressor",
                    HistGradientBoostingRegressor(loss=loss, **regressor_config),
                ),
            ]
        )
        quantity_model.fit(features.loc[positive_mask], target.loc[positive_mask])
        model = HurdleModel(
            occurrence_model=occurrence_model,
            quantity_model=quantity_model,
            conditional_loss=loss,
            threshold=threshold,
        )
        candidates[model.model_name] = model
    return candidates


def select_occurrence_threshold(
    actual_occurrence: Iterable[int | bool],
    probabilities: Iterable[float],
    *,
    thresholds: Iterable[float] = OCCURRENCE_THRESHOLD_GRID,
) -> dict[str, float]:
    actual = np.asarray(list(actual_occurrence), dtype="int8")
    predicted_probability = np.asarray(list(probabilities), dtype=float)
    if actual.shape != predicted_probability.shape or actual.size == 0:
        raise ValueError("Ocorrências e probabilidades devem ter o mesmo tamanho positivo")
    if not np.isfinite(predicted_probability).all():
        raise ValueError("Probabilidades devem ser finitas")
    candidates = tuple(float(value) for value in thresholds)
    if not candidates or any(not 0 < value < 1 for value in candidates):
        raise ValueError("Todos os thresholds devem estar entre zero e um")

    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = predicted_probability >= threshold
        f1 = float(f1_score(actual, predicted, zero_division=0))
        recall = float(recall_score(actual, predicted, zero_division=0))
        scored.append((f1, recall, threshold))
    best_f1, best_recall, best_threshold = max(
        scored, key=lambda values: (values[0], values[1], -values[2])
    )
    return {
        "threshold": best_threshold,
        "validation_f1": best_f1,
        "validation_recall": best_recall,
    }


def recursive_hurdle_forecast(
    history_grid: pd.DataFrame,
    model: HurdleModel,
    horizon: int,
    *,
    active_only: bool = True,
    generated_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon deve ser positivo")
    history = _validate_history_grid(history_grid)
    first_history_date = history["date"].min()
    origin_date = history["date"].max()
    latest_products = (
        history.sort_values("date")
        .groupby("product_id", observed=True, sort=True)
        .tail(1)
        .set_index("product_id")
    )
    if active_only:
        latest_products = latest_products.loc[latest_products["is_active"].astype(bool)]
    if latest_products.empty:
        raise ValueError("Nenhum produto elegível para previsão hurdle")

    product_ids = latest_products.index.tolist()
    quantities = {
        product_id: group["quantity_sold"].astype(float).tolist()
        for product_id, group in history.groupby(
            "product_id", observed=True, sort=True
        )
        if product_id in latest_products.index
    }
    timestamp = generated_at or pd.Timestamp.now(tz="UTC")
    output: list[dict[str, Any]] = []
    for horizon_day in range(1, horizon + 1):
        forecast_date = origin_date + pd.Timedelta(days=horizon_day)
        features = _build_step_features(
            product_ids,
            quantities,
            latest_products,
            forecast_date,
            first_history_date,
        )
        components = model.predict_components(features)
        for index, product_id in enumerate(product_ids):
            prediction = float(components["predicted_quantity"][index])
            quantities[product_id].append(prediction)
            output.append(
                {
                    "product_id": product_id,
                    "forecast_date": forecast_date,
                    "horizon_day": horizon_day,
                    "predicted_quantity": prediction,
                    "sale_probability": float(components["sale_probability"][index]),
                    "conditional_quantity": float(
                        components["conditional_quantity"][index]
                    ),
                    "predicted_occurrence": int(
                        components["predicted_occurrence"][index]
                    ),
                    "occurrence_threshold": model.threshold,
                    "model_name": model.model_name,
                    "model_version": HURDLE_VERSION,
                    "generated_at": timestamp,
                }
            )
    return pd.DataFrame.from_records(output)
