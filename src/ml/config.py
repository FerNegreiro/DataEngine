from __future__ import annotations

from datetime import date
from pathlib import Path

GCP_PROJECT_ID = "dataengine-fernando-2026"
DBT_DATASET_ID = "dataengine_dbt"
BIGQUERY_LOCATION = "southamerica-east1"

PRIMARY_FORECAST_HORIZON = 14
FORECAST_HORIZONS = (7, 14, 30)
RANDOM_STATE = 42

ARTIFACTS_DIR = Path("artifacts/ml")
ML_STAGING_DIR = Path("data/ml_staging")
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"

EXPECTED_DATA_START_DATE = date(2023, 1, 6)
INITIAL_TRAIN_END_DATE = date(2026, 3, 30)
VALIDATION_WINDOWS = (
    (date(2026, 3, 31), date(2026, 4, 29)),
    (date(2026, 4, 30), date(2026, 5, 29)),
    (date(2026, 5, 30), date(2026, 6, 28)),
)
FINAL_TEST_START_DATE = date(2026, 6, 29)
FINAL_TEST_END_DATE = date(2026, 7, 28)

LAG_DAYS = (1, 7, 14, 28, 30)
ROLLING_WINDOWS = (7, 14, 28, 30)

CATEGORICAL_FEATURE_COLUMNS = (
    "product_id",
    "category",
    "brand",
)

NUMERIC_FEATURE_COLUMNS = (
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "time_index",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_std_30",
    "sales_last_7_days",
    "sales_last_14_days",
    "sales_last_30_days",
    "unit_price",
    "unit_cost",
)

MODEL_FEATURE_COLUMNS = CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS
INTERMITTENT_FEATURE_COLUMNS = (
    "days_since_last_sale",
    "sale_days_last_7",
    "sale_days_last_14",
    "sale_days_last_28",
    "sale_days_last_30",
    "historical_sale_probability",
    "mean_positive_demand",
    "causal_adi",
    "current_zero_streak",
)
HURDLE_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + INTERMITTENT_FEATURE_COLUMNS
TARGET_COLUMN = "quantity_sold"

MODEL_NAME = "hist_gradient_boosting_poisson"
MODEL_VERSION = "1.0.0"

DEFAULT_MODEL_PARAMETERS = {
    "loss": "poisson",
    "learning_rate": 0.08,
    "max_iter": 120,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}

ADI_THRESHOLD = 1.32
CV_SQUARED_THRESHOLD = 0.49

DEFAULT_CROSTON_ALPHA = 0.1
DEFAULT_TSB_ALPHA = 0.1
DEFAULT_TSB_BETA = 0.1

HURDLE_NAME_PREFIX = "hurdle"
HURDLE_VERSION = "2.0.0"
DEFAULT_HURDLE_CLASSIFIER_PARAMETERS = {
    "learning_rate": 0.08,
    "max_iter": 80,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}
DEFAULT_HURDLE_REGRESSOR_PARAMETERS = {
    "learning_rate": 0.08,
    "max_iter": 80,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}
HURDLE_CONDITIONAL_LOSSES = ("poisson", "squared_error")
OCCURRENCE_THRESHOLD_GRID = tuple(index / 100 for index in range(1, 51))

PROMOTION_MAX_RELATIVE_DEGRADATION = 0.10
PROMOTION_MAX_ABSOLUTE_BIAS = 0.25
PROMOTION_MAX_WORSE_SEGMENT_SHARE = 0.50
