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
