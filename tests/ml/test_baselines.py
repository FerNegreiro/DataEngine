from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.baselines import (
    predict_last_observation,
    predict_moving_average_28,
    predict_seasonal_lag_7,
)


def test_baselines_select_the_expected_causal_features() -> None:
    features = pd.DataFrame(
        {"lag_1": [2.0, -1.0], "lag_7": [3.0, np.nan], "rolling_mean_28": [4.0, 5.0]}
    )

    assert predict_last_observation(features).tolist() == [2.0, 0.0]
    assert predict_seasonal_lag_7(features).tolist() == [3.0, 0.0]
    assert predict_moving_average_28(features).tolist() == [4.0, 5.0]
