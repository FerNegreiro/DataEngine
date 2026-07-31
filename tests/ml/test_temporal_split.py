from __future__ import annotations

from datetime import date

import pandas as pd

from src.ml.temporal_split import (
    build_expanding_window_folds,
    build_final_test_fold,
    temporal_holdout,
)


def _date_frame() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2023-01-06", "2026-07-28", freq="D")})


def test_expanding_window_fold_boundaries_are_exact() -> None:
    folds = build_expanding_window_folds(_date_frame())

    assert [(fold.validation_start_date, fold.validation_end_date) for fold in folds] == [
        (date(2026, 3, 31), date(2026, 4, 29)),
        (date(2026, 4, 30), date(2026, 5, 29)),
        (date(2026, 5, 30), date(2026, 6, 28)),
    ]
    assert [fold.train_end_date for fold in folds] == [
        date(2026, 3, 30),
        date(2026, 4, 29),
        date(2026, 5, 29),
    ]


def test_final_test_fold_boundaries_are_exact() -> None:
    fold = build_final_test_fold(_date_frame())
    assert fold.train_end_date == date(2026, 6, 28)
    assert fold.validation_start_date == date(2026, 6, 29)
    assert fold.validation_end_date == date(2026, 7, 28)


def test_temporal_holdout_has_no_overlap() -> None:
    holdout = temporal_holdout(
        _date_frame(),
        train_end_date=date(2026, 6, 28),
        test_start_date=date(2026, 6, 29),
        test_end_date=date(2026, 7, 28),
    )
    assert holdout.train["date"].max() < holdout.test["date"].min()
