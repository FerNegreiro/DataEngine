from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.ml.config import (
    FINAL_TEST_END_DATE,
    FINAL_TEST_START_DATE,
    INITIAL_TRAIN_END_DATE,
    VALIDATION_WINDOWS,
)


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_start_date: date
    train_end_date: date
    validation_start_date: date
    validation_end_date: date


@dataclass(frozen=True)
class TemporalHoldout:
    train: pd.DataFrame
    test: pd.DataFrame


def _date_bounds(dataframe: pd.DataFrame) -> tuple[date, date]:
    if dataframe.empty:
        raise ValueError("O dataframe temporal não pode ser vazio")
    if "date" not in dataframe:
        raise ValueError("O dataframe temporal deve possuir a coluna date")
    dates = pd.to_datetime(dataframe["date"], errors="raise").dt.date
    return dates.min(), dates.max()


def validate_required_date_range(
    dataframe: pd.DataFrame,
    required_start: date = FINAL_TEST_START_DATE,
    required_end: date = FINAL_TEST_END_DATE,
) -> None:
    minimum_date, maximum_date = _date_bounds(dataframe)
    if minimum_date > required_start or maximum_date < required_end:
        raise ValueError(
            "Período incompatível com os cortes temporais: "
            f"dados={minimum_date}..{maximum_date}, "
            f"necessário até={required_end}"
        )


def temporal_holdout(
    dataframe: pd.DataFrame,
    train_end_date: date,
    test_start_date: date,
    test_end_date: date,
) -> TemporalHoldout:
    if train_end_date >= test_start_date or test_start_date > test_end_date:
        raise ValueError("Limites inválidos para holdout temporal")
    dates = pd.to_datetime(dataframe["date"], errors="raise").dt.date
    train = dataframe.loc[dates <= train_end_date].copy()
    test = dataframe.loc[
        (dates >= test_start_date) & (dates <= test_end_date)
    ].copy()
    if train.empty or test.empty:
        raise ValueError("Treino e teste devem conter linhas")
    if pd.to_datetime(train["date"]).max() >= pd.to_datetime(test["date"]).min():
        raise ValueError("Treino e teste temporal se sobrepõem")
    return TemporalHoldout(train=train, test=test)


def build_expanding_window_folds(
    dataframe: pd.DataFrame,
    initial_train_end_date: date = INITIAL_TRAIN_END_DATE,
    validation_windows: tuple[tuple[date, date], ...] = VALIDATION_WINDOWS,
) -> list[TemporalFold]:
    minimum_date, maximum_date = _date_bounds(dataframe)
    folds: list[TemporalFold] = []
    previous_end = initial_train_end_date
    for index, (validation_start, validation_end) in enumerate(validation_windows, start=1):
        if validation_start != previous_end + timedelta(days=1):
            raise ValueError("Janelas expanding devem ser consecutivas")
        if validation_start > validation_end or validation_end > maximum_date:
            raise ValueError("Janela de validação incompatível com os dados")
        folds.append(
            TemporalFold(
                name=f"validation_fold_{index}",
                train_start_date=minimum_date,
                train_end_date=validation_start - timedelta(days=1),
                validation_start_date=validation_start,
                validation_end_date=validation_end,
            )
        )
        previous_end = validation_end
    return folds


def build_final_test_fold(dataframe: pd.DataFrame) -> TemporalFold:
    validate_required_date_range(dataframe)
    minimum_date, _ = _date_bounds(dataframe)
    return TemporalFold(
        name="final_test",
        train_start_date=minimum_date,
        train_end_date=FINAL_TEST_START_DATE - timedelta(days=1),
        validation_start_date=FINAL_TEST_START_DATE,
        validation_end_date=FINAL_TEST_END_DATE,
    )
