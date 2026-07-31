from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.config import ADI_THRESHOLD, CV_SQUARED_THRESHOLD


def demand_statistics(quantity: pd.Series) -> dict[str, float | int | str]:
    """Calcula ADI e CV² no contrato Syntetos-Boylan-Croston.

    ADI é o número de períodos dividido pelo número de demandas positivas. CV² é
    o quadrado do coeficiente de variação populacional dos tamanhos positivos.
    Os limites clássicos são ADI=1,32 e CV²=0,49.
    """
    values = pd.to_numeric(quantity, errors="raise").to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("A série de demanda não pode ser vazia")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("A demanda deve conter somente valores finitos e não negativos")
    positive = values[values > 0]
    non_zero_days = int(positive.size)
    if non_zero_days == 0:
        adi = np.inf
        cv_squared = 0.0
    else:
        adi = float(values.size / non_zero_days)
        mean_positive = float(positive.mean())
        cv_squared = (
            float(np.square(positive.std(ddof=0) / mean_positive))
            if mean_positive > 0
            else 0.0
        )
    return {
        "non_zero_demand_days": non_zero_days,
        "adi": adi,
        "cv_squared": cv_squared,
        "demand_pattern": classify_demand_pattern(adi, cv_squared),
    }


def classify_demand_pattern(
    adi: float,
    cv_squared: float,
    *,
    adi_threshold: float = ADI_THRESHOLD,
    cv_squared_threshold: float = CV_SQUARED_THRESHOLD,
) -> str:
    if adi < 0 or cv_squared < 0:
        raise ValueError("ADI e CV² devem ser não negativos")
    frequent = adi < adi_threshold
    stable_size = cv_squared < cv_squared_threshold
    if frequent and stable_size:
        return "smooth"
    if not frequent and stable_size:
        return "intermittent"
    if frequent and not stable_size:
        return "erratic"
    return "lumpy"


def segment_product_demand(training_grid: pd.DataFrame) -> pd.DataFrame:
    required = {"product_id", "date", "quantity_sold"}
    missing = required.difference(training_grid.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para segmentação: {', '.join(sorted(missing))}")
    if training_grid.empty:
        raise ValueError("O período de treinamento não pode ser vazio")
    if training_grid.duplicated(["product_id", "date"]).any():
        raise ValueError("A segmentação exige uma linha por produto e data")

    records: list[dict[str, object]] = []
    for product_id, group in training_grid.groupby("product_id", observed=True, sort=True):
        records.append(
            {
                "product_id": product_id,
                **demand_statistics(group.sort_values("date")["quantity_sold"]),
            }
        )
    return pd.DataFrame.from_records(records)
