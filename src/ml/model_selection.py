from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ml.baselines import BASELINE_PREDICTORS
from src.ml.config import (
    CHAMPION_MODEL,
    FORECAST_HORIZONS,
    MODEL_NAME,
    MODEL_VERSION,
    PRIMARY_FORECAST_HORIZON,
    PROMOTION_MAX_ABSOLUTE_BIAS,
    PROMOTION_MAX_RELATIVE_DEGRADATION,
    PROMOTION_MAX_WORSE_SEGMENT_SHARE,
)
from src.ml.demand_segmentation import segment_product_demand
from src.ml.evaluate import (
    calculate_forecast_metrics,
    calculate_occurrence_metrics,
    seasonal_naive_scale,
)
from src.ml.feature_engineering import add_temporal_features
from src.ml.hurdle import (
    recursive_hurdle_forecast,
    select_occurrence_threshold,
    train_hurdle_candidates,
)
from src.ml.intermittent_demand import INTERMITTENT_METHODS, forecast_intermittent_grid
from src.ml.inventory_risk import classify_inventory_risk
from src.ml.predict import model_predictor, recursive_forecast
from src.ml.temporal_split import (
    TemporalFold,
    build_expanding_window_folds,
    build_final_test_fold,
)
from src.ml.train import train_forecasting_model


@dataclass
class ExperimentResult:
    aggregate_metrics: pd.DataFrame
    segment_metrics: pd.DataFrame
    product_metrics: pd.DataFrame
    occurrence_metrics: pd.DataFrame
    demand_segments: pd.DataFrame
    forecasts: pd.DataFrame
    inventory_risk_comparison: pd.DataFrame
    model_comparison: dict[str, Any]
    promotion_decision: dict[str, Any]
    models: dict[str, Any]


@dataclass
class _FoldOutput:
    comparison: pd.DataFrame
    segments: pd.DataFrame
    training_grid: pd.DataFrame


def _merge_actuals(
    grid: pd.DataFrame,
    forecast: pd.DataFrame,
    fold: TemporalFold,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.to_datetime(grid["date"]).dt.date
    actuals = grid.loc[
        (dates >= fold.validation_start_date) & (dates <= fold.validation_end_date),
        ["product_id", "date", "quantity_sold"],
    ].rename(columns={"date": "forecast_date", "quantity_sold": "actual_quantity"})
    comparison = forecast.merge(
        actuals,
        on=["product_id", "forecast_date"],
        how="left",
        validate="one_to_one",
    ).merge(
        segments[["product_id", "demand_pattern"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    if comparison[["actual_quantity", "demand_pattern"]].isna().any().any():
        raise ValueError(f"Dados de avaliação incompletos no fold {fold.name}")
    comparison["split"] = fold.name
    comparison["actual_occurrence"] = comparison["actual_quantity"].gt(0).astype("int8")
    return comparison


def _run_fold(
    grid: pd.DataFrame,
    fold: TemporalFold,
    *,
    occurrence_threshold: float,
    model_parameters: Mapping[str, Any] | None,
    hurdle_classifier_parameters: Mapping[str, Any] | None,
    hurdle_regressor_parameters: Mapping[str, Any] | None,
    croston_alpha: float,
    tsb_beta: float,
) -> _FoldOutput:
    dates = pd.to_datetime(grid["date"]).dt.date
    training_grid = grid.loc[dates <= fold.train_end_date].copy()
    evaluation_days = (fold.validation_end_date - fold.validation_start_date).days + 1
    featured_training = add_temporal_features(training_grid)
    segments = segment_product_demand(training_grid)
    segments.insert(0, "split", fold.name)
    segments.insert(1, "train_end_date", fold.train_end_date.isoformat())

    timestamp = pd.Timestamp.now(tz="UTC")
    forecasts: list[pd.DataFrame] = []
    poisson_model = train_forecasting_model(featured_training, model_parameters)
    forecasts.append(
        recursive_forecast(
            training_grid,
            model_predictor(poisson_model),
            evaluation_days,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            active_only=False,
            generated_at=timestamp,
        )
    )
    for baseline_name, predictor in BASELINE_PREDICTORS.items():
        forecasts.append(
            recursive_forecast(
                training_grid,
                predictor,
                evaluation_days,
                model_name=baseline_name,
                model_version="deterministic",
                active_only=False,
                generated_at=timestamp,
            )
        )
    for method in INTERMITTENT_METHODS:
        forecasts.append(
            forecast_intermittent_grid(
                training_grid,
                evaluation_days,
                method,
                alpha=croston_alpha,
                beta=tsb_beta,
                active_only=False,
                generated_at=timestamp,
            )
        )
    hurdle_models = train_hurdle_candidates(
        featured_training,
        classifier_parameters=hurdle_classifier_parameters,
        regressor_parameters=hurdle_regressor_parameters,
        threshold=occurrence_threshold,
    )
    for model in hurdle_models.values():
        forecasts.append(
            recursive_hurdle_forecast(
                training_grid,
                model,
                evaluation_days,
                active_only=False,
                generated_at=timestamp,
            )
        )

    comparisons = [
        _merge_actuals(grid, forecast, fold, segments) for forecast in forecasts
    ]
    return _FoldOutput(
        comparison=pd.concat(comparisons, ignore_index=True, sort=False),
        segments=segments,
        training_grid=training_grid,
    )


def _group_scale(training: pd.DataFrame) -> float | None:
    return seasonal_naive_scale(training)


def _metric_rows(
    comparisons: pd.DataFrame,
    fold_outputs: dict[str, _FoldOutput],
    horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    product_records: list[dict[str, Any]] = []
    normalized_horizons = tuple(sorted(set(horizons)))

    for split, split_data in comparisons.groupby("split", observed=True, sort=True):
        fold_output = fold_outputs[str(split)]
        global_scale = _group_scale(fold_output.training_grid)
        training_with_segment = fold_output.training_grid.merge(
            fold_output.segments[["product_id", "demand_pattern"]],
            on="product_id",
            how="left",
            validate="many_to_one",
        )
        product_scales = {
            product_id: _group_scale(group)
            for product_id, group in fold_output.training_grid.groupby(
                "product_id", observed=True, sort=True
            )
        }
        segment_scales = {
            pattern: _group_scale(group)
            for pattern, group in training_with_segment.groupby(
                "demand_pattern", observed=True, sort=True
            )
        }

        for model_name, model_data in split_data.groupby(
            "model_name", observed=True, sort=True
        ):
            for horizon in normalized_horizons:
                subset = model_data.loc[model_data["horizon_day"] <= horizon]
                if subset.empty:
                    continue
                aggregate_records.append(
                    {
                        "split": split,
                        "model_name": model_name,
                        "horizon": horizon,
                        **calculate_forecast_metrics(
                            subset["actual_quantity"],
                            subset["predicted_quantity"],
                            mase_denominator=global_scale,
                        ),
                        "actual_quantity_sum": float(subset["actual_quantity"].sum()),
                        "predicted_quantity_sum": float(
                            subset["predicted_quantity"].sum()
                        ),
                    }
                )
                for pattern, segment_data in subset.groupby(
                    "demand_pattern", observed=True, sort=True
                ):
                    segment_records.append(
                        {
                            "split": split,
                            "model_name": model_name,
                            "horizon": horizon,
                            "demand_pattern": pattern,
                            "product_count": int(segment_data["product_id"].nunique()),
                            **calculate_forecast_metrics(
                                segment_data["actual_quantity"],
                                segment_data["predicted_quantity"],
                                mase_denominator=segment_scales.get(pattern),
                            ),
                        }
                    )
                for product_id, product_data in subset.groupby(
                    "product_id", observed=True, sort=True
                ):
                    product_records.append(
                        {
                            "split": split,
                            "model_name": model_name,
                            "horizon": horizon,
                            "product_id": product_id,
                            "demand_pattern": product_data["demand_pattern"].iloc[0],
                            "actual_quantity_sum": float(
                                product_data["actual_quantity"].sum()
                            ),
                            **calculate_forecast_metrics(
                                product_data["actual_quantity"],
                                product_data["predicted_quantity"],
                                mase_denominator=product_scales.get(product_id),
                            ),
                        }
                    )
    return (
        pd.DataFrame.from_records(aggregate_records),
        pd.DataFrame.from_records(segment_records),
        pd.DataFrame.from_records(product_records),
    )


def _occurrence_metric_rows(
    comparisons: pd.DataFrame,
    threshold: float,
    horizons: Iterable[int],
) -> pd.DataFrame:
    hurdle_rows = comparisons.loc[comparisons["sale_probability"].notna()].copy()
    records: list[dict[str, Any]] = []
    for (split, model_name), model_data in hurdle_rows.groupby(
        ["split", "model_name"], observed=True, sort=True
    ):
        for horizon in tuple(sorted(set(horizons))):
            subset = model_data.loc[model_data["horizon_day"] <= horizon]
            records.append(
                {
                    "split": split,
                    "model_name": model_name,
                    "horizon": horizon,
                    **calculate_occurrence_metrics(
                        subset["actual_occurrence"],
                        subset["sale_probability"],
                        threshold=threshold,
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _product_win_counts(
    product_metrics: pd.DataFrame,
    splits: Iterable[str],
) -> dict[str, int]:
    selected = product_metrics.loc[
        product_metrics["split"].isin(list(splits))
        & product_metrics["horizon"].eq(PRIMARY_FORECAST_HORIZON)
        & product_metrics["wape"].notna()
    ]
    averaged = selected.groupby(
        ["product_id", "model_name"], as_index=False, observed=True
    )["wape"].mean()
    pivot = averaged.pivot(index="product_id", columns="model_name", values="wape")
    counts = {str(column): 0 for column in pivot.columns}
    for _, row in pivot.iterrows():
        minimum = row.min(skipna=True)
        for model_name, value in row.items():
            if pd.notna(value) and np.isclose(value, minimum):
                counts[str(model_name)] += 1
    return counts


def _select_challenger(aggregate_metrics: pd.DataFrame) -> tuple[str, dict[str, float]]:
    validation = aggregate_metrics.loc[
        aggregate_metrics["split"].str.startswith("validation_fold_")
        & aggregate_metrics["horizon"].eq(PRIMARY_FORECAST_HORIZON)
        & aggregate_metrics["wape"].notna()
    ]
    means = validation.groupby("model_name", observed=True)["wape"].mean().to_dict()
    candidates = {
        str(name): float(value)
        for name, value in means.items()
        if name != CHAMPION_MODEL
    }
    if not candidates:
        raise ValueError("Nenhum challenger possui WAPE de validação válido")
    return min(candidates, key=lambda name: (candidates[name], name)), candidates


def decide_promotion(
    aggregate_metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    challenger: str,
) -> dict[str, Any]:
    primary = aggregate_metrics.loc[aggregate_metrics["horizon"].eq(14)]
    validation = primary.loc[primary["split"].str.startswith("validation_fold_")]
    pivot = validation.pivot(index="split", columns="model_name", values="wape")
    comparable = pivot.dropna(subset=[challenger, CHAMPION_MODEL])
    fold_wins = int((comparable[challenger] < comparable[CHAMPION_MODEL]).sum())
    relative_degradation = (
        (comparable[challenger] - comparable[CHAMPION_MODEL])
        / comparable[CHAMPION_MODEL]
    )
    no_material_fold_degradation = bool(
        (relative_degradation <= PROMOTION_MAX_RELATIVE_DEGRADATION).all()
    )

    final_test = primary.loc[primary["split"].eq("final_test")].set_index("model_name")
    final_metrics_available = challenger in final_test.index and CHAMPION_MODEL in final_test.index
    final_not_worse = (
        bool(final_test.loc[challenger, "wape"] <= final_test.loc[CHAMPION_MODEL, "wape"])
        if final_metrics_available
        else None
    )
    acceptable_bias = (
        bool(abs(float(final_test.loc[challenger, "bias"])) <= PROMOTION_MAX_ABSOLUTE_BIAS)
        if final_metrics_available and pd.notna(final_test.loc[challenger, "bias"])
        else None
    )

    final_segments = segment_metrics.loc[
        segment_metrics["split"].eq("final_test")
        & segment_metrics["horizon"].eq(14)
        & segment_metrics["model_name"].isin([challenger, CHAMPION_MODEL])
    ]
    segment_pivot = final_segments.pivot(
        index="demand_pattern", columns="model_name", values="wape"
    ).dropna(subset=[challenger, CHAMPION_MODEL])
    if segment_pivot.empty:
        segment_condition: bool | None = None
        worse_segment_share: float | None = None
    else:
        segment_degradation = (
            (segment_pivot[challenger] - segment_pivot[CHAMPION_MODEL])
            / segment_pivot[CHAMPION_MODEL]
        )
        worse_segment_share = float(
            (segment_degradation > PROMOTION_MAX_RELATIVE_DEGRADATION).mean()
        )
        segment_condition = worse_segment_share <= PROMOTION_MAX_WORSE_SEGMENT_SHARE

    challenger_forecasts = comparisons.loc[comparisons["model_name"].eq(challenger)]
    quality_passed = bool(
        not challenger_forecasts.empty
        and challenger_forecasts["predicted_quantity"].notna().all()
        and np.isfinite(challenger_forecasts["predicted_quantity"]).all()
        and challenger_forecasts["predicted_quantity"].ge(0).all()
        and not challenger_forecasts.duplicated(
            ["split", "product_id", "forecast_date", "model_name"]
        ).any()
    )
    criteria: dict[str, bool | None] = {
        "wins_at_least_two_validation_folds": fold_wins >= 2,
        "no_material_validation_degradation": no_material_fold_degradation,
        "not_worse_on_final_test": final_not_worse,
        "acceptable_aggregate_bias": acceptable_bias,
        "does_not_worsen_most_segments": segment_condition,
        "forecast_quality_passed": quality_passed,
    }
    if any(value is None for value in criteria.values()):
        status = "inconclusive"
    elif all(bool(value) for value in criteria.values()):
        status = "promoted"
    else:
        status = "rejected"
    failed = [name for name, value in criteria.items() if value is False]
    unavailable = [name for name, value in criteria.items() if value is None]
    return {
        "decision": status,
        "challenger": challenger,
        "previous_champion": CHAMPION_MODEL,
        "final_champion": challenger if status == "promoted" else CHAMPION_MODEL,
        "criteria": criteria,
        "validation_fold_wins": fold_wins,
        "validation_fold_count": int(len(comparable)),
        "maximum_relative_fold_degradation": (
            float(relative_degradation.max()) if not relative_degradation.empty else None
        ),
        "worse_segment_share": worse_segment_share,
        "thresholds": {
            "material_relative_degradation": PROMOTION_MAX_RELATIVE_DEGRADATION,
            "maximum_absolute_bias": PROMOTION_MAX_ABSOLUTE_BIAS,
            "maximum_worse_segment_share": PROMOTION_MAX_WORSE_SEGMENT_SHARE,
        },
        "failed_criteria": failed,
        "unavailable_criteria": unavailable,
        "reason": (
            "Todos os critérios objetivos foram satisfeitos."
            if status == "promoted"
            else "Critérios não satisfeitos: " + ", ".join(failed or unavailable)
        ),
    }


def _future_forecasts_and_models(
    grid: pd.DataFrame,
    horizon: int,
    threshold: float,
    *,
    model_parameters: Mapping[str, Any] | None,
    hurdle_classifier_parameters: Mapping[str, Any] | None,
    hurdle_regressor_parameters: Mapping[str, Any] | None,
    croston_alpha: float,
    tsb_beta: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    featured = add_temporal_features(grid)
    timestamp = pd.Timestamp.now(tz="UTC")
    models: dict[str, Any] = {}
    poisson_model = train_forecasting_model(featured, model_parameters)
    models[MODEL_NAME] = poisson_model
    forecasts = [
        recursive_forecast(
            grid,
            model_predictor(poisson_model),
            horizon,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            active_only=True,
            generated_at=timestamp,
        )
    ]
    for baseline_name, predictor in BASELINE_PREDICTORS.items():
        forecasts.append(
            recursive_forecast(
                grid,
                predictor,
                horizon,
                model_name=baseline_name,
                model_version="deterministic",
                active_only=True,
                generated_at=timestamp,
            )
        )
    for method in INTERMITTENT_METHODS:
        forecasts.append(
            forecast_intermittent_grid(
                grid,
                horizon,
                method,
                alpha=croston_alpha,
                beta=tsb_beta,
                active_only=True,
                generated_at=timestamp,
            )
        )
    hurdle_models = train_hurdle_candidates(
        featured,
        classifier_parameters=hurdle_classifier_parameters,
        regressor_parameters=hurdle_regressor_parameters,
        threshold=threshold,
    )
    for name, model in hurdle_models.items():
        models[name] = model
        forecasts.append(
            recursive_hurdle_forecast(
                grid,
                model,
                horizon,
                active_only=True,
                generated_at=timestamp,
            )
        )
    future = pd.concat(forecasts, ignore_index=True, sort=False)
    future.insert(0, "split", "future")
    return future, models


def _inventory_risk_scenarios(
    future_forecasts: pd.DataFrame,
    products: pd.DataFrame,
    final_champion: str,
    challenger: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    scenarios: list[pd.DataFrame] = []
    for model_name, forecasts in future_forecasts.groupby(
        "model_name", observed=True, sort=True
    ):
        risk = classify_inventory_risk(
            forecasts,
            products,
            horizon=PRIMARY_FORECAST_HORIZON,
        )
        risk["model_name"] = model_name
        risk["risk_class"] = risk["risk_class"].astype(str)
        scenarios.append(risk)
    comparison = pd.concat(scenarios, ignore_index=True)
    champion_classes = comparison.loc[
        comparison["model_name"].eq(final_champion),
        ["product_id", "risk_class"],
    ].rename(columns={"risk_class": "champion_risk_class"})
    comparison = comparison.merge(
        champion_classes,
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    comparison["champion_model"] = final_champion
    comparison["is_champion"] = comparison["model_name"].eq(final_champion)
    comparison["is_selected_challenger"] = comparison["model_name"].eq(challenger)
    comparison["changed_vs_champion"] = (
        comparison["risk_class"] != comparison["champion_risk_class"]
    )
    change_counts = {
        str(model_name): int(group["changed_vs_champion"].sum())
        for model_name, group in comparison.groupby(
            "model_name", observed=True, sort=True
        )
    }
    return comparison, change_counts


def evaluate_iteration_02(
    grid: pd.DataFrame,
    products: pd.DataFrame,
    *,
    forecast_horizon: int = PRIMARY_FORECAST_HORIZON,
    model_parameters: Mapping[str, Any] | None = None,
    hurdle_classifier_parameters: Mapping[str, Any] | None = None,
    hurdle_regressor_parameters: Mapping[str, Any] | None = None,
    croston_alpha: float = 0.1,
    tsb_beta: float = 0.1,
) -> ExperimentResult:
    validation_folds = build_expanding_window_folds(grid)
    final_fold = build_final_test_fold(grid)
    fold_outputs: dict[str, _FoldOutput] = {}

    for fold in validation_folds:
        fold_outputs[fold.name] = _run_fold(
            grid,
            fold,
            occurrence_threshold=0.5,
            model_parameters=model_parameters,
            hurdle_classifier_parameters=hurdle_classifier_parameters,
            hurdle_regressor_parameters=hurdle_regressor_parameters,
            croston_alpha=croston_alpha,
            tsb_beta=tsb_beta,
        )
    validation_comparisons = pd.concat(
        [fold_outputs[fold.name].comparison for fold in validation_folds],
        ignore_index=True,
        sort=False,
    )
    threshold_rows = validation_comparisons.loc[
        validation_comparisons["model_name"].eq("hurdle_poisson")
    ]
    threshold_result = select_occurrence_threshold(
        threshold_rows["actual_occurrence"], threshold_rows["sale_probability"]
    )
    selected_threshold = threshold_result["threshold"]
    for output in fold_outputs.values():
        hurdle_mask = output.comparison["sale_probability"].notna()
        output.comparison.loc[hurdle_mask, "predicted_occurrence"] = (
            output.comparison.loc[hurdle_mask, "sale_probability"]
            >= selected_threshold
        ).astype("int8")
        output.comparison.loc[hurdle_mask, "occurrence_threshold"] = selected_threshold

    fold_outputs[final_fold.name] = _run_fold(
        grid,
        final_fold,
        occurrence_threshold=selected_threshold,
        model_parameters=model_parameters,
        hurdle_classifier_parameters=hurdle_classifier_parameters,
        hurdle_regressor_parameters=hurdle_regressor_parameters,
        croston_alpha=croston_alpha,
        tsb_beta=tsb_beta,
    )
    comparisons = pd.concat(
        [fold_outputs[fold.name].comparison for fold in [*validation_folds, final_fold]],
        ignore_index=True,
        sort=False,
    )
    demand_segments = pd.concat(
        [fold_outputs[fold.name].segments for fold in [*validation_folds, final_fold]],
        ignore_index=True,
    )
    aggregate, segments, products_metrics = _metric_rows(
        comparisons, fold_outputs, FORECAST_HORIZONS
    )
    occurrence = _occurrence_metric_rows(
        comparisons, selected_threshold, FORECAST_HORIZONS
    )
    challenger, challenger_validation_means = _select_challenger(aggregate)
    promotion = decide_promotion(aggregate, segments, comparisons, challenger)

    generation_horizon = max(forecast_horizon, PRIMARY_FORECAST_HORIZON)
    future_forecasts, models = _future_forecasts_and_models(
        grid,
        generation_horizon,
        selected_threshold,
        model_parameters=model_parameters,
        hurdle_classifier_parameters=hurdle_classifier_parameters,
        hurdle_regressor_parameters=hurdle_regressor_parameters,
        croston_alpha=croston_alpha,
        tsb_beta=tsb_beta,
    )
    risk, risk_changes = _inventory_risk_scenarios(
        future_forecasts,
        products,
        promotion["final_champion"],
        challenger,
    )
    validation_split_names = [fold.name for fold in validation_folds]
    model_comparison = {
        "primary_horizon": PRIMARY_FORECAST_HORIZON,
        "occurrence_threshold_selection": threshold_result,
        "selected_challenger": challenger,
        "validation_mean_wape_by_challenger": challenger_validation_means,
        "product_win_count_method": "tie_inclusive_minimum_valid_product_wape",
        "validation_product_win_counts": _product_win_counts(
            products_metrics, validation_split_names
        ),
        "final_test_product_win_counts": _product_win_counts(
            products_metrics, ["final_test"]
        ),
        "risk_changes_vs_final_champion": risk_changes,
        "final_champion": promotion["final_champion"],
    }
    return ExperimentResult(
        aggregate_metrics=aggregate,
        segment_metrics=segments,
        product_metrics=products_metrics,
        occurrence_metrics=occurrence,
        demand_segments=demand_segments,
        forecasts=future_forecasts,
        inventory_risk_comparison=risk,
        model_comparison=model_comparison,
        promotion_decision=promotion,
        models=models,
    )
