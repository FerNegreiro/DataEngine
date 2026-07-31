from __future__ import annotations

import pandas as pd

from src.ml.run_metadata import (
    build_pipeline_run,
    finish_pipeline_run,
    generate_run_id,
)


def test_run_id_is_deterministic_for_the_same_publication() -> None:
    timestamp = pd.Timestamp("2026-07-31T12:00:00Z")
    first = generate_run_id(timestamp, "2026-07-28")
    second = generate_run_id(timestamp, "2026-07-28")
    different = generate_run_id(timestamp + pd.Timedelta(seconds=1), "2026-07-28")
    assert first == second
    assert first != different


def test_pipeline_run_tracks_success_and_failure_without_losing_counts() -> None:
    started = pd.Timestamp("2026-07-31T12:00:00Z")
    running = build_pipeline_run(
        run_id="run-1",
        started_at=started,
        source_data_min_date="2023-01-06",
        source_data_max_date="2026-07-28",
        products_processed=52,
        forecast_rows=2652,
        risk_rows=52,
    )
    successful = finish_pipeline_run(
        running,
        status="success",
        finished_at=started + pd.Timedelta(seconds=10),
    )
    failed = finish_pipeline_run(
        running,
        status="failed",
        finished_at=started + pd.Timedelta(seconds=5),
        error_message="permission denied",
    )
    assert successful["status"] == "success"
    assert successful["duration_seconds"] == 10
    assert successful["forecast_rows"] == 2652
    assert failed["status"] == "failed"
    assert failed["error_message"] == "permission denied"
