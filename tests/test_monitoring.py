"""Tests of the Evidently drift report wrapper."""

from __future__ import annotations

import pandas as pd

from fishpond import monitoring

DRIFT_TYPE = "evidently:metric_v2:ValueDrift"


def test_no_drift_between_identical_samples(clean_frame: pd.DataFrame) -> None:
    result = monitoring.compute_drift(clean_frame, clean_frame.sample(frac=0.5, random_state=0))
    assert result.dataset_drift is False
    assert result.drift_share < 0.5
    assert set(result.column_scores) == {*monitoring.NUMERIC_MONITORED, *monitoring.CATEGORICAL_MONITORED}
    assert result.html == ""


def test_shifted_sensors_are_reported_as_drift(clean_frame: pd.DataFrame) -> None:
    current = clean_frame.copy()
    for column in ("temperature_c", "nitrate", "turbidity_ntu", "dissolved_oxygen", "ammonia"):
        current[column] = current[column] * 3 + 10
    result = monitoring.compute_drift(clean_frame, current, with_html=True)
    assert result.dataset_drift is True
    assert "temperature_c" in result.drifted_columns
    assert "<html" in result.html.lower()


def test_summarize_handles_distance_metrics_and_ignores_others() -> None:
    report = {
        "metrics": [
            {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {"count": 1}},
            {"config": {"type": DRIFT_TYPE, "column": "a", "method": "Wasserstein", "threshold": 0.1}, "value": 0.3},
            {"config": {"type": DRIFT_TYPE, "column": "b", "method": "K-S p_value", "threshold": 0.05}, "value": 0.5},
        ]
    }
    result = monitoring.summarize(report, drift_share_threshold=0.5)
    assert result.drifted_columns == ["a"]
    assert result.drift_share == 0.5
    assert result.dataset_drift is True


def test_summarize_empty_report() -> None:
    result = monitoring.summarize({})
    assert result.drift_share == 0.0
    assert result.dataset_drift is False
