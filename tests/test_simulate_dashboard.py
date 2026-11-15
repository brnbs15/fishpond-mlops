"""Tests of the traffic simulator and the dashboard logic."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fishpond import dashboard, simulate
from fishpond.config import Settings
from fishpond.data import save_processed
from fishpond.orchestration import FishpondClient
from fishpond.prediction_log import PredictionLogger
from fishpond.training import train_and_register


def test_sample_records_are_json_ready(clean_frame: pd.DataFrame) -> None:
    records = simulate.sample_records(clean_frame, 30, seed=1)
    assert len(records) == 30
    assert "fish_weight_g" not in records[0] and "fish_length_cm" not in records[0]
    for record in records:
        assert isinstance(record["timestamp"], str)
        assert all(value is None or not isinstance(value, float) or math.isfinite(value) for value in record.values())


def test_sample_records_with_drift_shift_values(clean_frame: pd.DataFrame) -> None:
    normal = pd.DataFrame(simulate.sample_records(clean_frame, 50, seed=2))
    drifted = pd.DataFrame(simulate.sample_records(clean_frame, 50, seed=2, drift=True))
    assert drifted["temperature_c"].mean() == pytest.approx(normal["temperature_c"].mean() + 4.0)
    assert len(simulate.sample_records(clean_frame.head(5), 20)) == 20


def test_send_records_in_batches() -> None:
    batches: list[int] = []

    def fake(url: str, method: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        assert payload is not None
        batches.append(len(payload["records"]))
        return {"predictions": [1.0] * len(payload["records"])}

    client = FishpondClient("http://api", fetch=fake)
    sent = simulate.send_records(client, [{"a": i} for i in range(120)], batch_size=50)
    assert sent == 120
    assert batches == [50, 50, 20]


def test_simulate_main(
    clean_frame: pd.DataFrame, mlflow_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_processed(clean_frame, mlflow_settings.processed_data_path)
    monkeypatch.setenv("FISHPOND_PROCESSED_DATA", str(mlflow_settings.processed_data_path))
    monkeypatch.setattr(simulate, "send_records", lambda client, records: len(records))
    assert simulate.main(["--n", "7", "--drift", "--api-url", "http://x"]) == 7


def test_dashboard_on_fresh_system(mlflow_settings: Settings) -> None:
    data = dashboard.load_monitoring_data(mlflow_settings)
    assert data.reference.empty and data.current.empty
    assert dashboard.prediction_summary(data.current)["predictions"] == 0
    assert dashboard.predictions_by_version(data.current).empty
    assert dashboard.drift_report(data) is None
    client = MlflowClient(tracking_uri=mlflow_settings.mlflow_tracking_uri)
    assert dashboard.registry_overview(client, "does-not-exist").empty


def test_dashboard_with_traffic_and_models(clean_frame: pd.DataFrame, mlflow_settings: Settings) -> None:
    save_processed(clean_frame, mlflow_settings.processed_data_path)
    log = PredictionLogger(mlflow_settings.prediction_log_path)
    records = simulate.sample_records(clean_frame, 40, drift=True)
    log.log(records[:30], [10.0] * 30, "1", "a")
    log.log(records[30:], [20.0] * 10, "2", "b")

    data = dashboard.load_monitoring_data(mlflow_settings)
    summary = dashboard.prediction_summary(data.current)
    assert summary == {"predictions": 40, "requests": 2, "mean_prediction": 12.5, "model_versions": ["1", "2"]}
    by_version = dashboard.predictions_by_version(data.current)
    assert by_version["predictions"].tolist() == [30, 10]

    report = dashboard.drift_report(data)
    assert report is not None
    assert report.html

    settings = dataclasses.replace(mlflow_settings)
    train_and_register(clean_frame, settings, candidate_names=["ridge"])
    train_and_register(clean_frame, settings, candidate_names=["ridge"])
    overview = dashboard.registry_overview(MlflowClient(settings.mlflow_tracking_uri), settings.model_name)
    assert overview["version"].tolist() == [2, 1]
    assert overview["status"].tolist() == ["rejected", "champion"]
    assert overview["aliases"].tolist() == ["", "champion"]
