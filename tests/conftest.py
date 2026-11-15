"""Shared pytest fixtures: small synthetic datasets that mimic the real sensor table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fishpond.config import Settings
from fishpond.data import clean_dataset


def make_raw_frame(rows: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw data: fish grow over time, sensors are noisy and sometimes wrong."""
    rng = np.random.default_rng(seed)
    ponds = np.array(["pond_01", "pond_02", "pond_03", "pond_07"])
    start = pd.Timestamp("2021-06-18")
    days = rng.uniform(0, 150, rows)
    weight = 3 + days**1.6 / 10 + rng.normal(0, 2, rows)
    frame = pd.DataFrame(
        {
            "timestamp": [(start + pd.Timedelta(days=float(d))).strftime("%Y-%m-%d %H:%M:%S") for d in days],
            "pond_id": rng.choice(ponds, rows),
            "temperature_c": rng.normal(24.7, 1.0, rows),
            "turbidity_ntu": rng.uniform(-50, 100, rows),
            "dissolved_oxygen": rng.uniform(0, 40, rows),
            "ph": rng.uniform(-3, 15, rows),
            "ammonia": rng.exponential(5, rows),
            "nitrate": rng.uniform(70, 2500, rows),
            "population": rng.choice([50.0, 75.0, np.nan], rows),
            "fish_length_cm": 6 + days / 3 + rng.normal(0, 0.5, rows),
            "fish_weight_g": weight,
        }
    )
    frame.loc[0, "ammonia"] = np.inf
    return frame


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def clean_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = clean_dataset(raw_frame)
    return cleaned


@pytest.fixture
def mlflow_settings(tmp_path: Path) -> Settings:
    """Settings pointing to a throw-away local MLflow (SQLite backend, artifacts in tmp_path)."""
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    settings = Settings(
        mlflow_tracking_uri=tracking_uri,
        processed_data_path=tmp_path / "processed.csv",
        prediction_log_path=tmp_path / "runtime" / "predictions.csv",
        test_size=0.25,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    for name in (settings.experiment_name, settings.prediction_experiment_name):
        client.create_experiment(name, artifact_location=(tmp_path / "artifacts" / name).as_uri())
    return settings


@pytest.fixture
def raw_csv(tmp_path: Path, raw_frame: pd.DataFrame) -> Path:
    path = tmp_path / "raw.csv"
    raw_frame.to_csv(path, index=False)
    return path
