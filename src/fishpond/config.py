"""Central, environment-driven configuration of the project."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

TARGET = "fish_weight_g"
TIMESTAMP = "timestamp"
POND_ID = "pond_id"
SENSOR_COLUMNS: tuple[str, ...] = (
    "temperature_c",
    "turbidity_ntu",
    "dissolved_oxygen",
    "ph",
    "ammonia",
    "nitrate",
)
LENGTH_COLUMN = "fish_length_cm"
POPULATION_COLUMN = "population"
RAW_COLUMNS: tuple[str, ...] = (
    TIMESTAMP,
    POND_ID,
    *SENSOR_COLUMNS,
    POPULATION_COLUMN,
    LENGTH_COLUMN,
    TARGET,
)
FEATURE_INPUT_COLUMNS: tuple[str, ...] = tuple(c for c in RAW_COLUMNS if c != TARGET)


@dataclass(frozen=True)
class Settings:
    """Runtime settings of the services (API, dashboard, training CLI)."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fishpond-training"
    prediction_experiment_name: str = "fishpond-predictions"
    model_name: str = "fishpond-weight-regressor"
    champion_alias: str = "champion"
    raw_data_path: Path = Path("data/raw/fish_pond_sample.csv")
    processed_data_path: Path = Path("data/processed/fish_pond_clean.csv")
    prediction_log_path: Path = Path("runtime/predictions.csv")
    api_url: str = "http://localhost:8000"
    auto_train_on_startup: bool = False
    log_predictions_to_mlflow: bool = True
    random_state: int = 42
    test_size: float = 0.2


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from environment variables (``FISHPOND_*`` and ``MLFLOW_TRACKING_URI``)."""
    source: Mapping[str, str] = os.environ if env is None else env
    defaults = Settings()
    return Settings(
        mlflow_tracking_uri=source.get("MLFLOW_TRACKING_URI", defaults.mlflow_tracking_uri),
        experiment_name=source.get("FISHPOND_EXPERIMENT", defaults.experiment_name),
        prediction_experiment_name=source.get(
            "FISHPOND_PREDICTION_EXPERIMENT", defaults.prediction_experiment_name
        ),
        model_name=source.get("FISHPOND_MODEL_NAME", defaults.model_name),
        champion_alias=source.get("FISHPOND_CHAMPION_ALIAS", defaults.champion_alias),
        raw_data_path=Path(source.get("FISHPOND_RAW_DATA", str(defaults.raw_data_path))),
        processed_data_path=Path(source.get("FISHPOND_PROCESSED_DATA", str(defaults.processed_data_path))),
        prediction_log_path=Path(source.get("FISHPOND_PREDICTION_LOG", str(defaults.prediction_log_path))),
        api_url=source.get("FISHPOND_API_URL", defaults.api_url),
        auto_train_on_startup=_as_bool(source.get("FISHPOND_AUTO_TRAIN", str(defaults.auto_train_on_startup))),
        log_predictions_to_mlflow=_as_bool(
            source.get("FISHPOND_LOG_PREDICTIONS_TO_MLFLOW", str(defaults.log_predictions_to_mlflow))
        ),
        random_state=int(source.get("FISHPOND_RANDOM_STATE", str(defaults.random_state))),
        test_size=float(source.get("FISHPOND_TEST_SIZE", str(defaults.test_size))),
    )
