"""Tests of the environment based configuration."""

from __future__ import annotations

from pathlib import Path

from fishpond.config import FEATURE_INPUT_COLUMNS, TARGET, Settings, get_settings


def test_defaults_without_environment() -> None:
    assert get_settings({}) == Settings()


def test_environment_overrides() -> None:
    settings = get_settings(
        {
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "FISHPOND_MODEL_NAME": "m",
            "FISHPOND_PROCESSED_DATA": "/data/x.csv",
            "FISHPOND_AUTO_TRAIN": "yes",
            "FISHPOND_LOG_PREDICTIONS_TO_MLFLOW": "0",
            "FISHPOND_RANDOM_STATE": "7",
            "FISHPOND_TEST_SIZE": "0.3",
        }
    )
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"
    assert settings.model_name == "m"
    assert settings.processed_data_path == Path("/data/x.csv")
    assert settings.auto_train_on_startup is True
    assert settings.log_predictions_to_mlflow is False
    assert settings.random_state == 7
    assert settings.test_size == 0.3


def test_target_is_not_an_input_feature() -> None:
    assert TARGET not in FEATURE_INPUT_COLUMNS
