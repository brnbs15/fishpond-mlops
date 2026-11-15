"""Tests of the MLflow training pipeline and the champion / challenger promotion."""

from __future__ import annotations

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fishpond import registry, training
from fishpond.config import Settings
from fishpond.data import save_processed


def test_decide_promotion_rules() -> None:
    assert registry.decide_promotion(10.0, None).promote is True
    assert registry.decide_promotion(9.0, 10.0).promote is True
    assert registry.decide_promotion(10.0, 10.0).promote is False
    strict = registry.decide_promotion(9.5, 10.0, min_relative_improvement=0.1)
    assert strict.promote is False
    assert "not better" in strict.reason


def test_dataset_fingerprint_is_deterministic(clean_frame: pd.DataFrame) -> None:
    assert training.dataset_fingerprint(clean_frame) == training.dataset_fingerprint(clean_frame.copy())
    assert training.dataset_fingerprint(clean_frame) != training.dataset_fingerprint(clean_frame.head(10))


def test_first_model_is_promoted_and_equal_challenger_is_rejected(
    clean_frame: pd.DataFrame, mlflow_settings: Settings
) -> None:
    client = MlflowClient(tracking_uri=mlflow_settings.mlflow_tracking_uri)

    first = training.train_and_register(clean_frame, mlflow_settings, candidate_names=["ridge"])
    assert first.promoted is True
    assert first.champion_version == first.model_version == "1"
    assert first.previous_champion_metrics is None
    assert [c.name for c in first.candidates] == ["ridge"]
    champion = registry.get_champion_version(client, mlflow_settings.model_name, mlflow_settings.champion_alias)
    assert champion is not None and str(champion.version) == "1"
    assert champion.tags["status"] == "champion"

    artifacts = {item.path for item in client.list_artifacts(first.run_id, "reports")}
    assert {
        "reports/candidate_comparison.csv",
        "reports/validation_predictions.csv",
        "reports/training_config.json",
        "reports/predicted_vs_actual.png",
    } <= artifacts

    # Same data, same algorithm -> same RMSE -> the challenger must not replace the champion.
    second = training.train_and_register(clean_frame, mlflow_settings, candidate_names=["ridge"])
    assert second.promoted is False
    assert second.model_version == "2"
    assert second.champion_version == "1"
    assert second.previous_champion_metrics is not None
    rejected = client.get_model_version(mlflow_settings.model_name, "2")
    assert rejected.tags["status"] == "rejected"
    assert "not better" in rejected.tags["rejection_reason"]
    assert second.to_dict()["champion_version"] == "1"


def test_unknown_candidate_fails_fast(clean_frame: pd.DataFrame, mlflow_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        training.train_and_register(clean_frame, mlflow_settings, candidate_names=["nope"])


def test_main_trains_from_processed_file(
    clean_frame: pd.DataFrame, mlflow_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_processed(clean_frame, mlflow_settings.processed_data_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_settings.mlflow_tracking_uri)
    result = training.main(
        ["--data", str(mlflow_settings.processed_data_path), "--include-length", "--candidates", "ridge"]
    )
    assert result.promoted is True
    assert result.metrics["rmse"] > 0


def test_promotion_marks_previous_champion_as_superseded(
    clean_frame: pd.DataFrame, mlflow_settings: Settings
) -> None:
    client = MlflowClient(tracking_uri=mlflow_settings.mlflow_tracking_uri)
    training.train_and_register(clean_frame, mlflow_settings, candidate_names=["ridge"])
    better = training.train_and_register(clean_frame, mlflow_settings, candidate_names=["random_forest"])
    assert better.promoted is True
    assert client.get_model_version(mlflow_settings.model_name, "1").tags["status"] == "superseded"
    assert client.get_model_version(mlflow_settings.model_name, "2").tags["status"] == "champion"
    # re-promoting the current champion keeps its status
    registry.promote(client, mlflow_settings.model_name, "2", mlflow_settings.champion_alias)
    assert client.get_model_version(mlflow_settings.model_name, "2").tags["status"] == "champion"
