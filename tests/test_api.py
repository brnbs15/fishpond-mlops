"""Tests of the REST API: real MLflow (SQLite) end-to-end flow plus the error paths."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from fishpond import api
from fishpond.config import Settings
from fishpond.data import save_processed
from fishpond.service import LoadedModel, ModelService
from fishpond.training import TrainingResult, train_and_register

READING: dict[str, Any] = {
    "timestamp": "2021-08-15 12:00:00",
    "pond_id": "pond_02",
    "temperature_c": 24.8,
    "turbidity_ntu": 85.0,
    "dissolved_oxygen": 6.5,
    "ph": 7.2,
    "ammonia": 2.4,
    "nitrate": 540.0,
    "population": 50,
}


def ridge_only_trainer(
    frame: pd.DataFrame, settings: Settings, include_length: bool, candidates: list[str] | None, improvement: float
) -> TrainingResult:
    """Real training, restricted to the fastest candidate to keep the tests quick."""
    return train_and_register(frame, settings, include_length, candidates or ["ridge"], improvement)


@pytest.fixture
def state(mlflow_settings: Settings, clean_frame: pd.DataFrame) -> api.AppState:
    save_processed(clean_frame, mlflow_settings.processed_data_path)
    app_state = api.build_state(mlflow_settings)
    app_state.trainer = ridge_only_trainer
    return app_state


@pytest.fixture
def client(state: api.AppState) -> Iterator[TestClient]:
    with TestClient(api.create_app(state)) as test_client:
        yield test_client


def test_end_to_end_train_predict_monitor(client: TestClient, state: api.AppState) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False, "model_version": None}
    assert client.post("/predict", json={"records": [READING]}).status_code == 503
    assert client.get("/monitoring/drift").status_code == 404

    trained = client.post("/train", json={"candidates": ["ridge"]})
    assert trained.status_code == 200, trained.text
    body = trained.json()
    assert body["promoted"] is True
    assert body["model_version"] == "1"
    assert body["candidates"][0]["name"] == "ridge"
    assert client.get("/health").json()["model_version"] == "1"
    assert client.get("/model").json()["version"] == "1"

    predicted = client.post("/predict", json={"records": [READING, {**READING, "pond_id": "pond_99"}]})
    assert predicted.status_code == 200, predicted.text
    prediction = predicted.json()
    assert prediction["model_version"] == "1"
    assert len(prediction["predictions"]) == 2

    logged = state.predictions.read()
    assert logged["request_id"].unique().tolist() == [prediction["request_id"]]
    assert {entry["path"] for entry in state.requests.read()} >= {"/health", "/train", "/predict"}

    mlflow_client = MlflowClient(tracking_uri=state.settings.mlflow_tracking_uri)
    experiment = mlflow_client.get_experiment_by_name(state.settings.prediction_experiment_name)
    assert experiment is not None
    runs = mlflow_client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert {a.path for a in mlflow_client.list_artifacts(runs[0].info.run_id)} == {"request.json", "predictions.csv"}

    drift = client.get("/monitoring/drift")
    assert drift.status_code == 200
    assert drift.json()["current_rows"] == 2

    again = client.post("/train", json={"candidates": ["ridge"]}).json()
    assert again["promoted"] is False
    assert again["champion_version"] == "1"


def test_prediction_experiment_is_created_when_missing(state: api.AppState) -> None:
    settings = dataclasses.replace(state.settings, prediction_experiment_name="brand-new-predictions")
    state.settings = settings
    with TestClient(api.create_app(state)) as test_client:
        test_client.post("/train", json={"candidates": ["ridge"]})
        assert test_client.post("/predict", json={"records": [READING]}).status_code == 200
    assert MlflowClient(settings.mlflow_tracking_uri).get_experiment_by_name("brand-new-predictions") is not None


def test_startup_auto_trains_first_model(state: api.AppState) -> None:
    state.settings = dataclasses.replace(state.settings, auto_train_on_startup=True)
    state.service.settings = state.settings
    with TestClient(api.create_app(state)) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is True


def test_startup_loads_existing_champion(state: api.AppState) -> None:
    ridge_only_trainer(state.load_data(), state.settings, False, None, 0.0)
    with TestClient(api.create_app(state)) as test_client:
        assert test_client.get("/health").json()["model_version"] == "1"


def test_startup_survives_unreachable_registry(state: api.AppState) -> None:
    class BrokenService(ModelService):
        def reload(self) -> LoadedModel:
            raise ConnectionError("registry down")

    state.service = BrokenService(state.settings, client=state.client)
    with TestClient(api.create_app(state)) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False


def test_auto_train_without_promotion_keeps_api_without_model(state: api.AppState) -> None:
    def not_promoting(*args: Any) -> TrainingResult:
        return TrainingResult("run", "ridge", {"rmse": 1.0}, "1", False, "test", None, None)

    state.settings = dataclasses.replace(state.settings, auto_train_on_startup=True)
    state.trainer = not_promoting
    with TestClient(api.create_app(state)) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False


def test_train_error_responses(client: TestClient, state: api.AppState) -> None:
    unknown = client.post("/train", json={"candidates": ["magic"]})
    assert unknown.status_code == 422
    assert "Unknown" in unknown.json()["detail"]
    assert client.post("/train", json={"min_relative_improvement": 2}).status_code == 422

    def missing() -> pd.DataFrame:
        raise FileNotFoundError("data/processed/fish_pond_clean.csv")

    state.load_data = missing
    assert client.post("/train", json={}).status_code == 503


def test_predict_validation_and_tracking_failure(
    client: TestClient, state: api.AppState, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.post("/predict", json={"records": []}).status_code == 422
    assert client.post("/predict", json={"records": [{"pond_id": "x"}]}).status_code == 422

    client.post("/train", json={"candidates": ["ridge"]})

    def broken_tracking(*args: Any, **kwargs: Any) -> str:
        raise ConnectionError("mlflow down")

    monkeypatch.setattr(api, "track_prediction", broken_tracking)
    response = client.post("/predict", json={"records": [READING]})
    assert response.status_code == 200
    assert len(state.predictions.read()) == 1


def test_create_app_uses_environment_settings(monkeypatch: pytest.MonkeyPatch, mlflow_settings: Settings) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_settings.mlflow_tracking_uri)
    app = api.create_app(run_startup=False)
    with TestClient(app) as test_client:
        assert test_client.get("/health").status_code == 200


def test_default_trainer_and_disabled_mlflow_prediction_logging(state: api.AppState) -> None:
    result = api.build_state(state.settings).trainer(state.load_data(), state.settings, False, ["ridge"], 0.0)
    assert result.promoted is True
    state.settings = dataclasses.replace(state.settings, log_predictions_to_mlflow=False)
    with TestClient(api.create_app(state)) as test_client:
        assert test_client.post("/predict", json={"records": [READING]}).status_code == 200
    experiment = state.client.get_experiment_by_name(state.settings.prediction_experiment_name)
    assert experiment is not None
    assert state.client.search_runs([experiment.experiment_id]) == []
