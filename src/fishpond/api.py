"""FastAPI REST service with ``/train`` and ``/predict`` endpoints (plus health, model info and drift)."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from fishpond import monitoring
from fishpond.config import Settings, get_settings
from fishpond.data import load_processed
from fishpond.prediction_log import PredictionLogger, RequestLogger
from fishpond.service import ModelNotAvailableError, ModelService, track_prediction
from fishpond.training import TrainingResult, train_and_register

logger = logging.getLogger(__name__)

Trainer = Callable[[pd.DataFrame, Settings, bool, list[str] | None, float], TrainingResult]
DataLoader = Callable[[], pd.DataFrame]


class PondReading(BaseModel):
    """One raw sensor reading of a pond (the same fields as the dataset, without the target)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "timestamp": "2021-08-15 12:00:00",
                    "pond_id": "pond_02",
                    "temperature_c": 24.8,
                    "turbidity_ntu": 85.0,
                    "dissolved_oxygen": 6.5,
                    "ph": 7.2,
                    "ammonia": 2.4,
                    "nitrate": 540.0,
                    "population": 50,
                    "fish_length_cm": None,
                }
            ]
        }
    )

    timestamp: datetime
    pond_id: str = Field(min_length=1)
    temperature_c: float | None = None
    turbidity_ntu: float | None = None
    dissolved_oxygen: float | None = None
    ph: float | None = None
    ammonia: float | None = None
    nitrate: float | None = None
    population: float | None = None
    fish_length_cm: float | None = None


class PredictRequest(BaseModel):
    records: list[PondReading] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    predictions: list[float]


class TrainRequest(BaseModel):
    include_length: bool = Field(False, description="Use the fish length as an input feature as well")
    candidates: list[str] | None = Field(None, description="Subset of model candidates, default: all")
    min_relative_improvement: float = Field(
        0.0, ge=0.0, lt=1.0, description="Required relative RMSE improvement over the champion to promote"
    )


class CandidateOut(BaseModel):
    name: str
    run_id: str
    cv_rmse: float
    metrics: dict[str, float]


class TrainResponse(BaseModel):
    run_id: str
    best_candidate: str
    metrics: dict[str, float]
    model_version: str
    promoted: bool
    reason: str
    champion_version: str | None
    previous_champion_metrics: dict[str, float] | None
    candidates: list[CandidateOut]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class DriftResponse(BaseModel):
    dataset_drift: bool
    drift_share: float
    drifted_columns: list[str]
    reference_rows: int
    current_rows: int


@dataclass
class AppState:
    """Everything the endpoints need; injected so tests can swap out MLflow and the data."""

    settings: Settings
    service: ModelService
    predictions: PredictionLogger
    requests: RequestLogger
    trainer: Trainer
    load_data: DataLoader
    client: MlflowClient


def _default_trainer(
    frame: pd.DataFrame, settings: Settings, include_length: bool, candidates: list[str] | None, improvement: float
) -> TrainingResult:
    return train_and_register(frame, settings, include_length, candidates, improvement)


def build_state(settings: Settings) -> AppState:
    """Production wiring of the application state."""
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    log_dir = settings.prediction_log_path.parent
    return AppState(
        settings=settings,
        service=ModelService(settings, client=client),
        predictions=PredictionLogger(settings.prediction_log_path),
        requests=RequestLogger(log_dir / "requests.jsonl"),
        trainer=_default_trainer,
        load_data=lambda: load_processed(settings.processed_data_path),
        client=client,
    )


def startup(state: AppState) -> None:
    """Load the champion; if there is none and auto training is enabled, train the first model."""
    try:
        state.service.reload()
        return
    except ModelNotAvailableError:
        logger.info("No champion model in the registry yet")
    except Exception:  # registry unreachable or broken model: keep the API up, /health reports it
        logger.exception("Could not load the champion model")
        return
    if state.settings.auto_train_on_startup:
        logger.info("Auto training the first model")
        result = state.trainer(state.load_data(), state.settings, False, None, 0.0)
        if result.promoted:
            state.service.reload()


def create_app(state: AppState | None = None, run_startup: bool = True) -> FastAPI:
    """Application factory."""
    app_state = state or build_state(get_settings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        if run_startup:
            startup(app_state)
        yield

    app = FastAPI(
        title="Fishpond weight prediction API",
        description="Train and serve models predicting the average catfish weight from pond sensor readings.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        app_state.requests.log(
            {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "client": request.client.host if request.client else None,
            }
        )
        return response

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        loaded = app_state.service.loaded
        return HealthResponse(
            status="ok", model_loaded=loaded is not None, model_version=loaded.version if loaded else None
        )

    @app.post("/train", response_model=TrainResponse, tags=["model"])
    def train(body: TrainRequest) -> TrainResponse:
        try:
            frame = app_state.load_data()
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=f"Training data not found: {error}") from error
        try:
            result = app_state.trainer(
                frame, app_state.settings, body.include_length, body.candidates, body.min_relative_improvement
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if result.promoted:
            app_state.service.reload()
        return TrainResponse.model_validate(result.to_dict())

    @app.post("/predict", response_model=PredictResponse, tags=["model"])
    def predict(body: PredictRequest) -> PredictResponse:
        records = [record.model_dump() for record in body.records]
        frame = pd.DataFrame(records)
        try:
            predictions, version = app_state.service.predict(frame)
        except ModelNotAvailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        request_id = uuid.uuid4().hex
        scored = app_state.predictions.log(records, predictions, version, request_id)
        if app_state.settings.log_predictions_to_mlflow:
            try:
                track_prediction(
                    app_state.client,
                    app_state.settings.prediction_experiment_name,
                    request_id,
                    body.model_dump(mode="json"),
                    scored,
                    version,
                )
            except Exception:  # tracking must never break serving; the CSV log above already has the data
                logger.exception("Could not log prediction %s to MLflow", request_id)
        return PredictResponse(request_id=request_id, model_version=version, predictions=predictions)

    @app.get("/model", tags=["model"])
    def model_info() -> dict[str, str | None]:
        loaded = app_state.service.loaded
        return {
            "model_name": app_state.settings.model_name,
            "alias": app_state.settings.champion_alias,
            "version": loaded.version if loaded else None,
        }

    @app.get("/monitoring/drift", response_model=DriftResponse, tags=["monitoring"])
    def drift() -> DriftResponse:
        current = app_state.predictions.read()
        if current.empty:
            raise HTTPException(status_code=404, detail="No predictions logged yet")
        reference = app_state.load_data()
        result = monitoring.compute_drift(reference, current)
        return DriftResponse(
            dataset_drift=result.dataset_drift,
            drift_share=result.drift_share,
            drifted_columns=result.drifted_columns,
            reference_rows=len(reference),
            current_rows=len(current),
        )

    return app
