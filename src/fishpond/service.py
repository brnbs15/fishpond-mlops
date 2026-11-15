"""Serving side: loads the champion model from the MLflow registry and tracks predictions in MLflow."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from fishpond import registry
from fishpond.config import Settings


class ModelNotAvailableError(RuntimeError):
    """Raised when no champion model is registered yet."""


@dataclass(frozen=True)
class LoadedModel:
    """A model loaded from the registry together with its version number."""

    model: Any
    version: str


class ModelService:
    """Caches the champion model; ``reload`` is called after every successful promotion."""

    def __init__(
        self,
        settings: Settings,
        client: MlflowClient | None = None,
        loader: Callable[[str], Any] = registry.load_sklearn_model,
    ) -> None:
        self.settings = settings
        self.client = client or MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        self._loader = loader
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None

    @property
    def loaded(self) -> LoadedModel | None:
        return self._loaded

    def reload(self) -> LoadedModel:
        """Load the version currently holding the champion alias."""
        version = registry.get_champion_version(self.client, self.settings.model_name, self.settings.champion_alias)
        if version is None:
            raise ModelNotAvailableError(
                f"No '{self.settings.champion_alias}' version of model '{self.settings.model_name}'; call /train first"
            )
        model = self._loader(f"models:/{self.settings.model_name}/{version.version}")
        loaded = LoadedModel(model=model, version=str(version.version))
        with self._lock:
            self._loaded = loaded
        return loaded

    def current(self) -> LoadedModel:
        """The cached champion, loading it on first use."""
        return self._loaded or self.reload()

    def predict(self, frame: pd.DataFrame) -> tuple[list[float], str]:
        """Predictions of the champion model for a frame of raw readings."""
        loaded = self.current()
        predictions = [float(value) for value in loaded.model.predict(frame)]
        return predictions, loaded.version


def track_prediction(
    client: MlflowClient,
    experiment_name: str,
    request_id: str,
    request_payload: dict[str, Any],
    scored: pd.DataFrame,
    model_version: str,
) -> str:
    """Log one ``/predict`` call as an MLflow run with the request and the predictions as artifacts.

    Uses the client API (no global active run), so it is safe to call from concurrent request threads.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
    run = client.create_run(experiment_id, run_name=f"predict-{request_id[:8]}", tags={"request_id": request_id})
    run_id = run.info.run_id
    client.log_param(run_id, "model_version", model_version)
    client.log_param(run_id, "n_records", len(scored))
    client.log_metric(run_id, "mean_prediction", float(np.mean(scored["prediction"].to_numpy(dtype=float))))
    client.log_dict(run_id, request_payload, "request.json")
    client.log_text(run_id, scored.to_csv(index=False), "predictions.csv")
    client.set_terminated(run_id)
    return run_id
