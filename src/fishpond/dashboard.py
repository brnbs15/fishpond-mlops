"""Business logic behind the Streamlit dashboard (the Streamlit script itself is only a thin view)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from fishpond import monitoring
from fishpond.config import Settings
from fishpond.data import load_processed
from fishpond.prediction_log import PREDICTION_COLUMN, PredictionLogger


@dataclass(frozen=True)
class MonitoringData:
    """Reference (training) data and current (logged production) data."""

    reference: pd.DataFrame
    current: pd.DataFrame


def load_monitoring_data(settings: Settings) -> MonitoringData:
    """Read the processed training data and the prediction log (both may be missing on a fresh system)."""
    path = settings.processed_data_path
    reference = load_processed(path) if path.exists() else pd.DataFrame()
    current = PredictionLogger(settings.prediction_log_path).read()
    return MonitoringData(reference=reference, current=current)


def prediction_summary(current: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers of the production traffic."""
    if current.empty:
        return {"predictions": 0, "requests": 0, "mean_prediction": None, "model_versions": []}
    versions = sorted({str(version) for version in current["model_version"].tolist()})
    return {
        "predictions": len(current),
        "requests": int(current["request_id"].nunique()),
        "mean_prediction": float(np.mean(current[PREDICTION_COLUMN].to_numpy(dtype=float))),
        "model_versions": versions,
    }


def predictions_by_version(current: pd.DataFrame) -> pd.DataFrame:
    """Number and mean of predictions per served model version."""
    if current.empty:
        return pd.DataFrame({"model_version": [], "predictions": [], "mean_prediction": []})
    grouped = current.groupby(current["model_version"].astype(str))[PREDICTION_COLUMN]
    return pd.DataFrame(
        {"predictions": grouped.count(), "mean_prediction": grouped.mean()}
    ).reset_index()


def registry_overview(client: MlflowClient, model_name: str) -> pd.DataFrame:
    """All registered versions with their status tags and aliases (empty if the model does not exist)."""
    columns = ["version", "status", "candidate", "val_rmse", "aliases", "run_id"]
    try:
        registered = client.get_registered_model(model_name)
        versions = client.search_model_versions(f"name='{model_name}'")
    except MlflowException:
        return pd.DataFrame({column: [] for column in columns})
    aliases: dict[str, Any] = dict(registered.aliases or {})
    rows: list[dict[str, Any]] = []
    for version in versions:
        tags: dict[str, str] = dict(version.tags or {})
        rows.append(
            {
                "version": int(version.version),
                "status": tags.get("status", ""),
                "candidate": tags.get("candidate", ""),
                "val_rmse": tags.get("val_rmse", ""),
                "aliases": ", ".join(alias for alias, number in aliases.items() if str(number) == str(version.version)),
                "run_id": version.run_id,
            }
        )
    frame = pd.DataFrame(rows).reindex(columns=columns)
    return frame.sort_values("version", ascending=False).reset_index(drop=True)


def drift_report(data: MonitoringData) -> monitoring.DriftResult | None:
    """Full Evidently report (with HTML) or ``None`` while there is nothing to compare."""
    if data.current.empty or data.reference.empty:
        return None
    return monitoring.compute_drift(data.reference, data.current, with_html=True)
