"""Append-only logs of incoming API requests and of the served predictions.

The prediction log is the "current" dataset of the Evidently drift report: it holds exactly the
feature values the model received in production, next to the prediction and the model version.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fishpond.config import FEATURE_INPUT_COLUMNS

PREDICTION_COLUMN = "prediction"
LOG_META_COLUMNS: tuple[str, ...] = ("logged_at", "request_id", "model_version")


def utc_now() -> str:
    """Current UTC time in ISO 8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class PredictionLogger:
    """Thread-safe CSV log of every record that was scored by the API."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    @property
    def columns(self) -> list[str]:
        return [*LOG_META_COLUMNS, *FEATURE_INPUT_COLUMNS, PREDICTION_COLUMN]

    def log(
        self,
        records: Sequence[Mapping[str, Any]],
        predictions: Sequence[float],
        model_version: str,
        request_id: str,
    ) -> pd.DataFrame:
        """Append one request (possibly several records) to the log and return the appended rows."""
        if len(records) != len(predictions):
            raise ValueError("records and predictions must have the same length")
        rows = pd.DataFrame(list(records)).reindex(columns=list(FEATURE_INPUT_COLUMNS))
        rows.insert(0, "model_version", model_version)
        rows.insert(0, "request_id", request_id)
        rows.insert(0, "logged_at", utc_now())
        rows[PREDICTION_COLUMN] = list(predictions)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.path.exists()
            rows.to_csv(self.path, mode="a", header=write_header, index=False)
        return rows

    def read(self) -> pd.DataFrame:
        """The whole prediction log (empty frame with the right columns if nothing was logged yet)."""
        if not self.path.exists():
            return pd.DataFrame({column: [] for column in self.columns})
        with self._lock:
            return pd.read_csv(self.path)


class RequestLogger:
    """Thread-safe JSON-lines log of every HTTP request reaching the API."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def log(self, entry: Mapping[str, Any]) -> None:
        line = json.dumps({"logged_at": utc_now(), **entry}, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
