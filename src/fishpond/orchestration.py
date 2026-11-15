"""Monitoring-driven retraining logic used by the Airflow DAG.

Only the standard library is used here, so the Airflow container can import this module
without installing the ML stack: all heavy work happens behind the REST API.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Fetch = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]


class ApiError(RuntimeError):
    """Non-successful HTTP answer of the fishpond API."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def http_json(
    url: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 900.0
) -> dict[str, Any]:
    """Minimal JSON-over-HTTP call; raises :class:`ApiError` for HTTP error codes."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return body
    except urllib.error.HTTPError as error:
        raise ApiError(error.code, error.read().decode("utf-8", errors="replace")) from error


@dataclass(frozen=True)
class DriftCheck:
    """Outcome of a drift check that decides about retraining."""

    retrain: bool
    reason: str
    drift_share: float | None = None


class FishpondClient:
    """Thin client of the fishpond REST API."""

    def __init__(self, base_url: str, fetch: Fetch = http_json) -> None:
        self.base_url = base_url.rstrip("/")
        self._fetch = fetch

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> dict[str, Any]:
        return self._fetch(self._url("/health"), "GET", None)

    def drift(self) -> dict[str, Any] | None:
        """Drift summary, or ``None`` if there are no logged predictions yet."""
        try:
            return self._fetch(self._url("/monitoring/drift"), "GET", None)
        except ApiError as error:
            if error.status == 404:
                return None
            raise

    def train(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._fetch(self._url("/train"), "POST", payload or {})

    def predict(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self._fetch(self._url("/predict"), "POST", {"records": records})


def decide_retraining(drift: dict[str, Any] | None, model_loaded: bool, drift_share_threshold: float) -> DriftCheck:
    """Retrain when there is no served model, or when the share of drifted features reaches the threshold."""
    if not model_loaded:
        return DriftCheck(True, "no model is served yet")
    if drift is None:
        return DriftCheck(False, "no production data logged yet")
    share = float(drift["drift_share"])
    if bool(drift["dataset_drift"]) or share >= drift_share_threshold:
        return DriftCheck(True, f"data drift detected (drift share {share:.2f})", share)
    return DriftCheck(False, f"no significant drift (drift share {share:.2f})", share)


def check_for_retraining(client: FishpondClient, drift_share_threshold: float = 0.5) -> DriftCheck:
    """Ask the API for its health and drift status and decide whether to retrain."""
    health = client.health()
    return decide_retraining(client.drift(), bool(health.get("model_loaded")), drift_share_threshold)


def retrain(client: FishpondClient) -> dict[str, Any]:
    """Start a training job; the champion/challenger rule in the API decides about promotion."""
    return client.train({"include_length": False})
