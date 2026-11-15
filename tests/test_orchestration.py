"""Tests of the retraining decision logic and the stdlib HTTP client used by Airflow."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from fishpond import orchestration
from fishpond.orchestration import ApiError, FishpondClient, decide_retraining


class _Handler(BaseHTTPRequestHandler):
    def _reply(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"status": "ok", "model_loaded": True})
        else:
            self._reply(404, {"detail": "No predictions logged yet"})

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self._reply(200, {"echo": body})

    def log_message(self, format: str, *args: Any) -> None:  # silence the test server
        return


@pytest.fixture
def server_url() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_http_json_get_and_post(server_url: str) -> None:
    assert orchestration.http_json(f"{server_url}/health")["model_loaded"] is True
    assert orchestration.http_json(f"{server_url}/x", "POST", {"a": 1}) == {"echo": {"a": 1}}


def test_http_json_raises_api_error(server_url: str) -> None:
    with pytest.raises(ApiError) as info:
        orchestration.http_json(f"{server_url}/monitoring/drift")
    assert info.value.status == 404


def test_client_against_real_http_server(server_url: str) -> None:
    client = FishpondClient(server_url + "/")
    assert client.health()["status"] == "ok"
    assert client.drift() is None
    assert client.train() == {"echo": {}}
    assert client.predict([{"pond_id": "pond_01"}]) == {"echo": {"records": [{"pond_id": "pond_01"}]}}


def test_client_drift_reraises_other_errors() -> None:
    def failing(url: str, method: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        raise ApiError(500, "boom")

    with pytest.raises(ApiError):
        FishpondClient("http://api", fetch=failing).drift()


@pytest.mark.parametrize(
    ("drift", "loaded", "expected"),
    [
        (None, False, True),
        (None, True, False),
        ({"dataset_drift": True, "drift_share": 0.1}, True, True),
        ({"dataset_drift": False, "drift_share": 0.6}, True, True),
        ({"dataset_drift": False, "drift_share": 0.2}, True, False),
    ],
)
def test_decide_retraining(drift: dict[str, Any] | None, loaded: bool, expected: bool) -> None:
    decision = decide_retraining(drift, loaded, drift_share_threshold=0.5)
    assert decision.retrain is expected
    assert decision.reason


def test_check_for_retraining_and_retrain() -> None:
    calls: list[tuple[str, str]] = []

    def fake(url: str, method: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        calls.append((method, url))
        if url.endswith("/health"):
            return {"model_loaded": True}
        if url.endswith("/monitoring/drift"):
            return {"dataset_drift": True, "drift_share": 0.75}
        return {"promoted": True, "payload": payload}

    client = FishpondClient("http://api", fetch=fake)
    decision = orchestration.check_for_retraining(client)
    assert decision.retrain is True
    assert decision.drift_share == 0.75
    assert orchestration.retrain(client) == {"promoted": True, "payload": {"include_length": False}}
    assert ("POST", "http://api/train") in calls
