"""Tests of the request and prediction logs."""

from __future__ import annotations

from pathlib import Path

import pytest

from fishpond.prediction_log import PREDICTION_COLUMN, PredictionLogger, RequestLogger


def test_prediction_log_appends_and_reads(tmp_path: Path) -> None:
    log = PredictionLogger(tmp_path / "logs" / "predictions.csv")
    assert log.read().empty
    assert PREDICTION_COLUMN in log.read().columns

    log.log([{"pond_id": "pond_01", "ph": 7.0}], [12.5], "3", "req-1")
    log.log([{"pond_id": "pond_02"}, {"pond_id": "pond_03", "unexpected": 1}], [1.0, 2.0], "4", "req-2")

    frame = log.read()
    assert len(frame) == 3
    assert list(frame.columns) == log.columns
    assert frame["request_id"].tolist() == ["req-1", "req-2", "req-2"]
    assert frame[PREDICTION_COLUMN].tolist() == [12.5, 1.0, 2.0]
    assert "unexpected" not in frame.columns


def test_prediction_log_rejects_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        PredictionLogger(tmp_path / "p.csv").log([{"pond_id": "pond_01"}], [1.0, 2.0], "1", "r")


def test_request_log_roundtrip(tmp_path: Path) -> None:
    log = RequestLogger(tmp_path / "requests.jsonl")
    assert log.read() == []
    log.log({"method": "GET", "path": "/health"})
    log.log({"method": "POST", "path": "/predict"})
    entries = log.read()
    assert [entry["path"] for entry in entries] == ["/health", "/predict"]
    assert all("logged_at" in entry for entry in entries)
