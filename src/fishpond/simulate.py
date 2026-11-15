"""Production traffic simulator: sends real (optionally drifted) readings to the ``/predict`` endpoint.

Used to fill the prediction log so the Evidently drift report and the Airflow retraining DAG have
"production" data to work with: ``python -m fishpond.simulate --n 200 --drift``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from fishpond.config import FEATURE_INPUT_COLUMNS, LENGTH_COLUMN, TIMESTAMP, get_settings
from fishpond.data import load_processed
from fishpond.orchestration import FishpondClient

logger = logging.getLogger(__name__)

# Multiplicative / additive shifts imitating a heat wave with poorer water quality.
DRIFT_SHIFTS: dict[str, tuple[float, float]] = {
    "temperature_c": (1.0, 4.0),
    "dissolved_oxygen": (0.5, 0.0),
    "nitrate": (1.8, 0.0),
    "turbidity_ntu": (1.0, -30.0),
    "ammonia": (3.0, 0.0),
}


def sample_records(reference: pd.DataFrame, n: int, seed: int = 0, drift: bool = False) -> list[dict[str, Any]]:
    """Random readings from the reference data as JSON-ready dicts (target and length are not sent)."""
    columns = [c for c in FEATURE_INPUT_COLUMNS if c != LENGTH_COLUMN]
    sample = reference.sample(n=n, replace=n > len(reference), random_state=seed).loc[:, columns].copy()
    if drift:
        for column, (factor, offset) in DRIFT_SHIFTS.items():
            sample[column] = sample[column] * factor + offset
    sample[TIMESTAMP] = pd.to_datetime(sample[TIMESTAMP]).dt.strftime("%Y-%m-%d %H:%M:%S")
    cleaned = sample.astype(object).where(pd.notna(sample), None)
    records: list[dict[str, Any]] = []
    for row in cleaned.to_dict(orient="records"):
        records.append({str(key): value for key, value in row.items()})
    return records


def send_records(client: FishpondClient, records: Sequence[dict[str, Any]], batch_size: int = 50) -> int:
    """Send the records in batches; returns the number of predictions received."""
    received = 0
    for start in range(0, len(records), batch_size):
        response = client.predict(list(records[start:start + batch_size]))
        received += len(response["predictions"])
    return received


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Send simulated production traffic to the fishpond API.")
    parser.add_argument("--n", type=int, default=200, help="number of readings to send")
    parser.add_argument("--drift", action="store_true", help="shift the sensor values to simulate data drift")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--api-url", default=settings.api_url)
    args = parser.parse_args(argv)
    records = sample_records(load_processed(settings.processed_data_path), args.n, args.seed, args.drift)
    sent = send_records(FishpondClient(args.api_url), records)
    logger.info("Sent %d readings (drift=%s), received %d predictions", len(records), args.drift, sent)
    return sent


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper, main() itself is tested
    logging.basicConfig(level=logging.INFO)
    main()
