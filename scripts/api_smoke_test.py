"""Manual end-to-end test of the running stack: python scripts/api_smoke_test.py [--api-url URL].

Calls every endpoint of the REST API started by `docker compose up` and prints the answers.
"""

from __future__ import annotations

import argparse
import json
import time

from fishpond.orchestration import FishpondClient, http_json

READING = {
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


def show(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2)[:1500])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    client = FishpondClient(args.api_url)

    show("GET /health", client.health())
    started = time.perf_counter()
    trained = client.train({"candidates": ["random_forest", "hist_gradient_boosting"]})
    summary = {key: value for key, value in trained.items() if key != "candidates"}
    show(f"POST /train ({time.perf_counter() - started:.1f} s)", summary)
    predicted = client.predict([READING, {**READING, "timestamp": "2021-10-01 08:00:00"}])
    show("POST /predict", predicted)
    assert len(predicted["predictions"]) == 2
    show("GET /model", http_json(f"{args.api_url}/model"))
    show("GET /monitoring/drift", client.drift())
    print("\nAll API calls succeeded.")


if __name__ == "__main__":
    main()
