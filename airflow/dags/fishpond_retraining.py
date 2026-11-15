"""Airflow DAG: daily data drift check and automatic retraining through the fishpond REST API.

Thin orchestration layer only - the decision logic lives in ``fishpond.orchestration`` (unit tested
under ``tests/``). The new model only gets served if it beats the current champion, which is
enforced by the champion/challenger rule of the ``/train`` endpoint.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from airflow.sdk import dag, task

from fishpond.orchestration import FishpondClient, check_for_retraining, retrain

API_URL = os.environ.get("FISHPOND_API_URL", "http://api:8000")
DRIFT_SHARE_THRESHOLD = float(os.environ.get("FISHPOND_DRIFT_SHARE_THRESHOLD", "0.5"))

logger = logging.getLogger(__name__)


@dag(
    dag_id="fishpond_drift_retraining",
    description="Check Evidently data drift of the production requests and retrain the model if needed",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["fishpond", "mlops"],
)
def fishpond_drift_retraining() -> None:
    @task.short_circuit
    def drift_check() -> bool:
        decision = check_for_retraining(FishpondClient(API_URL), DRIFT_SHARE_THRESHOLD)
        logger.info("Retraining decision: %s (%s)", decision.retrain, decision.reason)
        return decision.retrain

    @task
    def retrain_model() -> dict[str, Any]:
        result = retrain(FishpondClient(API_URL))
        logger.info("Training result: promoted=%s, %s", result.get("promoted"), result.get("reason"))
        return result

    drift_check() >> retrain_model()


fishpond_drift_retraining()
