"""Evidently data drift report: training data (reference) vs. logged production requests (current)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from fishpond.config import POND_ID, POPULATION_COLUMN, SENSOR_COLUMNS

NUMERIC_MONITORED: tuple[str, ...] = (*SENSOR_COLUMNS, POPULATION_COLUMN)
CATEGORICAL_MONITORED: tuple[str, ...] = (POND_ID,)
DEFAULT_DRIFT_SHARE = 0.5


@dataclass(frozen=True)
class DriftResult:
    """Condensed outcome of the Evidently drift report plus the full HTML report."""

    dataset_drift: bool
    drift_share: float
    drifted_columns: list[str]
    column_scores: dict[str, float] = field(default_factory=dict[str, float])
    html: str = ""


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [*NUMERIC_MONITORED, *CATEGORICAL_MONITORED]
    prepared = frame.reindex(columns=columns).copy()
    for column in NUMERIC_MONITORED:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared[POND_ID] = prepared[POND_ID].astype(str)
    return prepared


def _column_is_drifted(metric: dict[str, Any]) -> tuple[str, float, bool] | None:
    config: dict[str, Any] = metric.get("config", {})
    if not str(config.get("type", "")).endswith("ValueDrift"):
        return None
    column = str(config["column"])
    score = float(metric["value"])
    threshold = float(config["threshold"])
    # p-value based tests drift when the p-value is small, distance based ones when the distance is large
    drifted = score < threshold if "p_value" in str(config.get("method", "")) else score >= threshold
    return column, score, drifted


def summarize(report_dict: dict[str, Any], drift_share_threshold: float = DEFAULT_DRIFT_SHARE) -> DriftResult:
    """Extract dataset- and column-level drift from ``Snapshot.dict()``."""
    scores: dict[str, float] = {}
    drifted: list[str] = []
    for metric in report_dict.get("metrics", []):
        parsed = _column_is_drifted(metric)
        if parsed is None:
            continue
        column, score, is_drifted = parsed
        scores[column] = score
        if is_drifted:
            drifted.append(column)
    share = len(drifted) / len(scores) if scores else 0.0
    return DriftResult(
        dataset_drift=share >= drift_share_threshold,
        drift_share=share,
        drifted_columns=drifted,
        column_scores=scores,
    )


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE,
    with_html: bool = False,
) -> DriftResult:
    """Run the Evidently ``DataDriftPreset`` on the monitored input columns."""
    definition = DataDefinition(
        numerical_columns=list(NUMERIC_MONITORED), categorical_columns=list(CATEGORICAL_MONITORED)
    )
    snapshot = Report([DataDriftPreset(drift_share=drift_share_threshold)]).run(
        Dataset.from_pandas(_prepare(current), data_definition=definition),
        Dataset.from_pandas(_prepare(reference), data_definition=definition),
    )
    result = summarize(snapshot.dict(), drift_share_threshold)
    if not with_html:
        return result
    return DriftResult(
        dataset_drift=result.dataset_drift,
        drift_share=result.drift_share,
        drifted_columns=result.drifted_columns,
        column_scores=result.column_scores,
        html=snapshot.get_html_str(as_iframe=False),
    )
