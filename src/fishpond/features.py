"""Scikit-learn compatible feature engineering steps.

They are part of the logged MLflow model, so the REST API can send raw sensor readings
and get exactly the same preprocessing that was used during training.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fishpond.config import LENGTH_COLUMN, POND_ID, POPULATION_COLUMN, SENSOR_COLUMNS, TIMESTAMP
from fishpond.data import sanitize_values

# The ponds were stocked in June 2021; elapsed days since then is a proxy of fish age.
REFERENCE_DATE = pd.Timestamp("2021-06-01")
DATE_FEATURES: tuple[str, ...] = ("elapsed_days", "hour_sin", "hour_cos")


class SensorSanitizer(BaseEstimator, TransformerMixin):
    """Turns physically impossible sensor readings into NaN (same rules as the offline cleaning)."""

    def fit(self, x: pd.DataFrame, y: Any = None) -> Self:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        cleaned, _ = sanitize_values(pd.DataFrame(x))
        return cleaned


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Derives numeric time features from the ``timestamp`` column."""

    def fit(self, x: pd.DataFrame, y: Any = None) -> Self:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(x).copy()
        stamps = pd.to_datetime(frame[TIMESTAMP], errors="coerce")
        hours = stamps.dt.hour + stamps.dt.minute / 60.0
        frame["elapsed_days"] = (stamps - REFERENCE_DATE).dt.total_seconds() / 86400.0
        frame["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        frame["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        return frame.drop(columns=[TIMESTAMP])


def numeric_features(include_length: bool) -> list[str]:
    """Numeric model inputs after date feature extraction."""
    columns = [*SENSOR_COLUMNS, POPULATION_COLUMN, *DATE_FEATURES]
    if include_length:
        columns.append(LENGTH_COLUMN)
    return columns


def build_preprocessor(include_length: bool) -> Pipeline:
    """Sanitize -> date features -> impute/scale numeric, one-hot encode the pond id."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features(include_length)),
            ("pond", OneHotEncoder(handle_unknown="ignore"), [POND_ID]),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("sanitize", SensorSanitizer()),
            ("dates", DateFeatureExtractor()),
            ("columns", columns),
        ]
    )
