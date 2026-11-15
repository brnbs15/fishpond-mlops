"""Tests of the feature engineering steps and the model building helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fishpond import model
from fishpond.config import FEATURE_INPUT_COLUMNS, TARGET
from fishpond.features import DateFeatureExtractor, SensorSanitizer, build_preprocessor, numeric_features


def test_sensor_sanitizer_is_stateless(raw_frame: pd.DataFrame) -> None:
    sanitizer = SensorSanitizer()
    assert sanitizer.fit(raw_frame) is sanitizer
    result = sanitizer.transform(raw_frame)
    assert result["ph"].dropna().between(0, 14).all()
    assert len(result) == len(raw_frame)


def test_date_feature_extractor_creates_cyclic_hour_and_age() -> None:
    frame = pd.DataFrame({"timestamp": ["2021-06-02 00:00:00", "2021-06-11 06:00:00"], "x": [1, 2]})
    extractor = DateFeatureExtractor()
    result = extractor.fit(frame).transform(frame)
    assert "timestamp" not in result.columns
    assert result["elapsed_days"].tolist() == pytest.approx([1.0, 10.25])
    assert result["hour_sin"].tolist() == pytest.approx([0.0, 1.0], abs=1e-9)
    assert result["hour_cos"].tolist() == pytest.approx([1.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("include_length", [True, False])
def test_preprocessor_handles_missing_values_and_unknown_ponds(
    clean_frame: pd.DataFrame, include_length: bool
) -> None:
    features = clean_frame.loc[:, list(FEATURE_INPUT_COLUMNS)]
    preprocessor = build_preprocessor(include_length).fit(features)
    new = features.head(2).copy()
    new["pond_id"] = "pond_99"
    new["ammonia"] = np.nan
    output = preprocessor.transform(new)
    assert output.shape[0] == 2
    assert not np.isnan(np.asarray(output, dtype=float)).any()
    assert ("fish_length_cm" in numeric_features(include_length)) is include_length


def test_split_data_respects_test_size(clean_frame: pd.DataFrame) -> None:
    split = model.split_data(clean_frame, test_size=0.25, random_state=1)
    assert len(split.x_test) == pytest.approx(len(clean_frame) * 0.25, abs=1)
    assert TARGET not in split.x_train.columns
    assert len(split.x_train) == len(split.y_train)


def test_regression_metrics_perfect_prediction() -> None:
    metrics = model.regression_metrics([1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0]))
    assert metrics == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


def test_resolve_candidates() -> None:
    assert [c.name for c in model.resolve_candidates(None)] == list(model.CANDIDATES)
    assert [c.name for c in model.resolve_candidates(["ridge"])] == ["ridge"]
    with pytest.raises(ValueError, match="Unknown"):
        model.resolve_candidates(["ridge", "magic"])


@pytest.mark.parametrize("name", list(model.CANDIDATES))
def test_every_candidate_can_be_tuned_and_evaluated(clean_frame: pd.DataFrame, name: str) -> None:
    split = model.split_data(clean_frame, test_size=0.25, random_state=0)
    tuned = model.tune_candidate(
        model.CANDIDATES[name], split.x_train, split.y_train, include_length=False, random_state=0, cv_folds=2
    )
    assert tuned.name == name
    assert tuned.cv_rmse > 0
    assert set(tuned.best_params) <= {"alpha", "max_depth", "min_samples_leaf", "learning_rate", "max_leaf_nodes"}
    metrics = model.evaluate(tuned.pipeline, split.x_test, split.y_test)
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert np.isfinite(tuned.pipeline.predict(split.x_test)).all()
