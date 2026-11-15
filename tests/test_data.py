"""Tests of the raw data loading and cleaning step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishpond import data
from fishpond.config import RAW_COLUMNS, TARGET


def test_load_raw_reads_all_columns(raw_csv: Path) -> None:
    frame = data.load_raw(raw_csv)
    assert list(frame.columns) == list(RAW_COLUMNS)


def test_load_raw_rejects_missing_columns(tmp_path: Path, raw_frame: pd.DataFrame) -> None:
    path = tmp_path / "broken.csv"
    raw_frame.drop(columns=["ph"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ph"):
        data.load_raw(path)


def test_sanitize_values_invalidates_out_of_range_readings() -> None:
    frame = pd.DataFrame(
        {
            "ph": [7.0, -1.0, 15.0, 8.0],
            "turbidity_ntu": [10.0, -5.0, 50.0, 100.0],
            "ammonia": [0.5, np.inf, 1e9, "bad"],
        }
    )
    cleaned, invalidated = data.sanitize_values(frame)
    assert cleaned["ph"].isna().tolist() == [False, True, True, False]
    assert invalidated["ph"] == 2
    assert invalidated["turbidity_ntu"] == 1
    # inf and the unparsable string were already NaN before the range check; only 1e9 counts as out of range
    assert cleaned["ammonia"].isna().sum() == 3
    assert invalidated["ammonia"] == 1
    assert "temperature_c" not in invalidated


def test_clean_dataset_drops_bad_rows_and_sorts(raw_frame: pd.DataFrame) -> None:
    raw = raw_frame.copy()
    raw.loc[1, "timestamp"] = "not a date"
    raw.loc[2, TARGET] = -1.0
    raw.loc[3, TARGET] = np.nan
    raw = pd.concat([raw, raw.iloc[[10]]], ignore_index=True)

    cleaned, report = data.clean_dataset(raw)

    assert report.rows_in == len(raw)
    assert report.invalid_timestamps == 1
    assert report.invalid_targets == 2
    assert report.duplicates == 1
    assert report.rows_out == len(cleaned) == len(raw) - 4
    assert cleaned["timestamp"].is_monotonic_increasing
    assert (cleaned[TARGET] > 0).all()
    assert cleaned["ph"].dropna().between(0, 14).all()
    assert report.values_set_to_nan["ph"] > 0


def test_save_and_load_processed_roundtrip(tmp_path: Path, raw_frame: pd.DataFrame) -> None:
    cleaned, _ = data.clean_dataset(raw_frame)
    path = tmp_path / "nested" / "clean.csv"
    data.save_processed(cleaned, path)
    loaded = data.load_processed(path)
    assert len(loaded) == len(cleaned)
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])
    assert loaded["pond_id"].map(type).eq(str).all()


def test_main_writes_processed_file(tmp_path: Path, raw_csv: Path) -> None:
    output = tmp_path / "out.csv"
    report = data.main(["--input", str(raw_csv), "--output", str(output)])
    assert output.exists()
    assert report.rows_out == len(pd.read_csv(output))


def test_sensor_columns_lists_water_quality_fields() -> None:
    assert "dissolved_oxygen" in data.sensor_columns()
    assert TARGET not in data.sensor_columns()


def test_ensure_processed_builds_missing_file_once(tmp_path: Path, raw_csv: Path) -> None:
    processed = tmp_path / "processed" / "clean.csv"
    first = data.ensure_processed(raw_csv, processed)
    assert processed.exists()
    raw_csv.unlink()
    second = data.ensure_processed(raw_csv, processed)
    assert len(first) == len(second)


def test_ensure_processed_without_any_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        data.ensure_processed(tmp_path / "raw.csv", tmp_path / "clean.csv")
