"""Loading and cleaning of the raw aquaponics fish pond sensor data.

Run as a module (``python -m fishpond.data``) to produce the processed dataset; the
DVC ``preprocess`` stage calls exactly this entry point.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fishpond.config import POND_ID, RAW_COLUMNS, SENSOR_COLUMNS, TARGET, TIMESTAMP, get_settings

logger = logging.getLogger(__name__)

# Physically plausible value ranges; readings outside them are sensor errors and become NaN.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "temperature_c": (0.0, 40.0),
    "turbidity_ntu": (0.0, 1000.0),
    "dissolved_oxygen": (0.0, 20.0),
    "ph": (0.0, 14.0),
    "ammonia": (0.0, 100.0),
    "nitrate": (0.0, 5000.0),
    "population": (1.0, 10000.0),
    "fish_length_cm": (0.0, 200.0),
}


@dataclass(frozen=True)
class CleaningReport:
    """Row and value counts describing what the cleaning step changed."""

    rows_in: int
    rows_out: int
    invalid_timestamps: int
    invalid_targets: int
    duplicates: int
    values_set_to_nan: dict[str, int]


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV file and check that every expected column is present."""
    frame = pd.read_csv(path)
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return frame


def _to_float(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numeric conversion of a column; unparsable values become NaN."""
    return pd.Series(pd.to_numeric(frame[column], errors="coerce"), index=frame.index, dtype="float64")


def sanitize_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coerce numeric columns, turn infinities and out-of-range readings into NaN.

    Returns a new frame and the number of values that were invalidated per column.
    """
    result = frame.copy()
    invalidated: dict[str, int] = {}
    for column, (low, high) in VALID_RANGES.items():
        if column not in result.columns:
            continue
        values = _to_float(result, column).replace([np.inf, -np.inf], np.nan)
        before = int(values.isna().sum())
        values = values.where((values >= low) & (values <= high))
        invalidated[column] = int(values.isna().sum()) - before
        result[column] = values
    return result, invalidated


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the raw sensor table.

    * unparsable timestamps and rows without pond id are dropped,
    * sensor errors (negative turbidity, pH outside 0-14, ...) become NaN and are imputed later in the model pipeline,
    * rows without a positive target are dropped (they cannot be used for supervised training),
    * exact duplicates are removed and the table is sorted by time.
    """
    rows_in = len(frame)
    data = frame.loc[:, list(RAW_COLUMNS)].copy()

    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], errors="coerce")
    invalid_timestamps = int(data[TIMESTAMP].isna().sum())
    data[POND_ID] = data[POND_ID].astype("string").str.strip()
    data = data.dropna(subset=[TIMESTAMP, POND_ID])

    data, invalidated = sanitize_values(data)

    target = _to_float(data, TARGET)
    valid_target = target.notna() & (target > 0) & np.isfinite(target)
    invalid_targets = int((~valid_target).sum())
    data = data.loc[valid_target]

    before_dedup = len(data)
    data = data.drop_duplicates()
    duplicates = before_dedup - len(data)

    data = data.sort_values([TIMESTAMP, POND_ID]).reset_index(drop=True)
    data[POND_ID] = data[POND_ID].astype(str)

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(data),
        invalid_timestamps=invalid_timestamps,
        invalid_targets=invalid_targets,
        duplicates=duplicates,
        values_set_to_nan=invalidated,
    )
    return data, report


def save_processed(frame: pd.DataFrame, path: Path) -> None:
    """Write the processed dataset as UTF-8 CSV with ISO timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def load_processed(path: Path) -> pd.DataFrame:
    """Read the processed dataset back with proper dtypes."""
    frame = pd.read_csv(path, parse_dates=[TIMESTAMP])
    frame[POND_ID] = frame[POND_ID].astype(str)
    return frame


def sensor_columns() -> list[str]:
    """Names of the water quality sensor columns (handy for monitoring)."""
    return list(SENSOR_COLUMNS)


def main(argv: Sequence[str] | None = None) -> CleaningReport:
    """CLI entry point: raw CSV -> processed CSV."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Clean the raw fish pond dataset.")
    parser.add_argument("--input", type=Path, default=settings.raw_data_path)
    parser.add_argument("--output", type=Path, default=settings.processed_data_path)
    args = parser.parse_args(argv)

    raw = load_raw(args.input)
    cleaned, report = clean_dataset(raw)
    save_processed(cleaned, args.output)
    logger.info("Cleaned %d -> %d rows, written to %s", report.rows_in, report.rows_out, args.output)
    return report


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper, main() itself is tested
    logging.basicConfig(level=logging.INFO)
    main()
