"""Model candidates, hyperparameter search and evaluation (no MLflow code here)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

from fishpond.config import FEATURE_INPUT_COLUMNS, TARGET
from fishpond.features import build_preprocessor


@dataclass(frozen=True)
class Candidate:
    """A regressor family together with the hyperparameter grid searched for it."""

    name: str
    factory: Callable[[int], Any]
    param_grid: dict[str, list[Any]]


CANDIDATES: dict[str, Candidate] = {
    "ridge": Candidate(
        name="ridge",
        factory=lambda seed: Ridge(),
        param_grid={"model__regressor__alpha": [0.1, 1.0, 10.0]},
    ),
    "random_forest": Candidate(
        name="random_forest",
        factory=lambda seed: RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1),
        param_grid={"model__regressor__max_depth": [None, 12], "model__regressor__min_samples_leaf": [1, 3]},
    ),
    "hist_gradient_boosting": Candidate(
        name="hist_gradient_boosting",
        factory=lambda seed: HistGradientBoostingRegressor(random_state=seed),
        param_grid={"model__regressor__learning_rate": [0.05, 0.1], "model__regressor__max_leaf_nodes": [15, 31]},
    ),
}


@dataclass(frozen=True)
class TunedModel:
    """Outcome of the hyperparameter search of one candidate."""

    name: str
    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_rmse: float


@dataclass(frozen=True)
class DataSplit:
    """Train / validation split of the processed dataset."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def build_pipeline(regressor: Any, include_length: bool) -> Pipeline:
    """Full model: preprocessing + regressor trained on log(1 + weight) because the target is right skewed."""
    model = TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    return Pipeline(steps=[("preprocess", build_preprocessor(include_length)), ("model", model)])


def split_data(frame: pd.DataFrame, test_size: float, random_state: int) -> DataSplit:
    """Random train/validation split of the model inputs and the target."""
    features = frame.loc[:, list(FEATURE_INPUT_COLUMNS)]
    target = frame[TARGET].astype(float)
    parts = train_test_split(features, target, test_size=test_size, random_state=random_state)
    x_train, x_test, y_train, y_test = cast(tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], tuple(parts))
    return DataSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def regression_metrics(y_true: Sequence[float] | pd.Series, y_pred: Sequence[float] | np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R2 of a set of predictions."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Metrics of a fitted model on a dataset."""
    return regression_metrics(y, model.predict(x))


def resolve_candidates(names: Sequence[str] | None) -> list[Candidate]:
    """Map candidate names to definitions; ``None`` means all of them."""
    if names is None:
        return list(CANDIDATES.values())
    unknown = sorted(set(names) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown model candidates: {unknown}. Available: {sorted(CANDIDATES)}")
    return [CANDIDATES[name] for name in names]


def tune_candidate(
    candidate: Candidate,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    include_length: bool,
    random_state: int,
    cv_folds: int = 3,
) -> TunedModel:
    """Grid search with k-fold CV; the best configuration is refitted on the whole training set."""
    search = GridSearchCV(
        estimator=build_pipeline(candidate.factory(random_state), include_length),
        param_grid=candidate.param_grid,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        refit=True,
    )
    search.fit(x_train, y_train)
    return TunedModel(
        name=candidate.name,
        pipeline=search.best_estimator_,
        best_params={key.rsplit("__", 1)[-1]: value for key, value in search.best_params_.items()},
        cv_rmse=float(-search.best_score_),
    )
