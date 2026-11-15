"""MLflow model registry helpers and the champion / challenger promotion rule.

The rule is what keeps model quality from degrading automatically: a freshly trained model
(challenger) only receives the ``champion`` alias - and therefore only gets served by the API -
if it beats the current champion on the *same, current* validation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class PromotionDecision:
    """Whether the challenger replaces the champion, and why."""

    promote: bool
    reason: str


def decide_promotion(
    challenger_rmse: float, champion_rmse: float | None, min_relative_improvement: float = 0.0
) -> PromotionDecision:
    """Promote if there is no champion yet, or the challenger's RMSE is lower by at least the given ratio."""
    if champion_rmse is None:
        return PromotionDecision(True, "no champion registered yet")
    threshold = champion_rmse * (1.0 - min_relative_improvement)
    if challenger_rmse < threshold:
        return PromotionDecision(True, f"challenger RMSE {challenger_rmse:.4f} < champion RMSE {champion_rmse:.4f}")
    return PromotionDecision(
        False, f"challenger RMSE {challenger_rmse:.4f} is not better than champion RMSE {champion_rmse:.4f}"
    )


def champion_uri(model_name: str, alias: str) -> str:
    """Registry URI of the model version holding the alias."""
    return f"models:/{model_name}@{alias}"


def get_champion_version(client: MlflowClient, model_name: str, alias: str) -> ModelVersion | None:
    """The model version holding the champion alias, or ``None`` if there is none (yet)."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def load_sklearn_model(uri: str) -> Any:
    """Load a scikit-learn model from the registry / tracking server."""
    return load_model(uri)


def register_version(client: MlflowClient, model_uri: str, model_name: str, tags: dict[str, str]) -> ModelVersion:
    """Register a logged model as a new version and tag it."""
    version = mlflow.register_model(model_uri, model_name)
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version.version, key, value)
    return version


def promote(client: MlflowClient, model_name: str, version: str, alias: str) -> None:
    """Move the alias to the given version and mark the version as champion."""
    client.set_registered_model_alias(model_name, alias, version)
    client.set_model_version_tag(model_name, version, "status", "champion")


def reject(client: MlflowClient, model_name: str, version: str, reason: str) -> None:
    """Keep the version in the registry for traceability but mark it as rejected."""
    client.set_model_version_tag(model_name, version, "status", "rejected")
    client.set_model_version_tag(model_name, version, "rejection_reason", reason)
