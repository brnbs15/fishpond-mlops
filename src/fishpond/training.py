"""MLflow-tracked training pipeline: tune candidates, pick the best, register it, promote if better.

Run as ``python -m fishpond.training`` or through the REST API ``POST /train`` endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from matplotlib.figure import Figure
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from mlflow.tracking import MlflowClient

from fishpond import registry
from fishpond.config import Settings, get_settings
from fishpond.data import load_processed
from fishpond.model import DataSplit, TunedModel, evaluate, resolve_candidates, split_data, tune_candidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateResult:
    """Validation outcome of one tuned candidate (one nested MLflow run)."""

    name: str
    run_id: str
    params: dict[str, Any]
    cv_rmse: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a whole training job, returned by the ``/train`` endpoint."""

    run_id: str
    best_candidate: str
    metrics: dict[str, float]
    model_version: str
    promoted: bool
    reason: str
    champion_version: str | None
    previous_champion_metrics: dict[str, float] | None
    candidates: list[CandidateResult] = field(default_factory=list[CandidateResult])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Short content hash of the training data, logged so every run is traceable to a data version."""
    digest = hashlib.sha256(frame.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()[:12]


def _plot_predictions(y_true: pd.Series, y_pred: Any, path: Path) -> None:
    figure = Figure(figsize=(5, 5))
    axis = figure.subplots()
    axis.scatter(y_true, y_pred, s=8, alpha=0.5)
    upper = float(max(y_true.max(), max(y_pred)))
    axis.plot([0, upper], [0, upper], color="red", linewidth=1)
    axis.set_xlabel("actual weight (g)")
    axis.set_ylabel("predicted weight (g)")
    axis.set_title("Validation set: predicted vs actual")
    figure.tight_layout()
    figure.savefig(path, dpi=100)


def _train_candidates(
    split: DataSplit, candidate_names: Sequence[str] | None, include_length: bool, random_state: int
) -> list[tuple[CandidateResult, TunedModel]]:
    results: list[tuple[CandidateResult, TunedModel]] = []
    for candidate in resolve_candidates(candidate_names):
        with mlflow.start_run(run_name=candidate.name, nested=True) as child:
            tuned = tune_candidate(candidate, split.x_train, split.y_train, include_length, random_state)
            metrics = evaluate(tuned.pipeline, split.x_test, split.y_test)
            mlflow.log_param("candidate", candidate.name)
            mlflow.log_params(tuned.best_params)
            mlflow.log_metric("cv_rmse", tuned.cv_rmse)
            mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()})
            result = CandidateResult(candidate.name, child.info.run_id, tuned.best_params, tuned.cv_rmse, metrics)
            results.append((result, tuned))
            logger.info("Candidate %s: val RMSE %.3f", candidate.name, metrics["rmse"])
    return results


def _log_artifacts(split: DataSplit, best: TunedModel, results: list[CandidateResult], params: dict[str, Any]) -> None:
    predictions = best.pipeline.predict(split.x_test)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        comparison = pd.DataFrame(
            [{"candidate": r.name, "cv_rmse": r.cv_rmse, **{f"val_{k}": v for k, v in r.metrics.items()}}
             for r in results]
        )
        comparison.to_csv(folder / "candidate_comparison.csv", index=False)
        validation = split.x_test.assign(actual=split.y_test.to_numpy(), predicted=predictions)
        validation.to_csv(folder / "validation_predictions.csv", index=False)
        (folder / "training_config.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
        _plot_predictions(split.y_test, predictions, folder / "predicted_vs_actual.png")
        mlflow.log_artifacts(str(folder), artifact_path="reports")


def train_and_register(
    frame: pd.DataFrame,
    settings: Settings,
    include_length: bool = False,
    candidate_names: Sequence[str] | None = None,
    min_relative_improvement: float = 0.0,
) -> TrainingResult:
    """Train all candidates on the given processed data, register the best and promote it if it is better."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    split = split_data(frame, settings.test_size, settings.random_state)
    params: dict[str, Any] = {
        "include_length": include_length,
        "candidates": ",".join(candidate_names) if candidate_names else "all",
        "rows": len(frame),
        "train_rows": len(split.x_train),
        "validation_rows": len(split.x_test),
        "test_size": settings.test_size,
        "random_state": settings.random_state,
        "data_fingerprint": dataset_fingerprint(frame),
        "min_relative_improvement": min_relative_improvement,
    }

    with mlflow.start_run(run_name="training") as parent:
        mlflow.log_params(params)
        mlflow.log_input(from_pandas(frame, name="fish_pond_clean"), context="training")
        trained = _train_candidates(split, candidate_names, include_length, settings.random_state)
        best_result, best_model = min(trained, key=lambda item: item[0].metrics["rmse"])
        mlflow.set_tag("best_candidate", best_result.name)
        mlflow.log_metrics({f"val_{key}": value for key, value in best_result.metrics.items()})
        _log_artifacts(split, best_model, [result for result, _ in trained], params)

        signature = infer_signature(split.x_test, best_model.pipeline.predict(split.x_test))
        model_info = log_model(
            best_model.pipeline,
            name="model",
            signature=signature,
            serialization_format="cloudpickle",
        )

        champion = registry.get_champion_version(client, settings.model_name, settings.champion_alias)
        champion_metrics: dict[str, float] | None = None
        if champion is not None:
            champion_model = registry.load_sklearn_model(
                registry.champion_uri(settings.model_name, settings.champion_alias)
            )
            champion_metrics = evaluate(champion_model, split.x_test, split.y_test)
            mlflow.log_metrics({f"champion_val_{key}": value for key, value in champion_metrics.items()})

        decision = registry.decide_promotion(
            best_result.metrics["rmse"],
            None if champion_metrics is None else champion_metrics["rmse"],
            min_relative_improvement,
        )
        version = registry.register_version(
            client,
            model_info.model_uri,
            settings.model_name,
            {
                "candidate": best_result.name,
                "val_rmse": f"{best_result.metrics['rmse']:.4f}",
                "run_id": parent.info.run_id,
            },
        )
        version_number = str(version.version)
        if decision.promote:
            registry.promote(client, settings.model_name, version_number, settings.champion_alias)
            champion_version: str | None = version_number
        else:
            registry.reject(client, settings.model_name, version_number, decision.reason)
            champion_version = None if champion is None else str(champion.version)
        mlflow.set_tags({"promoted": str(decision.promote), "promotion_reason": decision.reason})
        logger.info("Model version %s registered, promoted=%s (%s)", version_number, decision.promote, decision.reason)

    return TrainingResult(
        run_id=parent.info.run_id,
        best_candidate=best_result.name,
        metrics=best_result.metrics,
        model_version=version_number,
        promoted=decision.promote,
        reason=decision.reason,
        champion_version=champion_version,
        previous_champion_metrics=champion_metrics,
        candidates=[result for result, _ in trained],
    )


def main(argv: Sequence[str] | None = None) -> TrainingResult:
    """CLI entry point: train on the processed dataset."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train, register and (maybe) promote a fish weight model.")
    parser.add_argument("--data", type=Path, default=settings.processed_data_path)
    parser.add_argument("--include-length", action="store_true", help="use fish length as a feature as well")
    parser.add_argument("--candidates", nargs="*", default=None, help="subset of model candidates to train")
    args = parser.parse_args(argv)
    result = train_and_register(load_processed(args.data), settings, args.include_length, args.candidates)
    logger.info("Training finished: %s", json.dumps(result.to_dict(), default=str))
    return result


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper, main() itself is tested
    logging.basicConfig(level=logging.INFO)
    main()
