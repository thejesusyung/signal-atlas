from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Any

import mlflow

from news_pipeline.config import get_settings


def configure_mlflow() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


@contextmanager
def tracked_run(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    nested: bool = False,
):
    configure_mlflow()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, nested=nested):
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags({key: str(value) for key, value in tags.items()})
        yield


def log_metrics(metrics: dict[str, float | int]) -> None:
    if metrics:
        mlflow.log_metrics(metrics)


def log_dict_artifact(
    payload: dict[str, Any],
    artifact_file: str,
    artifact_path: str = "reports",
) -> None:
    configure_mlflow()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="news-pipeline-",
        suffix=f"-{Path(artifact_file).name}",
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, indent=2, default=str))
        path = Path(handle.name)
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    finally:
        path.unlink(missing_ok=True)
