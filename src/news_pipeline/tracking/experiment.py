from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import re
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


def log_llm_call_trace(
    *,
    experiment_name: str,
    provider_name: str,
    model_name: str,
    trace_context: dict[str, Any] | None,
    request_payload: dict[str, Any],
    attempts: list[dict[str, Any]],
    response_payload: dict[str, Any] | None,
    error_message: str | None,
) -> None:
    if mlflow.active_run() is None:
        return

    normalized_context = {key: str(value) for key, value in (trace_context or {}).items()}
    operation = normalized_context.get("operation", "llm_call")
    run_name = f"llm_{_slugify(operation)}"
    params = {
        "provider_name": provider_name,
        "model_name": model_name,
        "operation": operation,
        **normalized_context,
    }
    metrics = {
        "attempt_count": len(attempts),
        "success": int(error_message is None),
    }
    if response_payload is not None:
        metrics["tokens_used"] = int(response_payload.get("tokens_used", 0))
        metrics["latency_ms"] = int(response_payload.get("latency_ms", 0))

    with tracked_run(
        experiment_name=experiment_name,
        run_name=run_name,
        params=params,
        tags={"tracking_scope": "llm_call", "provider_name": provider_name},
        nested=True,
    ):
        log_metrics(metrics)
        log_dict_artifact(
            {
                "trace_context": trace_context or {},
                "request": request_payload,
                "attempts": attempts,
                "response": response_payload,
                "error_message": error_message,
            },
            artifact_file=f"{_slugify(operation)}.json",
            artifact_path="llm_calls",
        )


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return normalized or "artifact"
