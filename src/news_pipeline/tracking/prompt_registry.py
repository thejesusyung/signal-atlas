from __future__ import annotations

import mlflow

from news_pipeline.tracking.experiment import configure_mlflow
from news_pipeline.llm.prompts import ENTITY_EXTRACTION_PROMPT, TOPIC_CLASSIFICATION_PROMPT, PromptSpec


def register_all_prompts() -> None:
    configure_mlflow()
    _register_if_absent("entity_extraction", ENTITY_EXTRACTION_PROMPT)
    _register_if_absent("topic_classification", TOPIC_CLASSIFICATION_PROMPT)
    # Imported inline to avoid circular import through detector.py
    from news_pipeline.signals.detector import SIGNAL_BRIEF_PROMPT
    _register_if_absent("signal_brief", SIGNAL_BRIEF_PROMPT)


def _register_if_absent(registry_name: str, spec: PromptSpec) -> None:
    try:
        mlflow.register_prompt(
            name=registry_name,
            template=spec.user_prompt_template,
            commit_message=f"Register {spec.version}",
            tags={"version": spec.version},
        )
    except mlflow.exceptions.MlflowException:
        pass  # Already exists — idempotent


def get_prompt_template(registry_name: str, fallback: PromptSpec) -> str:
    """Load from registry; fall back to the bundled constant (e.g. in tests without MLflow)."""
    try:
        configure_mlflow()
        prompt = mlflow.load_prompt(f"prompts:/{registry_name}/1")
        return prompt.template
    except Exception:
        return fallback.user_prompt_template
