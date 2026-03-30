from __future__ import annotations

from dataclasses import dataclass

from news_pipeline.db.models import ExtractionRunType


@dataclass(slots=True)
class ExtractionStepError(RuntimeError):
    run_type: ExtractionRunType
    llm_provider: str
    model_name: str
    prompt_version: str
    tokens_used: int
    latency_ms: int
    error_message: str

    def __str__(self) -> str:
        return self.error_message
