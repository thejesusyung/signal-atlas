from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from news_pipeline.contracts import LLMResponse


class LLMProviderError(RuntimeError):
    """Base provider error."""


@dataclass(frozen=True, slots=True)
class LLMTraceContext:
    operation: str
    article_id: str | None = None
    prompt_version: str | None = None
    article_title: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


class LLMProvider(ABC):
    provider_name: str

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 900,
        trace_context: LLMTraceContext | None = None,
    ) -> LLMResponse:
        raise NotImplementedError
