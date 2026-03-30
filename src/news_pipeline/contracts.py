from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ArticleCandidate:
    title: str
    summary: str
    url: str
    published_at: datetime | None
    source_name: str
    source_feed: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScrapeResult:
    full_text: str
    word_count: int
    extraction_method: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    tokens_used: int
    latency_ms: int
    provider_name: str


@dataclass(slots=True)
class EntityRecord:
    name: str
    entity_type: str
    role: str
    confidence: float


@dataclass(slots=True)
class TopicAssignment:
    topic_name: str
    confidence: float
    method: str = "llm"

