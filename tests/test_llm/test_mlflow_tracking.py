from __future__ import annotations

import httpx
import mlflow

from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.llm.provider import LLMTraceContext


class _NoOpLimiter:
    def acquire(self) -> None:
        return None

    def backoff(self, delay_seconds: float) -> None:
        return None


def test_groq_provider_sets_span_attributes_on_success(monkeypatch):
    """complete() sets tokens_used/latency_ms/model/attempts/operation on the active span."""
    captured_attrs: dict[str, object] = {}

    class FakeSpan:
        def set_attribute(self, key: str, value: object) -> None:
            captured_attrs[key] = value

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"content": '{"entities": []}'}}],
                "usage": {"total_tokens": 42},
            },
        )

    monkeypatch.setattr(mlflow, "get_current_active_span", lambda: FakeSpan())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GroqProvider(client=client, rate_limiter=_NoOpLimiter())
    provider.api_key = "test-key"

    with mlflow.start_span(name="test_parent"):
        provider.complete(
            prompt="hello",
            system_prompt="system",
            trace_context=LLMTraceContext(
                operation="entity_extraction",
                article_id="article-123",
                prompt_version="entity_v3",
                article_title="Acme opens a lab",
            ),
        )

    assert captured_attrs["tokens_used"] == 42
    assert captured_attrs["model"] == "llama-3.1-8b-instant"
    assert captured_attrs["attempts"] == 1
    assert captured_attrs["operation"] == "entity_extraction"
    assert "latency_ms" in captured_attrs


def test_groq_provider_sets_span_attributes_without_trace_context(monkeypatch):
    """complete() sets core span attributes even when trace_context is None."""
    captured_attrs: dict[str, object] = {}

    class FakeSpan:
        def set_attribute(self, key: str, value: object) -> None:
            captured_attrs[key] = value

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"content": '{"result": "ok"}'}}],
                "usage": {"total_tokens": 10},
            },
        )

    monkeypatch.setattr(mlflow, "get_current_active_span", lambda: FakeSpan())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GroqProvider(client=client, rate_limiter=_NoOpLimiter())
    provider.api_key = "test-key"

    provider.complete(prompt="hello", system_prompt="system")

    assert captured_attrs["tokens_used"] == 10
    assert captured_attrs["attempts"] == 1
    assert "operation" not in captured_attrs
