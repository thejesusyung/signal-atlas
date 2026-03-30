from __future__ import annotations

from contextlib import contextmanager

import httpx

from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.llm.provider import LLMTraceContext
from news_pipeline.tracking import experiment


class _NoOpLimiter:
    def acquire(self) -> None:
        return None

    def backoff(self, delay_seconds: float) -> None:
        return None


def test_groq_provider_logs_llm_trace(monkeypatch):
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"content": '{"entities": []}'}}],
                "usage": {"total_tokens": 42},
            },
        )

    def fake_log_llm_call_trace(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("news_pipeline.llm.groq_client.log_llm_call_trace", fake_log_llm_call_trace)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GroqProvider(client=client, rate_limiter=_NoOpLimiter())
    provider.api_key = "test-key"

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

    assert captured["experiment_name"] == "extraction_monitoring"
    assert captured["provider_name"] == "groq"
    assert captured["trace_context"] == {
        "operation": "entity_extraction",
        "article_id": "article-123",
        "prompt_version": "entity_v3",
        "article_title": "Acme opens a lab",
    }
    assert captured["request_payload"]["prompt"] == "hello"
    assert captured["response_payload"]["tokens_used"] == 42
    assert captured["error_message"] is None


def test_log_llm_call_trace_creates_nested_mlflow_run(monkeypatch):
    events: list[tuple[str, object]] = []

    class FakeMlflow:
        def __init__(self) -> None:
            self._active_run = object()

        def set_tracking_uri(self, uri: str) -> None:
            events.append(("set_tracking_uri", uri))

        def set_experiment(self, name: str) -> None:
            events.append(("set_experiment", name))

        def active_run(self):
            return self._active_run

        @contextmanager
        def start_run(self, run_name: str, nested: bool = False):
            events.append(("start_run", {"run_name": run_name, "nested": nested}))
            yield object()

        def log_params(self, params):
            events.append(("log_params", params))

        def set_tags(self, tags):
            events.append(("set_tags", tags))

        def log_metrics(self, metrics):
            events.append(("log_metrics", metrics))

        def log_artifact(self, path: str, artifact_path: str | None = None):
            events.append(("log_artifact", artifact_path))

    monkeypatch.setattr(experiment, "mlflow", FakeMlflow())

    experiment.log_llm_call_trace(
        experiment_name="extraction_monitoring",
        provider_name="groq",
        model_name="llama-3.1-8b-instant",
        trace_context={"operation": "entity_extraction", "article_id": "article-123"},
        request_payload={"prompt": "hello", "system_prompt": "system"},
        attempts=[{"attempt": 1, "latency_ms": 25, "status_code": 200}],
        response_payload={
            "text": '{"entities":[]}',
            "model": "llama-3.1-8b-instant",
            "tokens_used": 42,
            "latency_ms": 25,
            "provider_name": "groq",
        },
        error_message=None,
    )

    assert ("set_experiment", "extraction_monitoring") in events
    assert (
        "start_run",
        {"run_name": "llm_entity_extraction", "nested": True},
    ) in events
    assert ("log_artifact", "llm_calls") in events
