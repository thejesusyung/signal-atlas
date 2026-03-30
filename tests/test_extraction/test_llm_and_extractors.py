from __future__ import annotations

import importlib
from contextlib import nullcontext
import httpx
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock

from news_pipeline.db.models import Base, Entity, ExtractionRunType, ProcessingStatus, RawArticle, Topic
from news_pipeline.extraction.entity_extractor import EntityExtractor
from news_pipeline.extraction.topic_extractor import TopicExtractor
from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.llm.provider import LLMProvider
from news_pipeline.utils import normalize_title_for_dedup, utcnow


class FakeProvider(LLMProvider):
    provider_name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 900,
        trace_context=None,
    ):
        from news_pipeline.contracts import LLMResponse

        if not self.responses:
            raise RuntimeError("No more responses")
        return LLMResponse(
            text=self.responses.pop(0),
            model="fake-model",
            tokens_used=12,
            latency_ms=20,
            provider_name=self.provider_name,
        )


def _build_article():
    return RawArticle(
        title="Acme opens new lab in Lima",
        normalized_title=normalize_title_for_dedup("Acme opens new lab in Lima"),
        summary="Acme opens a new lab.",
        full_text="Acme opens a new lab in Lima and hires engineers.",
        cleaned_text="Acme opens a new lab in Lima and hires engineers.",
        url="https://example.com/acme-lab",
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        ingested_at=utcnow(),
        processing_status=ProcessingStatus.pending_extraction,
    )


def test_groq_provider_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"content": '{"entities": []}'}}],
                "usage": {"total_tokens": 42},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GroqProvider(client=client)
    provider.api_key = "test-key"

    response = provider.complete(prompt="hello", system_prompt="system")

    assert response.tokens_used == 42
    assert response.provider_name == "groq"


def test_groq_provider_honors_retry_after(monkeypatch):
    requests = {"count": 0}
    sleeps: list[float] = []

    class NoOpLimiter:
        def __init__(self) -> None:
            self.calls = 0
            self.backoffs: list[float] = []

        def acquire(self) -> None:
            self.calls += 1

        def backoff(self, delay_seconds: float) -> None:
            self.backoffs.append(delay_seconds)

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        requests["count"] += 1
        if requests["count"] == 1:
            return httpx.Response(429, headers={"retry-after": "3"})
        return httpx.Response(
            200,
            json={
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"content": '{"entities": []}'}}],
                "usage": {"total_tokens": 17},
            },
        )

    monkeypatch.setattr("news_pipeline.llm.groq_client.time.sleep", fake_sleep)

    limiter = NoOpLimiter()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GroqProvider(client=client, rate_limiter=limiter)
    provider.api_key = "test-key"

    response = provider.complete(prompt="hello", system_prompt="system")

    assert response.tokens_used == 17
    assert sleeps == [3.0]
    assert limiter.calls == 2
    assert limiter.backoffs == [3.0]


def test_entity_extractor_repairs_invalid_json(session):
    article = _build_article()
    session.add(article)
    session.commit()

    provider = FakeProvider(
        [
            '{"entities":[{"name":"Acme","type":"company","role":"subject","confidence":0.98}',
            '{"entities":[{"name":"Acme","type":"company","role":"subject","confidence":0.98}]}',
        ]
    )
    extractor = EntityExtractor(provider)

    records = extractor.extract_for_article(session, article)
    session.commit()

    assert len(records) == 1
    assert records[0].name == "Acme"
    assert article.entities[0].entity.name == "Acme"
    assert article.extraction_runs[-1].success is True


def test_entity_extractor_ignores_unsupported_entity_types(session):
    article = _build_article()
    session.add(article)
    session.commit()

    provider = FakeProvider(
        [
            '{"entities":['
            '{"name":"AI summit","type":"event","role":"mentioned","confidence":0.6},'
            '{"name":"2026","type":"year","role":"mentioned","confidence":0.5},'
            '{"name":"Acme","type":"company","role":"subject","confidence":0.98}'
            ']}'
        ]
    )
    extractor = EntityExtractor(provider)

    records = extractor.extract_for_article(session, article)
    session.commit()

    assert [record.name for record in records] == ["Acme"]
    assert [link.entity.name for link in article.entities] == ["Acme"]
    assert article.extraction_runs[-1].success is True


def test_entity_extractor_filters_entities_not_present_in_article_text(session):
    article = _build_article()
    session.add(article)
    session.commit()

    provider = FakeProvider(
        [
            '{"entities":['
            '{"name":"Elon Musk","type":"person","role":"mentioned","confidence":0.0},'
            '{"name":"Acme","type":"company","role":"subject","confidence":0.98},'
            '{"name":"Lima","type":"location","role":"mentioned","confidence":0.82}'
            ']}'
        ]
    )
    extractor = EntityExtractor(provider)

    records = extractor.extract_for_article(session, article)
    session.commit()

    assert [record.name for record in records] == ["Acme", "Lima"]
    assert sorted(link.entity.name for link in article.entities) == ["Acme", "Lima"]


def test_topic_extractor_parses_topics(session):
    article = _build_article()
    session.add(article)
    session.commit()

    provider = FakeProvider(
        ['{"topics":[{"topic_name":"technology","confidence":0.91},{"topic_name":"business","confidence":0.51}]}']
    )
    extractor = TopicExtractor(provider)

    assignments = extractor.extract_for_article(session, article)
    session.commit()

    assert [item.topic_name for item in assignments] == ["technology", "business"]
    assert len(article.topics) == 2


def test_entity_extractor_reconciles_reruns(session):
    article = _build_article()
    article.full_text = "Acme and Globex opened a new lab in Lima and hired engineers."
    article.cleaned_text = article.full_text
    other_article = RawArticle(
        title="Other article",
        normalized_title=normalize_title_for_dedup("Other article"),
        summary="Other",
        full_text="Other",
        cleaned_text="Other",
        url="https://example.com/other",
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        ingested_at=utcnow(),
        processing_status=ProcessingStatus.pending_extraction,
    )
    session.add_all([article, other_article])
    session.commit()

    first_provider = FakeProvider(
        ['{"entities":[{"name":"Acme","type":"company","role":"subject","confidence":0.98}]}']
    )
    first_extractor = EntityExtractor(first_provider)
    first_extractor.extract_for_article(session, article)
    session.commit()

    second_provider = FakeProvider(
        ['{"entities":[{"name":"Globex","type":"company","role":"mentioned","confidence":0.88}]}']
    )
    second_extractor = EntityExtractor(second_provider)
    second_extractor.extract_for_article(session, article)
    session.commit()

    session.refresh(article)
    assert [link.entity.name for link in article.entities] == ["Globex"]
    assert session.query(Entity).filter_by(name="Acme").one().article_count == 0
    assert session.query(Entity).filter_by(name="Globex").one().article_count == 1


def test_topic_extractor_reconciles_reruns(session):
    article = _build_article()
    session.add(article)
    session.commit()

    first_provider = FakeProvider(['{"topics":[{"topic_name":"technology","confidence":0.91}]}'])
    first_extractor = TopicExtractor(first_provider)
    first_extractor.extract_for_article(session, article)
    session.commit()

    second_provider = FakeProvider(['{"topics":[{"topic_name":"world","confidence":0.72}]}'])
    second_extractor = TopicExtractor(second_provider)
    second_extractor.extract_for_article(session, article)
    session.commit()

    session.refresh(article)
    assert [link.topic.name for link in article.topics] == ["world"]


def test_duplicate_entities_only_increment_once(session):
    article_one = _build_article()
    article_two = RawArticle(
        title="Second article",
        normalized_title=normalize_title_for_dedup("Second article"),
        summary="Second",
        full_text="Acme announced a second launch systems project in Lima.",
        cleaned_text="Acme announced a second launch systems project in Lima.",
        url="https://example.com/acme-2",
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        ingested_at=utcnow(),
        processing_status=ProcessingStatus.pending_extraction,
    )
    session.add_all([article_one, article_two])
    session.commit()

    seed_provider = FakeProvider(
        ['{"entities":[{"name":"Acme","type":"company","role":"subject","confidence":0.90}]}']
    )
    EntityExtractor(seed_provider).extract_for_article(session, article_one)
    session.commit()

    duplicate_provider = FakeProvider(
        [
            '{"entities":[{"name":"Acme","type":"company","role":"subject","confidence":0.92},'
            '{"name":"Acme","type":"company","role":"mentioned","confidence":0.80}]}'
        ]
    )
    EntityExtractor(duplicate_provider).extract_for_article(session, article_two)
    session.commit()

    acme = session.query(Entity).filter_by(name="Acme").one()
    assert acme.article_count == 2
    assert len(article_two.entities) == 1


def test_topic_get_or_create_recovers_from_integrity_error():
    existing = Topic(name="technology")
    session = MagicMock()
    session.scalar.side_effect = [None, existing]
    session.begin_nested.return_value = nullcontext()
    session.flush.side_effect = IntegrityError("stmt", "params", Exception("duplicate"))

    topic = TopicExtractor._get_or_create_topic(session, "technology")

    assert topic is existing
    session.flush.assert_called_once()


def test_entity_get_or_create_recovers_from_integrity_error():
    existing = Entity(name="Acme", normalized_name="acme", entity_type="company", article_count=1)
    session = MagicMock()
    session.scalar.side_effect = [None, existing]
    session.begin_nested.return_value = nullcontext()
    session.flush.side_effect = IntegrityError("stmt", "params", Exception("duplicate"))

    entity = EntityExtractor._get_or_create_entity(session, "Acme", "acme", "company")

    assert entity is existing
    session.flush.assert_called_once()


def test_extraction_dag_rolls_back_partial_state(monkeypatch, tmp_path):
    airflow_home = tmp_path / "airflow-home"
    logs = airflow_home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIRFLOW_HOME", str(airflow_home))
    monkeypatch.setenv("AIRFLOW__LOGGING__BASE_LOG_FOLDER", str(logs))
    monkeypatch.setenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", f"sqlite:///{airflow_home / 'airflow.db'}")
    monkeypatch.setenv("AIRFLOW__CORE__DAGS_FOLDER", str(Path(__file__).resolve().parents[2] / "dags"))
    monkeypatch.setenv("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

    extraction_dag = importlib.import_module("dags.extraction_dag")
    monkeypatch.setattr(extraction_dag, "tracked_run", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(extraction_dag, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(extraction_dag, "log_dict_artifact", lambda *args, **kwargs: None)

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr("news_pipeline.db.session.SessionLocal", SessionLocal)
    monkeypatch.setattr("dags.extraction_dag.SessionLocal", SessionLocal)

    article = RawArticle(
        title="Pending article",
        normalized_title=normalize_title_for_dedup("Pending article"),
        summary="Pending",
        full_text="Pending",
        cleaned_text="Pending",
        url="https://example.com/pending",
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        ingested_at=utcnow(),
        processing_status=ProcessingStatus.pending_extraction,
    )
    session = SessionLocal()
    session.add(article)
    session.commit()
    article_id = article.id
    session.close()

    provider = FakeProvider(
        [
            '{"entities":[{"name":"Acme","type":"company","role":"subject","confidence":0.9}]}',
            '{"topics":[{"topic_name":"technology","confidence":0.9}',  # invalid JSON
            '{"topics":[{"topic_name":"technology","confidence":0.9}',  # repair also invalid
        ]
    )
    monkeypatch.setattr(extraction_dag, "GroqProvider", lambda: provider)

    with pytest.raises(Exception):
        extraction_dag._process_article(str(article_id))

    session = SessionLocal()
    try:
        stored_article = session.query(RawArticle).filter_by(id=article_id).one()
        assert stored_article.processing_status == ProcessingStatus.failed
        assert stored_article.entities == []
        assert stored_article.topics == []
        assert len(stored_article.extraction_runs) == 1
        assert stored_article.extraction_runs[0].run_type == ExtractionRunType.topic
        assert stored_article.extraction_runs[0].success is False
    finally:
        session.close()
        engine.dispose()
