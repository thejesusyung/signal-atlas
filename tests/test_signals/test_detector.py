"""
Tests for detect_and_persist_signals() baseline-gating logic.

The key regression: before the fix, a single ArticleEntity row count guarded
*both* entity and topic signal detection.  A dataset with sufficient topic
history but sparse entity history would never produce topic signals.

Each test controls which baseline window has >= _MIN_*_BASELINE_ROWS rows and
asserts that only the correctly-gated signal type is produced.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from news_pipeline.contracts import LLMResponse
from news_pipeline.db.models import (
    ArticleEntity,
    ArticleTopic,
    Base,
    Entity,
    EntityType,
    ProcessingStatus,
    RawArticle,
    Signal,
    Topic,
    TopicMethod,
)
from news_pipeline.llm.provider import LLMProvider, LLMTraceContext
from news_pipeline.utils import normalize_entity_name, normalize_title_for_dedup
from news_pipeline.signals.detector import detect_and_persist_signals


# ---------------------------------------------------------------------------
# Shared test settings (override get_settings() in the module under test)
# ---------------------------------------------------------------------------

TEST_SETTINGS = SimpleNamespace(
    signal_current_window_hours=24,
    signal_baseline_window_hours=72,
    signal_zscore_threshold=1.5,
    signal_top_n=10,
)


class StubLLMProvider(LLMProvider):
    """Returns a minimal valid JSON summary without hitting any API."""

    provider_name = "stub"

    def complete(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 900,
        trace_context: LLMTraceContext | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text=json.dumps({"summary": "Stub summary."}),
            model="stub-model",
            tokens_used=0,
            latency_ms=0,
            provider_name=self.provider_name,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    sess = SessionLocal()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


@pytest.fixture()
def provider() -> StubLLMProvider:
    return StubLLMProvider()


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _baseline_ts(offset_hours: int = 48) -> datetime:
    """A timestamp that falls inside the baseline window (24–72 h ago)."""
    return _now() - timedelta(hours=offset_hours)


def _current_ts(offset_hours: int = 1) -> datetime:
    """A timestamp that falls inside the current window (< 24 h ago)."""
    return _now() - timedelta(hours=offset_hours)


# ---------------------------------------------------------------------------
# Helpers that insert rows with explicit timestamps
# ---------------------------------------------------------------------------

def _make_article(session: Session, title: str, ingested_at: datetime) -> RawArticle:
    article = RawArticle(
        title=title,
        normalized_title=normalize_title_for_dedup(title),
        summary="",
        url=f"https://example.com/{title.replace(' ', '-')}",
        source_name="Test Feed",
        source_feed="https://example.com/feed",
        category="test",
        published_at=ingested_at,
        ingested_at=ingested_at,
        processing_status=ProcessingStatus.extracted,
        word_count=5,
    )
    session.add(article)
    session.flush()
    return article


def _make_entity(session: Session, name: str) -> Entity:
    entity = Entity(
        name=name,
        entity_type=EntityType.company,
        normalized_name=normalize_entity_name(name),
        article_count=1,
    )
    session.add(entity)
    session.flush()
    return entity


def _make_topic(session: Session, name: str) -> Topic:
    topic = Topic(name=name)
    session.add(topic)
    session.flush()
    return topic


def _link_entity(
    session: Session, article: RawArticle, entity: Entity, extracted_at: datetime
) -> None:
    session.add(
        ArticleEntity(
            article_id=article.id,
            entity_id=entity.id,
            role="subject",
            confidence=0.9,
            extracted_at=extracted_at,
        )
    )
    session.flush()


def _link_topic(session: Session, article: RawArticle, topic: Topic) -> None:
    session.add(
        ArticleTopic(
            article_id=article.id,
            topic_id=topic.id,
            confidence=0.9,
            method=TopicMethod.llm,
        )
    )
    session.flush()


def _seed_entity_baseline(session: Session, entity: Entity, n: int = 10) -> None:
    """Insert *n* entity rows in the baseline window to satisfy the entity gate."""
    for i in range(n):
        article = _make_article(session, f"Baseline entity article {i}", _baseline_ts())
        _link_entity(session, article, entity, extracted_at=_baseline_ts())


def _seed_topic_baseline(session: Session, topic: Topic, n: int = 10) -> None:
    """Insert *n* article-topic rows in the baseline window to satisfy the topic gate."""
    for i in range(n):
        article = _make_article(session, f"Baseline topic article {i}", _baseline_ts())
        _link_topic(session, article, topic)


def _seed_entity_spike(session: Session, entity: Entity, n: int = 3) -> None:
    """Insert *n* entity rows in the current window (triggers zero-baseline score ≥ 1.5)."""
    for i in range(n):
        article = _make_article(session, f"Spike entity article {i}", _current_ts())
        _link_entity(session, article, entity, extracted_at=_current_ts())


def _seed_topic_spike(session: Session, topic: Topic, n: int = 3) -> None:
    """Insert *n* article-topic rows in the current window (triggers zero-baseline score ≥ 1.5)."""
    for i in range(n):
        article = _make_article(session, f"Spike topic article {i}", _current_ts())
        _link_topic(session, article, topic)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselineGating:
    """
    Verify that entity and topic signals are gated on their own baseline counts,
    not on a shared count.
    """

    # Stub out summary generation so tests don't need a live MLflow server.
    @pytest.fixture(autouse=True)
    def _patch_generate_summary(self):
        with patch(
            "news_pipeline.signals.detector._generate_summary",
            return_value="stub summary",
        ):
            yield

    @patch("news_pipeline.signals.detector.get_settings", return_value=TEST_SETTINGS)
    def test_entity_signals_skipped_when_entity_baseline_insufficient(
        self, _mock_settings, session: Session, provider: StubLLMProvider
    ) -> None:
        """Fewer than _MIN_ENTITY_BASELINE_ROWS entity rows → no entity signals,
        even when there is current-window entity activity."""
        background_entity = _make_entity(session, "Background Corp")
        spike_entity = _make_entity(session, "NewCo")

        # Only 5 baseline entity rows (below threshold of 10)
        _seed_entity_baseline(session, background_entity, n=5)

        # Ample current-window entity activity — would fire if gate passed
        _seed_entity_spike(session, spike_entity, n=3)

        signals = detect_and_persist_signals(session, provider)

        entity_signals = [s for s in signals if s.signal_type == "entity_velocity"]
        assert entity_signals == [], (
            "Entity signals should be suppressed when entity baseline < threshold"
        )

    @patch("news_pipeline.signals.detector.get_settings", return_value=TEST_SETTINGS)
    def test_topic_signals_run_when_topic_baseline_sufficient_despite_no_entity_data(
        self, _mock_settings, session: Session, provider: StubLLMProvider
    ) -> None:
        """Topic baseline ≥ threshold → topic signals are generated even when
        there are zero entity rows in the baseline (entity gate fails)."""
        bg_topic = _make_topic(session, "technology")
        spike_topic = _make_topic(session, "climate")

        # No entity data at all — entity gate will fail
        # But topic baseline is satisfied
        _seed_topic_baseline(session, bg_topic, n=10)

        # New topic with current-window activity → zero-baseline spike score = 3.0
        _seed_topic_spike(session, spike_topic, n=3)

        signals = detect_and_persist_signals(session, provider)

        entity_signals = [s for s in signals if s.signal_type == "entity_velocity"]
        topic_signals = [s for s in signals if s.signal_type == "topic_velocity"]

        assert entity_signals == [], "No entity signals expected (no entity baseline data)"
        assert len(topic_signals) >= 1, "Topic signals should run when topic baseline is sufficient"
        assert any(s.topic_name == "climate" for s in topic_signals)

    @patch("news_pipeline.signals.detector.get_settings", return_value=TEST_SETTINGS)
    def test_entity_signals_run_when_entity_baseline_sufficient_despite_no_topic_data(
        self, _mock_settings, session: Session, provider: StubLLMProvider
    ) -> None:
        """Entity baseline ≥ threshold → entity signals are generated even when
        there are zero topic rows in the baseline (topic gate fails)."""
        bg_entity = _make_entity(session, "Background Corp")
        spike_entity = _make_entity(session, "NewCo")

        # Entity baseline is satisfied; no topic data at all
        _seed_entity_baseline(session, bg_entity, n=10)

        # New entity with current-window activity → zero-baseline spike score = 3.0
        _seed_entity_spike(session, spike_entity, n=3)

        signals = detect_and_persist_signals(session, provider)

        entity_signals = [s for s in signals if s.signal_type == "entity_velocity"]
        topic_signals = [s for s in signals if s.signal_type == "topic_velocity"]

        assert len(entity_signals) >= 1, "Entity signals should run when entity baseline is sufficient"
        assert any(s.entity_id == spike_entity.id for s in entity_signals)
        assert topic_signals == [], "No topic signals expected (no topic baseline data)"

    @patch("news_pipeline.signals.detector.get_settings", return_value=TEST_SETTINGS)
    def test_both_signal_types_run_when_both_baselines_sufficient(
        self, _mock_settings, session: Session, provider: StubLLMProvider
    ) -> None:
        """When both entity and topic baselines meet the threshold, both signal
        types are evaluated and can fire independently."""
        bg_entity = _make_entity(session, "Background Corp")
        spike_entity = _make_entity(session, "NewCo")
        bg_topic = _make_topic(session, "technology")
        spike_topic = _make_topic(session, "climate")

        _seed_entity_baseline(session, bg_entity, n=10)
        _seed_topic_baseline(session, bg_topic, n=10)

        _seed_entity_spike(session, spike_entity, n=3)
        _seed_topic_spike(session, spike_topic, n=3)

        signals = detect_and_persist_signals(session, provider)

        entity_signals = [s for s in signals if s.signal_type == "entity_velocity"]
        topic_signals = [s for s in signals if s.signal_type == "topic_velocity"]

        assert len(entity_signals) >= 1, "Entity signals should fire when both baselines are met"
        assert len(topic_signals) >= 1, "Topic signals should fire when both baselines are met"

    @patch("news_pipeline.signals.detector.get_settings", return_value=TEST_SETTINGS)
    def test_returns_empty_when_neither_baseline_is_sufficient(
        self, _mock_settings, session: Session, provider: StubLLMProvider
    ) -> None:
        """When both baselines are below threshold, no signals are produced."""
        bg_entity = _make_entity(session, "Background Corp")
        spike_entity = _make_entity(session, "NewCo")
        bg_topic = _make_topic(session, "technology")
        spike_topic = _make_topic(session, "climate")

        # Only 5 rows each — both below threshold of 10
        _seed_entity_baseline(session, bg_entity, n=5)
        _seed_topic_baseline(session, bg_topic, n=5)

        _seed_entity_spike(session, spike_entity, n=3)
        _seed_topic_spike(session, spike_topic, n=3)

        signals = detect_and_persist_signals(session, provider)
        assert signals == [], "No signals should be produced when both baselines are insufficient"
