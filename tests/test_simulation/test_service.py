"""Tests for simulation_service.py — DB query functions over in-memory SQLite."""
from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Import sim models so they register with Base.metadata before create_all
import news_pipeline.simulation.models  # noqa: F401
from news_pipeline.db.models import Base
from news_pipeline.simulation.models import (
    SimCycle,
    SimEngagement,
    SimPersona,
    SimPromptVersion,
    SimTweet,
    SimWriter,
    SimWriterCycleScore,
)
from news_pipeline.services.simulation_service import (
    get_cycle_detail,
    get_latest_cycle_with_leaderboard,
    get_writer_evolution,
    list_cycles,
    list_writers,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def session() -> Generator[Session, None, None]:
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


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _seed_writer(session: Session, name: str = "TheBreakingWire") -> SimWriter:
    writer = SimWriter(name=name, persona_description=f"{name} persona")
    session.add(writer)
    session.flush()
    version = SimPromptVersion(
        writer_id=writer.id,
        version_number=1,
        style_prompt="Lead with facts.",
        cycle_introduced=0,
    )
    session.add(version)
    session.flush()
    writer.current_version_id = version.id
    session.flush()
    return writer


def _seed_cycle(session: Session, cycle_number: int = 1) -> SimCycle:
    cycle = SimCycle(
        cycle_number=cycle_number,
        week_number=1,
        started_at=_utcnow(),
        completed_at=_utcnow(),
        story_ids=["story-1"],
        mlflow_run_id="abc123",
    )
    session.add(cycle)
    session.flush()
    return cycle


def _seed_score(
    session: Session,
    cycle: SimCycle,
    writer: SimWriter,
    score: float = 0.5,
) -> SimWriterCycleScore:
    version = session.get(SimPromptVersion, writer.current_version_id)
    cs = SimWriterCycleScore(
        cycle_id=cycle.id,
        writer_id=writer.id,
        prompt_version_id=version.id if version else None,
        engagement_score=score,
        repost_count=3,
        like_count=2,
        comment_count=1,
        skip_count=4,
        tweet_count=3,
        reader_sample_count=10,
    )
    session.add(cs)
    session.flush()
    return cs


def _seed_tweet(
    session: Session,
    cycle: SimCycle,
    writer: SimWriter,
    content: str = "Breaking news tweet.",
) -> SimTweet:
    version = session.get(SimPromptVersion, writer.current_version_id)
    tweet = SimTweet(
        cycle_id=cycle.id,
        writer_id=writer.id,
        prompt_version_id=version.id if version else None,
        content=content,
    )
    session.add(tweet)
    session.flush()
    return tweet


def _seed_persona(session: Session, name: str = "TestPersona") -> SimPersona:
    persona = SimPersona(
        name=name,
        archetype_group="test_group",
        description="A test persona.",
    )
    session.add(persona)
    session.flush()
    return persona


def _seed_engagement(
    session: Session,
    tweet: SimTweet,
    persona: SimPersona,
    action: str = "like",
) -> SimEngagement:
    eng = SimEngagement(
        tweet_id=tweet.id,
        persona_id=persona.id,
        action=action,
        reason="interesting",
    )
    session.add(eng)
    session.flush()
    return eng


# ── get_latest_cycle_with_leaderboard ─────────────────────────────────────────


class TestGetLatestCycle:
    def test_returns_none_when_no_cycles(self, session: Session):
        assert get_latest_cycle_with_leaderboard(session) is None

    def test_returns_none_when_only_incomplete_cycle(self, session: Session):
        cycle = SimCycle(
            cycle_number=1, week_number=1, story_ids=[], completed_at=None
        )
        session.add(cycle)
        session.commit()
        assert get_latest_cycle_with_leaderboard(session) is None

    def test_returns_latest_completed_cycle(self, session: Session):
        writer = _seed_writer(session)
        cycle = _seed_cycle(session, cycle_number=1)
        _seed_score(session, cycle, writer, score=0.42)
        session.commit()

        result = get_latest_cycle_with_leaderboard(session)
        assert result is not None
        assert result["cycle"]["cycle_number"] == 1
        assert result["cycle"]["story_count"] == 1

    def test_leaderboard_sorted_by_score_desc(self, session: Session):
        w1 = _seed_writer(session, "WriterA")
        w2 = _seed_writer(session, "WriterB")
        cycle = _seed_cycle(session)
        _seed_score(session, cycle, w1, score=0.3)
        _seed_score(session, cycle, w2, score=0.7)
        session.commit()

        result = get_latest_cycle_with_leaderboard(session)
        lb = result["leaderboard"]
        assert lb[0]["writer_name"] == "WriterB"
        assert lb[0]["rank"] == 1
        assert lb[1]["writer_name"] == "WriterA"

    def test_mutations_empty_for_cycle_zero(self, session: Session):
        writer = _seed_writer(session)
        cycle = SimCycle(
            cycle_number=0, week_number=0,
            started_at=_utcnow(), completed_at=_utcnow(), story_ids=[]
        )
        session.add(cycle)
        session.flush()
        _seed_score(session, cycle, writer)
        session.commit()

        result = get_latest_cycle_with_leaderboard(session)
        assert result["mutations"] == []

    def test_picks_highest_cycle_number_not_insertion_order(self, session: Session):
        writer = _seed_writer(session)
        cycle1 = _seed_cycle(session, cycle_number=1)
        cycle2 = _seed_cycle(session, cycle_number=3)
        cycle3 = _seed_cycle(session, cycle_number=2)
        for c in [cycle1, cycle2, cycle3]:
            _seed_score(session, c, writer)
        session.commit()

        result = get_latest_cycle_with_leaderboard(session)
        assert result["cycle"]["cycle_number"] == 3


# ── list_cycles ───────────────────────────────────────────────────────────────


class TestListCycles:
    def test_empty_db_returns_zero_total(self, session: Session):
        items, total = list_cycles(session)
        assert items == []
        assert total == 0

    def test_returns_all_cycles_newest_first(self, session: Session):
        writer = _seed_writer(session)
        for n in [1, 2, 3]:
            c = _seed_cycle(session, cycle_number=n)
            _seed_score(session, c, writer)
        session.commit()

        items, total = list_cycles(session)
        assert total == 3
        assert [item["cycle_number"] for item in items] == [3, 2, 1]

    def test_pagination(self, session: Session):
        writer = _seed_writer(session)
        for n in range(1, 6):
            c = _seed_cycle(session, cycle_number=n)
            _seed_score(session, c, writer)
        session.commit()

        items, total = list_cycles(session, limit=2, offset=0)
        assert total == 5
        assert len(items) == 2
        assert items[0]["cycle_number"] == 5

        page2, _ = list_cycles(session, limit=2, offset=2)
        assert page2[0]["cycle_number"] == 3


# ── get_cycle_detail ──────────────────────────────────────────────────────────


class TestGetCycleDetail:
    def test_returns_none_for_missing_cycle(self, session: Session):
        assert get_cycle_detail(session, cycle_number=999) is None

    def test_returns_cycle_with_tweets_and_leaderboard(self, session: Session):
        writer = _seed_writer(session)
        cycle = _seed_cycle(session)
        tweet = _seed_tweet(session, cycle, writer, "Breaking: big story.")
        persona = _seed_persona(session)
        _seed_engagement(session, tweet, persona, action="repost")
        _seed_score(session, cycle, writer, score=0.6)
        session.commit()

        result = get_cycle_detail(session, cycle_number=1)
        assert result is not None
        assert result["cycle"]["cycle_number"] == 1
        assert len(result["tweets"]) == 1
        assert result["tweets"][0]["content"] == "Breaking: big story."
        assert result["tweets"][0]["repost_count"] == 1
        assert len(result["leaderboard"]) == 1


# ── list_writers ──────────────────────────────────────────────────────────────


class TestListWriters:
    def test_empty_db_returns_empty_list(self, session: Session):
        assert list_writers(session) == []

    def test_returns_writer_with_current_prompt(self, session: Session):
        _seed_writer(session, "TheBreakingWire")
        session.commit()

        writers = list_writers(session)
        assert len(writers) == 1
        assert writers[0]["name"] == "TheBreakingWire"
        assert writers[0]["current_version"] == 1
        assert writers[0]["current_style_prompt"] == "Lead with facts."

    def test_total_mutations_counts_versions_beyond_first(self, session: Session):
        writer = _seed_writer(session)
        # Add a second version (mutation)
        v2 = SimPromptVersion(
            writer_id=writer.id,
            version_number=2,
            style_prompt="Be more provocative.",
            cycle_introduced=1,
            triggered_by_score=0.12,
        )
        session.add(v2)
        session.flush()
        writer.current_version_id = v2.id
        session.commit()

        writers = list_writers(session)
        assert writers[0]["total_mutations"] == 1


# ── get_writer_evolution ──────────────────────────────────────────────────────


class TestGetWriterEvolution:
    def test_returns_none_for_unknown_writer(self, session: Session):
        assert get_writer_evolution(session, writer_name="NoSuchWriter") is None

    def test_returns_single_version_on_fresh_writer(self, session: Session):
        _seed_writer(session, "SharpTake")
        session.commit()

        result = get_writer_evolution(session, writer_name="SharpTake")
        assert result is not None
        assert result["writer_name"] == "SharpTake"
        assert result["total_mutations"] == 0
        assert len(result["versions"]) == 1
        assert result["versions"][0]["version_number"] == 1

    def test_returns_full_lineage_after_mutations(self, session: Session):
        writer = _seed_writer(session, "DataDeskDaily")
        v2 = SimPromptVersion(
            writer_id=writer.id,
            version_number=2,
            style_prompt="Include a statistic.",
            cycle_introduced=2,
            triggered_by_score=0.09,
        )
        v3 = SimPromptVersion(
            writer_id=writer.id,
            version_number=3,
            style_prompt="Lead with the number, explain after.",
            cycle_introduced=4,
            triggered_by_score=0.15,
        )
        session.add_all([v2, v3])
        session.commit()

        result = get_writer_evolution(session, writer_name="DataDeskDaily")
        assert result["total_mutations"] == 2
        assert len(result["versions"]) == 3
        # Versions ordered by version_number ascending
        assert [v["version_number"] for v in result["versions"]] == [1, 2, 3]
        assert result["versions"][1]["triggered_by_score"] == pytest.approx(0.09, abs=1e-4)

    def test_current_version_reflects_latest(self, session: Session):
        writer = _seed_writer(session, "FeedChronos")
        v2 = SimPromptVersion(
            writer_id=writer.id,
            version_number=2,
            style_prompt="Be ironic.",
            cycle_introduced=1,
        )
        session.add(v2)
        session.flush()
        writer.current_version_id = v2.id
        session.commit()

        result = get_writer_evolution(session, writer_name="FeedChronos")
        assert result["current_version"] == 2
