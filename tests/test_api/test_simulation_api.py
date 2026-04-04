"""Tests for the /simulation/* FastAPI endpoints."""
from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Import sim models so they register with Base.metadata before create_all
import news_pipeline.simulation.models  # noqa: F401
from news_pipeline.api.app import app
from news_pipeline.db.models import Base
from news_pipeline.db.session import get_session
from news_pipeline.simulation.models import (
    SimCycle,
    SimEngagement,
    SimPersona,
    SimPromptVersion,
    SimTweet,
    SimWriter,
    SimWriterCycleScore,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@pytest.fixture()
def sim_session() -> Generator[Session, None, None]:
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
def empty_client(sim_session: Session) -> Generator[TestClient, None, None]:
    """Client with an empty simulation database."""
    def _override() -> Generator[Session, None, None]:
        yield sim_session

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _seed_simulation(session: Session) -> dict:
    """Seed one complete simulation cycle with a writer, tweet, and score."""
    writer = SimWriter(name="TheBreakingWire", persona_description="Neutral wire service.")
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

    cycle = SimCycle(
        cycle_number=1,
        week_number=1,
        started_at=_utcnow(),
        completed_at=_utcnow(),
        story_ids=["story-abc"],
        mlflow_run_id="run123",
    )
    session.add(cycle)
    session.flush()

    tweet = SimTweet(
        cycle_id=cycle.id,
        writer_id=writer.id,
        prompt_version_id=version.id,
        content="Breaking: major event confirmed.",
    )
    session.add(tweet)
    session.flush()

    persona = SimPersona(
        name="TestReader",
        archetype_group="casual_lurker",
        description="Just scrolling.",
    )
    session.add(persona)
    session.flush()

    session.add(SimEngagement(
        tweet_id=tweet.id, persona_id=persona.id, action="repost", reason="good"
    ))
    session.flush()

    score = SimWriterCycleScore(
        cycle_id=cycle.id,
        writer_id=writer.id,
        prompt_version_id=version.id,
        engagement_score=0.75,
        repost_count=3,
        like_count=2,
        comment_count=1,
        skip_count=4,
        tweet_count=2,
        reader_sample_count=10,
    )
    session.add(score)
    session.commit()

    return {"writer": writer, "cycle": cycle, "version": version}


@pytest.fixture()
def seeded_client(sim_session: Session) -> Generator[TestClient, None, None]:
    """Client with one seeded simulation cycle."""
    _seed_simulation(sim_session)

    def _override() -> Generator[Session, None, None]:
        yield sim_session

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── GET /simulation/latest ────────────────────────────────────────────────────


def test_latest_returns_404_when_no_cycles(empty_client: TestClient):
    response = empty_client.get("/simulation/latest")
    assert response.status_code == 404


def test_latest_returns_200_with_seeded_data(seeded_client: TestClient):
    response = seeded_client.get("/simulation/latest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["cycle"]["cycle_number"] == 1
    assert payload["cycle"]["story_count"] == 1
    assert len(payload["leaderboard"]) == 1
    assert payload["leaderboard"][0]["writer_name"] == "TheBreakingWire"
    assert payload["leaderboard"][0]["engagement_score"] == pytest.approx(0.75, abs=0.001)


def test_latest_includes_mlflow_run_id(seeded_client: TestClient):
    payload = seeded_client.get("/simulation/latest").json()
    assert payload["cycle"]["mlflow_run_id"] == "run123"


# ── GET /simulation/cycles ────────────────────────────────────────────────────


def test_cycles_list_returns_200(seeded_client: TestClient):
    response = seeded_client.get("/simulation/cycles")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert len(payload["items"]) == 1
    assert payload["items"][0]["cycle_number"] == 1


def test_cycles_list_empty_when_no_data(empty_client: TestClient):
    response = empty_client.get("/simulation/cycles")
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_cycles_list_pagination_params(seeded_client: TestClient):
    response = seeded_client.get("/simulation/cycles", params={"page": 1, "page_size": 5})
    assert response.status_code == 200
    assert response.json()["page"] == 1


# ── GET /simulation/cycles/{cycle_number} ─────────────────────────────────────


def test_cycle_detail_returns_404_for_missing(seeded_client: TestClient):
    response = seeded_client.get("/simulation/cycles/999")
    assert response.status_code == 404


def test_cycle_detail_returns_200_with_tweets(seeded_client: TestClient):
    response = seeded_client.get("/simulation/cycles/1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["cycle"]["cycle_number"] == 1
    assert len(payload["tweets"]) == 1
    assert payload["tweets"][0]["content"] == "Breaking: major event confirmed."
    assert payload["tweets"][0]["repost_count"] == 1


def test_cycle_detail_includes_leaderboard(seeded_client: TestClient):
    payload = seeded_client.get("/simulation/cycles/1").json()
    assert len(payload["leaderboard"]) == 1
    assert payload["leaderboard"][0]["rank"] == 1


# ── GET /simulation/writers ───────────────────────────────────────────────────


def test_writers_returns_200(seeded_client: TestClient):
    response = seeded_client.get("/simulation/writers")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    writer = payload["writers"][0]
    assert writer["name"] == "TheBreakingWire"
    assert writer["current_version"] == 1
    assert writer["current_style_prompt"] == "Lead with facts."


def test_writers_empty_list_when_no_data(empty_client: TestClient):
    response = empty_client.get("/simulation/writers")
    assert response.status_code == 200
    assert response.json()["total"] == 0


# ── GET /simulation/writers/{name}/evolution ──────────────────────────────────


def test_writer_evolution_returns_404_for_unknown(seeded_client: TestClient):
    response = seeded_client.get("/simulation/writers/NoSuchWriter/evolution")
    assert response.status_code == 404


def test_writer_evolution_returns_200_with_versions(seeded_client: TestClient):
    response = seeded_client.get("/simulation/writers/TheBreakingWire/evolution")
    assert response.status_code == 200
    payload = response.json()
    assert payload["writer_name"] == "TheBreakingWire"
    assert payload["total_mutations"] == 0
    assert len(payload["versions"]) == 1
    assert payload["versions"][0]["version_number"] == 1
    assert payload["versions"][0]["style_prompt"] == "Lead with facts."
