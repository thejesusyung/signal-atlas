from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select

from news_pipeline.db.models import Base, LLMRateLimitReservation
from news_pipeline.llm.rate_limit import DatabaseRequestRateLimiter, get_shared_rate_limiter


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.current = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)


def test_database_rate_limiter_shares_reservations_across_instances(tmp_path):
    database_path = tmp_path / "rate-limit.sqlite"
    database_url = f"sqlite:///{database_path}"
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    engine.dispose()

    clock = FakeClock(datetime(2026, 3, 25, tzinfo=timezone.utc))
    limiter_one = DatabaseRequestRateLimiter(
        provider_name="groq",
        requests_per_minute=1,
        database_url=database_url,
        window_seconds=10.0,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )
    limiter_two = DatabaseRequestRateLimiter(
        provider_name="groq",
        requests_per_minute=1,
        database_url=database_url,
        window_seconds=10.0,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    limiter_one.acquire()
    limiter_two.acquire()

    assert clock.sleeps == [10.0]

    verification_engine = create_engine(database_url, future=True)
    try:
        with verification_engine.connect() as connection:
            reservations = connection.execute(
                select(LLMRateLimitReservation).order_by(LLMRateLimitReservation.reserved_at.asc())
            ).all()
    finally:
        verification_engine.dispose()
    assert len(reservations) == 2
    assert reservations[1].reserved_at - reservations[0].reserved_at == timedelta(seconds=10)


def test_in_memory_rate_limiter_paces_requests(monkeypatch):
    from news_pipeline.llm.rate_limit import InMemoryRequestRateLimiter

    current = {"now": 100.0}
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return current["now"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current["now"] += seconds

    monkeypatch.setattr("news_pipeline.llm.rate_limit.time.monotonic", fake_monotonic)
    monkeypatch.setattr("news_pipeline.llm.rate_limit.time.sleep", fake_sleep)

    limiter = InMemoryRequestRateLimiter(requests_per_minute=30)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert sleeps == [2.0, 2.0]


def test_database_rate_limiter_backoff_delays_next_slot(tmp_path):
    database_path = tmp_path / "rate-limit-backoff.sqlite"
    database_url = f"sqlite:///{database_path}"
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    engine.dispose()

    clock = FakeClock(datetime(2026, 3, 25, tzinfo=timezone.utc))
    limiter = DatabaseRequestRateLimiter(
        provider_name="groq",
        requests_per_minute=30,
        database_url=database_url,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    limiter.backoff(12.0)
    limiter.acquire()

    assert clock.sleeps == [14.0]


def test_get_shared_rate_limiter_builds_memory_backend(monkeypatch):
    monkeypatch.setenv("LLM_RATE_LIMIT_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    from news_pipeline.config import get_settings
    from news_pipeline.llm import rate_limit

    get_settings.cache_clear()
    rate_limit._LIMITERS.clear()

    try:
        limiter = get_shared_rate_limiter("groq", 30)
        assert limiter.__class__.__name__ == "InMemoryRequestRateLimiter"
    finally:
        rate_limit._LIMITERS.clear()
        get_settings.cache_clear()
