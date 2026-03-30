from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Callable

from sqlalchemy import delete, func, insert, select, text
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from news_pipeline.config import get_settings
from news_pipeline.db.models import LLMRateLimitReservation


class RequestRateLimiter:
    def acquire(self) -> None:
        raise NotImplementedError

    def backoff(self, delay_seconds: float) -> None:
        return None


class InMemoryRequestRateLimiter(RequestRateLimiter):
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self.min_interval_seconds = self.window_seconds / self.requests_per_minute
        self._next_available_at = 0.0
        self._lock = Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            reserved_for = max(now, self._next_available_at)
            self._next_available_at = reserved_for + self.min_interval_seconds
            sleep_for = reserved_for - now

        if sleep_for > 0:
            time.sleep(max(sleep_for, 0.01))

    def backoff(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        with self._lock:
            self._next_available_at = max(self._next_available_at, time.monotonic() + delay_seconds)


class DatabaseRequestRateLimiter(RequestRateLimiter):
    def __init__(
        self,
        provider_name: str,
        requests_per_minute: int,
        database_url: str,
        *,
        window_seconds: float = 60.0,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        fallback_limiter: RequestRateLimiter | None = None,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.provider_name = provider_name
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.min_interval_seconds = self.window_seconds / self.requests_per_minute
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.sleep_fn = sleep_fn or time.sleep
        self.fallback_limiter = fallback_limiter
        self.engine = create_engine(database_url, future=True, pool_pre_ping=True)
        self.table = LLMRateLimitReservation.__table__

    def acquire(self) -> None:
        try:
            sleep_for = self._reserve_slot()
        except OperationalError:
            self._acquire_with_fallback()
            return
        except SQLAlchemyError:
            self._acquire_with_fallback()
            return

        if sleep_for > 0:
            self.sleep_fn(max(sleep_for, 0.01))

    def backoff(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        try:
            self._reserve_backoff(delay_seconds)
        except OperationalError:
            self._backoff_with_fallback(delay_seconds)
        except SQLAlchemyError:
            self._backoff_with_fallback(delay_seconds)

    def _reserve_slot(self) -> float:
        with self.engine.begin() as connection:
            now, latest_reservation = self._prepare_provider_window(connection)

            if latest_reservation is None:
                reserved_at = now
            else:
                latest_reservation = self._normalize_timestamp(latest_reservation)
                reserved_at = max(
                    now,
                    latest_reservation + timedelta(seconds=self.min_interval_seconds),
                )

            connection.execute(
                insert(self.table).values(provider_name=self.provider_name, reserved_at=reserved_at)
            )
            return (reserved_at - now).total_seconds()

    def _reserve_backoff(self, delay_seconds: float) -> None:
        with self.engine.begin() as connection:
            now, latest_reservation = self._prepare_provider_window(connection)
            reserved_at = now + timedelta(seconds=delay_seconds)
            if latest_reservation is not None:
                latest_reservation = self._normalize_timestamp(latest_reservation)
                reserved_at = max(reserved_at, latest_reservation)
            connection.execute(
                insert(self.table).values(provider_name=self.provider_name, reserved_at=reserved_at)
            )

    def _prepare_provider_window(self, connection) -> tuple[datetime, datetime | None]:
        self._lock_provider_window(connection)
        now = self._normalize_timestamp(self.now_fn())
        cutoff = now - timedelta(seconds=self.window_seconds)

        connection.execute(
            delete(self.table).where(
                self.table.c.provider_name == self.provider_name,
                self.table.c.reserved_at <= cutoff,
            )
        )

        latest_reservation = connection.execute(
            select(func.max(self.table.c.reserved_at)).where(
                self.table.c.provider_name == self.provider_name
            )
        ).scalar_one()
        return now, latest_reservation

    def _lock_provider_window(self, connection) -> None:
        if connection.dialect.name != "postgresql":
            return
        connection.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": self._advisory_lock_key()},
        )

    def _advisory_lock_key(self) -> int:
        digest = hashlib.blake2b(self.provider_name.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=True)

    def _acquire_with_fallback(self) -> None:
        if self.fallback_limiter is None:
            raise
        self.fallback_limiter.acquire()

    def _backoff_with_fallback(self, delay_seconds: float) -> None:
        if self.fallback_limiter is None:
            raise
        self.fallback_limiter.backoff(delay_seconds)

    @staticmethod
    def _normalize_timestamp(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


_LIMITERS: dict[tuple[str, str, int, str], RequestRateLimiter] = {}
_LIMITERS_LOCK = Lock()


def get_shared_rate_limiter(
    name: str,
    requests_per_minute: int,
    *,
    backend: str | None = None,
    database_url: str | None = None,
) -> RequestRateLimiter:
    settings = get_settings()
    resolved_backend = (backend or settings.llm_rate_limit_backend).strip().lower()
    resolved_database_url = database_url or settings.database_url
    key = (resolved_backend, name, requests_per_minute, resolved_database_url)

    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = _build_rate_limiter(
                backend=resolved_backend,
                name=name,
                requests_per_minute=requests_per_minute,
                database_url=resolved_database_url,
            )
            _LIMITERS[key] = limiter
        return limiter


def _build_rate_limiter(
    *,
    backend: str,
    name: str,
    requests_per_minute: int,
    database_url: str,
) -> RequestRateLimiter:
    if backend == "memory":
        return InMemoryRequestRateLimiter(requests_per_minute=requests_per_minute)
    if backend == "database":
        return DatabaseRequestRateLimiter(
            provider_name=name,
            requests_per_minute=requests_per_minute,
            database_url=database_url,
        )
    if backend == "auto":
        return DatabaseRequestRateLimiter(
            provider_name=name,
            requests_per_minute=requests_per_minute,
            database_url=database_url,
            fallback_limiter=InMemoryRequestRateLimiter(requests_per_minute=requests_per_minute),
        )
    raise ValueError(f"Unsupported LLM rate-limit backend: {backend}")
