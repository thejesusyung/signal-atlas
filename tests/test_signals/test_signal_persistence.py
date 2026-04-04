"""Regression tests for Signal.article_ids cross-dialect persistence.

JSONBList stores a Python list as native JSONB on PostgreSQL and as JSON text
on SQLite.  These tests run against the SQLite in-memory fixture and verify
that the round-trip behaviour (write list → flush → expire → reload) is correct.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from news_pipeline.db.models import Signal


def _make_signal(**kwargs) -> Signal:
    defaults = dict(signal_type="entity_spike", score=0.75, article_ids=[])
    defaults.update(kwargs)
    return Signal(**defaults)


def test_article_ids_round_trip(session: Session) -> None:
    """article_ids written as a list is returned as an equal list after reload."""
    ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
    sig = _make_signal(article_ids=ids)
    session.add(sig)
    session.flush()
    session.expire(sig)  # force a SELECT on next access

    reloaded = session.get(Signal, sig.id)
    assert isinstance(reloaded.article_ids, list), "article_ids must be a list, not a raw string"
    assert reloaded.article_ids == ids


def test_article_ids_empty_list(session: Session) -> None:
    """Empty list round-trips correctly (guards against NULL or '[]' string leaking through)."""
    sig = _make_signal(article_ids=[])
    session.add(sig)
    session.flush()
    session.expire(sig)

    reloaded = session.get(Signal, sig.id)
    assert reloaded.article_ids == []


def test_article_ids_single_entry(session: Session) -> None:
    """Single-element list is not collapsed to a scalar."""
    ids = [str(uuid.uuid4())]
    sig = _make_signal(article_ids=ids)
    session.add(sig)
    session.flush()
    session.expire(sig)

    reloaded = session.get(Signal, sig.id)
    assert reloaded.article_ids == ids
