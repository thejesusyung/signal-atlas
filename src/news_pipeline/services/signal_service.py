from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from news_pipeline.db.models import Signal


def get_latest_signals(session: Session, limit: int = 10) -> list[Signal]:
    return session.scalars(
        select(Signal).order_by(Signal.detected_at.desc()).limit(limit)
    ).all()
