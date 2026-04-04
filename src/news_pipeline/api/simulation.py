"""FastAPI router for simulation endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from news_pipeline.config import get_settings
from news_pipeline.db.session import get_session
from news_pipeline.services.simulation_service import (
    get_cycle_detail,
    get_latest_cycle_with_leaderboard,
    get_writer_evolution,
    list_cycles,
    list_writers,
)

settings = get_settings()
router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.get("/latest")
def get_latest(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Most recent completed cycle: leaderboard + tweet table + mutation log."""
    result = get_latest_cycle_with_leaderboard(session)
    if result is None:
        raise HTTPException(status_code=404, detail="No completed simulation cycles yet")
    return result


@router.get("/writers")
def get_writers(session: Session = Depends(get_session)) -> dict[str, Any]:
    """All writers with current prompt version, all-time stats, and mutation count."""
    writers = list_writers(session)
    return {"writers": writers, "total": len(writers)}


@router.get("/writers/{name}/evolution")
def get_evolution(
    name: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Full prompt version lineage for one writer, with performance per version."""
    result = get_writer_evolution(session, writer_name=name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Writer {name!r} not found")
    return result


@router.get("/cycles")
def get_cycles(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(
        default=settings.api_page_size, ge=1, le=settings.api_max_page_size
    ),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Paginated list of all simulation cycles, newest first."""
    offset = (page - 1) * page_size
    items, total = list_cycles(session, limit=page_size, offset=offset)
    return {"items": items, "total": total, "page": page}


@router.get("/cycles/{cycle_number}")
def get_cycle(
    cycle_number: int, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """One cycle: per-writer leaderboard + every tweet with engagement breakdown."""
    result = get_cycle_detail(session, cycle_number=cycle_number)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Cycle {cycle_number} not found"
        )
    return result
