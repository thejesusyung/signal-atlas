"""DB query functions for the simulation API endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import case, desc, func, select
from sqlalchemy.orm import Session

from news_pipeline.simulation.models import (
    SimCycle,
    SimEngagement,
    SimPromptVersion,
    SimTweet,
    SimWriter,
    SimWriterCycleScore,
)


# ── Latest cycle ──────────────────────────────────────────────────────────────


def get_latest_cycle_with_leaderboard(session: Session) -> dict[str, Any] | None:
    """Return the most recent completed cycle with its per-writer leaderboard.

    Returns None if no cycles exist yet.
    """
    cycle = session.scalars(
        select(SimCycle)
        .where(SimCycle.completed_at.isnot(None))
        .order_by(desc(SimCycle.cycle_number))
        .limit(1)
    ).first()

    if cycle is None:
        return None

    leaderboard = _cycle_leaderboard(session, cycle.id, cycle.cycle_number)
    mutations = _mutations_for_cycle(session, cycle.cycle_number)
    tweets = _tweets_for_cycle(session, cycle.id)

    return {
        "cycle": _serialize_cycle(cycle),
        "leaderboard": leaderboard,
        "tweets": tweets,
        "mutations": mutations,
    }


# ── Cycle list ────────────────────────────────────────────────────────────────


def list_cycles(
    session: Session, limit: int = 20, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """Paginated list of cycles (newest first) with aggregate stats."""
    total = session.scalar(select(func.count()).select_from(SimCycle)) or 0

    cycles = session.scalars(
        select(SimCycle)
        .order_by(desc(SimCycle.cycle_number))
        .limit(limit)
        .offset(offset)
    ).all()

    items = []
    for cycle in cycles:
        avg_score = session.scalar(
            select(func.avg(SimWriterCycleScore.engagement_score)).where(
                SimWriterCycleScore.cycle_id == cycle.id
            )
        )
        mutation_count = session.scalar(
            select(func.count())
            .select_from(SimPromptVersion)
            .where(SimPromptVersion.cycle_introduced == cycle.cycle_number)
            .where(SimPromptVersion.cycle_introduced > 0)
        )
        items.append(
            {
                **_serialize_cycle(cycle),
                "avg_engagement_score": round(float(avg_score), 4) if avg_score else None,
                "mutation_count": mutation_count or 0,
            }
        )

    return items, total


# ── Cycle detail (tweets) ─────────────────────────────────────────────────────


def get_cycle_detail(session: Session, cycle_number: int) -> dict[str, Any] | None:
    """Return one cycle with all its tweets and per-tweet engagement breakdown."""
    cycle = session.scalars(
        select(SimCycle).where(SimCycle.cycle_number == cycle_number)
    ).first()

    if cycle is None:
        return None

    # Single query: tweets + engagement counts via conditional aggregation
    tweets = _tweets_for_cycle(session, cycle.id)
    leaderboard = _cycle_leaderboard(session, cycle.id, cycle.cycle_number)

    return {
        "cycle": _serialize_cycle(cycle),
        "leaderboard": leaderboard,
        "tweets": tweets,
    }


# ── Writer list ───────────────────────────────────────────────────────────────


def list_writers(session: Session) -> list[dict[str, Any]]:
    """Return all writers with their current prompt version and all-time stats."""
    writers = session.scalars(
        select(SimWriter).order_by(SimWriter.name)
    ).all()

    result = []
    for writer in writers:
        current_version = (
            session.get(SimPromptVersion, writer.current_version_id)
            if writer.current_version_id
            else None
        )

        stats = session.execute(
            select(
                func.avg(SimWriterCycleScore.engagement_score).label("avg_score"),
                func.count(SimWriterCycleScore.cycle_id).label("cycles"),
                func.count(SimTweet.id).label("total_tweets"),
            )
            .select_from(SimWriter)
            .outerjoin(SimWriterCycleScore, SimWriterCycleScore.writer_id == SimWriter.id)
            .outerjoin(SimTweet, SimTweet.writer_id == SimWriter.id)
            .where(SimWriter.id == writer.id)
        ).one()

        mutation_count = session.scalar(
            select(func.count())
            .select_from(SimPromptVersion)
            .where(SimPromptVersion.writer_id == writer.id)
            .where(SimPromptVersion.version_number > 1)
        )

        result.append(
            {
                "name": writer.name,
                "persona": writer.persona_description,
                "current_version": (
                    current_version.version_number if current_version else None
                ),
                "current_style_prompt": (
                    current_version.style_prompt if current_version else None
                ),
                "all_time_avg_score": (
                    round(float(stats.avg_score), 4) if stats.avg_score else None
                ),
                "cycles_participated": stats.cycles or 0,
                "total_tweets": stats.total_tweets or 0,
                "total_mutations": mutation_count or 0,
            }
        )

    return result


# ── Writer evolution ──────────────────────────────────────────────────────────


def get_writer_evolution(session: Session, writer_name: str) -> dict[str, Any] | None:
    """Return a writer's full prompt version lineage with performance per version."""
    writer = session.scalars(
        select(SimWriter).where(SimWriter.name == writer_name)
    ).first()

    if writer is None:
        return None

    versions = session.scalars(
        select(SimPromptVersion)
        .where(SimPromptVersion.writer_id == writer.id)
        .order_by(SimPromptVersion.version_number)
    ).all()

    version_list = []
    for version in versions:
        avg_score = session.scalar(
            select(func.avg(SimWriterCycleScore.engagement_score)).where(
                SimWriterCycleScore.prompt_version_id == version.id
            )
        )
        cycles_active = session.scalar(
            select(func.count()).select_from(SimWriterCycleScore).where(
                SimWriterCycleScore.prompt_version_id == version.id
            )
        )
        version_list.append(
            {
                "version_number": version.version_number,
                "style_prompt": version.style_prompt,
                "cycle_introduced": version.cycle_introduced,
                "triggered_by_score": (
                    round(version.triggered_by_score, 4)
                    if version.triggered_by_score is not None
                    else None
                ),
                "avg_score_while_active": (
                    round(float(avg_score), 4) if avg_score else None
                ),
                "cycles_active": cycles_active or 0,
            }
        )

    return {
        "writer_name": writer.name,
        "persona": writer.persona_description,
        "current_version": (
            version_list[-1]["version_number"] if version_list else None
        ),
        "total_mutations": max(0, len(version_list) - 1),
        "versions": version_list,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _tweets_for_cycle(session: Session, cycle_uuid: UUID) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            SimTweet.id.label("tweet_id"),
            SimTweet.content,
            SimWriter.name.label("writer_name"),
            SimPromptVersion.version_number.label("prompt_version"),
            func.count(
                case((SimEngagement.action == "repost", 1))
            ).label("repost_count"),
            func.count(
                case((SimEngagement.action == "like", 1))
            ).label("like_count"),
            func.count(
                case((SimEngagement.action == "comment", 1))
            ).label("comment_count"),
            func.count(
                case((SimEngagement.action == "skip", 1))
            ).label("skip_count"),
        )
        .join(SimWriter, SimTweet.writer_id == SimWriter.id)
        .outerjoin(SimEngagement, SimEngagement.tweet_id == SimTweet.id)
        .outerjoin(SimPromptVersion, SimTweet.prompt_version_id == SimPromptVersion.id)
        .where(SimTweet.cycle_id == cycle_uuid)
        .group_by(
            SimTweet.id,
            SimTweet.content,
            SimWriter.name,
            SimPromptVersion.version_number,
        )
        .order_by(SimWriter.name, SimTweet.created_at)
    ).all()
    return [_serialize_tweet_row(row) for row in rows]


def _cycle_leaderboard(
    session: Session, cycle_uuid: UUID, cycle_number: int
) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            SimWriterCycleScore,
            SimWriter.name.label("writer_name"),
            SimPromptVersion.version_number.label("prompt_version"),
        )
        .join(SimWriter, SimWriterCycleScore.writer_id == SimWriter.id)
        .outerjoin(
            SimPromptVersion,
            SimWriterCycleScore.prompt_version_id == SimPromptVersion.id,
        )
        .where(SimWriterCycleScore.cycle_id == cycle_uuid)
        .order_by(desc(SimWriterCycleScore.engagement_score))
    ).all()

    leaderboard = []
    for rank, row in enumerate(rows, start=1):
        score_obj = row.SimWriterCycleScore
        leaderboard.append(
            {
                "rank": rank,
                "writer_name": row.writer_name,
                "prompt_version": row.prompt_version,
                "engagement_score": round(score_obj.engagement_score, 4),
                "repost_count": score_obj.repost_count,
                "like_count": score_obj.like_count,
                "comment_count": score_obj.comment_count,
                "skip_count": score_obj.skip_count,
                "tweet_count": score_obj.tweet_count,
                "readers_sampled": score_obj.reader_sample_count,
            }
        )
    return leaderboard


def _mutations_for_cycle(
    session: Session, cycle_number: int
) -> list[dict[str, Any]]:
    """Return prompt versions introduced during this cycle (mutations only)."""
    if cycle_number == 0:
        return []
    rows = session.execute(
        select(
            SimPromptVersion.version_number,
            SimPromptVersion.triggered_by_score,
            SimWriter.name.label("writer_name"),
        )
        .join(SimWriter, SimPromptVersion.writer_id == SimWriter.id)
        .where(SimPromptVersion.cycle_introduced == cycle_number)
        .where(SimPromptVersion.version_number > 1)
    ).all()
    return [
        {
            "writer_name": row.writer_name,
            "new_version": row.version_number,
            "triggered_by_score": (
                round(row.triggered_by_score, 4)
                if row.triggered_by_score is not None
                else None
            ),
        }
        for row in rows
    ]


def _serialize_cycle(cycle: SimCycle) -> dict[str, Any]:
    return {
        "cycle_number": cycle.cycle_number,
        "week_number": cycle.week_number,
        "started_at": cycle.started_at.isoformat() if cycle.started_at else None,
        "completed_at": (
            cycle.completed_at.isoformat() if cycle.completed_at else None
        ),
        "story_count": len(cycle.story_ids) if cycle.story_ids else 0,
        "mlflow_run_id": cycle.mlflow_run_id,
    }


def _serialize_tweet_row(row: Any) -> dict[str, Any]:
    readers = row.repost_count + row.like_count + row.comment_count + row.skip_count
    weighted = row.repost_count * 3 + row.comment_count * 2 + row.like_count * 1
    score = weighted / (readers * 3) if readers > 0 else 0.0
    return {
        "tweet_id": str(row.tweet_id),
        "writer_name": row.writer_name,
        "prompt_version": row.prompt_version,
        "content": row.content,
        "repost_count": row.repost_count,
        "like_count": row.like_count,
        "comment_count": row.comment_count,
        "skip_count": row.skip_count,
        "readers_sampled": readers,
        "engagement_score": round(score, 4),
    }
