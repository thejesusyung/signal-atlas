from __future__ import annotations

import json
import logging
import math
from datetime import timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from news_pipeline.config import get_settings
from news_pipeline.db.models import ArticleEntity, ArticleTopic, Entity, RawArticle, Signal, Topic

from news_pipeline.llm.provider import LLMProvider, LLMTraceContext
from news_pipeline.llm.prompts import PromptSpec
from news_pipeline.utils import utcnow

LOGGER = logging.getLogger(__name__)

SIGNAL_BRIEF_PROMPT = PromptSpec(
    name="signal_brief",
    version="brief_v1",
    system_prompt=(
        "You are a news intelligence analyst. "
        "Write concise, factual intelligence briefs based strictly on the provided headlines. "
        "Never speculate beyond what the headlines state. "
        "Return JSON only."
    ),
    user_prompt_template=(
        """
Generate a 2-3 sentence intelligence brief for this emerging signal.

Signal type: {{ signal_type }}
Subject: {{ subject_name }}
Anomaly score: {{ "%.2f"|format(score) }}

Supporting headlines:
{% for title in titles %}- {{ title }}
{% endfor %}

Return JSON: {"summary": "2-3 sentence brief here."}
        """.strip()
    ),
)

# Minimum baseline-window rows required before the corresponding signal type
# is attempted.  Entity and topic thresholds are checked independently so that
# a dataset with rich topic history but sparse entity data (or vice-versa)
# still produces signals for the well-covered signal type.
_MIN_ENTITY_BASELINE_ROWS = 10
_MIN_TOPIC_BASELINE_ROWS = 10


def detect_and_persist_signals(session: Session, provider: LLMProvider) -> list[Signal]:
    settings = get_settings()
    now = utcnow()
    current_cutoff = now - timedelta(hours=settings.signal_current_window_hours)
    baseline_cutoff = now - timedelta(hours=settings.signal_baseline_window_hours)

    baseline_entity_row_count = session.scalar(
        select(func.count(ArticleEntity.article_id)).where(
            ArticleEntity.extracted_at >= baseline_cutoff,
            ArticleEntity.extracted_at < current_cutoff,
        )
    ) or 0

    baseline_topic_row_count = session.scalar(
        select(func.count(ArticleTopic.article_id))
        .join(RawArticle, RawArticle.id == ArticleTopic.article_id)
        .where(
            RawArticle.ingested_at >= baseline_cutoff,
            RawArticle.ingested_at < current_cutoff,
        )
    ) or 0

    run_entity = baseline_entity_row_count >= _MIN_ENTITY_BASELINE_ROWS
    run_topic = baseline_topic_row_count >= _MIN_TOPIC_BASELINE_ROWS

    if not run_entity:
        LOGGER.info(
            "Skipping entity signals: only %d entity baseline rows in %d-%d h window (need %d)",
            baseline_entity_row_count,
            settings.signal_current_window_hours,
            settings.signal_baseline_window_hours,
            _MIN_ENTITY_BASELINE_ROWS,
        )
    if not run_topic:
        LOGGER.info(
            "Skipping topic signals: only %d topic baseline rows in %d-%d h window (need %d)",
            baseline_topic_row_count,
            settings.signal_current_window_hours,
            settings.signal_baseline_window_hours,
            _MIN_TOPIC_BASELINE_ROWS,
        )

    if not run_entity and not run_topic:
        return []

    candidates = _score_candidates(
        session, now, current_cutoff, baseline_cutoff, settings,
        run_entity=run_entity, run_topic=run_topic,
    )
    top = candidates[: settings.signal_top_n]

    signals = []
    for candidate in top:
        summary = _generate_summary(provider, candidate, settings)
        sig = Signal(
            entity_id=candidate.get("entity_id"),
            topic_name=candidate.get("topic_name"),
            signal_type=candidate["signal_type"],
            score=round(candidate["score"], 4),
            summary=summary,
            article_ids=candidate["article_ids"],
        )
        session.add(sig)
        signals.append(sig)

    if signals:
        session.flush()
    return signals


def _score_candidates(
    session,
    now,
    current_cutoff,
    baseline_cutoff,
    settings,
    *,
    run_entity: bool = True,
    run_topic: bool = True,
) -> list[dict]:
    """Score per-entity and per-topic velocity in the current window vs the baseline window.

    Both windows cover the same number of hours so counts are directly comparable
    without normalisation.  Z-scores use a Poisson-variance approximation
    (``stdev ≈ sqrt(baseline_count)``) which is appropriate for count data with
    no reliable historical stdev estimate.
    """
    candidates: list[dict] = []
    window_hours = settings.signal_current_window_hours

    # Normalise baseline to the same window length as current
    baseline_window_hours = settings.signal_baseline_window_hours
    normalisation = window_hours / max(baseline_window_hours - window_hours, 1)

    # ------------------------------------------------------------------ #
    # Entity velocity                                                       #
    # ------------------------------------------------------------------ #
    if run_entity:
        current_entity_counts: dict[str, int] = {
            str(row.entity_id): row.cnt
            for row in session.execute(
                select(ArticleEntity.entity_id, func.count(ArticleEntity.article_id).label("cnt"))
                .where(ArticleEntity.extracted_at >= current_cutoff)
                .group_by(ArticleEntity.entity_id)
            ).all()
        }

        baseline_entity_counts: dict[str, int] = {
            str(row.entity_id): row.cnt
            for row in session.execute(
                select(ArticleEntity.entity_id, func.count(ArticleEntity.article_id).label("cnt"))
                .where(
                    ArticleEntity.extracted_at >= baseline_cutoff,
                    ArticleEntity.extracted_at < current_cutoff,
                )
                .group_by(ArticleEntity.entity_id)
            ).all()
        }

        for entity_id_str, current_cnt in current_entity_counts.items():
            raw_baseline = baseline_entity_counts.get(entity_id_str, 0)
            baseline_rate = raw_baseline * normalisation

            z = _poisson_zscore(current_cnt, baseline_rate)
            if z < settings.signal_zscore_threshold:
                continue

            entity_id = UUID(entity_id_str)
            entity = session.get(Entity, entity_id)
            if entity is None:
                continue

            article_ids = _get_entity_article_ids(session, entity_id, since=current_cutoff)
            candidates.append({
                "signal_type": "entity_velocity",
                "entity_id": entity_id,
                "topic_name": None,
                "subject_name": entity.name,
                "score": z,
                "article_ids": article_ids,
                "titles": _get_article_titles(session, article_ids),
            })

    # ------------------------------------------------------------------ #
    # Topic velocity                                                        #
    # ------------------------------------------------------------------ #
    if run_topic:
        current_topic_counts: dict[str, int] = {
            row.name: row.cnt
            for row in session.execute(
                select(Topic.name, func.count(ArticleTopic.article_id).label("cnt"))
                .join(ArticleTopic, ArticleTopic.topic_id == Topic.id)
                .join(RawArticle, RawArticle.id == ArticleTopic.article_id)
                .where(RawArticle.ingested_at >= current_cutoff)
                .group_by(Topic.name)
            ).all()
        }

        baseline_topic_counts: dict[str, int] = {
            row.name: row.cnt
            for row in session.execute(
                select(Topic.name, func.count(ArticleTopic.article_id).label("cnt"))
                .join(ArticleTopic, ArticleTopic.topic_id == Topic.id)
                .join(RawArticle, RawArticle.id == ArticleTopic.article_id)
                .where(
                    RawArticle.ingested_at >= baseline_cutoff,
                    RawArticle.ingested_at < current_cutoff,
                )
                .group_by(Topic.name)
            ).all()
        }

        for topic_name, current_cnt in current_topic_counts.items():
            raw_baseline = baseline_topic_counts.get(topic_name, 0)
            baseline_rate = raw_baseline * normalisation

            z = _poisson_zscore(current_cnt, baseline_rate)
            if z < settings.signal_zscore_threshold:
                continue

            article_ids = _get_topic_article_ids(session, topic_name, since=current_cutoff)
            candidates.append({
                "signal_type": "topic_velocity",
                "entity_id": None,
                "topic_name": topic_name,
                "subject_name": topic_name,
                "score": z,
                "article_ids": article_ids,
                "titles": _get_article_titles(session, article_ids),
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _poisson_zscore(current: int, baseline_rate: float) -> float:
    """Z-score using Poisson variance approximation: stdev ≈ sqrt(baseline_rate).

    When there is no baseline (new entity/topic), any mention in the current
    window is treated as a spike; score is capped at a meaningful constant.
    """
    if baseline_rate <= 0:
        # No history: score proportionally to current count, capped
        return min(float(current), 5.0) if current >= 3 else 0.0
    return (current - baseline_rate) / math.sqrt(baseline_rate)


def _get_entity_article_ids(session: Session, entity_id: UUID, since) -> list[str]:
    rows = session.execute(
        select(ArticleEntity.article_id)
        .where(
            ArticleEntity.entity_id == entity_id,
            ArticleEntity.extracted_at >= since,
        )
        .limit(8)
    ).all()
    return [str(r.article_id) for r in rows]


def _get_topic_article_ids(session: Session, topic_name: str, since) -> list[str]:
    rows = session.execute(
        select(ArticleTopic.article_id)
        .join(Topic, Topic.id == ArticleTopic.topic_id)
        .join(RawArticle, RawArticle.id == ArticleTopic.article_id)
        .where(
            Topic.name == topic_name,
            RawArticle.ingested_at >= since,
        )
        .limit(8)
    ).all()
    return [str(r.article_id) for r in rows]


def _get_article_titles(session: Session, article_ids: list[str]) -> list[str]:
    if not article_ids:
        return []
    rows = session.execute(
        select(RawArticle.title)
        .where(RawArticle.id.in_([UUID(aid) for aid in article_ids]))
        .limit(6)
    ).all()
    return [r.title for r in rows]


def _generate_summary(provider: LLMProvider, candidate: dict, settings) -> str:
    try:
        from news_pipeline.tracking.prompt_registry import get_prompt_template
        prompt = PromptSpec(
            name=SIGNAL_BRIEF_PROMPT.name,
            version=SIGNAL_BRIEF_PROMPT.version,
            system_prompt=SIGNAL_BRIEF_PROMPT.system_prompt,
            user_prompt_template=get_prompt_template("signal_brief", SIGNAL_BRIEF_PROMPT),
        )
        system_prompt, user_prompt = prompt.render(
            signal_type=candidate["signal_type"].replace("_", " "),
            subject_name=candidate["subject_name"],
            score=candidate["score"],
            titles=candidate.get("titles", []),
        )
        trace = LLMTraceContext(
            operation="signal_brief",
            article_id=None,
            prompt_version=prompt.version,
        )
        response = provider.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            trace_context=trace,
        )
        payload = json.loads(response.text)
        return payload.get("summary", "")
    except Exception as exc:
        LOGGER.warning("Signal brief generation failed for %s: %s", candidate["subject_name"], exc)
        return ""
