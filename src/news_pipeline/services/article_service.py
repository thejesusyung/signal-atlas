from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from news_pipeline.contracts import ArticleCandidate, ScrapeResult
from news_pipeline.db.models import (
    ArticleEntity,
    ArticleTopic,
    Entity,
    EntityType,
    ExtractionRun,
    ProcessingStatus,
    RawArticle,
    Topic,
)
from news_pipeline.utils import clean_article_text, normalize_entity_name, normalize_title_for_dedup, utcnow


def insert_article(
    session: Session,
    candidate: ArticleCandidate,
    scrape_result: ScrapeResult,
    duplicate_of: UUID | None = None,
) -> RawArticle:
    article = RawArticle(
        url=candidate.url,
        title=candidate.title,
        normalized_title=normalize_title_for_dedup(candidate.title),
        summary=candidate.summary,
        full_text=scrape_result.full_text or None,
        cleaned_text=clean_article_text(scrape_result.full_text) or None,
        source_name=candidate.source_name,
        source_feed=candidate.source_feed,
        published_at=candidate.published_at,
        category=candidate.category,
        word_count=scrape_result.word_count or None,
        duplicate_of=duplicate_of,
    )
    session.add(article)
    session.flush()
    return article


def get_recent_articles(session: Session, hours: int) -> list[RawArticle]:
    cutoff = utcnow() - timedelta(hours=hours)
    return session.scalars(select(RawArticle).where(RawArticle.ingested_at >= cutoff)).all()


def get_pending_articles(session: Session, limit: int) -> list[RawArticle]:
    statement = (
        select(RawArticle)
        .where(RawArticle.processing_status == ProcessingStatus.pending_extraction)
        .order_by(RawArticle.ingested_at.asc())
        .limit(limit)
    )
    return session.scalars(statement).all()


def get_article(session: Session, article_id: UUID) -> RawArticle | None:
    return session.get(RawArticle, article_id)


def list_articles(
    session: Session,
    q: str | None = None,
    source: str | None = None,
    topic: str | None = None,
    entity: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[RawArticle], int]:
    statement = select(RawArticle).distinct()
    count_statement = select(func.count(func.distinct(RawArticle.id)))

    if topic:
        statement = statement.join(ArticleTopic).join(Topic)
        count_statement = count_statement.join(ArticleTopic).join(Topic)
        topic_filter = Topic.name == topic
        statement = statement.where(topic_filter)
        count_statement = count_statement.where(topic_filter)

    if entity:
        statement = statement.join(ArticleEntity).join(Entity)
        count_statement = count_statement.join(ArticleEntity).join(Entity)
        entity_filter = Entity.normalized_name == normalize_entity_name(entity)
        statement = statement.where(entity_filter)
        count_statement = count_statement.where(entity_filter)

    filters = _build_article_filters(q=q, source=source, date_from=date_from, date_to=date_to)
    if filters:
        statement = statement.where(and_(*filters))
        count_statement = count_statement.where(and_(*filters))

    statement = statement.order_by(RawArticle.published_at.desc().nullslast(), RawArticle.ingested_at.desc())
    items = session.scalars(statement.offset(offset).limit(limit)).all()
    total = session.scalar(count_statement) or 0
    return items, total


def list_entities(
    session: Session,
    entity_type: EntityType | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[Entity], int]:
    statement = select(Entity)
    count_statement = select(func.count(Entity.id))
    if entity_type:
        statement = statement.where(Entity.entity_type == entity_type)
        count_statement = count_statement.where(Entity.entity_type == entity_type)
    statement = statement.order_by(Entity.article_count.desc(), Entity.name.asc())
    items = session.scalars(statement.offset(offset).limit(limit)).all()
    total = session.scalar(count_statement) or 0
    return items, total


def get_articles_for_entity(
    session: Session, entity_id: UUID, offset: int = 0, limit: int = 20
) -> tuple[list[RawArticle], int]:
    statement = (
        select(RawArticle)
        .join(ArticleEntity)
        .where(ArticleEntity.entity_id == entity_id)
        .order_by(RawArticle.published_at.desc().nullslast())
    )
    count_statement = (
        select(func.count(RawArticle.id))
        .join(ArticleEntity)
        .where(ArticleEntity.entity_id == entity_id)
    )
    items = session.scalars(statement.offset(offset).limit(limit)).all()
    total = session.scalar(count_statement) or 0
    return items, total


def list_topics(session: Session) -> list[tuple[str, int]]:
    statement = (
        select(Topic.name, func.count(ArticleTopic.article_id))
        .join(ArticleTopic, ArticleTopic.topic_id == Topic.id)
        .group_by(Topic.name)
        .order_by(func.count(ArticleTopic.article_id).desc(), Topic.name.asc())
    )
    return list(session.execute(statement).all())


def pipeline_stats(session: Session) -> dict[str, int]:
    return {
        "total_articles": session.scalar(select(func.count(RawArticle.id))) or 0,
        "extracted_articles": session.scalar(
            select(func.count(RawArticle.id)).where(RawArticle.processing_status == ProcessingStatus.extracted)
        )
        or 0,
        "total_entities": session.scalar(select(func.count(Entity.id))) or 0,
        "topic_assignments": session.scalar(select(func.count(ArticleTopic.article_id))) or 0,
        "extraction_runs": session.scalar(select(func.count(ExtractionRun.id))) or 0,
    }


def _build_article_filters(
    q: str | None,
    source: str | None,
    date_from: date | None,
    date_to: date | None,
) -> Sequence:
    filters = []
    if q:
        pattern = f"%{q}%"
        filters.append(
            or_(
                RawArticle.title.ilike(pattern),
                RawArticle.summary.ilike(pattern),
                RawArticle.cleaned_text.ilike(pattern),
                RawArticle.full_text.ilike(pattern),
            )
        )
    if source:
        filters.append(RawArticle.source_name == source)
    if date_from:
        filters.append(RawArticle.published_at >= datetime.combine(date_from, time.min))
    if date_to:
        filters.append(RawArticle.published_at <= datetime.combine(date_to, time.max))
    return filters
