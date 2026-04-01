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


def get_graph_data(session: Session, min_articles: int = 1, max_entities: int = 80) -> dict:
    topics_q = (
        select(
            Topic.id, Topic.name,
            func.count(ArticleTopic.article_id).label("article_count"),
            func.max(RawArticle.published_at).label("latest_at"),
        )
        .join(ArticleTopic, ArticleTopic.topic_id == Topic.id)
        .join(RawArticle, RawArticle.id == ArticleTopic.article_id)
        .group_by(Topic.id, Topic.name)
    )
    topics = session.execute(topics_q).all()
    topic_id_to_name = {str(t.id): t.name for t in topics}

    entities_q = (
        select(
            Entity.id, Entity.name, Entity.entity_type, Entity.article_count,
            func.max(RawArticle.published_at).label("latest_at"),
        )
        .join(ArticleEntity, ArticleEntity.entity_id == Entity.id)
        .join(RawArticle, RawArticle.id == ArticleEntity.article_id)
        .where(Entity.article_count >= min_articles)
        .group_by(Entity.id, Entity.name, Entity.entity_type, Entity.article_count)
        .order_by(Entity.article_count.desc())
        .limit(max_entities)
    )
    entities = session.execute(entities_q).all()
    entity_ids = {str(e.id) for e in entities}

    links_q = (
        select(
            ArticleEntity.entity_id,
            ArticleTopic.topic_id,
            func.count(ArticleEntity.article_id).label("weight"),
            func.max(RawArticle.published_at).label("latest_at"),
        )
        .join(ArticleTopic, ArticleTopic.article_id == ArticleEntity.article_id)
        .join(RawArticle, RawArticle.id == ArticleEntity.article_id)
        .group_by(ArticleEntity.entity_id, ArticleTopic.topic_id)
    )
    raw_links = session.execute(links_q).all()

    nodes = [
        {
            "id": f"topic:{t.name}", "label": t.name, "group": "topic",
            "article_count": t.article_count,
            "latest_at": t.latest_at.isoformat() if t.latest_at else None,
        }
        for t in topics
    ] + [
        {
            "id": f"entity:{e.id}", "label": e.name, "group": e.entity_type.value,
            "article_count": e.article_count,
            "latest_at": e.latest_at.isoformat() if e.latest_at else None,
        }
        for e in entities
    ]
    links = [
        {
            "source": f"entity:{lnk.entity_id}",
            "target": f"topic:{topic_id_to_name[str(lnk.topic_id)]}",
            "value": lnk.weight,
            "latest_at": lnk.latest_at.isoformat() if lnk.latest_at else None,
        }
        for lnk in raw_links
        if str(lnk.entity_id) in entity_ids and str(lnk.topic_id) in topic_id_to_name
    ]
    # Semantic cluster nodes (additive layer)
    clusters_q = (
        select(
            RawArticle.semantic_cluster_id,
            RawArticle.cluster_label,
            func.count(RawArticle.id).label("article_count"),
            func.max(RawArticle.published_at).label("latest_at"),
        )
        .where(RawArticle.semantic_cluster_id.isnot(None))
        .group_by(RawArticle.semantic_cluster_id, RawArticle.cluster_label)
        .order_by(func.count(RawArticle.id).desc())
    )
    clusters = session.execute(clusters_q).all()
    cluster_nodes = [
        {
            "id": f"cluster:{c.semantic_cluster_id}",
            "label": c.cluster_label or f"Cluster {c.semantic_cluster_id}",
            "group": "cluster",
            "article_count": c.article_count,
            "latest_at": c.latest_at.isoformat() if c.latest_at else None,
        }
        for c in clusters
    ]

    stats = pipeline_stats(session)
    return {
        "nodes": nodes,
        "links": links,
        "cluster_nodes": cluster_nodes,
        "meta": {"total_articles": stats["total_articles"], "extracted_articles": stats["extracted_articles"]},
    }


def get_similar_articles(
    session: Session,
    article_id: UUID,
    limit: int = 10,
) -> list[tuple[RawArticle, float]]:
    """Find nearest neighbours by cosine similarity.

    On PostgreSQL uses a pgvector indexed ``<=>`` query (fast, scalable).
    On SQLite (tests) falls back to an in-memory numpy dot-product scan.
    """
    anchor = session.get(RawArticle, article_id)
    if anchor is None or anchor.embedding is None:
        return []

    dialect = session.bind.dialect.name
    if dialect == "postgresql":
        return _pgvector_similar(session, anchor, article_id, limit)
    return _numpy_similar(session, anchor, article_id, limit)


def _pgvector_similar(
    session: Session,
    anchor: RawArticle,
    article_id: UUID,
    limit: int,
) -> list[tuple[RawArticle, float]]:
    from sqlalchemy import text as sa_text

    # anchor.embedding is a list[float] coming from VectorType
    anchor_vec = anchor.embedding if isinstance(anchor.embedding, list) else list(anchor.embedding)
    vec_str = "[" + ",".join(str(v) for v in anchor_vec) + "]"

    rows = session.execute(
        sa_text(
            "SELECT id, (embedding <=> :vec)::float AS distance"
            " FROM raw_articles"
            " WHERE id != :aid AND embedding IS NOT NULL"
            " ORDER BY embedding <=> :vec"
            " LIMIT :lim"
        ),
        {"vec": vec_str, "aid": str(article_id), "lim": limit},
    ).all()

    if not rows:
        return []

    id_to_distance = {row.id: float(row.distance) for row in rows}
    articles = session.scalars(
        select(RawArticle).where(RawArticle.id.in_(list(id_to_distance)))
    ).all()
    articles_sorted = sorted(articles, key=lambda a: id_to_distance.get(a.id, 999))
    return [(a, id_to_distance[a.id]) for a in articles_sorted]


def _numpy_similar(
    session: Session,
    anchor: RawArticle,
    article_id: UUID,
    limit: int,
) -> list[tuple[RawArticle, float]]:
    import numpy as np
    from news_pipeline.embeddings.encoder import vector_from_db

    anchor_vec = vector_from_db(anchor.embedding)

    rows = session.scalars(
        select(RawArticle).where(
            RawArticle.id != article_id,
            RawArticle.embedding.isnot(None),
        )
    ).all()

    if not rows:
        return []

    matrix = np.array([vector_from_db(r.embedding) for r in rows], dtype=np.float32)
    sims = matrix @ anchor_vec
    top_idx = np.argsort(sims)[::-1][:limit]
    return [(rows[i], float(1.0 - sims[i])) for i in top_idx]


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
