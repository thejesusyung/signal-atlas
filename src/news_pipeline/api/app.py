from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from news_pipeline.config import get_settings
from news_pipeline.db.models import Entity, EntityType, RawArticle
from news_pipeline.db.session import get_session
from news_pipeline.db.models import Signal
from news_pipeline.services.article_service import (
    get_article,
    get_articles_for_entity,
    get_graph_data,
    get_similar_articles,
    list_articles,
    list_entities,
    list_topics,
    pipeline_stats,
)
from news_pipeline.services.signal_service import get_latest_signals

_STATIC = Path(__file__).parent / "static"

settings = get_settings()
app = FastAPI(title="News Intelligence Pipeline API", version="0.1.0")


@app.get("/brief")
def get_brief(session: Session = Depends(get_session)) -> dict[str, Any]:
    signals = get_latest_signals(session, limit=10)
    return {"signals": [_serialize_signal(s) for s in signals], "count": len(signals)}


@app.get("/similar/{article_id}")
def get_similar(
    article_id: UUID,
    limit: int = Query(default=10, ge=1, le=50),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    pairs = get_similar_articles(session, article_id, limit=limit)
    return {
        "article_id": str(article_id),
        "similar": [
            {**_serialize_article_summary(article), "distance": round(distance, 4)}
            for article, distance in pairs
        ],
    }


@app.get("/", include_in_schema=False)
def serve_map() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/graph")
def get_graph(
    min_articles: int = Query(default=1, ge=1),
    max_entities: int = Query(default=80, ge=1, le=300),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    return get_graph_data(session, min_articles=min_articles, max_entities=max_entities)


@app.get("/articles")
def get_articles(
    q: str | None = None,
    source: str | None = None,
    topic: str | None = None,
    entity: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=settings.api_page_size, ge=1, le=settings.api_max_page_size),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    offset = (page - 1) * page_size
    items, total = list_articles(
        session,
        q=q,
        source=source,
        topic=topic,
        entity=entity,
        date_from=date_from,
        date_to=date_to,
        offset=offset,
        limit=page_size,
    )
    return {"items": [_serialize_article_summary(item) for item in items], "total": total, "page": page}


@app.get("/articles/{article_id}")
def get_article_detail(article_id: UUID, session: Session = Depends(get_session)) -> dict[str, Any]:
    article = get_article(session, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return _serialize_article_detail(article)


@app.get("/entities")
def get_entities(
    entity_type: EntityType | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=settings.api_page_size, ge=1, le=settings.api_max_page_size),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    offset = (page - 1) * page_size
    items, total = list_entities(session, entity_type=entity_type, offset=offset, limit=page_size)
    return {"items": [_serialize_entity(item) for item in items], "total": total, "page": page}


@app.get("/entities/{entity_id}/articles")
def get_entity_articles(
    entity_id: UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=settings.api_page_size, ge=1, le=settings.api_max_page_size),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    entity = session.get(Entity, entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    offset = (page - 1) * page_size
    items, total = get_articles_for_entity(session, entity_id=entity_id, offset=offset, limit=page_size)
    return {
        "entity": _serialize_entity(entity),
        "items": [_serialize_article_summary(item) for item in items],
        "total": total,
        "page": page,
    }


@app.get("/topics")
def get_topics(session: Session = Depends(get_session)) -> dict[str, Any]:
    items = [{"name": name, "article_count": count} for name, count in list_topics(session)]
    return {"items": items, "total": len(items)}


@app.get("/stats")
def get_stats(session: Session = Depends(get_session)) -> dict[str, int]:
    return pipeline_stats(session)


def _serialize_article_summary(article: RawArticle) -> dict[str, Any]:
    return {
        "id": str(article.id),
        "title": article.title,
        "summary": article.summary,
        "url": article.url,
        "source_name": article.source_name,
        "category": article.category,
        "published_at": article.published_at,
        "word_count": article.word_count,
        "processing_status": article.processing_status.value,
    }


def _serialize_article_detail(article: RawArticle) -> dict[str, Any]:
    return {
        **_serialize_article_summary(article),
        "full_text": article.full_text,
        "cleaned_text": article.cleaned_text,
        "entities": [
            {
                "id": str(link.entity.id),
                "name": link.entity.name,
                "entity_type": link.entity.entity_type.value,
                "role": link.role,
                "confidence": link.confidence,
            }
            for link in article.entities
        ],
        "topics": [
            {"name": link.topic.name, "confidence": link.confidence, "method": link.method.value}
            for link in article.topics
        ],
    }


def _serialize_entity(entity: Entity) -> dict[str, Any]:
    return {
        "id": str(entity.id),
        "name": entity.name,
        "entity_type": entity.entity_type.value,
        "normalized_name": entity.normalized_name,
        "article_count": entity.article_count,
    }


def _serialize_signal(sig: Signal) -> dict[str, Any]:
    return {
        "id": str(sig.id),
        "signal_type": sig.signal_type,
        "entity_id": str(sig.entity_id) if sig.entity_id else None,
        "entity_name": sig.entity.name if sig.entity_id and sig.entity else None,
        "topic_name": sig.topic_name,
        "score": sig.score,
        "summary": sig.summary,
        "detected_at": sig.detected_at.isoformat() if sig.detected_at else None,
        "article_ids": sig.article_ids,
    }
