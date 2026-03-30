from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from news_pipeline.api.app import app
from news_pipeline.db.models import (
    ArticleEntity,
    ArticleTopic,
    Base,
    Entity,
    EntityType,
    ProcessingStatus,
    RawArticle,
    Topic,
    TopicMethod,
)
from news_pipeline.db.session import get_session
from news_pipeline.utils import normalize_entity_name, normalize_title_for_dedup, utcnow


@pytest.fixture
def session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def seeded_session(session: Session) -> Session:
    article = RawArticle(
        title="Acme launches orbital product",
        normalized_title=normalize_title_for_dedup("Acme launches orbital product"),
        summary="Acme announced a new orbital platform.",
        full_text="Acme announced a new orbital platform in Lima.",
        cleaned_text="Acme announced a new orbital platform in Lima.",
        url="https://example.com/articles/acme-orbit",
        source_name="Example News",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        processing_status=ProcessingStatus.extracted,
        word_count=9,
    )
    entity = Entity(
        name="Acme",
        entity_type=EntityType.company,
        normalized_name=normalize_entity_name("Acme"),
        article_count=1,
    )
    topic = Topic(name="technology")
    session.add_all([article, entity, topic])
    session.flush()
    session.add(ArticleEntity(article_id=article.id, entity_id=entity.id, role="subject", confidence=0.97))
    session.add(ArticleTopic(article_id=article.id, topic_id=topic.id, confidence=0.93, method=TopicMethod.llm))
    session.commit()
    return session


@pytest.fixture
def client(seeded_session: Session) -> Generator[TestClient, None, None]:
    def override_get_session() -> Generator[Session, None, None]:
        yield seeded_session

    app.dependency_overrides[get_session] = override_get_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
