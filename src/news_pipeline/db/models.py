from __future__ import annotations

import enum
import json
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import CHAR, TypeDecorator


class GUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value if isinstance(value, uuid.UUID) else uuid.UUID(str(value)))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class VectorType(TypeDecorator):
    """Cross-dialect vector column.

    On PostgreSQL: delegates to pgvector's Vector type for native storage and
    indexed cosine-distance queries via the ``<=>`` operator.
    On SQLite (tests): stores as JSON text, exactly like the previous Text column.
    """

    impl = Text
    cache_ok = True

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            # pgvector accepts a plain Python list directly
            if hasattr(value, "tolist"):
                return value.tolist()
            return list(value)
        # SQLite: store as JSON text
        if hasattr(value, "tolist"):
            return json.dumps(value.tolist())
        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            # pgvector returns a list already; return as-is for callers to wrap
            return value
        # SQLite: deserialise from JSON text
        return json.loads(value)


Base = declarative_base()


class ProcessingStatus(str, enum.Enum):
    pending_extraction = "pending_extraction"
    extracted = "extracted"
    failed = "failed"


class EntityType(str, enum.Enum):
    person = "person"
    company = "company"
    organization = "organization"
    location = "location"
    product = "product"


class TopicMethod(str, enum.Enum):
    llm = "llm"


class ExtractionRunType(str, enum.Enum):
    entity = "entity"
    topic = "topic"


class RawArticle(Base):
    __tablename__ = "raw_articles"
    __table_args__ = (
        Index("ix_raw_articles_url", "url"),
        Index("ix_raw_articles_published_at", "published_at"),
        Index("ix_raw_articles_processing_status", "processing_status"),
        Index("ix_raw_articles_normalized_title", "normalized_title"),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    url = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    normalized_title = Column(Text, nullable=False)
    summary = Column(Text, nullable=False, default="")
    full_text = Column(Text)
    cleaned_text = Column(Text)
    source_name = Column(String(255), nullable=False)
    source_feed = Column(String(255), nullable=False)
    published_at = Column(DateTime(timezone=True))
    ingested_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    category = Column(String(100), nullable=False)
    word_count = Column(Integer)
    processing_status = Column(
        Enum(ProcessingStatus, name="processing_status"),
        default=ProcessingStatus.pending_extraction,
        nullable=False,
    )
    duplicate_of = Column(GUID(), ForeignKey("raw_articles.id", ondelete="SET NULL"))
    embedding = Column(VectorType(dim=384), nullable=True)
    semantic_cluster_id = Column(Integer, nullable=True)
    cluster_label = Column(String(200), nullable=True)

    entities = relationship("ArticleEntity", back_populates="article", cascade="all, delete-orphan")
    topics = relationship("ArticleTopic", back_populates="article", cascade="all, delete-orphan")
    extraction_runs = relationship("ExtractionRun", back_populates="article", cascade="all, delete-orphan")


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("normalized_name", "entity_type", name="uq_entity_normalized_name_type"),
        Index("ix_entities_normalized_name", "normalized_name"),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    entity_type = Column(Enum(EntityType, name="entity_type"), nullable=False)
    normalized_name = Column(Text, nullable=False)
    first_seen_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    article_count = Column(Integer, default=1, nullable=False)

    articles = relationship("ArticleEntity", back_populates="entity", cascade="all, delete-orphan")


class ArticleEntity(Base):
    __tablename__ = "article_entities"
    __table_args__ = (
        UniqueConstraint("article_id", "entity_id", name="uq_article_entity"),
    )

    article_id = Column(GUID(), ForeignKey("raw_articles.id", ondelete="CASCADE"), primary_key=True)
    entity_id = Column(GUID(), ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True)
    role = Column(String(100))
    confidence = Column(Float, nullable=False)
    extracted_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    article = relationship("RawArticle", back_populates="entities")
    entity = relationship("Entity", back_populates="articles")


class Topic(Base):
    __tablename__ = "topics"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)

    articles = relationship("ArticleTopic", back_populates="topic", cascade="all, delete-orphan")


class ArticleTopic(Base):
    __tablename__ = "article_topics"
    __table_args__ = (
        UniqueConstraint("article_id", "topic_id", name="uq_article_topic"),
        Index("ix_article_topics_topic_id", "topic_id"),
    )

    article_id = Column(GUID(), ForeignKey("raw_articles.id", ondelete="CASCADE"), primary_key=True)
    topic_id = Column(GUID(), ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True)
    confidence = Column(Float, nullable=False)
    method = Column(Enum(TopicMethod, name="topic_method"), default=TopicMethod.llm, nullable=False)

    article = relationship("RawArticle", back_populates="topics")
    topic = relationship("Topic", back_populates="articles")


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    article_id = Column(GUID(), ForeignKey("raw_articles.id", ondelete="CASCADE"), nullable=False)
    run_type = Column(Enum(ExtractionRunType, name="extraction_run_type"), nullable=False)
    llm_provider = Column(String(100), nullable=False)
    model_name = Column(String(255), nullable=False)
    prompt_version = Column(String(100), nullable=False)
    tokens_used = Column(Integer, default=0, nullable=False)
    latency_ms = Column(Integer, default=0, nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    article = relationship("RawArticle", back_populates="extraction_runs")


class Signal(Base):
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signals_detected_at", "detected_at"),
        Index("ix_signals_entity_id", "entity_id"),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    entity_id = Column(GUID(), ForeignKey("entities.id", ondelete="SET NULL"), nullable=True)
    topic_name = Column(String(200), nullable=True)
    signal_type = Column(String(50), nullable=False)
    score = Column(Float, nullable=False)
    summary = Column(Text, nullable=True)
    detected_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    article_ids = Column(JSONB, nullable=False, default=list)

    entity = relationship("Entity", foreign_keys=[entity_id])


class LLMRateLimitReservation(Base):
    __tablename__ = "llm_rate_limit_reservations"
    __table_args__ = (
        Index(
            "ix_llm_rate_limit_reservations_provider_reserved_at",
            "provider_name",
            "reserved_at",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_name = Column(String(100), nullable=False)
    reserved_at = Column(DateTime(timezone=True), nullable=False)
