"""Initial v1 schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260324_01"
down_revision = None
branch_labels = None
depends_on = None


processing_status = postgresql.ENUM(
    "pending_extraction", "extracted", "failed", name="processing_status", create_type=False
)
entity_type = postgresql.ENUM(
    "person", "company", "organization", "location", "product", name="entity_type", create_type=False
)
topic_method = postgresql.ENUM("llm", name="topic_method", create_type=False)
extraction_run_type = postgresql.ENUM("entity", "topic", name="extraction_run_type", create_type=False)


def upgrade() -> None:
    processing_status.create(op.get_bind(), checkfirst=True)
    entity_type.create(op.get_bind(), checkfirst=True)
    topic_method.create(op.get_bind(), checkfirst=True)
    extraction_run_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "raw_articles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("normalized_title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("full_text", sa.Text(), nullable=True),
        sa.Column("source_name", sa.String(length=255), nullable=False),
        sa.Column("source_feed", sa.String(length=255), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("processing_status", processing_status, nullable=False),
        sa.Column("duplicate_of", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["duplicate_of"], ["raw_articles.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url"),
    )
    op.create_index("ix_raw_articles_url", "raw_articles", ["url"])
    op.create_index("ix_raw_articles_published_at", "raw_articles", ["published_at"])
    op.create_index("ix_raw_articles_processing_status", "raw_articles", ["processing_status"])
    op.create_index("ix_raw_articles_normalized_title", "raw_articles", ["normalized_title"])

    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("entity_type", entity_type, nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("article_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name", "entity_type", name="uq_entity_normalized_name_type"),
    )
    op.create_index("ix_entities_normalized_name", "entities", ["normalized_name"])

    op.create_table(
        "topics",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "article_entities",
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=100), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("extracted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["raw_articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["entity_id"], ["entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("article_id", "entity_id"),
        sa.UniqueConstraint("article_id", "entity_id", name="uq_article_entity"),
    )

    op.create_table(
        "article_topics",
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("method", topic_method, nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["raw_articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("article_id", "topic_id"),
        sa.UniqueConstraint("article_id", "topic_id", name="uq_article_topic"),
    )
    op.create_index("ix_article_topics_topic_id", "article_topics", ["topic_id"])

    op.create_table(
        "extraction_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_type", extraction_run_type, nullable=False),
        sa.Column("llm_provider", sa.String(length=100), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.String(length=100), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["raw_articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("extraction_runs")
    op.drop_index("ix_article_topics_topic_id", table_name="article_topics")
    op.drop_table("article_topics")
    op.drop_table("article_entities")
    op.drop_table("topics")
    op.drop_index("ix_entities_normalized_name", table_name="entities")
    op.drop_table("entities")
    op.drop_index("ix_raw_articles_normalized_title", table_name="raw_articles")
    op.drop_index("ix_raw_articles_processing_status", table_name="raw_articles")
    op.drop_index("ix_raw_articles_published_at", table_name="raw_articles")
    op.drop_index("ix_raw_articles_url", table_name="raw_articles")
    op.drop_table("raw_articles")

    extraction_run_type.drop(op.get_bind(), checkfirst=True)
    topic_method.drop(op.get_bind(), checkfirst=True)
    entity_type.drop(op.get_bind(), checkfirst=True)
    processing_status.drop(op.get_bind(), checkfirst=True)
