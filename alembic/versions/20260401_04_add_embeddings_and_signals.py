"""Add embeddings, semantic clusters, and signals table."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "20260401_04"
down_revision = "20260328_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    # pgvector extension (PostgreSQL only)
    if is_pg:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Embedding + cluster columns on raw_articles
    op.add_column("raw_articles", sa.Column("embedding", sa.Text(), nullable=True))
    op.add_column("raw_articles", sa.Column("semantic_cluster_id", sa.Integer(), nullable=True))
    op.add_column("raw_articles", sa.Column("cluster_label", sa.String(200), nullable=True))
    op.create_index("ix_raw_articles_semantic_cluster_id", "raw_articles", ["semantic_cluster_id"])

    # article_ids: JSONB on PostgreSQL (native indexed JSON storage), TEXT on SQLite (tests).
    # The ORM model uses the JSONBList TypeDecorator which handles serialisation on both dialects.
    if is_pg:
        article_ids_col = sa.Column(
            "article_ids", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        )
    else:
        article_ids_col = sa.Column(
            "article_ids", sa.Text(), nullable=False, server_default=sa.text("'[]'")
        )

    # Signals table
    op.create_table(
        "signals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="SET NULL"), nullable=True),
        sa.Column("topic_name", sa.String(200), nullable=True),
        sa.Column("signal_type", sa.String(50), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        article_ids_col,
    )
    op.create_index("ix_signals_detected_at", "signals", ["detected_at"])
    op.create_index("ix_signals_entity_id", "signals", ["entity_id"])


def downgrade() -> None:
    op.drop_table("signals")
    op.drop_index("ix_raw_articles_semantic_cluster_id", table_name="raw_articles")
    op.drop_column("raw_articles", "cluster_label")
    op.drop_column("raw_articles", "semantic_cluster_id")
    op.drop_column("raw_articles", "embedding")
    op.execute("DROP EXTENSION IF EXISTS vector")
