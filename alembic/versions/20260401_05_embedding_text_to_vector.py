"""Convert raw_articles.embedding from Text (JSON) to pgvector vector(384).

Copies existing JSON-serialised embeddings into the native vector column via
a direct cast (pgvector accepts the ``[x,y,...]`` format produced by
``json.dumps``), then creates an HNSW cosine-distance index.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260401_05"
down_revision = "20260401_04"
branch_labels = None
depends_on = None

_EMBEDDING_DIM = 384


def upgrade() -> None:
    # Add new native vector column alongside the old text column
    op.add_column(
        "raw_articles",
        sa.Column("embedding_vec", sa.Text(), nullable=True),  # placeholder; cast below
    )
    # Cast existing JSON text to vector in one pass; rows with NULL are left NULL
    op.execute(
        f"ALTER TABLE raw_articles"
        f" ALTER COLUMN embedding_vec TYPE vector({_EMBEDDING_DIM})"
        f" USING NULL"
    )
    op.execute(
        "UPDATE raw_articles"
        " SET embedding_vec = embedding::vector"
        " WHERE embedding IS NOT NULL"
    )

    # Swap: drop old text column, rename new vector column
    op.drop_column("raw_articles", "embedding")
    op.execute("ALTER TABLE raw_articles RENAME COLUMN embedding_vec TO embedding")

    # HNSW index for cosine-distance queries (``<=>`` operator)
    op.execute(
        "CREATE INDEX ix_raw_articles_embedding_cosine"
        " ON raw_articles USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_raw_articles_embedding_cosine")

    # Reverse: add text column, cast vector back to JSON text, swap
    op.add_column(
        "raw_articles",
        sa.Column("embedding_text", sa.Text(), nullable=True),
    )
    op.execute(
        "UPDATE raw_articles"
        " SET embedding_text = embedding::text"
        " WHERE embedding IS NOT NULL"
    )
    op.drop_column("raw_articles", "embedding")
    op.execute("ALTER TABLE raw_articles RENAME COLUMN embedding_text TO embedding")
