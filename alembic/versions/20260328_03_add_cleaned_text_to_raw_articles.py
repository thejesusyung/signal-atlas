"""Add cleaned_text to raw_articles."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260328_03"
down_revision = "20260325_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("raw_articles", sa.Column("cleaned_text", sa.Text(), nullable=True))
    op.execute("UPDATE raw_articles SET cleaned_text = full_text WHERE full_text IS NOT NULL AND cleaned_text IS NULL")


def downgrade() -> None:
    op.drop_column("raw_articles", "cleaned_text")
