"""Add LLM rate limit reservation table."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260325_02"
down_revision = "20260324_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_rate_limit_reservations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider_name", sa.String(length=100), nullable=False),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_llm_rate_limit_reservations_provider_reserved_at",
        "llm_rate_limit_reservations",
        ["provider_name", "reserved_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_llm_rate_limit_reservations_provider_reserved_at",
        table_name="llm_rate_limit_reservations",
    )
    op.drop_table("llm_rate_limit_reservations")
