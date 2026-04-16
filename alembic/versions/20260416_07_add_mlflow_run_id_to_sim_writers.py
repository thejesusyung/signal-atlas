"""Add mlflow_run_id to sim_writers for persistent per-writer experiment tracking."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260416_07"
down_revision = "20260402_06"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sim_writers",
        sa.Column("mlflow_run_id", sa.String(50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sim_writers", "mlflow_run_id")
