"""Add simulation tables for the Twitter simulation layer."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260402_06"
down_revision = "20260401_05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    # sim_writers — the 5 competing publisher agents
    op.create_table(
        "sim_writers",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("persona_description", sa.Text(), nullable=False),
        # No FK — avoids circular dependency with sim_prompt_versions.
        sa.Column("current_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("name", name="uq_sim_writers_name"),
    )

    # sim_prompt_versions — full evolutionary lineage of each writer's style prompt
    op.create_table(
        "sim_prompt_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "writer_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_writers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("style_prompt", sa.Text(), nullable=False),
        # Self-referential: which version was mutated to produce this one.
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_prompt_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("cycle_introduced", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("triggered_by_score", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "writer_id", "version_number", name="uq_sim_prompt_writer_version"
        ),
    )
    op.create_index(
        "ix_sim_prompt_versions_writer_id", "sim_prompt_versions", ["writer_id"]
    )

    # sim_personas — the 100 reader personas, seeded once from personas.yaml
    op.create_table(
        "sim_personas",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("archetype_group", sa.String(50), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.UniqueConstraint("name", name="uq_sim_personas_name"),
    )
    op.create_index(
        "ix_sim_personas_archetype_group", "sim_personas", ["archetype_group"]
    )

    # sim_cycles — one row per simulation run
    if is_pg:
        story_ids_col = sa.Column(
            "story_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        )
    else:
        story_ids_col = sa.Column(
            "story_ids", sa.Text(), nullable=False, server_default=sa.text("'[]'")
        )

    op.create_table(
        "sim_cycles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("cycle_number", sa.Integer(), nullable=False),
        sa.Column("week_number", sa.Integer(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        story_ids_col,
        sa.Column("mlflow_run_id", sa.String(50), nullable=True),
        sa.UniqueConstraint("cycle_number", name="uq_sim_cycles_cycle_number"),
    )
    op.create_index("ix_sim_cycles_cycle_number", "sim_cycles", ["cycle_number"])
    op.create_index("ix_sim_cycles_started_at", "sim_cycles", ["started_at"])

    # sim_tweets — tweets generated each cycle
    op.create_table(
        "sim_tweets",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "cycle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_cycles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "writer_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_writers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "prompt_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_prompt_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "article_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("raw_articles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_sim_tweets_cycle_id", "sim_tweets", ["cycle_id"])
    op.create_index("ix_sim_tweets_writer_id", "sim_tweets", ["writer_id"])

    # sim_engagements — one persona's reaction to one tweet
    op.create_table(
        "sim_engagements",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "tweet_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_tweets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "persona_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_personas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_sim_engagements_tweet_id", "sim_engagements", ["tweet_id"])
    op.create_index("ix_sim_engagements_persona_id", "sim_engagements", ["persona_id"])

    # sim_writer_cycle_scores — aggregate per writer per cycle
    op.create_table(
        "sim_writer_cycle_scores",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "cycle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_cycles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "writer_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_writers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "prompt_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sim_prompt_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "engagement_score", sa.Float(), nullable=False, server_default="0.0"
        ),
        sa.Column("repost_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("like_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("comment_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skip_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tweet_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "reader_sample_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.UniqueConstraint(
            "cycle_id", "writer_id", name="uq_sim_score_cycle_writer"
        ),
    )
    op.create_index(
        "ix_sim_writer_cycle_scores_writer_id",
        "sim_writer_cycle_scores",
        ["writer_id"],
    )
    op.create_index(
        "ix_sim_writer_cycle_scores_cycle_id",
        "sim_writer_cycle_scores",
        ["cycle_id"],
    )


def downgrade() -> None:
    op.drop_table("sim_writer_cycle_scores")
    op.drop_table("sim_engagements")
    op.drop_table("sim_tweets")
    op.drop_table("sim_cycles")
    op.drop_table("sim_personas")
    op.drop_table("sim_prompt_versions")
    op.drop_table("sim_writers")
