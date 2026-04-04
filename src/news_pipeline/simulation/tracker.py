"""MLflow logging for the simulation layer.

All functions are pure with respect to the DB — they take plain dicts and
call the MLflow API. DB reads and writes stay in the DAG task functions.
"""

from __future__ import annotations

import logging

import mlflow

from news_pipeline.tracking.experiment import configure_mlflow

LOGGER = logging.getLogger(__name__)

# MLflow experiment name for the simulation.
SIMULATION_EXPERIMENT = "simulation_cycles"


# ── Cycle logging ─────────────────────────────────────────────────────────────


def log_cycle_to_mlflow(
    *,
    cycle_number: int,
    week_number: int,
    story_count: int,
    personas_per_tweet: int,
    writer_scores: list[dict],
    mutations: list[dict],
    tweet_rows: list[dict],
) -> str:
    """Create a parent MLflow run for this cycle and nested runs per writer.

    Args:
        cycle_number: monotonically increasing cycle counter.
        week_number: weeks elapsed since epoch (for week-level grouping).
        story_count: number of news stories used as input this cycle.
        personas_per_tweet: how many personas evaluated each tweet.
        writer_scores: list of per-writer score dicts (sorted by score desc).
            Each dict must have: writer_name, prompt_version_number,
            engagement_score, repost_count, like_count, comment_count,
            skip_count, tweet_count, readers_sampled.
        mutations: list of mutation dicts from score_and_mutate.
        tweet_rows: list of {writer_name, content} dicts for the tweet table.

    Returns:
        The MLflow run_id of the parent run.
    """
    configure_mlflow()
    mlflow.set_experiment(SIMULATION_EXPERIMENT)

    # Cycle-level aggregate stats
    scores = [w["engagement_score"] for w in writer_scores]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    top_score = scores[0] if scores else 0.0
    bottom_score = scores[-1] if scores else 0.0

    total_readers = sum(w.get("readers_sampled", 0) for w in writer_scores)
    total_skips = sum(w.get("skip_count", 0) for w in writer_scores)
    skip_rate = total_skips / total_readers if total_readers > 0 else 0.0

    with mlflow.start_run(run_name=f"cycle_{cycle_number:04d}") as parent_run:
        # ── Cycle-level params (constant within a run) ────────────────────────
        mlflow.log_params(
            {
                "cycle_number": cycle_number,
                "week_number": week_number,
                "story_count": story_count,
                "writers_count": len(writer_scores),
                "mutations_count": len(mutations),
                "personas_per_tweet": personas_per_tweet,
            }
        )
        mlflow.set_tags(
            {
                "dag_id": "simulation_dag",
                "week": str(week_number),
            }
        )

        # ── Cycle-level metrics (logged with step for time-series view) ───────
        mlflow.log_metrics(
            {
                "avg_engagement_score": avg_score,
                "top_engagement_score": top_score,
                "bottom_engagement_score": bottom_score,
                "cycle_skip_rate": skip_rate,
                "mutation_count": float(len(mutations)),
            },
            step=cycle_number,
        )

        # ── Tweet table artifact ──────────────────────────────────────────────
        if tweet_rows:
            mlflow.log_table(
                data={
                    "writer_name": [r["writer_name"] for r in tweet_rows],
                    "content": [r["content"] for r in tweet_rows],
                },
                artifact_file="cycle_tweets.json",
            )

        # ── Nested run per writer ─────────────────────────────────────────────
        for writer_data in writer_scores:
            writer_name = writer_data["writer_name"]
            readers = writer_data.get("readers_sampled", 0)
            score = writer_data["engagement_score"]

            with mlflow.start_run(
                run_name=f"{writer_name}_c{cycle_number:04d}",
                nested=True,
            ):
                mlflow.log_params(
                    {
                        "writer_name": writer_name,
                        "prompt_version": writer_data.get("prompt_version_number", "?"),
                    }
                )
                mlflow.set_tags(
                    {
                        "writer_name": writer_name,
                        "week": str(week_number),
                    }
                )
                mlflow.log_metrics(
                    {
                        "engagement_score": score,
                        "repost_rate": writer_data["repost_count"] / max(readers, 1),
                        "like_rate": writer_data["like_count"] / max(readers, 1),
                        "comment_rate": writer_data["comment_count"] / max(readers, 1),
                        "skip_rate": writer_data["skip_count"] / max(readers, 1),
                        "tweet_count": float(writer_data["tweet_count"]),
                        "readers_sampled": float(readers),
                    },
                    step=cycle_number,
                )

        run_id = parent_run.info.run_id

    LOGGER.info(
        "Logged cycle %d to MLflow (run_id=%s): avg=%.3f top=%.3f bottom=%.3f mutations=%d",
        cycle_number,
        run_id,
        avg_score,
        top_score,
        bottom_score,
        len(mutations),
    )
    return run_id


# ── Prompt Registry ───────────────────────────────────────────────────────────


def register_prompt(
    *,
    writer_name: str,
    style_prompt: str,
    version_number: int,
    cycle_number: int,
    triggered_by_score: float | None,
) -> None:
    """Register or update a writer's style_prompt in the MLflow Prompt Registry.

    Registry name: ``sim_{writer_name}`` (e.g. ``sim_TheBreakingWire``).
    Each call creates a new version in the registry, so call only when
    a genuinely new version exists (initial seed or post-mutation).
    """
    configure_mlflow()
    registry_name = f"sim_{writer_name}"

    tags: dict[str, str] = {
        "writer": writer_name,
        "version_number": str(version_number),
        "cycle_introduced": str(cycle_number),
    }
    if triggered_by_score is not None:
        tags["triggered_by_score"] = f"{triggered_by_score:.4f}"

    label = "initial" if version_number == 1 else f"mutation_v{version_number}"
    commit_message = f"{label} — cycle {cycle_number}"

    try:
        mlflow.register_prompt(
            name=registry_name,
            template=style_prompt,
            commit_message=commit_message,
            tags=tags,
        )
        LOGGER.info(
            "Registered prompt %r v%d in MLflow Prompt Registry",
            registry_name,
            version_number,
        )
    except Exception as exc:
        # Prompt registry requires MLflow ≥ 2.20. Log a warning but don't
        # crash the task if the server version is older.
        LOGGER.warning(
            "Could not register prompt %r: %s", registry_name, exc
        )


# ── Weekly champion tagging ───────────────────────────────────────────────────


def tag_weekly_champion(
    *,
    writer_name: str,
    week_number: int,
    avg_score: float,
    cycle_numbers: list[int],
) -> None:
    """Find and tag the parent runs belonging to the winning writer's week.

    Searches ``simulation_cycles`` for runs tagged with ``writer_name`` and
    sets a ``week_champion`` tag on the run with the highest engagement_score
    for that writer in the given week.
    """
    configure_mlflow()
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(SIMULATION_EXPERIMENT)
    if experiment is None:
        return

    # Search nested writer runs for this writer + week
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.writer_name = '{writer_name}' "
            f"AND tags.week = '{week_number}'"
        ),
        order_by=["metrics.engagement_score DESC"],
        max_results=1,
    )

    if not runs:
        LOGGER.warning(
            "No MLflow runs found for writer %r week %d", writer_name, week_number
        )
        return

    best_run = runs[0]
    client.set_tag(best_run.info.run_id, "week_champion", "true")
    client.set_tag(
        best_run.info.run_id, "champion_week", str(week_number)
    )
    LOGGER.info(
        "Tagged run %s as week %d champion (%s, avg_score=%.3f, cycles=%s)",
        best_run.info.run_id,
        week_number,
        writer_name,
        avg_score,
        cycle_numbers,
    )
