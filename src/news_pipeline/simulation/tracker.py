"""MLflow logging for the simulation layer.

Each writer has ONE persistent MLflow run (stored on SimWriter.mlflow_run_id).
Every cycle appends metrics to that run using step=cycle_number, so the MLflow
UI shows a real trend line for each writer across the full simulation.

Call order per simulation lifecycle:
  1. ensure_writer_run()   — once at seed time, stores run_id in DB
  2. log_writer_cycle()    — every cycle, resumes run and appends one step
  3. register_prompt()     — on mutation, attaches new prompt artifact to run
  4. tag_weekly_champion() — on week boundary, tags the winning run
"""

from __future__ import annotations

import logging

import mlflow

from news_pipeline.tracking.experiment import configure_mlflow

LOGGER = logging.getLogger(__name__)

SIMULATION_EXPERIMENT = "simulation_cycles"


# ── Run creation (seed time) ──────────────────────────────────────────────────


def ensure_writer_run(
    *,
    writer_name: str,
    persona_description: str,
    initial_prompt: str,
) -> str:
    """Create the persistent MLflow run for a writer. Call once when the writer is seeded.

    The run is immediately ended — MLflow keeps it in FINISHED state and allows
    resuming it later via start_run(run_id=...).

    Returns the run_id to be stored on SimWriter.mlflow_run_id.
    """
    configure_mlflow()
    mlflow.set_experiment(SIMULATION_EXPERIMENT)
    run = mlflow.start_run(run_name=f"writer_{writer_name}")
    mlflow.log_params({"writer_name": writer_name, "persona": persona_description})
    mlflow.log_text(initial_prompt, "prompt_v1.txt")
    mlflow.end_run()
    LOGGER.info("Created MLflow run %s for writer %r", run.info.run_id, writer_name)
    return run.info.run_id


# ── Per-cycle logging ─────────────────────────────────────────────────────────


def log_writer_cycle(
    *,
    writer_run_id: str,
    cycle_number: int,
    engagement_score: float,
    repost_rate: float,
    like_rate: float,
    comment_rate: float,
    skip_rate: float,
    prompt_version: int,
) -> None:
    """Resume a writer's persistent run and append this cycle's metrics as one step.

    Using step=cycle_number means MLflow plots a continuous trend line across all
    cycles for each metric in the writer's run view.
    """
    configure_mlflow()
    with mlflow.start_run(run_id=writer_run_id):
        mlflow.log_metrics(
            {
                "engagement_score": engagement_score,
                "repost_rate": repost_rate,
                "like_rate": like_rate,
                "comment_rate": comment_rate,
                "skip_rate": skip_rate,
                # Tracked as a metric so jumps are visible in the chart alongside scores.
                "prompt_version": float(prompt_version),
            },
            step=cycle_number,
        )


# ── Prompt registry + artifact ────────────────────────────────────────────────


def register_prompt(
    *,
    writer_name: str,
    style_prompt: str,
    version_number: int,
    cycle_number: int,
    triggered_by_score: float | None,
    writer_run_id: str | None = None,
) -> None:
    """Register a new prompt version in the MLflow Prompt Registry and attach it to the writer's run.

    Registry name: ``sim_{writer_name}`` (e.g. ``sim_TheBreakingWire``).
    Also logs the prompt text as an artifact (``prompt_v{N}.txt``) on the writer's
    persistent run so prompt evolution is visible alongside the score trend.
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
        # Prompt registry requires MLflow ≥ 2.20.
        LOGGER.warning("Could not register prompt %r: %s", registry_name, exc)

    # Attach the prompt text to the writer's persistent run so it's viewable in the UI.
    if writer_run_id:
        with mlflow.start_run(run_id=writer_run_id):
            mlflow.log_text(style_prompt, f"prompt_v{version_number}.txt")
        LOGGER.debug(
            "Attached prompt_v%d.txt to run %s for writer %r",
            version_number,
            writer_run_id,
            writer_name,
        )


# ── Weekly champion tagging ───────────────────────────────────────────────────


def tag_weekly_champion(
    *,
    writer_run_id: str,
    writer_name: str,
    week_number: int,
    avg_score: float,
) -> None:
    """Tag the winning writer's persistent run as the week champion."""
    configure_mlflow()
    with mlflow.start_run(run_id=writer_run_id):
        mlflow.set_tags(
            {
                "champion": "true",
                "champion_week": str(week_number),
                "champion_avg_score": f"{avg_score:.4f}",
            }
        )
    LOGGER.info(
        "Tagged run %s as week %d champion (%s, avg_score=%.3f)",
        writer_run_id,
        week_number,
        writer_name,
        avg_score,
    )
