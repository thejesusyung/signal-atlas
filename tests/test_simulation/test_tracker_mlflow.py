"""Tests for the per-writer persistent MLflow tracking in tracker.py.

Uses a local file-based MLflow backend (via tmp_path) so tests are self-contained
and fast — no server required, no artifact permissions issues.

Run with:
    pytest tests/test_simulation/test_tracker_mlflow.py -v
"""

from __future__ import annotations

import pytest
import mlflow
from mlflow.tracking import MlflowClient

from news_pipeline.simulation.tracker import (
    SIMULATION_EXPERIMENT,
    ensure_writer_run,
    log_writer_cycle,
    register_prompt,
    tag_weekly_champion,
)


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a temp local store for each test. Fully isolated, no server needed."""
    uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(uri)
    # ensure_writer_run / log_writer_cycle call configure_mlflow() internally,
    # which reads from settings. Patch it out so it doesn't overwrite our test URI.
    monkeypatch.setattr(
        "news_pipeline.simulation.tracker.configure_mlflow", lambda: None
    )
    yield
    mlflow.set_tracking_uri("")


@pytest.fixture
def client():
    return MlflowClient()


@pytest.fixture
def writer_run(client):
    """Create a writer run for use in tests."""
    run_id = ensure_writer_run(
        writer_name="_test_writer",
        persona_description="A test persona for automated testing.",
        initial_prompt="Write concise breaking-news tweets.",
    )
    return run_id


# ── ensure_writer_run ─────────────────────────────────────────────────────────


def test_ensure_writer_run_creates_finished_run(client, writer_run):
    """ensure_writer_run() creates a FINISHED run with params and the initial prompt artifact."""
    run = client.get_run(writer_run)

    assert run.info.status == "FINISHED"
    assert run.data.params["writer_name"] == "_test_writer"
    assert "persona" in run.data.params

    artifacts = [a.path for a in client.list_artifacts(writer_run)]
    assert "prompt_v1.txt" in artifacts


def test_ensure_writer_run_is_in_simulation_experiment(client, writer_run):
    """The run lands in the simulation_cycles experiment."""
    run = client.get_run(writer_run)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == SIMULATION_EXPERIMENT


def test_ensure_writer_run_is_idempotent_in_db(client):
    """Calling ensure_writer_run twice creates two separate runs (DB dedup is the caller's job)."""
    run_id_1 = ensure_writer_run(
        writer_name="_test_writer",
        persona_description="desc",
        initial_prompt="prompt",
    )
    run_id_2 = ensure_writer_run(
        writer_name="_test_writer",
        persona_description="desc",
        initial_prompt="prompt",
    )
    # The DB layer prevents double-calling by checking mlflow_run_id != None.
    # At the MLflow level, two calls produce two distinct runs.
    assert run_id_1 != run_id_2


# ── log_writer_cycle ──────────────────────────────────────────────────────────


def test_log_writer_cycle_appends_metrics(client, writer_run):
    """log_writer_cycle() adds metrics to the writer's existing run."""
    log_writer_cycle(
        writer_run_id=writer_run,
        cycle_number=1,
        engagement_score=0.42,
        repost_rate=0.10,
        like_rate=0.20,
        comment_rate=0.05,
        skip_rate=0.65,
        prompt_version=1,
    )

    run = client.get_run(writer_run)
    assert run.data.metrics["engagement_score"] == pytest.approx(0.42)
    assert run.data.metrics["prompt_version"] == pytest.approx(1.0)


def test_log_writer_cycle_produces_trend_line(client, writer_run):
    """Each cycle call appends one step — metric history has N entries for N cycles."""
    scores = [0.30, 0.38, 0.47]
    for cycle, score in enumerate(scores, start=1):
        log_writer_cycle(
            writer_run_id=writer_run,
            cycle_number=cycle,
            engagement_score=score,
            repost_rate=0.1,
            like_rate=0.2,
            comment_rate=0.05,
            skip_rate=0.65,
            prompt_version=1,
        )

    history = client.get_metric_history(writer_run, "engagement_score")
    steps_and_values = [(m.step, m.value) for m in sorted(history, key=lambda m: m.step)]

    assert len(steps_and_values) == 3
    assert steps_and_values[0] == (1, pytest.approx(0.30))
    assert steps_and_values[1] == (2, pytest.approx(0.38))
    assert steps_and_values[2] == (3, pytest.approx(0.47))


def test_log_writer_cycle_tracks_prompt_version_changes(client, writer_run):
    """prompt_version metric jumps when a mutation fires — visible in the chart."""
    log_writer_cycle(
        writer_run_id=writer_run, cycle_number=1, engagement_score=0.3,
        repost_rate=0.1, like_rate=0.1, comment_rate=0.05, skip_rate=0.75,
        prompt_version=1,
    )
    log_writer_cycle(
        writer_run_id=writer_run, cycle_number=2, engagement_score=0.52,
        repost_rate=0.15, like_rate=0.2, comment_rate=0.07, skip_rate=0.58,
        prompt_version=2,  # mutation fired between cycle 1 and 2
    )

    history = client.get_metric_history(writer_run, "prompt_version")
    versions = [m.value for m in sorted(history, key=lambda m: m.step)]
    assert versions == [pytest.approx(1.0), pytest.approx(2.0)]


# ── register_prompt ───────────────────────────────────────────────────────────


def test_register_prompt_attaches_artifact_to_writer_run(client, writer_run):
    """register_prompt() with writer_run_id logs prompt_v{N}.txt to the run."""
    register_prompt(
        writer_name="_test_writer",
        style_prompt="Write punchy tweets that cut through noise.",
        version_number=2,
        cycle_number=5,
        triggered_by_score=0.31,
        writer_run_id=writer_run,
    )

    artifacts = [a.path for a in client.list_artifacts(writer_run)]
    assert "prompt_v2.txt" in artifacts


def test_register_prompt_without_run_id_does_not_crash():
    """register_prompt() with no writer_run_id skips artifact logging gracefully."""
    # Should not raise even though there's no active run to attach the artifact to.
    register_prompt(
        writer_name="_test_writer",
        style_prompt="Some prompt.",
        version_number=1,
        cycle_number=0,
        triggered_by_score=None,
        writer_run_id=None,
    )


# ── tag_weekly_champion ───────────────────────────────────────────────────────


def test_tag_weekly_champion_sets_tags(client, writer_run):
    """tag_weekly_champion() sets champion/week/score tags on the writer's run."""
    tag_weekly_champion(
        writer_run_id=writer_run,
        writer_name="_test_writer",
        week_number=3,
        avg_score=0.61,
    )

    run = client.get_run(writer_run)
    assert run.data.tags["champion"] == "true"
    assert run.data.tags["champion_week"] == "3"
    assert "0.61" in run.data.tags["champion_avg_score"]


# ── end-to-end lifecycle ──────────────────────────────────────────────────────


def test_full_writer_lifecycle(client):
    """Seed → 3 cycles → mutation → champion. Mirrors the full DAG without Airflow or DB."""
    run_id = ensure_writer_run(
        writer_name="_lifecycle_writer",
        persona_description="Covers fintech and emerging markets.",
        initial_prompt="Short, data-driven takes on market moves.",
    )

    # Cycles 1–2 on the initial prompt
    for cycle, score in enumerate([0.33, 0.41], start=1):
        log_writer_cycle(
            writer_run_id=run_id, cycle_number=cycle, engagement_score=score,
            repost_rate=0.08, like_rate=0.18, comment_rate=0.04, skip_rate=0.70,
            prompt_version=1,
        )

    # Mutation fires after cycle 2
    register_prompt(
        writer_name="_lifecycle_writer",
        style_prompt="Lead with the stakes. Keep it under 200 chars.",
        version_number=2,
        cycle_number=3,
        triggered_by_score=0.41,
        writer_run_id=run_id,
    )

    # Cycle 3 on the mutated prompt
    log_writer_cycle(
        writer_run_id=run_id, cycle_number=3, engagement_score=0.58,
        repost_rate=0.14, like_rate=0.27, comment_rate=0.07, skip_rate=0.52,
        prompt_version=2,
    )

    # Week ends — this writer won
    tag_weekly_champion(
        writer_run_id=run_id, writer_name="_lifecycle_writer",
        week_number=1, avg_score=0.44,
    )

    # Assert the full run state
    run = client.get_run(run_id)
    history = sorted(
        client.get_metric_history(run_id, "engagement_score"), key=lambda m: m.step
    )
    artifacts = [a.path for a in client.list_artifacts(run_id)]

    assert len(history) == 3, "one metric point per cycle"
    assert history[0].value == pytest.approx(0.33)
    assert history[2].value == pytest.approx(0.58), "score should improve after mutation"
    assert "prompt_v1.txt" in artifacts, "initial prompt attached at seed"
    assert "prompt_v2.txt" in artifacts, "mutated prompt attached after mutation"
    assert run.data.tags["champion"] == "true"
    assert run.data.tags["champion_week"] == "1"
