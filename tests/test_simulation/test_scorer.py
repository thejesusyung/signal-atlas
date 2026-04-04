"""Tests for simulation/scorer.py — pure functions, no DB or LLM needed."""
from __future__ import annotations

import pytest

from news_pipeline.simulation.scorer import (
    compute_engagement_score,
    aggregate_writer_scores,
    select_writers_to_mutate,
)


# ── compute_engagement_score ──────────────────────────────────────────────────


def test_compute_score_all_reposts():
    # All 10 readers reposted → max possible score = 1.0
    score = compute_engagement_score(
        repost_count=10, like_count=0, comment_count=0, skip_count=0, readers_sampled=10
    )
    assert score == pytest.approx(1.0)


def test_compute_score_all_skips():
    score = compute_engagement_score(
        repost_count=0, like_count=0, comment_count=0, skip_count=10, readers_sampled=10
    )
    assert score == 0.0


def test_compute_score_zero_readers():
    score = compute_engagement_score(
        repost_count=0, like_count=0, comment_count=0, skip_count=0, readers_sampled=0
    )
    assert score == 0.0


def test_compute_score_all_likes():
    # Like weight=1, max weight=3 → 10 likes / (10 * 3) = 1/3
    score = compute_engagement_score(
        repost_count=0, like_count=10, comment_count=0, skip_count=0, readers_sampled=10
    )
    assert score == pytest.approx(1 / 3)


def test_compute_score_mixed():
    # 3 reposts (×3) + 2 comments (×2) + 1 like (×1) = 14; readers=10 → 14/30
    score = compute_engagement_score(
        repost_count=3, like_count=1, comment_count=2, skip_count=4, readers_sampled=10
    )
    assert score == pytest.approx(14 / 30)


def test_compute_score_normalized_between_zero_and_one():
    score = compute_engagement_score(
        repost_count=2, like_count=3, comment_count=1, skip_count=4, readers_sampled=10
    )
    assert 0.0 <= score <= 1.0


# ── aggregate_writer_scores ───────────────────────────────────────────────────


def _make_eval(
    writer_id: str,
    writer_name: str = "Writer A",
    prompt_version_id: str = "pv-1",
    repost_count: int = 0,
    like_count: int = 0,
    comment_count: int = 0,
    skip_count: int = 0,
    readers_sampled: int = 10,
) -> dict:
    return {
        "writer_id": writer_id,
        "writer_name": writer_name,
        "prompt_version_id": prompt_version_id,
        "repost_count": repost_count,
        "like_count": like_count,
        "comment_count": comment_count,
        "skip_count": skip_count,
        "readers_sampled": readers_sampled,
    }


def test_aggregate_groups_by_writer():
    evals = [
        _make_eval("w1", writer_name="Alice", repost_count=1),
        _make_eval("w1", writer_name="Alice", repost_count=2),
        _make_eval("w2", writer_name="Bob", like_count=5),
    ]
    result = aggregate_writer_scores(evals)
    assert set(result.keys()) == {"w1", "w2"}
    assert result["w1"]["repost_count"] == 3
    assert result["w1"]["tweet_count"] == 2
    assert result["w1"]["writer_name"] == "Alice"


def test_aggregate_sums_readers():
    evals = [
        _make_eval("w1", readers_sampled=10),
        _make_eval("w1", readers_sampled=10),
    ]
    result = aggregate_writer_scores(evals)
    assert result["w1"]["readers_sampled"] == 20


def test_aggregate_computes_engagement_score():
    # One tweet, all reposts — score should be 1.0
    evals = [_make_eval("w1", repost_count=10, readers_sampled=10)]
    result = aggregate_writer_scores(evals)
    assert result["w1"]["engagement_score"] == pytest.approx(1.0)


def test_aggregate_empty_returns_empty():
    assert aggregate_writer_scores([]) == {}


# ── select_writers_to_mutate ──────────────────────────────────────────────────


def _make_scores(scores: dict[str, float]) -> dict[str, dict]:
    return {
        wid: {
            "engagement_score": score,
            "writer_name": wid,
            "prompt_version_id": "pv",
            "repost_count": 0,
            "like_count": 0,
            "comment_count": 0,
            "skip_count": 0,
            "tweet_count": 1,
            "readers_sampled": 10,
        }
        for wid, score in scores.items()
    }


def test_select_returns_bottom_two():
    scores = _make_scores({"w1": 0.9, "w2": 0.5, "w3": 0.1, "w4": 0.2})
    result = select_writers_to_mutate(scores, bottom_n=2)
    assert set(result) == {"w3", "w4"}


def test_select_empty_returns_empty():
    assert select_writers_to_mutate({}, bottom_n=2) == []


def test_select_all_zero_scores_still_returns_bottom_n():
    # Cold start — everyone at 0.0, should still select bottom_n
    scores = _make_scores({"w1": 0.0, "w2": 0.0, "w3": 0.0})
    result = select_writers_to_mutate(scores, bottom_n=2)
    assert len(result) == 2


def test_select_respects_bottom_n_parameter():
    scores = _make_scores({"w1": 0.9, "w2": 0.7, "w3": 0.3, "w4": 0.1})
    result = select_writers_to_mutate(scores, bottom_n=1)
    assert result == ["w4"]


def test_select_fewer_writers_than_bottom_n():
    scores = _make_scores({"w1": 0.3})
    result = select_writers_to_mutate(scores, bottom_n=2)
    assert result == ["w1"]
