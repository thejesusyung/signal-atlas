"""Engagement scoring — pure functions, no DB or LLM calls."""

from __future__ import annotations

from collections import defaultdict

# Weights applied to each reader action.
# Repost = broadest reach, comment = friction/controversy, like = quiet approval.
ACTION_WEIGHTS: dict[str, int] = {
    "repost": 3,
    "comment": 2,
    "like": 1,
    "skip": 0,
}

_MAX_WEIGHT = max(ACTION_WEIGHTS.values())  # 3


def compute_engagement_score(
    repost_count: int,
    like_count: int,
    comment_count: int,
    skip_count: int,
    readers_sampled: int,
) -> float:
    """Return a normalised 0–1 score.

    Score = weighted_sum / (readers_sampled × max_weight).
    Returns 0.0 when readers_sampled is 0.
    """
    if readers_sampled == 0:
        return 0.0
    weighted = (
        repost_count * ACTION_WEIGHTS["repost"]
        + like_count * ACTION_WEIGHTS["like"]
        + comment_count * ACTION_WEIGHTS["comment"]
    )
    return weighted / (readers_sampled * _MAX_WEIGHT)


def aggregate_writer_scores(evaluations: list[dict]) -> dict[str, dict]:
    """Group a flat list of tweet-evaluation dicts by writer and compute scores.

    Each evaluation dict is the return value of the ``evaluate_tweet`` task:
    {writer_id, writer_name, prompt_version_id, repost_count, like_count,
     comment_count, skip_count, readers_sampled, ...}

    Returns a dict keyed by writer_id:
    {
        writer_name, prompt_version_id,
        repost_count, like_count, comment_count, skip_count,
        tweet_count, readers_sampled,
        engagement_score,           # 0–1 normalised
    }
    """
    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "writer_name": "",
            "prompt_version_id": "",
            "repost_count": 0,
            "like_count": 0,
            "comment_count": 0,
            "skip_count": 0,
            "tweet_count": 0,
            "readers_sampled": 0,
        }
    )

    for ev in evaluations:
        wid = ev["writer_id"]
        b = buckets[wid]
        b["writer_name"] = ev["writer_name"]
        b["prompt_version_id"] = ev["prompt_version_id"]
        b["repost_count"] += ev.get("repost_count", 0)
        b["like_count"] += ev.get("like_count", 0)
        b["comment_count"] += ev.get("comment_count", 0)
        b["skip_count"] += ev.get("skip_count", 0)
        b["tweet_count"] += 1
        b["readers_sampled"] += ev.get("readers_sampled", 0)

    for b in buckets.values():
        b["engagement_score"] = compute_engagement_score(
            b["repost_count"],
            b["like_count"],
            b["comment_count"],
            b["skip_count"],
            b["readers_sampled"],
        )

    return dict(buckets)


def select_writers_to_mutate(
    writer_scores: dict[str, dict],
    bottom_n: int = 2,
) -> list[str]:
    """Return writer_ids of the bottom_n performers, skipping any above average.

    If all writers are equal (e.g., first cycle with all zeros), still returns
    the bottom_n to ensure the simulation evolves even from a cold start.
    """
    if not writer_scores:
        return []

    ranked = sorted(writer_scores.items(), key=lambda kv: kv[1]["engagement_score"])
    scores = [v["engagement_score"] for v in writer_scores.values()]
    avg = sum(scores) / len(scores)

    candidates = ranked[:bottom_n]

    # If every writer is strictly above average (shouldn't happen with bottom_n=2
    # but guard anyway), return nothing — no mutation needed.
    all_above = all(score > avg for _, d in candidates for score in [d["engagement_score"]])
    if all_above and len(writer_scores) > bottom_n:
        return []

    return [wid for wid, _ in candidates]
