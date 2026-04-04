"""PersonaReader: evaluates one tweet from the perspective of one persona."""

from __future__ import annotations

import json
import logging
import re

from news_pipeline.llm.provider import LLMProvider, LLMTraceContext

LOGGER = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"like", "repost", "comment", "skip"})

_SYSTEM_PROMPT = """\
You are roleplaying as a specific Twitter user.
Read the tweet and decide authentically how this person would react to it.
Respond with valid JSON only — no other text:
{"action": "skip", "reason": "one sentence"}

The action must be exactly one of: like, repost, comment, skip
Be true to the personality. Most real users skip most tweets.\
"""


class PersonaReader:
    def evaluate(
        self,
        *,
        tweet_content: str,
        persona_name: str,
        persona_description: str,
        provider: LLMProvider,
    ) -> dict:
        """Return {"action": str, "reason": str | None} for this persona + tweet.

        Never raises — falls back to {"action": "skip", "reason": None} on any
        LLM or parse failure so one bad call does not abort the whole tweet eval.
        """
        user_prompt = (
            f"About you:\n{persona_description}\n\n"
            f"Tweet:\n{tweet_content}\n\n"
            f"How do you react?"
        )
        try:
            response = provider.complete(
                prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=80,
                trace_context=LLMTraceContext(
                    operation="persona_evaluation",
                    prompt_version=persona_name,
                ),
            )
            return _parse_response(response.text)
        except Exception as exc:
            LOGGER.warning(
                "Persona evaluation failed for %r: %s — defaulting to skip",
                persona_name,
                exc,
            )
            return {"action": "skip", "reason": None}


def _parse_response(raw: str) -> dict:
    """Parse the model's JSON response into {"action", "reason"}.

    Handles:
    - Clean JSON: {"action": "like", "reason": "..."}
    - JSON wrapped in markdown fences
    - Partial or malformed JSON (falls back to skip)
    """
    text = raw.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try to extract the first JSON object if there is surrounding text
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"action": "skip", "reason": None}

    action = str(data.get("action", "skip")).lower().strip()
    if action not in _VALID_ACTIONS:
        action = "skip"

    reason = data.get("reason")
    if isinstance(reason, str):
        reason = reason.strip()[:500] or None
    else:
        reason = None

    return {"action": action, "reason": reason}
