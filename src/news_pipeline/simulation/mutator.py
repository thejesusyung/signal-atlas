"""PromptMutator: rewrites a writer's style_prompt based on performance history."""

from __future__ import annotations

import logging
import re

from news_pipeline.llm.provider import LLMProvider, LLMTraceContext

LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a social media strategist advising a Twitter publisher.
You will see the publisher's persona, their current writing strategy, their recent
engagement scores, and what the top-performing publisher is doing differently.

Your task: write an evolved version of their strategy.
Rules:
- Make exactly one meaningful change (tone, angle, framing, structure, or focus).
- Keep the writer's core persona intact — do not change who they fundamentally are.
- Do not copy the top performer word-for-word; borrow a tactic, not an identity.
- A strategy describes voice, tone, angle, and structure. It must NOT contain example tweets, specific facts, names, or quoted content.
- Output only the new strategy text. No preamble, no explanation, no labels.\
"""

# Prefixes the model sometimes adds before the actual strategy text.
_PREAMBLE_PATTERNS = re.compile(
    r"^(here(?:'s| is)(?: the| an?)?(?: evolved| updated| new)?(?: strategy| approach| version)?[:\s]*"
    r"|evolved strategy[:\s]*"
    r"|new strategy[:\s]*"
    r"|updated strategy[:\s]*"
    r"|strategy[:\s]*)",
    re.IGNORECASE,
)


class PromptMutator:
    def mutate(
        self,
        *,
        writer_name: str,
        persona_description: str,
        current_prompt: str,
        recent_scores: list[float],
        top_performer_name: str,
        top_performer_prompt: str,
        provider: LLMProvider,
    ) -> str:
        """Return an evolved style_prompt string for the given writer.

        Args:
            writer_name: used for trace labelling only.
            persona_description: the writer's stable persona (never changes).
            current_prompt: the style_prompt being mutated.
            recent_scores: engagement scores oldest-first, e.g. [0.12, 0.09, 0.11].
            top_performer_name: name of the best writer this cycle.
            top_performer_prompt: their current style_prompt for inspiration.
            provider: LLM provider instance.
        """
        scores_str = " → ".join(f"{s:.3f}" for s in recent_scores)
        user_prompt = (
            f"Publisher persona:\n{persona_description}\n\n"
            f"Current writing strategy:\n{current_prompt}\n\n"
            f"Recent engagement scores (oldest → newest): {scores_str}\n\n"
            f"This cycle's top performer ({top_performer_name}) uses this strategy:\n"
            f"{top_performer_prompt}\n\n"
            f"Write the evolved strategy for {writer_name}:"
        )

        response = provider.complete(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.6,
            max_tokens=250,
            trace_context=LLMTraceContext(
                operation="prompt_mutation",
                prompt_version=writer_name,
            ),
        )

        new_prompt = _clean_output(response.text)
        LOGGER.info(
            "Mutated prompt for %s (scores %s) — %d → %d chars",
            writer_name,
            scores_str,
            len(current_prompt),
            len(new_prompt),
        )
        return new_prompt


def _clean_output(raw: str) -> str:
    """Strip preamble lines the model sometimes prepends to the strategy text."""
    text = raw.strip()
    # Remove a leading preamble on the first line only
    text = _PREAMBLE_PATTERNS.sub("", text, count=1).strip()
    # Remove any surrounding quotes
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text
