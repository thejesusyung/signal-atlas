"""TweetWriter: generates one tweet per (writer, story) pair."""

from __future__ import annotations

import logging

from news_pipeline.llm.provider import LLMProvider, LLMTraceContext

LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Twitter publisher with a defined writing strategy.
Write exactly one tweet about the news story provided.
Follow your writing strategy precisely — it defines your voice and angle.
Output only the tweet text. No hashtags unless they feel natural. Maximum 280 characters."""


class TweetWriter:
    def generate(
        self,
        *,
        story: dict,
        style_prompt: str,
        writer_name: str,
        prompt_version: int,
        provider: LLMProvider,
    ) -> str:
        """Return a tweet string (≤ 280 chars) for the given story.

        Args:
            story: dict with keys title, summary, entities, signal_type.
            style_prompt: the writer's current evolving strategy text.
            writer_name: used only for trace labelling.
            prompt_version: version number, used for trace labelling.
            provider: LLM provider instance.
        """
        entities_line = (
            ", ".join(story["entities"]) if story.get("entities") else "not specified"
        )
        user_prompt = (
            f"Your writing strategy:\n{style_prompt}\n\n"
            f"News story:\n"
            f"Title: {story['title']}\n"
            f"Summary: {story['summary']}\n"
            f"Key entities: {entities_line}\n\n"
            f"Write your tweet:"
        )

        trace_context = LLMTraceContext(
            operation="tweet_generation",
            prompt_version=f"{writer_name}_v{prompt_version}",
        )

        response = provider.complete(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.8,
            max_tokens=120,
            trace_context=trace_context,
        )

        content = _clean_tweet(response.text)
        LOGGER.debug(
            "Writer %s v%d generated %d-char tweet",
            writer_name,
            prompt_version,
            len(content),
        )
        return content


def _clean_tweet(raw: str) -> str:
    """Strip surrounding quotes the model sometimes adds, enforce 280-char cap."""
    text = raw.strip()
    # Remove wrapping quotes if the model returned "tweet text" or 'tweet text'
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text[:280]
