"""Tests for simulation reader, writer, and mutator — LLM calls stubbed out."""
from __future__ import annotations

import pytest

from news_pipeline.contracts import LLMResponse
from news_pipeline.llm.provider import LLMProvider, LLMTraceContext


# ── Stub provider ─────────────────────────────────────────────────────────────


class StubProvider(LLMProvider):
    """Returns a configurable fixed response without any network call."""

    provider_name = "stub"

    def __init__(self, text: str = "stub response") -> None:
        self._text = text

    def complete(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 900,
        trace_context: LLMTraceContext | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text=self._text,
            model="stub-model",
            tokens_used=5,
            latency_ms=1,
            provider_name=self.provider_name,
        )


class ErrorProvider(LLMProvider):
    """Always raises so we can test fallback paths."""

    provider_name = "error"

    def complete(self, *args, **kwargs) -> LLMResponse:
        raise RuntimeError("provider unavailable")


# ── _parse_response ───────────────────────────────────────────────────────────


class TestParseResponse:
    """Tests for reader._parse_response — the JSON extraction helper."""

    def _parse(self, text: str) -> dict:
        from news_pipeline.simulation.reader import _parse_response
        return _parse_response(text)

    def test_clean_json_like(self):
        result = self._parse('{"action": "like", "reason": "great content"}')
        assert result["action"] == "like"
        assert result["reason"] == "great content"

    def test_clean_json_repost(self):
        result = self._parse('{"action": "repost", "reason": null}')
        assert result["action"] == "repost"
        assert result["reason"] is None

    def test_fenced_json_block(self):
        raw = "```json\n{\"action\": \"comment\", \"reason\": \"interesting\"}\n```"
        result = self._parse(raw)
        assert result["action"] == "comment"

    def test_fenced_without_language_tag(self):
        raw = "```\n{\"action\": \"like\", \"reason\": \"neat\"}\n```"
        result = self._parse(raw)
        assert result["action"] == "like"

    def test_invalid_action_falls_back_to_skip(self):
        result = self._parse('{"action": "downvote", "reason": "nope"}')
        assert result["action"] == "skip"

    def test_missing_action_falls_back_to_skip(self):
        result = self._parse('{"reason": "no action key"}')
        assert result["action"] == "skip"

    def test_malformed_json_falls_back_to_skip(self):
        result = self._parse("not json at all")
        assert result["action"] == "skip"
        assert result["reason"] is None

    def test_reason_truncated_to_500_chars(self):
        long_reason = "x" * 600
        result = self._parse(f'{{"action": "like", "reason": "{long_reason}"}}')
        assert len(result["reason"]) == 500

    def test_empty_reason_becomes_none(self):
        result = self._parse('{"action": "skip", "reason": ""}')
        assert result["reason"] is None

    def test_non_string_reason_becomes_none(self):
        result = self._parse('{"action": "like", "reason": 42}')
        assert result["reason"] is None

    def test_json_embedded_in_text(self):
        raw = 'Some preamble {"action": "repost", "reason": "good"} trailing text'
        result = self._parse(raw)
        assert result["action"] == "repost"


# ── PersonaReader ─────────────────────────────────────────────────────────────


class TestPersonaReader:
    def test_evaluate_returns_action_from_provider(self):
        from news_pipeline.simulation.reader import PersonaReader
        provider = StubProvider('{"action": "like", "reason": "nice"}')
        reader = PersonaReader()
        result = reader.evaluate(
            tweet_content="Big news today.",
            persona_name="TestUser",
            persona_description="A regular Twitter user.",
            provider=provider,
        )
        assert result["action"] == "like"
        assert result["reason"] == "nice"

    def test_evaluate_defaults_to_skip_on_provider_error(self):
        from news_pipeline.simulation.reader import PersonaReader
        reader = PersonaReader()
        result = reader.evaluate(
            tweet_content="tweet",
            persona_name="TestUser",
            persona_description="desc",
            provider=ErrorProvider(),
        )
        assert result == {"action": "skip", "reason": None}


# ── _clean_tweet ──────────────────────────────────────────────────────────────


class TestCleanTweet:
    def _clean(self, text: str) -> str:
        from news_pipeline.simulation.writer import _clean_tweet
        return _clean_tweet(text)

    def test_strips_double_quotes(self):
        assert self._clean('"Hello world"') == "Hello world"

    def test_strips_single_quotes(self):
        assert self._clean("'Hello world'") == "Hello world"

    def test_no_quotes_unchanged(self):
        assert self._clean("Hello world") == "Hello world"

    def test_mismatched_quotes_unchanged(self):
        assert self._clean('"mismatched\'') == '"mismatched\''

    def test_enforces_280_char_cap(self):
        long_text = "x" * 300
        result = self._clean(long_text)
        assert len(result) == 280

    def test_empty_string_unchanged(self):
        assert self._clean("") == ""

    def test_strips_surrounding_whitespace(self):
        assert self._clean("  hello  ") == "hello"


# ── TweetWriter ───────────────────────────────────────────────────────────────


class TestTweetWriter:
    def test_generate_returns_cleaned_text(self):
        from news_pipeline.simulation.writer import TweetWriter
        provider = StubProvider('"Quoted tweet text here"')
        writer = TweetWriter()
        result = writer.generate(
            story={
                "title": "Big event happened",
                "summary": "Something major occurred today.",
                "entities": ["Acme Corp"],
                "signal_type": "entity_velocity",
            },
            style_prompt="Be direct and factual.",
            writer_name="TheBreakingWire",
            prompt_version=1,
            provider=provider,
        )
        # Surrounding quotes should be stripped
        assert result == "Quoted tweet text here"

    def test_generate_enforces_280_char_limit(self):
        from news_pipeline.simulation.writer import TweetWriter
        provider = StubProvider("x" * 300)
        writer = TweetWriter()
        result = writer.generate(
            story={"title": "t", "summary": "s", "entities": [], "signal_type": "x"},
            style_prompt="write stuff",
            writer_name="w",
            prompt_version=1,
            provider=provider,
        )
        assert len(result) <= 280

    def test_generate_no_entities_uses_fallback_text(self):
        """generate() should not raise when entities list is empty."""
        from news_pipeline.simulation.writer import TweetWriter
        provider = StubProvider("tweet content")
        TweetWriter().generate(
            story={"title": "Title", "summary": "Summary", "entities": [], "signal_type": "x"},
            style_prompt="style",
            writer_name="writer",
            prompt_version=1,
            provider=provider,
        )


# ── _clean_output (mutator) ───────────────────────────────────────────────────


class TestCleanOutput:
    def _clean(self, text: str) -> str:
        from news_pipeline.simulation.mutator import _clean_output
        return _clean_output(text)

    def test_strips_here_is_preamble(self):
        result = self._clean("Here is the evolved strategy: Lead with data.")
        assert result == "Lead with data."

    def test_strips_evolved_strategy_preamble(self):
        result = self._clean("Evolved strategy: Ask provocative questions.")
        assert result == "Ask provocative questions."

    def test_strips_new_strategy_preamble(self):
        result = self._clean("New strategy: Be more concise.")
        assert result == "Be more concise."

    def test_clean_input_passes_through(self):
        result = self._clean("Lead with data and context.")
        assert result == "Lead with data and context."

    def test_strips_surrounding_double_quotes(self):
        result = self._clean('"Lead with hard numbers."')
        assert result == "Lead with hard numbers."

    def test_strips_surrounding_single_quotes(self):
        result = self._clean("'Lead with hard numbers.'")
        assert result == "Lead with hard numbers."

    def test_preamble_only_first_line_stripped(self):
        text = "Here is the evolved strategy: Line one.\nLine two unchanged."
        result = self._clean(text)
        assert "Line two unchanged." in result
        assert "Here is" not in result


# ── PromptMutator ─────────────────────────────────────────────────────────────


class TestPromptMutator:
    def test_mutate_returns_string(self):
        from news_pipeline.simulation.mutator import PromptMutator
        provider = StubProvider("Lean into controversy to drive comments.")
        mutator = PromptMutator()
        result = mutator.mutate(
            writer_name="SharpTake",
            persona_description="A contrarian columnist.",
            current_prompt="Lead with the uncomfortable take.",
            recent_scores=[0.12, 0.09, 0.11],
            top_performer_name="TheBreakingWire",
            top_performer_prompt="Be first with the fact.",
            provider=provider,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mutate_strips_preamble_from_provider_response(self):
        from news_pipeline.simulation.mutator import PromptMutator
        provider = StubProvider("Here is the evolved strategy: Use vivid language.")
        mutator = PromptMutator()
        result = mutator.mutate(
            writer_name="SharpTake",
            persona_description="desc",
            current_prompt="original",
            recent_scores=[0.1],
            top_performer_name="top",
            top_performer_prompt="top prompt",
            provider=provider,
        )
        assert result == "Use vivid language."
