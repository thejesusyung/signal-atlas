from __future__ import annotations

from sqlalchemy.orm import Session

from news_pipeline.contracts import LLMResponse
from news_pipeline.db.models import ProcessingStatus, RawArticle
from news_pipeline.extraction.entity_extractor import EntityExtractor
from news_pipeline.extraction.topic_extractor import TopicExtractor
from news_pipeline.llm.provider import LLMProvider
from news_pipeline.utils import (
    DEFAULT_LLM_ARTICLE_TEXT_CHARS,
    clean_article_text,
    choose_article_text,
    normalize_title_for_dedup,
    utcnow,
)


class CaptureProvider(LLMProvider):
    provider_name = "capture"

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 900,
        trace_context=None,
    ):
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return LLMResponse(
            text=self.response_text,
            model="capture-model",
            tokens_used=10,
            latency_ms=5,
            provider_name=self.provider_name,
        )


def _build_long_article() -> RawArticle:
    return RawArticle(
        title="Acme expands satellite program",
        normalized_title=normalize_title_for_dedup("Acme expands satellite program"),
        summary="Acme is expanding its satellite program in Lima with a new launch systems team.",
        full_text=("Acme built a new launch systems lab in Lima. " * 500),
        cleaned_text=("Acme built a new launch systems lab in Lima. " * 500),
        url="https://example.com/articles/acme-satellites",
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        ingested_at=utcnow(),
        processing_status=ProcessingStatus.pending_extraction,
    )


def test_choose_article_text_bounds_large_articles() -> None:
    full_text = ("Acme built a new launch systems lab in Lima. " * 500).strip()
    summary = "Acme is expanding its satellite program in Lima with a new launch systems team."

    selected = choose_article_text(full_text=full_text, summary=summary, title="Acme expands satellite program")

    assert len(selected) <= DEFAULT_LLM_ARTICLE_TEXT_CHARS
    assert selected.startswith("Summary:")
    assert "Article excerpt:" in selected
    assert "[truncated]" in selected
    assert "Acme is expanding its satellite program" in selected


def test_clean_article_text_removes_leading_author_bio() -> None:
    raw_text = (
        "is a senior reviewer with over twenty years of experience. "
        "She covers smart home, IoT, and connected tech. "
        "Posts from this author will be added to your daily email digest. "
        "All the smart home news, reviews, and gadgets you need to know about "
        "Reolink's new solar-powered floodlight camera is now available. "
        "Reolink's new solar-powered floodlight camera is now available."
    )

    cleaned = clean_article_text(raw_text)

    assert cleaned == "Reolink's new solar-powered floodlight camera is now available."


def test_entity_extractor_sends_bounded_context(session: Session) -> None:
    article = _build_long_article()
    session.add(article)
    session.commit()

    provider = CaptureProvider('{"entities":[]}')
    extractor = EntityExtractor(provider)

    extractor.extract_for_article(session, article)

    assert len(provider.prompts) == 1
    assert "Summary:" in provider.prompts[0]
    assert "Article excerpt:" in provider.prompts[0]
    assert "[truncated]" in provider.prompts[0]
    assert len(provider.prompts[0]) < len(article.full_text)


def test_entity_extractor_prefers_cleaned_text(session: Session) -> None:
    article = RawArticle(
        title="Reolink camera launch",
        normalized_title=normalize_title_for_dedup("Reolink camera launch"),
        summary="Reolink launched a camera.",
        full_text=(
            "is a senior reviewer with over twenty years of experience. "
            "She covers smart home technology. "
            "Reolink launched a new camera with solar charging."
        ),
        cleaned_text="Reolink launched a new camera with solar charging.",
        url="https://example.com/articles/reolink-camera",
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        ingested_at=utcnow(),
        processing_status=ProcessingStatus.pending_extraction,
    )
    session.add(article)
    session.commit()

    provider = CaptureProvider('{"entities":[]}')
    extractor = EntityExtractor(provider)

    extractor.extract_for_article(session, article)

    assert "senior reviewer" not in provider.prompts[0]
    assert "Reolink launched a new camera with solar charging." in provider.prompts[0]


def test_topic_extractor_sends_bounded_context(session: Session) -> None:
    article = _build_long_article()
    session.add(article)
    session.commit()

    provider = CaptureProvider('{"topics":[{"topic_name":"technology","confidence":0.91}]}')
    extractor = TopicExtractor(provider)

    extractor.extract_for_article(session, article)

    assert len(provider.prompts) == 1
    assert "Summary:" in provider.prompts[0]
    assert "Article excerpt:" in provider.prompts[0]
    assert "[truncated]" in provider.prompts[0]
    assert len(provider.prompts[0]) < len(article.full_text)
