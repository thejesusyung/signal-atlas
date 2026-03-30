from __future__ import annotations

from news_pipeline.contracts import ArticleCandidate
from news_pipeline.db.models import ProcessingStatus, RawArticle
from news_pipeline.ingestion.dedup import ArticleDeduplicator
from news_pipeline.utils import normalize_title_for_dedup, utcnow


def test_dedup_matches_exact_url(session):
    article = RawArticle(
        title="Acme launches new platform",
        normalized_title=normalize_title_for_dedup("Acme launches new platform"),
        summary="",
        url="https://example.com/acme",
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        ingested_at=utcnow(),
        processing_status=ProcessingStatus.pending_extraction,
    )
    session.add(article)
    session.commit()

    candidate = ArticleCandidate(
        title="Different title",
        summary="",
        url="https://example.com/acme",
        published_at=utcnow(),
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
    )

    result = ArticleDeduplicator().check(session, candidate)

    assert result.is_duplicate is True
    assert result.duplicate_of == article.id


def test_dedup_matches_normalized_title(session):
    article = RawArticle(
        title="Acme launches new platform in Lima",
        normalized_title=normalize_title_for_dedup("Acme launches new platform in Lima"),
        summary="",
        url="https://example.com/acme-1",
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
        published_at=utcnow(),
        ingested_at=utcnow(),
        processing_status=ProcessingStatus.pending_extraction,
    )
    session.add(article)
    session.commit()

    candidate = ArticleCandidate(
        title="Acme launches a new platform in Lima",
        summary="",
        url="https://example.com/acme-2",
        published_at=utcnow(),
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
    )

    result = ArticleDeduplicator(similarity_threshold=0.8).check(session, candidate)

    assert result.is_duplicate is True
    assert result.duplicate_of == article.id


def test_dedup_allows_distinct_title(session):
    candidate = ArticleCandidate(
        title="Different topic entirely",
        summary="",
        url="https://example.com/distinct",
        published_at=utcnow(),
        source_name="Example",
        source_feed="https://example.com/feed",
        category="technology",
    )

    result = ArticleDeduplicator().check(session, candidate)

    assert result.is_duplicate is False
    assert result.duplicate_of is None

