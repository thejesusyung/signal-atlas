from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from news_pipeline.config import get_settings
from news_pipeline.contracts import ArticleCandidate
from news_pipeline.db.models import RawArticle
from news_pipeline.utils import normalize_title_for_dedup, title_similarity, utcnow


@dataclass(slots=True)
class DuplicateCheckResult:
    is_duplicate: bool
    duplicate_of: UUID | None = None
    matched_title: str | None = None
    similarity: float | None = None


class ArticleDeduplicator:
    def __init__(self, similarity_threshold: float | None = None, recent_hours: int | None = None) -> None:
        settings = get_settings()
        self.similarity_threshold = similarity_threshold or settings.dedup_title_similarity
        self.recent_hours = recent_hours or settings.dedup_recent_hours

    def check(self, session: Session, candidate: ArticleCandidate) -> DuplicateCheckResult:
        exact = session.scalar(select(RawArticle).where(RawArticle.url == candidate.url))
        if exact is not None:
            return DuplicateCheckResult(True, exact.id, exact.title, 1.0)

        cutoff = utcnow() - timedelta(hours=self.recent_hours)
        normalized_title = normalize_title_for_dedup(candidate.title)
        recent_articles = session.scalars(
            select(RawArticle).where(RawArticle.ingested_at >= cutoff)
        ).all()

        best_match: RawArticle | None = None
        best_score = 0.0
        for article in recent_articles:
            if article.normalized_title == normalized_title:
                return DuplicateCheckResult(True, article.id, article.title, 1.0)

            score = title_similarity(normalized_title, article.normalized_title)
            if score > best_score:
                best_score = score
                best_match = article

        if best_match and best_score >= self.similarity_threshold:
            return DuplicateCheckResult(True, best_match.id, best_match.title, best_score)
        return DuplicateCheckResult(False)

