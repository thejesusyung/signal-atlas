#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_pipeline.db.models import RawArticle
from news_pipeline.db.session import session_scope
from news_pipeline.utils import clean_article_text


def main() -> int:
    updated = 0
    with session_scope() as session:
        articles = session.scalars(
            select(RawArticle).where(RawArticle.full_text.is_not(None)).order_by(RawArticle.ingested_at.asc())
        ).all()
        for article in articles:
            cleaned = clean_article_text(article.full_text)
            if article.cleaned_text != cleaned:
                article.cleaned_text = cleaned or None
                updated += 1
    print(f"updated_articles={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
