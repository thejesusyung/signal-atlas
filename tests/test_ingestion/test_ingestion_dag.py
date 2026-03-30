from __future__ import annotations

import importlib
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from news_pipeline.contracts import ScrapeResult
from news_pipeline.db.models import Base, RawArticle


def test_persist_articles_keeps_successful_inserts_when_one_candidate_fails(monkeypatch, tmp_path):
    airflow_home = tmp_path / "airflow-home"
    logs = airflow_home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIRFLOW_HOME", str(airflow_home))
    monkeypatch.setenv("AIRFLOW__LOGGING__BASE_LOG_FOLDER", str(logs))
    monkeypatch.setenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", f"sqlite:///{airflow_home / 'airflow.db'}")
    monkeypatch.setenv("AIRFLOW__CORE__DAGS_FOLDER", str(Path(__file__).resolve().parents[2] / "dags"))
    monkeypatch.setenv("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

    ingestion_dag = importlib.import_module("dags.ingestion_dag")

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr("news_pipeline.db.session.SessionLocal", SessionLocal)

    def scrape(self, url: str) -> ScrapeResult:
        if url.endswith("/bad"):
            raise RuntimeError("boom")
        return ScrapeResult("body text", 2, "beautifulsoup", True, None)

    monkeypatch.setattr("news_pipeline.ingestion.scraper.ArticleScraper.scrape", scrape)

    payloads = [
        {
            "title": "Good article",
            "summary": "",
            "url": "https://example.com/good",
            "published_at": None,
            "source_name": "Example",
            "source_feed": "https://example.com/feed",
            "category": "technology",
        },
        {
            "title": "Bad article",
            "summary": "",
            "url": "https://example.com/bad",
            "published_at": None,
            "source_name": "Example",
            "source_feed": "https://example.com/feed",
            "category": "technology",
        },
    ]

    stats = ingestion_dag._persist_articles(payloads)

    session = SessionLocal()
    try:
        articles = session.query(RawArticle).order_by(RawArticle.url.asc()).all()
        assert stats["new_articles"] == 1
        assert stats["failed_articles"] == 1
        assert [article.url for article in articles] == ["https://example.com/good"]
    finally:
        session.close()
        engine.dispose()
