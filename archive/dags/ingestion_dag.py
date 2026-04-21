from __future__ import annotations

import logging
from datetime import datetime

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from news_pipeline.contracts import ArticleCandidate
from news_pipeline.db.session import session_scope
from news_pipeline.ingestion.dedup import ArticleDeduplicator
from news_pipeline.ingestion.rss import RSSFeedParser
from news_pipeline.ingestion.scraper import ArticleScraper
from news_pipeline.services.article_service import insert_article
from news_pipeline.tracking.experiment import log_dict_artifact, log_metrics, tracked_run

LOGGER = logging.getLogger(__name__)


def _failure_callback(context) -> None:
    LOGGER.exception("Task failed in ingestion DAG", extra={"context": str(context)})


@dag(
    dag_id="ingestion_dag",
    schedule="0 */2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 2, "on_failure_callback": _failure_callback},
    tags=["news", "ingestion"],
)
def build_ingestion_dag():
    @task
    def fetch_candidates() -> list[dict]:
        parser = RSSFeedParser()
        return [_serialize_candidate(item) for item in parser.parse_all()]

    @task
    def persist_articles(candidate_payloads: list[dict]) -> dict:
        return _persist_articles(candidate_payloads)

    @task
    def log_ingestion_stats(stats: dict) -> dict:
        with tracked_run(
            experiment_name="ingestion_monitoring",
            run_name="ingestion_dag",
            params={"source_type": "rss"},
        ):
            log_metrics(
                {
                    "articles_found": stats["articles_found"],
                    "duplicates_skipped": stats["duplicates_skipped"],
                    "new_articles": stats["new_articles"],
                    "scraped_articles": stats["scraped_articles"],
                }
            )
            log_dict_artifact(stats, "ingestion_stats.json")
        return stats

    candidates = fetch_candidates()
    ingestion_stats = persist_articles(candidates)
    logged = log_ingestion_stats(ingestion_stats)

    trigger_extraction = TriggerDagRunOperator(
        task_id="trigger_extraction",
        trigger_dag_id="extraction_dag",
        wait_for_completion=False,
    )

    logged >> trigger_extraction


def _persist_articles(candidate_payloads: list[dict]) -> dict:
    deduplicator = ArticleDeduplicator()
    scraper = ArticleScraper()
    new_articles = 0
    duplicates = 0
    scraped = 0
    failed = 0
    inserted_ids: list[str] = []

    for payload in candidate_payloads:
        try:
            candidate = _deserialize_candidate(payload)
            with session_scope() as session:
                duplicate = deduplicator.check(session, candidate)
                if duplicate.is_duplicate:
                    duplicates += 1
                    continue

                scrape_result = scraper.scrape(candidate.url)
                article = insert_article(session, candidate, scrape_result, duplicate.duplicate_of)
                article_id = str(article.id)
            if scrape_result.success:
                scraped += 1
            inserted_ids.append(article_id)
            new_articles += 1
        except Exception as error:  # pragma: no cover - defensive path
            failed += 1
            LOGGER.exception(
                "Failed to ingest article candidate %s: %s",
                payload.get("url", "<unknown>"),
                error,
            )

    return {
        "articles_found": len(candidate_payloads),
        "duplicates_skipped": duplicates,
        "new_articles": new_articles,
        "scraped_articles": scraped,
        "failed_articles": failed,
        "article_ids": inserted_ids,
    }


def _serialize_candidate(candidate: ArticleCandidate) -> dict:
    return {
        "title": candidate.title,
        "summary": candidate.summary,
        "url": candidate.url,
        "published_at": candidate.published_at.isoformat() if candidate.published_at else None,
        "source_name": candidate.source_name,
        "source_feed": candidate.source_feed,
        "category": candidate.category,
    }


def _deserialize_candidate(payload: dict) -> ArticleCandidate:
    return ArticleCandidate(
        title=payload["title"],
        summary=payload["summary"],
        url=payload["url"],
        published_at=datetime.fromisoformat(payload["published_at"]) if payload["published_at"] else None,
        source_name=payload["source_name"],
        source_feed=payload["source_feed"],
        category=payload["category"],
    )


ingestion_dag = build_ingestion_dag()
