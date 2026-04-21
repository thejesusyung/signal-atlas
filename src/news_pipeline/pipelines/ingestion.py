"""Ingestion pipeline — standalone ECS Fargate entrypoint.

Replaces ingestion_dag.py. Runs the full ingestion cycle and then triggers
the extraction ECS task via boto3 (replacing TriggerDagRunOperator).

Required env vars for ECS trigger:
  ECS_CLUSTER              — e.g. pet-signal-atlas
  ECS_EXTRACTION_TASK_DEF  — e.g. pet-signal-atlas-extraction:1
  ECS_SUBNET_IDS           — comma-separated subnet IDs
  ECS_SECURITY_GROUP_ID    — single security group ID
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from news_pipeline.contracts import ArticleCandidate
from news_pipeline.db.session import session_scope
from news_pipeline.ingestion.dedup import ArticleDeduplicator
from news_pipeline.ingestion.rss import RSSFeedParser
from news_pipeline.ingestion.scraper import ArticleScraper
from news_pipeline.services.article_service import insert_article
from news_pipeline.tracking.experiment import log_dict_artifact, log_metrics, tracked_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)


def run() -> None:
    LOGGER.info("Ingestion pipeline starting")

    candidates = _fetch_candidates()
    LOGGER.info("Fetched %d RSS candidates", len(candidates))

    stats = _persist_articles(candidates)
    LOGGER.info(
        "Ingestion complete: found=%d new=%d scraped=%d duplicates=%d failed=%d",
        stats["articles_found"],
        stats["new_articles"],
        stats["scraped_articles"],
        stats["duplicates_skipped"],
        stats["failed_articles"],
    )

    _log_stats(stats)
    _trigger_extraction()


def _fetch_candidates() -> list[dict]:
    parser = RSSFeedParser()
    return [_serialize_candidate(item) for item in parser.parse_all()]


def _persist_articles(candidate_payloads: list[dict]) -> dict:
    deduplicator = ArticleDeduplicator()
    scraper = ArticleScraper()
    new_articles = 0
    duplicates = 0
    scraped = 0
    failed = 0

    for payload in candidate_payloads:
        try:
            candidate = _deserialize_candidate(payload)
            with session_scope() as session:
                duplicate = deduplicator.check(session, candidate)
                if duplicate.is_duplicate:
                    duplicates += 1
                    continue

                scrape_result = scraper.scrape(candidate.url)
                insert_article(session, candidate, scrape_result, duplicate.duplicate_of)

            if scrape_result.success:
                scraped += 1
            new_articles += 1
        except Exception:
            failed += 1
            LOGGER.exception("Failed to ingest candidate %s", payload.get("url", "<unknown>"))

    return {
        "articles_found": len(candidate_payloads),
        "duplicates_skipped": duplicates,
        "new_articles": new_articles,
        "scraped_articles": scraped,
        "failed_articles": failed,
    }


def _log_stats(stats: dict) -> None:
    with tracked_run(
        experiment_name="ingestion_monitoring",
        run_name="ingestion_pipeline",
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


def _trigger_extraction() -> None:
    cluster = os.environ.get("ECS_CLUSTER", "")
    task_def = os.environ.get("ECS_EXTRACTION_TASK_DEF", "")
    subnets_raw = os.environ.get("ECS_SUBNET_IDS", "")
    security_group = os.environ.get("ECS_SECURITY_GROUP_ID", "")

    if not all([cluster, task_def, subnets_raw, security_group]):
        LOGGER.warning(
            "ECS trigger env vars not set (ECS_CLUSTER, ECS_EXTRACTION_TASK_DEF, "
            "ECS_SUBNET_IDS, ECS_SECURITY_GROUP_ID) — skipping extraction trigger"
        )
        return

    import boto3

    subnets = [s.strip() for s in subnets_raw.split(",") if s.strip()]
    ecs = boto3.client("ecs")
    response = ecs.run_task(
        cluster=cluster,
        taskDefinition=task_def,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": subnets,
                "securityGroups": [security_group],
                "assignPublicIp": "ENABLED",
            }
        },
    )
    tasks = response.get("tasks", [])
    failures = response.get("failures", [])
    if failures:
        LOGGER.error("Failed to trigger extraction task: %s", failures)
    else:
        task_arn = tasks[0]["taskArn"] if tasks else "<unknown>"
        LOGGER.info("Triggered extraction ECS task: %s", task_arn)


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


if __name__ == "__main__":
    run()
