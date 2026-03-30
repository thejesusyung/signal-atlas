from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from airflow.decorators import dag, task

from news_pipeline.config import get_settings
from news_pipeline.db.models import ExtractionRun, ProcessingStatus
from news_pipeline.db.session import SessionLocal, session_scope
from news_pipeline.extraction.errors import ExtractionStepError
from news_pipeline.extraction.entity_extractor import EntityExtractor
from news_pipeline.extraction.topic_extractor import TopicExtractor
from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.services.article_service import get_article, get_pending_articles
from news_pipeline.tracking.experiment import log_dict_artifact, log_metrics, tracked_run

LOGGER = logging.getLogger(__name__)


def _failure_callback(context) -> None:
    LOGGER.exception("Task failed in extraction DAG", extra={"context": str(context)})


@dag(
    dag_id="extraction_dag",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 1, "on_failure_callback": _failure_callback},
    tags=["news", "extraction"],
)
def build_extraction_dag():
    @task
    def fetch_pending_article_ids() -> list[str]:
        settings = get_settings()
        with session_scope() as session:
            articles = get_pending_articles(session, limit=settings.extraction_batch_size)
            return [str(article.id) for article in articles]

    @task(pool="llm_pool")
    def process_article(article_id: str) -> dict:
        return _process_article(article_id)

    @task
    def log_batch(results: list[dict]) -> None:
        processed = [item for item in results if item]
        with tracked_run(
            experiment_name="extraction_monitoring",
            run_name="extraction_dag",
            params={"provider": "groq"},
        ):
            log_metrics(
                {
                    "articles_processed": len(processed),
                    "articles_extracted": sum(1 for item in processed if item["status"] == "extracted"),
                    "articles_failed": sum(1 for item in processed if item["status"] == "failed"),
                    "entity_count": sum(item["entities"] for item in processed),
                    "topic_count": sum(item["topics"] for item in processed),
                    "tokens_used": sum(item["tokens"] for item in processed),
                }
            )
            log_dict_artifact({"results": processed}, "extraction_batch.json")

    article_ids = fetch_pending_article_ids()
    processed = process_article.expand(article_id=article_ids)
    log_batch(processed)


def _process_article(article_id: str) -> dict:
    settings = get_settings()
    provider = GroqProvider()
    entity_extractor = EntityExtractor(provider)
    topic_extractor = TopicExtractor(provider)
    session = SessionLocal()
    article = None

    try:
        article = get_article(session, UUID(article_id))
        if article is None:
            return {"article_id": article_id, "status": "missing", "entities": 0, "topics": 0, "tokens": 0}

        with tracked_run(
            experiment_name=settings.mlflow_experiment_extraction,
            run_name=f"article_{article_id}",
            params={
                "article_id": article_id,
                "provider_name": provider.provider_name,
            },
            tags={
                "tracking_scope": "article_extraction",
                "dag_id": "extraction_dag",
                "article_id": article_id,
            },
        ):
            try:
                entities = entity_extractor.extract_for_article(session, article)
                topics = topic_extractor.extract_for_article(session, article)
                article.processing_status = ProcessingStatus.extracted
                session.commit()
                tokens = sum(run.tokens_used for run in article.extraction_runs[-2:])
                result = {
                    "article_id": article_id,
                    "status": "extracted",
                    "entities": len(entities),
                    "topics": len(topics),
                    "tokens": tokens,
                }
                log_metrics(
                    {
                        "article_success": 1,
                        "entity_count": len(entities),
                        "topic_count": len(topics),
                        "tokens_used": tokens,
                    }
                )
                log_dict_artifact(
                    {
                        "article_id": article_id,
                        "title": article.title,
                        "status": "extracted",
                        "entity_names": [record.name for record in entities],
                        "topic_names": [assignment.topic_name for assignment in topics],
                    },
                    "article_result.json",
                )
                return result
            except Exception as error:
                log_metrics({"article_success": 0})
                log_dict_artifact(
                    {
                        "article_id": article_id,
                        "title": article.title,
                        "status": "failed",
                        "error": str(error),
                    },
                    "article_failure.json",
                )
                raise
    except ExtractionStepError as error:
        session.rollback()
        _persist_failed_run(article_id, error)
        _mark_article_failed(article_id)
        LOGGER.exception("Extraction failed for article %s: %s", article_id, error)
        raise
    except Exception as error:
        session.rollback()
        _mark_article_failed(article_id)
        LOGGER.exception("Extraction failed for article %s: %s", article_id, error)
        raise
    finally:
        session.close()


def _mark_article_failed(article_id: str) -> None:
    with session_scope() as session:
        article = get_article(session, UUID(article_id))
        if article is not None:
            article.processing_status = ProcessingStatus.failed


def _persist_failed_run(article_id: str, error: ExtractionStepError) -> None:
    with session_scope() as session:
        session.add(
            ExtractionRun(
                article_id=UUID(article_id),
                run_type=error.run_type,
                llm_provider=error.llm_provider,
                model_name=error.model_name,
                prompt_version=error.prompt_version,
                tokens_used=error.tokens_used,
                latency_ms=error.latency_ms,
                success=False,
                error_message=error.error_message,
            )
        )


extraction_dag = build_extraction_dag()
