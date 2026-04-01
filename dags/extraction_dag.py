from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

import mlflow

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
            mlflow.log_table(
                data={
                    "article_id":  [r["article_id"] for r in processed],
                    "status":      [r["status"]     for r in processed],
                    "entities":    [r["entities"]   for r in processed],
                    "topics":      [r["topics"]     for r in processed],
                    "tokens_used": [r["tokens"]     for r in processed],
                },
                artifact_file="extraction_batch.json",
            )

    @task
    def embed_articles() -> dict:
        """Embed all extracted articles that have no embedding yet."""
        import numpy as np
        from sqlalchemy import select
        from news_pipeline.db.models import RawArticle
        from news_pipeline.embeddings.encoder import encode_texts, vector_to_db

        settings = get_settings()
        with session_scope() as session:
            articles = session.scalars(
                select(RawArticle)
                .where(
                    RawArticle.processing_status == ProcessingStatus.extracted,
                    RawArticle.embedding.is_(None),
                    RawArticle.cleaned_text.isnot(None),
                )
                .order_by(RawArticle.ingested_at.asc())
                .limit(settings.extraction_batch_size * 5)
            ).all()

            if not articles:
                return {"embedded": 0}

            texts = [a.cleaned_text or a.summary for a in articles]
            vectors = encode_texts(texts, batch_size=settings.embedding_batch_size)
            for article, vec in zip(articles, vectors):
                article.embedding = vector_to_db(vec)

        LOGGER.info("Embedded %d articles", len(articles))
        return {"embedded": len(articles)}

    @task
    def cluster_articles() -> dict:
        """Re-cluster all embedded articles and update cluster assignments."""
        import numpy as np
        from sqlalchemy import select, update
        from news_pipeline.db.models import RawArticle
        from news_pipeline.embeddings.encoder import vector_from_db
        from news_pipeline.embeddings.clustering import NOISE_LABEL, cluster_embeddings, label_cluster

        settings = get_settings()
        with session_scope() as session:
            rows = session.execute(
                select(RawArticle.id, RawArticle.embedding, RawArticle.title)
                .where(
                    RawArticle.embedding.isnot(None),
                    RawArticle.processing_status == ProcessingStatus.extracted,
                )
            ).all()

            if len(rows) < settings.embedding_min_cluster_size:
                LOGGER.info("Too few embedded articles (%d) to cluster; skipping", len(rows))
                return {"clusters": 0, "articles_clustered": 0}

            ids = [r.id for r in rows]
            titles = [r.title for r in rows]
            matrix = np.array([vector_from_db(r.embedding) for r in rows], dtype=np.float32)
            labels = cluster_embeddings(matrix, settings.embedding_min_cluster_size)

            cluster_titles: dict[int, list[str]] = {}
            for title, label in zip(titles, labels):
                if label != NOISE_LABEL:
                    cluster_titles.setdefault(int(label), []).append(title)

            cluster_label_map = {
                cid: label_cluster(cid, ctitles) for cid, ctitles in cluster_titles.items()
            }

            for article_id, label in zip(ids, labels):
                label_int = int(label)
                session.execute(
                    update(RawArticle)
                    .where(RawArticle.id == article_id)
                    .values(
                        semantic_cluster_id=label_int if label_int != NOISE_LABEL else None,
                        cluster_label=cluster_label_map.get(label_int, "unclustered"),
                    )
                )

        n_clusters = len(set(int(l) for l in labels if int(l) != NOISE_LABEL))
        LOGGER.info("Clustered %d articles into %d clusters", len(rows), n_clusters)
        return {"clusters": n_clusters, "articles_clustered": len(rows)}

    @task(pool="llm_pool")
    def generate_signals() -> dict:
        """Detect anomalous entity/topic velocity and generate LLM briefs."""
        from news_pipeline.signals.detector import detect_and_persist_signals

        settings = get_settings()
        provider = GroqProvider()
        with session_scope() as session:
            with tracked_run(
                experiment_name=settings.mlflow_experiment_signals,
                run_name="signal_detection",
                params={"provider": provider.provider_name},
                tags={"tracking_scope": "signal_detection", "dag_id": "extraction_dag"},
            ):
                signals = detect_and_persist_signals(session, provider)
                log_metrics({"signals_generated": len(signals)})

        LOGGER.info("Generated %d signals", len(signals))
        return {"signals_generated": len(signals)}

    @task
    def register_prompts() -> None:
        from news_pipeline.tracking.prompt_registry import register_all_prompts
        register_all_prompts()

    registered = register_prompts()
    article_ids = fetch_pending_article_ids()
    processed = process_article.expand(article_id=article_ids)
    logged = log_batch(processed)
    embedded = embed_articles()
    clustered = cluster_articles()
    signaled = generate_signals()

    registered >> article_ids
    logged >> embedded >> clustered >> signaled


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
