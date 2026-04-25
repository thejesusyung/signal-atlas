"""Extraction pipeline — standalone ECS Fargate entrypoint.

Replaces extraction_dag.py. Processes all pending articles through LLM
extraction, then embeds, clusters, and generates signals.

LLM fan-out uses ThreadPoolExecutor. Rate limiting is handled by the
existing get_shared_rate_limiter (thread-safe, DB-backed advisory locks).
"""

from __future__ import annotations

import logging
import mlflow
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from uuid import UUID

from news_pipeline.config import get_settings
from news_pipeline.db.models import ExtractionRun, ProcessingStatus
from news_pipeline.db.session import SessionLocal, session_scope
from news_pipeline.extraction.errors import ExtractionStepError
from news_pipeline.extraction.entity_extractor import EntityExtractor
from news_pipeline.extraction.topic_extractor import TopicExtractor
from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.services.article_service import get_article, get_pending_articles
from news_pipeline.tracking.experiment import log_dict_artifact, log_metrics, tracked_run
from news_pipeline.tracking.prompt_registry import register_all_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)


def run() -> None:
    LOGGER.info("Extraction pipeline starting")

    register_all_prompts()

    settings = get_settings()
    with session_scope() as session:
        articles = get_pending_articles(session, limit=settings.extraction_batch_size)
        article_ids = [str(a.id) for a in articles]

    LOGGER.info("Processing %d pending articles", len(article_ids))

    results = _process_articles_parallel(article_ids, settings.extraction_batch_size)
    _log_batch(results)

    embed_stats = _embed_articles()
    cluster_stats = _cluster_articles()
    signal_stats = _generate_signals()

    LOGGER.info(
        "Extraction pipeline complete: processed=%d embedded=%d clusters=%d signals=%d",
        len(results),
        embed_stats["embedded"],
        cluster_stats["clusters"],
        signal_stats["signals_generated"],
    )


def _process_articles_parallel(article_ids: list[str], max_workers: int) -> list[dict]:
    results: list[dict] = []
    # Max workers capped to avoid overwhelming the LLM rate limiter queue
    workers = min(max_workers, 10)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_article, aid): aid for aid in article_ids}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                aid = futures[future]
                LOGGER.exception("Unhandled error processing article %s", aid)
                results.append({"article_id": aid, "status": "failed", "entities": 0, "topics": 0, "tokens": 0})
    return results


def _process_article(article_id: str) -> dict:
    settings = get_settings()
    provider = GroqProvider()
    entity_extractor = EntityExtractor(provider)
    topic_extractor = TopicExtractor(provider)
    session = SessionLocal()

    try:
        article = get_article(session, UUID(article_id))
        if article is None:
            return {"article_id": article_id, "status": "missing", "entities": 0, "topics": 0, "tokens": 0}

        with tracked_run(
            experiment_name=settings.mlflow_experiment_extraction,
            run_name=f"article_{article_id}",
            params={"article_id": article_id, "provider_name": provider.provider_name},
            tags={"tracking_scope": "article_extraction", "article_id": article_id},
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
                        "entity_names": [r.name for r in entities],
                        "topic_names": [a.topic_name for a in topics],
                    },
                    "article_result.json",
                )
                return result
            except Exception:
                log_metrics({"article_success": 0})
                log_dict_artifact(
                    {"article_id": article_id, "title": article.title, "status": "failed"},
                    "article_failure.json",
                )
                raise
    except ExtractionStepError as error:
        session.rollback()
        _persist_failed_run(article_id, error)
        _mark_article_failed(article_id)
        LOGGER.exception("Extraction failed for article %s", article_id)
        return {"article_id": article_id, "status": "failed", "entities": 0, "topics": 0, "tokens": 0}
    except Exception:
        session.rollback()
        _mark_article_failed(article_id)
        LOGGER.exception("Extraction failed for article %s", article_id)
        return {"article_id": article_id, "status": "failed", "entities": 0, "topics": 0, "tokens": 0}
    finally:
        session.close()


def _log_batch(results: list[dict]) -> None:
    processed = [r for r in results if r]
    with tracked_run(
        experiment_name="extraction_monitoring",
        run_name="extraction_pipeline",
        params={"provider": "groq"},
    ):
        log_metrics(
            {
                "articles_processed": len(processed),
                "articles_extracted": sum(1 for r in processed if r["status"] == "extracted"),
                "articles_failed": sum(1 for r in processed if r["status"] == "failed"),
                "entity_count": sum(r["entities"] for r in processed),
                "topic_count": sum(r["topics"] for r in processed),
                "tokens_used": sum(r["tokens"] for r in processed),
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


def _embed_articles() -> dict:
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


def _cluster_articles() -> dict:
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


def _generate_signals() -> dict:
    from news_pipeline.signals.detector import detect_and_persist_signals
    from news_pipeline.llm.openrouter_client import OpenRouterProvider

    settings = get_settings()
    provider = OpenRouterProvider()
    with session_scope() as session:
        with tracked_run(
            experiment_name=settings.mlflow_experiment_signals,
            run_name="signal_detection",
            params={"provider": provider.provider_name},
            tags={"tracking_scope": "signal_detection"},
        ):
            signals = detect_and_persist_signals(session, provider)
            log_metrics({"signals_generated": len(signals)})

    LOGGER.info("Generated %d signals", len(signals))
    return {"signals_generated": len(signals)}


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


if __name__ == "__main__":
    run()
