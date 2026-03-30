#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_pipeline.config import get_settings
from news_pipeline.contracts import EntityRecord
from news_pipeline.db.models import RawArticle
from news_pipeline.db.session import SessionLocal
from news_pipeline.extraction.entity_extractor import EntityExtractor
from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.llm.prompts import ENTITY_EXTRACTION_PROMPT, JSON_REPAIR_PROMPT
from news_pipeline.llm.provider import LLMTraceContext
from news_pipeline.tracking.experiment import log_dict_artifact, log_metrics, tracked_run
from news_pipeline.utils import (
    DEFAULT_LLM_ARTICLE_TEXT_CHARS,
    DEFAULT_LLM_SUMMARY_TEXT_CHARS,
    choose_article_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a batch of direct entity-extraction calls against real articles."
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of articles to evaluate.")
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=80,
        help="Minimum article word count to include in the sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/assessment_reports",
        help="Directory for the saved assessment JSON report.",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip MLflow tracking for host-local assessment runs.",
    )
    return parser.parse_args()


def load_articles(limit: int, min_word_count: int) -> list[RawArticle]:
    session = SessionLocal()
    try:
        statement = (
            select(RawArticle)
            .where(RawArticle.full_text.is_not(None))
            .where(RawArticle.word_count.is_not(None))
            .where(RawArticle.word_count >= min_word_count)
            .order_by(desc(RawArticle.published_at), desc(RawArticle.ingested_at))
            .limit(limit)
        )
        return list(session.execute(statement).scalars())
    finally:
        session.close()


def entity_to_dict(record: EntityRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "type": record.entity_type,
        "role": record.role,
        "confidence": record.confidence,
    }


def print_result(index: int, total: int, result: dict[str, Any]) -> None:
    status = "OK" if result["success"] else "FAIL"
    print(
        (
            f"[{index:02d}/{total:02d}] {status} "
            f"latency={result['latency_ms']}ms "
            f"tokens={result['tokens_used']} "
            f"raw_json={result['raw_json_valid']} "
            f"repair={result['used_repair']} "
            f"entities={result['entity_count']} "
            f"title={result['title']}"
        ),
        flush=True,
    )
    if result["success"]:
        preview = "; ".join(
            f"{item['name']} ({item['type']}, {item['role']}, {item['confidence']:.2f})"
            for item in result["entities"][:5]
        )
        print(f"  {preview or '[no entities]'}", flush=True)
    else:
        print(f"  error={result['error_message']}", flush=True)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    articles = load_articles(limit=args.limit, min_word_count=args.min_word_count)
    if not articles:
        print("No articles matched the assessment filters.", flush=True)
        return 1

    provider = GroqProvider()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"entity_assessment_{timestamp}.json"

    batch_results: list[dict[str, Any]] = []
    latencies: list[int] = []
    token_totals: list[int] = []
    entity_counts: list[int] = []
    repair_count = 0
    raw_json_success_count = 0
    empty_entity_count = 0

    print(
        json.dumps(
            {
                "provider": "groq",
                "model": settings.groq_model,
                "rpm_limit": settings.llm_requests_per_minute,
                "rate_limit_backend": settings.llm_rate_limit_backend,
                "sample_size": len(articles),
                "min_word_count": args.min_word_count,
            },
            indent=2,
        ),
        flush=True,
    )

    run_context = nullcontext()
    if not args.skip_mlflow:
        run_context = tracked_run(
            settings.mlflow_experiment_extraction,
            run_name=f"entity_batch_assessment_{timestamp}",
            params={
                "assessment_sample_size": len(articles),
                "assessment_min_word_count": args.min_word_count,
                "assessment_model": settings.groq_model,
            },
            tags={"tracking_scope": "entity_batch_assessment"},
        )

    with run_context:
        for index, article in enumerate(articles, start=1):
            article_text = choose_article_text(
                article.full_text,
                article.summary,
                article.title,
                max_chars=DEFAULT_LLM_ARTICLE_TEXT_CHARS,
                summary_max_chars=DEFAULT_LLM_SUMMARY_TEXT_CHARS,
            )
            system_prompt, prompt = ENTITY_EXTRACTION_PROMPT.render(
                title=article.title,
                article_text=article_text,
            )
            result: dict[str, Any] = {
                "article_id": str(article.id),
                "title": article.title,
                "word_count": article.word_count,
                "prompt_version": ENTITY_EXTRACTION_PROMPT.version,
                "success": False,
                "raw_json_valid": False,
                "used_repair": False,
                "entity_count": 0,
                "entities": [],
                "latency_ms": 0,
                "tokens_used": 0,
                "error_message": None,
            }

            try:
                response = provider.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    trace_context=LLMTraceContext(
                        operation="entity_extraction_assessment",
                        article_id=str(article.id),
                        prompt_version=ENTITY_EXTRACTION_PROMPT.version,
                        article_title=article.title,
                    ),
                )
                result["latency_ms"] = response.latency_ms
                result["tokens_used"] = response.tokens_used

                try:
                    records = EntityExtractor._parse_entities(response.text)
                    result["raw_json_valid"] = True
                except Exception:
                    repair_count += 1
                    result["used_repair"] = True
                    repair_system_prompt, repair_prompt = JSON_REPAIR_PROMPT.render(
                        broken_output=response.text
                    )
                    repaired = provider.complete(
                        prompt=repair_prompt,
                        system_prompt=repair_system_prompt,
                        temperature=0.0,
                        trace_context=LLMTraceContext(
                            operation="entity_json_repair_assessment",
                            article_id=str(article.id),
                            prompt_version=JSON_REPAIR_PROMPT.version,
                            article_title=article.title,
                        ),
                    )
                    result["latency_ms"] += repaired.latency_ms
                    result["tokens_used"] += repaired.tokens_used
                    records = EntityExtractor._parse_entities(repaired.text)

                if result["raw_json_valid"]:
                    raw_json_success_count += 1

                result["success"] = True
                result["entities"] = [entity_to_dict(record) for record in records]
                result["entity_count"] = len(result["entities"])

                if result["entity_count"] == 0:
                    empty_entity_count += 1

                latencies.append(int(result["latency_ms"]))
                token_totals.append(int(result["tokens_used"]))
                entity_counts.append(int(result["entity_count"]))
            except Exception as error:
                result["error_message"] = str(error)

            batch_results.append(result)
            print_result(index, len(articles), result)

        success_count = sum(1 for item in batch_results if item["success"])
        failure_count = len(batch_results) - success_count
        summary = {
            "sample_size": len(batch_results),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": round(success_count / len(batch_results), 4),
            "raw_json_success_count": raw_json_success_count,
            "raw_json_success_rate": round(raw_json_success_count / len(batch_results), 4),
            "repair_count": repair_count,
            "repair_rate": round(repair_count / len(batch_results), 4),
            "empty_entity_count": empty_entity_count,
            "empty_entity_rate": round(empty_entity_count / len(batch_results), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "avg_tokens_used": round(sum(token_totals) / len(token_totals), 2)
            if token_totals
            else 0.0,
            "avg_entities_per_success": round(sum(entity_counts) / len(entity_counts), 2)
            if entity_counts
            else 0.0,
            "report_path": str(report_path),
        }

        report_payload = {"summary": summary, "results": batch_results}
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        if not args.skip_mlflow:
            log_metrics(
                {
                    "assessment_success_rate": summary["success_rate"],
                    "assessment_raw_json_success_rate": summary["raw_json_success_rate"],
                    "assessment_repair_rate": summary["repair_rate"],
                    "assessment_empty_entity_rate": summary["empty_entity_rate"],
                    "assessment_avg_latency_ms": summary["avg_latency_ms"],
                    "assessment_avg_tokens_used": summary["avg_tokens_used"],
                    "assessment_avg_entities_per_success": summary["avg_entities_per_success"],
                }
            )
            log_dict_artifact(
                report_payload,
                artifact_file=report_path.name,
                artifact_path="assessment_reports",
            )
        print(json.dumps({"summary": summary}, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
