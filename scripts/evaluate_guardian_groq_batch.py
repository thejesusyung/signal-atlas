#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_pipeline.config import get_settings
from news_pipeline.extraction.entity_extractor import EntityExtractor
from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.llm.prompts import ENTITY_EXTRACTION_PROMPT, JSON_REPAIR_PROMPT
from news_pipeline.llm.provider import LLMTraceContext
from news_pipeline.utils import (
    DEFAULT_LLM_ARTICLE_TEXT_CHARS,
    DEFAULT_LLM_SUMMARY_TEXT_CHARS,
    choose_article_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Guardian articles and evaluate Groq entity extraction on them."
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of Guardian articles to fetch.")
    parser.add_argument(
        "--section",
        default="",
        help="Optional Guardian section filter, for example politics or world.",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Optional Guardian search query.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/assessment_reports",
        help="Directory for fetched samples and assessment reports.",
    )
    parser.add_argument(
        "--guardian-api-key",
        default="test",
        help="Guardian API key. Defaults to the public test key.",
    )
    return parser.parse_args()


def fetch_guardian_articles(limit: int, section: str, query: str, api_key: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "page-size": limit,
        "order-by": "newest",
        "show-fields": "trailText,headline,bodyText,byline,thumbnail,shortUrl",
        "api-key": api_key,
    }
    if section:
        params["section"] = section
    if query:
        params["q"] = query

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get("https://content.guardianapis.com/search", params=params)
        response.raise_for_status()
        payload = response.json()
    return payload


def canonical_articles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("response", {}).get("results", [])
    articles: list[dict[str, Any]] = []
    for item in results:
        fields = item.get("fields", {}) or {}
        body_text = fields.get("bodyText") or ""
        article = {
            "source_name": "The Guardian",
            "provider": "guardian",
            "provider_item_id": item.get("id", ""),
            "content_type": item.get("type", ""),
            "section_id": item.get("sectionId", ""),
            "section_name": item.get("sectionName", ""),
            "title": item.get("webTitle", ""),
            "summary": fields.get("trailText", ""),
            "headline": fields.get("headline", ""),
            "url": item.get("webUrl", ""),
            "short_url": fields.get("shortUrl", ""),
            "thumbnail": fields.get("thumbnail", ""),
            "byline": fields.get("byline", ""),
            "published_at": item.get("webPublicationDate"),
            "full_text": body_text,
            "word_count": len(body_text.split()) if body_text else 0,
        }
        articles.append(article)
    return articles


def preview_entities(records: list[dict[str, Any]], limit: int = 5) -> str:
    if not records:
        return "[no entities]"
    return "; ".join(
        f"{item['name']} ({item['type']}, {item['role']}, {item['confidence']:.2f})"
        for item in records[:limit]
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()
    provider = GroqProvider()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"guardian_sample_{timestamp}.json"
    report_path = output_dir / f"guardian_groq_entity_assessment_{timestamp}.json"

    payload = fetch_guardian_articles(
        limit=args.limit,
        section=args.section,
        query=args.query,
        api_key=args.guardian_api_key,
    )
    sample_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    articles = canonical_articles(payload)
    if not articles:
        print("No Guardian articles were returned.", flush=True)
        return 1

    results: list[dict[str, Any]] = []
    latencies: list[int] = []
    tokens_used: list[int] = []
    entity_counts: list[int] = []
    raw_json_ok = 0
    repairs = 0

    print(
        json.dumps(
            {
                "source": "guardian",
                "model": settings.groq_model,
                "sample_size": len(articles),
                "section": args.section or None,
                "query": args.query or None,
                "sample_path": str(sample_path.relative_to(ROOT)),
            },
            indent=2,
        ),
        flush=True,
    )

    for index, article in enumerate(articles, start=1):
        article_text = choose_article_text(
            article["full_text"],
            article["summary"],
            article["title"],
            max_chars=DEFAULT_LLM_ARTICLE_TEXT_CHARS,
            summary_max_chars=DEFAULT_LLM_SUMMARY_TEXT_CHARS,
        )
        system_prompt, prompt = ENTITY_EXTRACTION_PROMPT.render(
            title=article["title"],
            article_text=article_text,
        )

        result: dict[str, Any] = {
            "provider_item_id": article["provider_item_id"],
            "content_type": article["content_type"],
            "section_name": article["section_name"],
            "title": article["title"],
            "url": article["url"],
            "word_count": article["word_count"],
            "llm_input_chars": len(article_text),
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
                    operation="guardian_entity_extraction_assessment",
                    article_id=article["provider_item_id"],
                    prompt_version=ENTITY_EXTRACTION_PROMPT.version,
                    article_title=article["title"],
                ),
            )
            result["latency_ms"] = response.latency_ms
            result["tokens_used"] = response.tokens_used

            try:
                records = EntityExtractor._parse_entities(response.text)
                result["raw_json_valid"] = True
                raw_json_ok += 1
            except Exception:
                repairs += 1
                result["used_repair"] = True
                repair_system_prompt, repair_prompt = JSON_REPAIR_PROMPT.render(
                    broken_output=response.text
                )
                repaired = provider.complete(
                    prompt=repair_prompt,
                    system_prompt=repair_system_prompt,
                    temperature=0.0,
                    trace_context=LLMTraceContext(
                        operation="guardian_entity_json_repair_assessment",
                        article_id=article["provider_item_id"],
                        prompt_version=JSON_REPAIR_PROMPT.version,
                        article_title=article["title"],
                    ),
                )
                result["latency_ms"] += repaired.latency_ms
                result["tokens_used"] += repaired.tokens_used
                records = EntityExtractor._parse_entities(repaired.text)

            result["success"] = True
            result["entities"] = [
                {
                    "name": record.name,
                    "type": record.entity_type,
                    "role": record.role,
                    "confidence": record.confidence,
                }
                for record in records
            ]
            result["entity_count"] = len(result["entities"])

            latencies.append(int(result["latency_ms"]))
            tokens_used.append(int(result["tokens_used"]))
            entity_counts.append(int(result["entity_count"]))

            print(
                f"[{index:02d}/{len(articles):02d}] OK "
                f"tokens={result['tokens_used']} latency={result['latency_ms']}ms "
                f"repair={result['used_repair']} entities={result['entity_count']} "
                f"title={article['title']}",
                flush=True,
            )
            print(f"  {preview_entities(result['entities'])}", flush=True)
        except Exception as error:
            result["error_message"] = str(error)
            print(
                f"[{index:02d}/{len(articles):02d}] FAIL title={article['title']}",
                flush=True,
            )
            print(f"  error={result['error_message']}", flush=True)

        results.append(result)

    successes = [item for item in results if item["success"]]
    summary = {
        "source": "guardian",
        "model": settings.groq_model,
        "sample_size": len(results),
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "success_rate": round(len(successes) / len(results), 4),
        "raw_json_success_count": raw_json_ok,
        "raw_json_success_rate": round(raw_json_ok / len(results), 4),
        "repair_count": repairs,
        "repair_rate": round(repairs / len(results), 4),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "avg_tokens_used": round(statistics.mean(tokens_used), 2) if tokens_used else 0.0,
        "avg_entities_per_success": round(statistics.mean(entity_counts), 2) if entity_counts else 0.0,
        "avg_word_count": round(
            statistics.mean(item["word_count"] for item in results),
            2,
        ),
        "avg_llm_input_chars": round(
            statistics.mean(item["llm_input_chars"] for item in results),
            2,
        ),
        "content_type_breakdown": {
            key: sum(1 for item in results if item["content_type"] == key)
            for key in sorted({item["content_type"] for item in results})
        },
    }
    report = {
        "summary": summary,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummary", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved sample to {sample_path}", flush=True)
    print(f"Saved report to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
