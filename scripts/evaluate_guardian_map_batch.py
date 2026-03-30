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
from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.llm.prompts import JSON_REPAIR_PROMPT, parse_json_payload
from news_pipeline.llm.provider import LLMTraceContext
from news_pipeline.utils import (
    DEFAULT_LLM_ARTICLE_TEXT_CHARS,
    DEFAULT_LLM_SUMMARY_TEXT_CHARS,
    choose_article_text,
    normalize_entity_name,
)

MAP_THEME_LABELS = [
    "geopolitics",
    "conflict",
    "elections",
    "public_policy",
    "law_and_justice",
    "economy",
    "markets",
    "business",
    "technology",
    "science",
    "energy",
    "climate",
    "health",
    "infrastructure",
    "society",
]

THEME_ALIASES = {
    "politics": "public_policy",
    "policy": "public_policy",
    "regulation": "public_policy",
    "justice": "law_and_justice",
    "legal": "law_and_justice",
    "security": "geopolitics",
    "international": "geopolitics",
    "international_relations": "geopolitics",
    "economics": "economy",
    "finance": "markets",
    "environment": "climate",
    "sustainability": "climate",
    "innovation": "technology",
}

MIN_ENTITY_CONFIDENCE = 0.6

EVENT_TYPE_LABELS = [
    "military_action",
    "diplomatic_move",
    "policy_change",
    "regulatory_action",
    "court_case",
    "crime_incident",
    "election",
    "legislation",
    "labor_action",
    "corporate_action",
    "product_or_research",
    "market_movement",
    "disaster_or_weather",
    "public_health",
    "infrastructure",
    "other",
]

ISSUE_TAG_LABELS = [
    "sanctions",
    "migration",
    "inflation",
    "energy_prices",
    "social_media",
    "child_safety",
    "platform_regulation",
    "cybersecurity",
    "quantum_computing",
    "artificial_intelligence",
    "transport",
    "roads",
    "housing",
    "healthcare",
    "labor_rights",
    "elections",
    "war",
    "ceasefire",
    "trade",
    "consumer_prices",
    "climate",
    "waste",
    "public_safety",
    "sexual_violence",
    "charity_governance",
    "corporate_governance",
]

COUNTRY_ALIASES = {
    "united states": "us",
    "usa": "us",
    "u.s.": "us",
    "u.s": "us",
    "united kingdom": "uk",
    "britain": "uk",
    "england": "uk",
    "european union": "eu",
}

NEWS_SECTIONS = ["world", "politics", "business", "technology"]

MAP_EXTRACTION_SYSTEM_PROMPT = (
    "You structure news articles for graph-based topic mapping. "
    "Return JSON only. Never include prose outside JSON."
)

MAP_EXTRACTION_PROMPT_TEMPLATE = """
Extract a compact structured representation of this news article for downstream clustering and graph layout.

Return JSON with this shape:
{
  "story_kind": "hard_news",
  "event_type": "regulatory_action",
  "event_summary": "One sentence summarizing the concrete news event.",
  "primary_theme": "geopolitics",
  "secondary_themes": ["conflict", "public_policy"],
  "issue_tags": ["platform_regulation", "child_safety"],
  "countries": ["uk", "eu"],
  "geo_scope": "regional",
  "entities": [
    {
      "name": "Keir Starmer",
      "type": "person",
      "role": "actor",
      "confidence": 0.96
    }
  ],
  "relations": [
    {
      "subject": "European Commission",
      "predicate": "investigates",
      "object": "Snapchat"
    }
  ]
}

Allowed story_kind values:
- hard_news
- explainer
- analysis
- opinion
- review
- liveblog
- other

Allowed primary/secondary theme values:
{{ theme_labels }}

Allowed event_type values:
{{ event_type_labels }}

Allowed issue_tags values:
{{ issue_tag_labels }}

Allowed geo_scope values:
- local
- national
- regional
- global

Allowed entity type values:
- person
- company
- organization
- government_body
- location
- product
- media_title
- event
- sports_team
- weather_event
- vehicle_or_mission

Allowed role values:
- actor
- target
- quoted
- mentioned

Rules:
- Prefer concrete entities that help group related stories.
- Include 4 to 12 entities when possible.
- Use "government_body" for ministries, courts, police forces, armies, regulators, parliaments, and official agencies.
- Use "event" for wars, elections, strikes, summits, inquiries, court cases, disasters, and similar named or clearly defined events.
- Use "weather_event" for storms, cyclones, hurricanes, wildfires, and similar phenomena.
- Use "media_title" for albums, films, shows, books, and named creative works.
- Use "vehicle_or_mission" for ships, spacecraft, missions, and named transport systems.
- Do not invent entities not supported by the provided text.
- Prefer the most central entities, not every passing mention.
- Event summary must be specific and factual, not generic.
- issue_tags should be concrete issue labels, not broad themes.
- countries should use short normalized forms such as uk, us, eu, iran, israel when obvious.
- Include 1 to 5 relations when the article supports them.
- Relations should use short present-tense predicates such as investigates, attacks, announces, rejects, backs, warns, sues.

Article title: {{ title }}
Article section: {{ section_name }}
Article summary: {{ summary }}
Article text:
{{ article_text }}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Guardian news articles and evaluate a map-oriented Groq extraction prompt."
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of Guardian articles to fetch.")
    parser.add_argument(
        "--sections",
        default=",".join(NEWS_SECTIONS),
        help="Comma-separated Guardian sections to include.",
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


def fetch_guardian_articles(limit: int, sections: list[str], api_key: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "page-size": min(limit, 50),
        "order-by": "newest",
        "show-fields": "trailText,headline,bodyText,byline,thumbnail,shortUrl",
        "section": "|".join(sections),
        "type": "article",
        "api-key": api_key,
    }
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get("https://content.guardianapis.com/search", params=params)
        response.raise_for_status()
        return response.json()


def canonical_articles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("response", {}).get("results", [])
    articles: list[dict[str, Any]] = []
    for item in results:
        fields = item.get("fields", {}) or {}
        body_text = fields.get("bodyText") or ""
        if not body_text:
            continue
        articles.append(
            {
                "provider_item_id": item.get("id", ""),
                "content_type": item.get("type", ""),
                "section_id": item.get("sectionId", ""),
                "section_name": item.get("sectionName", ""),
                "title": item.get("webTitle", ""),
                "summary": fields.get("trailText", ""),
                "url": item.get("webUrl", ""),
                "published_at": item.get("webPublicationDate"),
                "full_text": body_text,
                "word_count": len(body_text.split()),
            }
        )
    return articles


def render_prompt(article: dict[str, Any]) -> tuple[str, str]:
    article_text = choose_article_text(
        article["full_text"],
        article["summary"],
        article["title"],
        max_chars=DEFAULT_LLM_ARTICLE_TEXT_CHARS,
        summary_max_chars=DEFAULT_LLM_SUMMARY_TEXT_CHARS,
    )
    prompt = MAP_EXTRACTION_PROMPT_TEMPLATE.replace("{{ theme_labels }}", ", ".join(MAP_THEME_LABELS))
    prompt = prompt.replace("{{ event_type_labels }}", ", ".join(EVENT_TYPE_LABELS))
    prompt = prompt.replace("{{ issue_tag_labels }}", ", ".join(ISSUE_TAG_LABELS))
    prompt = prompt.replace("{{ title }}", article["title"])
    prompt = prompt.replace("{{ section_name }}", article["section_name"])
    prompt = prompt.replace("{{ summary }}", article["summary"])
    prompt = prompt.replace("{{ article_text }}", article_text)
    return MAP_EXTRACTION_SYSTEM_PROMPT, prompt


def normalize_type(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    allowed = {
        "person",
        "company",
        "organization",
        "government_body",
        "location",
        "product",
        "media_title",
        "event",
        "sports_team",
        "weather_event",
        "vehicle_or_mission",
    }
    return normalized if normalized in allowed else None


def normalize_role(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"actor", "target", "quoted", "mentioned"}:
        return normalized
    return "mentioned"


def normalize_theme(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    normalized = THEME_ALIASES.get(normalized, normalized)
    if normalized in MAP_THEME_LABELS:
        return normalized
    return None


def normalize_event_type(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in EVENT_TYPE_LABELS else "other"


def normalize_issue_tag(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in ISSUE_TAG_LABELS:
        return normalized
    return None


def normalize_country(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    normalized = COUNTRY_ALIASES.get(normalized, normalized)
    if not normalized or len(normalized) > 24:
        return None
    return normalized


def normalize_geo_scope(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"local", "national", "regional", "global"} else "national"


def parse_map_payload(raw_text: str) -> dict[str, Any]:
    payload = parse_json_payload(raw_text)
    story_kind = str(payload.get("story_kind", "other")).strip().lower()
    if story_kind not in {"hard_news", "explainer", "analysis", "opinion", "review", "liveblog", "other"}:
        raise ValueError(f"Unsupported story_kind: {story_kind}")

    primary_theme = normalize_theme(payload.get("primary_theme"))
    if primary_theme is None:
        raise ValueError(f"Unsupported primary_theme: {primary_theme}")
    event_type = normalize_event_type(payload.get("event_type"))

    secondary_themes_raw = payload.get("secondary_themes", [])
    if not isinstance(secondary_themes_raw, list):
        raise ValueError("secondary_themes must be a list")
    secondary_themes: list[str] = []
    for item in secondary_themes_raw:
        normalized = normalize_theme(item)
        if normalized is None:
            continue
        if normalized != primary_theme and normalized not in secondary_themes:
            secondary_themes.append(normalized)

    issue_tags_raw = payload.get("issue_tags", [])
    if not isinstance(issue_tags_raw, list):
        raise ValueError("issue_tags must be a list")
    issue_tags: list[str] = []
    for item in issue_tags_raw:
        normalized = normalize_issue_tag(item)
        if normalized and normalized not in issue_tags:
            issue_tags.append(normalized)

    countries_raw = payload.get("countries", [])
    if not isinstance(countries_raw, list):
        raise ValueError("countries must be a list")
    countries: list[str] = []
    for item in countries_raw:
        normalized = normalize_country(item)
        if normalized and normalized not in countries:
            countries.append(normalized)

    geo_scope = normalize_geo_scope(payload.get("geo_scope"))

    entities_raw = payload.get("entities", [])
    if not isinstance(entities_raw, list):
        raise ValueError("entities must be a list")

    deduped_entities: dict[tuple[str, str], dict[str, Any]] = {}
    for item in entities_raw:
        if not isinstance(item, dict):
            raise ValueError("entity item must be an object")
        entity_type = normalize_type(item.get("type"))
        if entity_type is None:
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        entity = {
            "name": name,
            "type": entity_type,
            "role": normalize_role(item.get("role")),
            "confidence": float(item.get("confidence", 0.0)),
        }
        if entity["confidence"] < MIN_ENTITY_CONFIDENCE:
            continue
        key = (normalize_entity_name(name), entity_type)
        existing = deduped_entities.get(key)
        if existing is None or entity["confidence"] >= existing["confidence"]:
            deduped_entities[key] = entity

    event_summary = str(payload.get("event_summary", "")).strip()
    if not event_summary:
        raise ValueError("event_summary is required")

    relations_raw = payload.get("relations", [])
    if not isinstance(relations_raw, list):
        raise ValueError("relations must be a list")
    relations: list[dict[str, str]] = []
    for item in relations_raw[:5]:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip().lower()
        obj = str(item.get("object", "")).strip()
        if subject and predicate and obj:
            relations.append({"subject": subject, "predicate": predicate, "object": obj})

    return {
        "story_kind": story_kind,
        "event_type": event_type,
        "event_summary": event_summary,
        "primary_theme": primary_theme,
        "secondary_themes": secondary_themes,
        "issue_tags": issue_tags,
        "countries": countries,
        "geo_scope": geo_scope,
        "entities": list(deduped_entities.values()),
        "relations": relations,
    }


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
    sections = [item.strip() for item in args.sections.split(",") if item.strip()]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"guardian_news_sample_{timestamp}.json"
    report_path = output_dir / f"guardian_map_assessment_{timestamp}.json"

    payload = fetch_guardian_articles(limit=args.limit, sections=sections, api_key=args.guardian_api_key)
    sample_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    articles = canonical_articles(payload)
    if not articles:
        print("No Guardian articles were returned.", flush=True)
        return 1

    results: list[dict[str, Any]] = []
    latencies: list[int] = []
    token_counts: list[int] = []
    entity_counts: list[int] = []
    raw_json_ok = 0
    repair_count = 0

    print(
        json.dumps(
            {
                "source": "guardian",
                "model": settings.groq_model,
                "sample_size": len(articles),
                "sections": sections,
                "sample_path": str(sample_path.relative_to(ROOT)),
            },
            indent=2,
        ),
        flush=True,
    )

    for index, article in enumerate(articles, start=1):
        system_prompt, prompt = render_prompt(article)
        article_text = choose_article_text(
            article["full_text"],
            article["summary"],
            article["title"],
            max_chars=DEFAULT_LLM_ARTICLE_TEXT_CHARS,
            summary_max_chars=DEFAULT_LLM_SUMMARY_TEXT_CHARS,
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
            "story_kind": None,
            "event_type": None,
            "event_summary": None,
            "primary_theme": None,
            "secondary_themes": [],
            "issue_tags": [],
            "countries": [],
            "geo_scope": None,
            "entity_count": 0,
            "entities": [],
            "relation_count": 0,
            "relations": [],
            "latency_ms": 0,
            "tokens_used": 0,
            "error_message": None,
        }

        try:
            response = provider.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                trace_context=LLMTraceContext(
                    operation="guardian_map_extraction_assessment",
                    article_id=article["provider_item_id"],
                    prompt_version="map_v1",
                    article_title=article["title"],
                ),
            )
            result["latency_ms"] = response.latency_ms
            result["tokens_used"] = response.tokens_used
            try:
                parsed = parse_map_payload(response.text)
                result["raw_json_valid"] = True
                raw_json_ok += 1
            except Exception:
                repair_count += 1
                result["used_repair"] = True
                repair_system_prompt, repair_prompt = JSON_REPAIR_PROMPT.render(broken_output=response.text)
                repaired = provider.complete(
                    prompt=repair_prompt,
                    system_prompt=repair_system_prompt,
                    temperature=0.0,
                    trace_context=LLMTraceContext(
                        operation="guardian_map_json_repair_assessment",
                        article_id=article["provider_item_id"],
                        prompt_version="json_fix_v1",
                        article_title=article["title"],
                    ),
                )
                result["latency_ms"] += repaired.latency_ms
                result["tokens_used"] += repaired.tokens_used
                parsed = parse_map_payload(repaired.text)

            result["success"] = True
            result["story_kind"] = parsed["story_kind"]
            result["event_type"] = parsed["event_type"]
            result["event_summary"] = parsed["event_summary"]
            result["primary_theme"] = parsed["primary_theme"]
            result["secondary_themes"] = parsed["secondary_themes"]
            result["issue_tags"] = parsed["issue_tags"]
            result["countries"] = parsed["countries"]
            result["geo_scope"] = parsed["geo_scope"]
            result["entities"] = parsed["entities"]
            result["entity_count"] = len(parsed["entities"])
            result["relations"] = parsed["relations"]
            result["relation_count"] = len(parsed["relations"])

            latencies.append(int(result["latency_ms"]))
            token_counts.append(int(result["tokens_used"]))
            entity_counts.append(int(result["entity_count"]))

            print(
                f"[{index:02d}/{len(articles):02d}] OK "
                f"tokens={result['tokens_used']} latency={result['latency_ms']}ms "
                f"repair={result['used_repair']} kind={result['story_kind']} "
                f"type={result['event_type']} theme={result['primary_theme']} "
                f"entities={result['entity_count']} relations={result['relation_count']} "
                f"title={article['title']}",
                flush=True,
            )
            print(f"  summary={result['event_summary']}", flush=True)
            if result["issue_tags"] or result["countries"]:
                print(
                    f"  issue_tags={result['issue_tags']} countries={result['countries']} geo_scope={result['geo_scope']}",
                    flush=True,
                )
            print(f"  {preview_entities(result['entities'])}", flush=True)
        except Exception as error:
            result["error_message"] = str(error)
            print(f"[{index:02d}/{len(articles):02d}] FAIL title={article['title']}", flush=True)
            print(f"  error={result['error_message']}", flush=True)

        results.append(result)

    successes = [item for item in results if item["success"]]
    summary = {
        "source": "guardian",
        "model": settings.groq_model,
        "sample_size": len(results),
        "sections": sections,
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "success_rate": round(len(successes) / len(results), 4),
        "raw_json_success_count": raw_json_ok,
        "raw_json_success_rate": round(raw_json_ok / len(results), 4),
        "repair_count": repair_count,
        "repair_rate": round(repair_count / len(results), 4),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "avg_tokens_used": round(statistics.mean(token_counts), 2) if token_counts else 0.0,
        "avg_entities_per_success": round(statistics.mean(entity_counts), 2) if entity_counts else 0.0,
        "avg_word_count": round(statistics.mean(item["word_count"] for item in results), 2),
        "avg_llm_input_chars": round(statistics.mean(item["llm_input_chars"] for item in results), 2),
        "story_kind_breakdown": {
            key: sum(1 for item in successes if item["story_kind"] == key)
            for key in sorted({item["story_kind"] for item in successes if item["story_kind"]})
        },
        "event_type_breakdown": {
            key: sum(1 for item in successes if item["event_type"] == key)
            for key in sorted({item["event_type"] for item in successes if item["event_type"]})
        },
        "primary_theme_breakdown": {
            key: sum(1 for item in successes if item["primary_theme"] == key)
            for key in sorted({item["primary_theme"] for item in successes if item["primary_theme"]})
        },
        "avg_relations_per_success": round(
            statistics.mean(item["relation_count"] for item in successes),
            2,
        ) if successes else 0.0,
    }
    report = {"summary": summary, "results": results}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummary", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved sample to {sample_path}", flush=True)
    print(f"Saved report to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
