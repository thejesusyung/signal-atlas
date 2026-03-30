#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_pipeline.config import get_settings
from news_pipeline.llm.groq_client import GroqProvider
from news_pipeline.llm.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    TOPIC_CLASSIFICATION_PROMPT,
    parse_json_payload,
)

SAMPLE_TITLE = "Acme opens new robotics lab in Lima"
SAMPLE_ARTICLE = (
    "Acme Corp said it opened a robotics lab in Lima, Peru. "
    "Chief executive Elena Torres said the site will hire 120 engineers over the next year. "
    "The company expects the facility to support logistics and industrial automation projects."
)


def build_request(task: str) -> tuple[str, str]:
    settings = get_settings()
    if task == "entity":
        return ENTITY_EXTRACTION_PROMPT.render(title=SAMPLE_TITLE, article_text=SAMPLE_ARTICLE)
    if task == "topic":
        return TOPIC_CLASSIFICATION_PROMPT.render(
            title=SAMPLE_TITLE,
            article_text=SAMPLE_ARTICLE,
            topic_labels=settings.topic_names,
        )
    system_prompt = "You are a precise assistant that returns only valid JSON."
    prompt = (
        "Return a JSON object with keys "
        '"status", "model_family", and "summary". '
        'Set "status" to "ok", "model_family" to "llm_smoke_test", '
        'and "summary" to one short sentence confirming JSON mode works.'
    )
    return system_prompt, prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a direct Groq smoke test outside Airflow.")
    parser.add_argument(
        "--task",
        choices=("ping", "entity", "topic"),
        default="entity",
        help="Prompt shape to test against the Groq API.",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of direct requests to send.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args()

    settings = get_settings()
    provider = GroqProvider()
    system_prompt, prompt = build_request(args.task)

    print(
        json.dumps(
            {
                "provider": "groq",
                "base_url": settings.groq_base_url,
                "model": settings.groq_model,
                "rpm_limit": settings.llm_requests_per_minute,
                "task": args.task,
                "count": args.count,
            },
            indent=2,
        )
    )

    for index in range(args.count):
        response = provider.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        payload = parse_json_payload(response.text)
        print(
            json.dumps(
                {
                    "request_index": index,
                    "ok": True,
                    "model": response.model,
                    "provider_name": response.provider_name,
                    "tokens_used": response.tokens_used,
                    "latency_ms": response.latency_ms,
                    "payload": payload,
                },
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
