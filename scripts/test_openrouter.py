"""
Quick smoke test for OpenRouter with openai/gpt-oss-120b:free via open-inference/int8.

Run from the project root:
    python scripts/test_openrouter.py

Requires OPENROUTER_API_KEY in .env (or exported in the shell).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-oss-120b:free"
PROVIDER = "open-inference/int8"

SYSTEM_PROMPT = (
    "You extract named entities from news articles. "
    "Return JSON only. Never include prose."
)
USER_PROMPT = (
    "Extract named entities from this headline:\n"
    "\"Apple and Microsoft both announced record quarterly earnings, "
    "with CEO Tim Cook crediting strong iPhone sales in Europe.\"\n\n"
    "Return JSON in this format: "
    "{\"entities\": [{\"name\": ..., \"type\": ..., \"role\": ..., \"confidence\": ...}]}"
)


def main() -> None:
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set in .env or environment.")
        sys.exit(1)

    print(f"Model   : {MODEL}")
    print(f"Provider: {PROVIDER}")
    print(f"Endpoint: {BASE_URL}/chat/completions")
    print()

    payload = {
        "model": MODEL,
        "temperature": 0.0,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        # Medium reasoning effort
        "reasoning": {
            "effort": "medium",
        },
        # Pin to specific inference provider
        "provider": {
            "order": [PROVIDER],
            "allow_fallbacks": False,
        },
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/signal-atlas",
        "X-Title": "Signal Atlas Test",
    }

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
    except httpx.HTTPError as exc:
        print(f"HTTP error: {exc}")
        sys.exit(1)

    latency_ms = int((time.perf_counter() - started) * 1000)

    print(f"Status  : {response.status_code}  ({latency_ms}ms)")

    if response.status_code != 200:
        print("Response body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        sys.exit(1)

    data = response.json()
    usage = data.get("usage", {})
    choice = data["choices"][0]
    content = choice["message"]["content"]
    model_used = data.get("model", "unknown")

    print(f"Model used       : {model_used}")
    print(f"Tokens (prompt)  : {usage.get('prompt_tokens', '?')}")
    print(f"Tokens (completion): {usage.get('completion_tokens', '?')}")
    print(f"Tokens (total)   : {usage.get('total_tokens', '?')}")
    print(f"Finish reason    : {choice.get('finish_reason')}")
    print()
    print("--- Response content ---")
    print(content)
    print("------------------------")

    # Validate it's parseable JSON
    try:
        parsed = json.loads(content)
        entities = parsed.get("entities", [])
        print(f"\nParsed OK — {len(entities)} entities found")
        for e in entities:
            print(f"  {e.get('name')} ({e.get('type')}) — confidence {e.get('confidence')}")
        print("\nSMOKE TEST PASSED")
    except json.JSONDecodeError:
        print("\nWARNING: Response is not valid JSON — may need JSON repair in production")
        print("SMOKE TEST PARTIAL (connection OK, JSON format issue)")


if __name__ == "__main__":
    main()
