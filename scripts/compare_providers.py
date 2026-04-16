"""
Quality + speed comparison: Groq llama-3.1-8b-instant  vs  OpenRouter openai/gpt-oss-120b:free

Tests the three tasks that affect demo impressiveness:
  1. Tweet generation   (creative, visible in UI)
  2. Signal brief       (prose intelligence summary, visible in UI)
  3. Entity extraction  (structured JSON, affects data quality)

Run from project root:
    source /home/ubuntu/venv/bin/activate
    python scripts/compare_providers.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── provider configs ──────────────────────────────────────────────────────────

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_URL = "https://openrouter.ai/api/v1"
OR_MODEL = "openai/gpt-oss-120b:free"
OR_PROVIDER = "open-inference/int8"
OR_REASONING = "medium"

# ── test payloads ─────────────────────────────────────────────────────────────

TWEET_SYSTEM = (
    "You are a Twitter publisher with a defined writing strategy.\n"
    "Write exactly one tweet about the news story provided.\n"
    "Follow your writing strategy precisely — it defines your voice and angle.\n"
    "Output only the tweet text. No hashtags unless they feel natural. Maximum 280 characters."
)
TWEET_USER = (
    "Your writing strategy:\n"
    "You are TheBreakingWire — a no-nonsense wire journalist who strips every story "
    "to its essential facts and delivers them fast. No opinions, no editorialising. "
    "Short punchy sentences. Precision over flair.\n\n"
    "News story:\n"
    "Title: Fed raises rates by 50 basis points amid persistent inflation\n"
    "Summary: The Federal Reserve announced a half-point rate hike today, its most aggressive "
    "move in two decades, signalling it will keep tightening until inflation falls to 2%.\n"
    "Key entities: Federal Reserve, Jerome Powell\n\n"
    "Write your tweet:"
)

SIGNAL_SYSTEM = (
    "You are a news intelligence analyst. "
    "Write concise, factual intelligence briefs based strictly on the provided headlines. "
    "Never speculate beyond what the headlines state. "
    "Return JSON only."
)
SIGNAL_USER = (
    "Generate a 2-3 sentence intelligence brief for this emerging signal.\n\n"
    "Signal type: entity velocity\n"
    "Subject: Nvidia\n"
    "Anomaly score: 3.42\n\n"
    "Supporting headlines:\n"
    "- Nvidia surpasses $3 trillion market cap for the first time\n"
    "- Nvidia CEO Jensen Huang: AI demand is 'insane'\n"
    "- Nvidia stock rises 10% after blowout earnings\n"
    "- Nvidia wins $4bn Pentagon AI contract\n"
    "- Nvidia H100 chips still in short supply, says analyst\n"
    "- Nvidia partners with Saudi Aramco on AI data centre\n\n"
    "Return JSON: {\"summary\": \"2-3 sentence brief here.\"}"
)

ENTITY_SYSTEM = (
    "You extract named entities from news articles. "
    "Return JSON only. Never include prose."
)
ENTITY_USER = (
    "Extract named entities from the article below.\n\n"
    "Return JSON with this shape:\n"
    "{\n"
    "  \"entities\": [\n"
    "    {\"name\": \"Elon Musk\", \"type\": \"person\", \"role\": \"mentioned\", \"confidence\": 0.96}\n"
    "  ]\n"
    "}\n\n"
    "Allowed type values: person, company, organization, location, product.\n"
    "Allowed role values: subject, mentioned, quoted.\n\n"
    "Article title: OpenAI and Google clash over AI safety standards at Senate hearing\n"
    "Article text:\n"
    "OpenAI CEO Sam Altman and Google DeepMind chief Demis Hassabis testified before the US Senate "
    "Commerce Committee on Tuesday, presenting conflicting views on how to regulate large language "
    "models. Senator Maria Cantwell pressed both executives on the risks of autonomous AI agents. "
    "Anthropic, represented by Dario Amodei, called for mandatory third-party audits. "
    "The hearing took place in Washington D.C."
)

TESTS = [
    {
        "name": "Tweet generation",
        "system": TWEET_SYSTEM,
        "user": TWEET_USER,
        "temperature": 0.8,
        "max_tokens": 120,
        "is_json": False,
    },
    {
        "name": "Signal brief",
        "system": SIGNAL_SYSTEM,
        "user": SIGNAL_USER,
        "temperature": 0.1,
        "max_tokens": 300,
        "is_json": True,
        "json_key": "summary",
    },
    {
        "name": "Entity extraction",
        "system": ENTITY_SYSTEM,
        "user": ENTITY_USER,
        "temperature": 0.0,
        "max_tokens": 600,
        "is_json": True,
        "json_key": "entities",
    },
]

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def call_groq(system: str, user: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": GROQ_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }
    started = time.perf_counter()
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{GROQ_URL}/chat/completions", json=payload, headers=headers)
    latency = int((time.perf_counter() - started) * 1000)
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "model": data.get("model", GROQ_MODEL),
        "tokens": data.get("usage", {}).get("total_tokens", 0),
        "latency_ms": latency,
        "finish": data["choices"][0].get("finish_reason"),
    }


def call_openrouter(system: str, user: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": OR_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "reasoning": {"effort": OR_REASONING},
        "provider": {"order": [OR_PROVIDER], "allow_fallbacks": False},
    }
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/signal-atlas",
        "X-Title": "Signal Atlas",
    }
    started = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{OR_URL}/chat/completions", json=payload, headers=headers)
    latency = int((time.perf_counter() - started) * 1000)
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "model": data.get("model", OR_MODEL),
        "tokens": data.get("usage", {}).get("total_tokens", 0),
        "latency_ms": latency,
        "finish": data["choices"][0].get("finish_reason"),
    }

# ── runner ────────────────────────────────────────────────────────────────────

def run_test(test: dict) -> None:
    name = test["name"]
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"{'='*70}")

    results = {}
    for label, fn in [("Groq  llama-3.1-8b-instant", call_groq), ("OpenRouter gpt-oss-120b", call_openrouter)]:
        print(f"\n[{label}] calling...", end=" ", flush=True)
        try:
            r = fn(test["system"], test["user"], test["temperature"], test["max_tokens"])
            print(f"{r['latency_ms']}ms  |  {r['tokens']} tokens  |  finish={r['finish']}")
            results[label] = r
        except Exception as exc:
            print(f"ERROR: {exc}")
            results[label] = None

    print()
    for label, r in results.items():
        if r is None:
            continue
        print(f"── {label} ──")
        if test["is_json"]:
            try:
                parsed = json.loads(r["text"])
                key = test.get("json_key")
                if key and key in parsed:
                    value = parsed[key]
                    if isinstance(value, list):
                        print(f"  ({len(value)} items)")
                        for item in value:
                            print(f"    {item}")
                    else:
                        print(f"  {value}")
                else:
                    print(f"  {json.dumps(parsed, indent=2)}")
                print("  [JSON: valid]")
            except json.JSONDecodeError:
                print(f"  [JSON: INVALID]  raw: {r['text'][:200]}")
        else:
            print(f"  {r['text'][:500]}")
        print(f"  latency={r['latency_ms']}ms  tokens={r['tokens']}")
        print()


def main() -> None:
    missing = []
    if not GROQ_KEY:
        missing.append("GROQ_API_KEY")
    if not OR_KEY:
        missing.append("OPENROUTER_API_KEY")
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}")
        sys.exit(1)

    print("Signal Atlas — Provider Quality Comparison")
    print(f"  Groq model     : {GROQ_MODEL}")
    print(f"  OpenRouter model: {OR_MODEL}  [{OR_PROVIDER}  reasoning={OR_REASONING}]")

    for test in TESTS:
        run_test(test)

    print(f"\n{'='*70}")
    print("  THROUGHPUT NOTES")
    print(f"{'='*70}")
    print("  OpenRouter gpt-oss-120b:free  ~30 tok/s output,  2000 calls/day cap")
    print("  Groq llama-3.1-8b-instant     ~800 tok/s output, 30 req/min rate limit")
    print()
    print("  Current daily call budget:")
    print("    Extraction (entity+topic+signal): ~516 calls/day")
    print("    Simulation (tweets+personas+mutations): ~3663 calls/day")
    print("    Total: ~4179 calls/day")
    print()
    print("  At 2000 calls/day cap, OpenRouter can cover extraction fully (~516)")
    print("  but NOT full simulation (3663). To fit simulation in 1484 remaining")
    print("  calls, persona count would need to drop from 60 to ~4 per tweet.")
    print()


if __name__ == "__main__":
    main()
