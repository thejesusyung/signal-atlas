from __future__ import annotations

import html
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from difflib import SequenceMatcher

from slugify import slugify

TITLE_PREFIXES = ("mr", "mrs", "ms", "dr", "sir", "madam", "ceo", "president")
DEFAULT_LLM_ARTICLE_TEXT_CHARS = 4_000
DEFAULT_LLM_SUMMARY_TEXT_CHARS = 600
TRUNCATION_MARKER = " ...[truncated]"
LEADING_BOILERPLATE_PATTERNS = (
    re.compile(r"^is a (?:senior |staff |executive |managing )?(?:editor|reviewer|writer|reporter|journalist)\b", re.I),
    re.compile(r"^(?:he|she|they) (?:covers|covered|writes|wrote|spent|has written)\b", re.I),
    re.compile(r"^posts from this author\b", re.I),
    re.compile(r"^all the .* you need to know about\b", re.I),
    re.compile(r"^this (?:article|story) (?:was|has been) updated\b", re.I),
)
BOILERPLATE_PATTERNS = (
    re.compile(r"^posts from this author\b", re.I),
    re.compile(r"^all the .* you need to know about\b", re.I),
    re.compile(r"^previous next(?: \d+ / \d+)?$", re.I),
    re.compile(r"^(?:sign up|subscribe)\b", re.I),
    re.compile(r"^(?:read more|related)\b", re.I),
    re.compile(r"^(?:image|photo|source):\b", re.I),
)
INLINE_BOILERPLATE_PATTERNS = (
    re.compile(r"\bprevious next(?: \d+ / \d+)?\b", re.I),
    re.compile(r"\bposts from this author[^.?!]{0,200}(?=[.?!]|$)", re.I),
    re.compile(r"\bimage:\s*[^.?!]{0,160}(?=[.?!]|$)", re.I),
    re.compile(r"\bphoto:\s*[^.?!]{0,160}(?=[.?!]|$)", re.I),
)


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_text_for_match(value: str | None) -> str:
    normalized = html.unescape(normalize_whitespace(value or "")).lower()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return normalize_whitespace(normalized)


def normalize_title_for_dedup(title: str) -> str:
    normalized = slugify(title or "", separator=" ")
    return normalize_whitespace(normalized)


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_title_for_dedup(left), normalize_title_for_dedup(right)).ratio()


def normalize_entity_name(name: str) -> str:
    cleaned = normalize_whitespace(name).strip(" ,.;:!?")
    lowered = cleaned.lower()
    parts = lowered.split()
    while parts and parts[0].rstrip(".") in TITLE_PREFIXES:
        parts = parts[1:]
    return normalize_whitespace(" ".join(parts))


def clean_summary_text(summary: str | None) -> str:
    normalized = html.unescape(normalize_whitespace(summary or ""))
    normalized = normalized.replace("[&#8230;]", "").replace("[…]", "")
    normalized = re.sub(r"\[\s*…\s*\]|\[\s*&?#?\w+;\s*\]", "", normalized)
    return normalize_whitespace(normalized)


def clean_article_text(text: str | None) -> str:
    normalized = html.unescape(normalize_whitespace(text or ""))
    if not normalized:
        return ""

    for pattern in INLINE_BOILERPLATE_PATTERNS:
        normalized = pattern.sub(" ", normalized)
    normalized = normalize_whitespace(normalized)

    sentences = _split_sentences(normalized)
    cleaned_sentences: list[str] = []
    skipped_leading = 0
    still_at_lead = True

    for sentence in sentences:
        if still_at_lead and skipped_leading < 5 and _matches_any(sentence, LEADING_BOILERPLATE_PATTERNS):
            skipped_leading += 1
            continue
        still_at_lead = False
        if _matches_any(sentence, BOILERPLATE_PATTERNS):
            continue
        if cleaned_sentences and sentence == cleaned_sentences[-1]:
            continue
        cleaned_sentences.append(sentence)

    cleaned = normalize_whitespace(" ".join(cleaned_sentences)).lstrip(" .,:;!-")
    return cleaned or normalized


def text_contains_entity(entity_name: str, text: str | None) -> bool:
    haystack = normalize_text_for_match(text)
    if not haystack:
        return False

    candidates = {
        normalize_text_for_match(entity_name),
        normalize_text_for_match(normalize_entity_name(entity_name)),
    }
    padded_haystack = f" {haystack} "
    return any(candidate and f" {candidate} " in padded_haystack for candidate in candidates)


def truncate_for_llm(value: str | None, max_chars: int) -> str:
    normalized = normalize_whitespace(value or "")
    if not normalized or max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:max_chars]

    cutoff = max_chars - len(TRUNCATION_MARKER)
    excerpt = normalized[:cutoff]
    boundary = max(
        excerpt.rfind(". "),
        excerpt.rfind("! "),
        excerpt.rfind("? "),
        excerpt.rfind("; "),
        excerpt.rfind(", "),
        excerpt.rfind(" "),
    )
    if boundary > cutoff // 2:
        excerpt = excerpt[:boundary]
    return f"{excerpt.rstrip()}{TRUNCATION_MARKER}"


def choose_article_text(
    full_text: str | None,
    summary: str | None,
    title: str,
    cleaned_text: str | None = None,
    max_chars: int = DEFAULT_LLM_ARTICLE_TEXT_CHARS,
    summary_max_chars: int = DEFAULT_LLM_SUMMARY_TEXT_CHARS,
) -> str:
    normalized_title = normalize_whitespace(title or "")
    normalized_summary = truncate_for_llm(clean_summary_text(summary), min(summary_max_chars, max_chars))
    normalized_full_text = normalize_whitespace(cleaned_text or full_text or "")

    if not normalized_full_text:
        return normalized_summary or normalized_title
    if len(normalized_full_text) <= max_chars:
        return normalized_full_text

    context_parts: list[str] = []
    if normalized_summary and normalized_summary != normalized_title:
        context_parts.append(f"Summary: {normalized_summary}")
    context_parts.append(f"Article excerpt: {normalized_full_text}")
    return truncate_for_llm("\n\n".join(context_parts), max_chars)


def safe_average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _split_sentences(text: str) -> list[str]:
    return [normalize_whitespace(part) for part in re.split(r"(?<=[.!?])\s+", text) if normalize_whitespace(part)]


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)
