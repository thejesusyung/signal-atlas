from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import feedparser
import httpx
import yaml

from news_pipeline.config import get_settings
from news_pipeline.contracts import ArticleCandidate

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FeedDefinition:
    name: str
    url: str
    category: str


class FeedStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            LOGGER.warning("Ignoring corrupted feed state file %s: %s", self.path, error)
            return {}
        if not isinstance(payload, dict):
            LOGGER.warning("Ignoring malformed feed state file %s: expected object", self.path)
            return {}
        return payload

    def save(self, state: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            delete=False,
        ) as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(self.path)


class RSSFeedParser:
    def __init__(
        self,
        client: httpx.Client | None = None,
        config_path: Path | None = None,
        state_store: FeedStateStore | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or httpx.Client(timeout=10.0, follow_redirects=True)
        self.config_path = config_path or settings.feeds_config_path
        self.state_store = state_store or FeedStateStore(settings.feed_state_path)

    def load_feeds(self) -> list[FeedDefinition]:
        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        return [FeedDefinition(**item) for item in payload.get("feeds", [])]

    def parse_all(self) -> list[ArticleCandidate]:
        feed_state = self.state_store.load()
        articles: list[ArticleCandidate] = []

        for feed in self.load_feeds():
            try:
                articles.extend(self.parse_feed(feed, feed_state))
            except httpx.TimeoutException:
                LOGGER.warning("Timed out fetching feed %s", feed.url)
            except httpx.HTTPError as error:
                LOGGER.warning("HTTP error fetching feed %s: %s", feed.url, error)
            except Exception as error:  # pragma: no cover - last resort guard
                LOGGER.exception("Unexpected error parsing %s: %s", feed.url, error)

        self.state_store.save(feed_state)
        return articles

    def parse_feed(
        self, feed: FeedDefinition, feed_state: dict[str, dict[str, str]]
    ) -> list[ArticleCandidate]:
        cached = feed_state.get(feed.url, {})
        headers = {
            "User-Agent": "NewsIntelligencePipeline/0.1",
        }
        if etag := cached.get("etag"):
            headers["If-None-Match"] = etag
        if last_modified := cached.get("last_modified"):
            headers["If-Modified-Since"] = last_modified

        response = self.client.get(feed.url, headers=headers)
        if response.status_code == 304:
            LOGGER.info("Feed unchanged: %s", feed.url)
            return []
        response.raise_for_status()

        parsed = feedparser.parse(response.content)
        if getattr(parsed, "bozo", False):
            LOGGER.warning("Malformed feed %s: %s", feed.url, getattr(parsed, "bozo_exception", ""))

        feed_state[feed.url] = {
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
        }

        articles: list[ArticleCandidate] = []
        for entry in parsed.entries:
            articles.append(
                ArticleCandidate(
                    title=_coerce_text(entry.get("title")),
                    summary=_coerce_text(entry.get("summary") or entry.get("description")),
                    url=_coerce_text(entry.get("link")),
                    published_at=_parse_date(entry),
                    source_name=feed.name,
                    source_feed=feed.url,
                    category=feed.category,
                )
            )
        return articles


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_date(entry: Any) -> datetime | None:
    raw_value = entry.get("published") or entry.get("updated")
    if not raw_value:
        return None
    try:
        parsed = parsedate_to_datetime(str(raw_value))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None
