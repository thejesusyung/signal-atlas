from __future__ import annotations

import httpx

from news_pipeline.ingestion.rss import FeedStateStore, RSSFeedParser

RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Example Feed</title>
    <item>
      <title>Acme announces launch</title>
      <description>Acme summary</description>
      <link>https://example.com/acme-launch</link>
      <pubDate>Tue, 24 Mar 2026 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


def test_rss_parser_parses_feed_and_saves_cache(tmp_path):
    config_path = tmp_path / "feeds.yaml"
    config_path.write_text(
        "feeds:\n  - name: Example\n    url: https://example.com/feed.xml\n    category: technology\n",
        encoding="utf-8",
    )
    state_store = FeedStateStore(tmp_path / "state.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["User-Agent"] == "NewsIntelligencePipeline/0.1"
        return httpx.Response(
            200,
            text=RSS_XML,
            headers={"ETag": "abc123", "Last-Modified": "Tue, 24 Mar 2026 12:00:00 GMT"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    parser = RSSFeedParser(client=client, config_path=config_path, state_store=state_store)

    articles = parser.parse_all()

    assert len(articles) == 1
    assert articles[0].title == "Acme announces launch"
    state = state_store.load()
    assert state["https://example.com/feed.xml"]["etag"] == "abc123"


def test_rss_parser_handles_304_without_errors(tmp_path):
    config_path = tmp_path / "feeds.yaml"
    config_path.write_text(
        "feeds:\n  - name: Example\n    url: https://example.com/feed.xml\n    category: technology\n",
        encoding="utf-8",
    )
    state_store = FeedStateStore(tmp_path / "state.json")
    state_store.save(
        {
            "https://example.com/feed.xml": {
                "etag": "abc123",
                "last_modified": "Tue, 24 Mar 2026 12:00:00 GMT",
            }
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["If-None-Match"] == "abc123"
        return httpx.Response(304)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    parser = RSSFeedParser(client=client, config_path=config_path, state_store=state_store)

    assert parser.parse_all() == []


def test_feed_state_store_ignores_corrupted_json(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{not valid json", encoding="utf-8")

    store = FeedStateStore(state_path)

    assert store.load() == {}
