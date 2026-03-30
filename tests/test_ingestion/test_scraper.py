from __future__ import annotations

import httpx

from news_pipeline.ingestion.scraper import ArticleScraper

ARTICLE_HTML = """
<html>
  <body>
    <article>
      <p>Acme launched a new orbital platform.</p>
      <p>The company expects commercial tests this summer.</p>
    </article>
  </body>
</html>
"""


def test_scraper_uses_bs4_fallback(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, text=ARTICLE_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    scraper = ArticleScraper(client=client)
    monkeypatch.setattr(scraper, "_extract_with_newspaper", staticmethod(lambda url, html: ""))

    result = scraper.scrape("https://example.com/article")

    assert result.success is True
    assert result.extraction_method == "beautifulsoup"
    assert "Acme launched" in result.full_text


def test_scraper_respects_robots_txt():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
        return httpx.Response(200, text=ARTICLE_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    scraper = ArticleScraper(client=client)

    result = scraper.scrape("https://example.com/blocked/story")

    assert result.success is False
    assert result.error == "Blocked by robots.txt"

