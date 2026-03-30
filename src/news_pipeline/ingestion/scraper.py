from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from newspaper import Article

from news_pipeline.config import get_settings
from news_pipeline.contracts import ScrapeResult
from news_pipeline.utils import normalize_whitespace

LOGGER = logging.getLogger(__name__)


class DomainRateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self._last_seen: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def wait(self, domain: str) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last_seen[domain]
            if delta < self.delay_seconds:
                time.sleep(self.delay_seconds - delta)
            self._last_seen[domain] = time.monotonic()


class RobotsCache:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client
        self._cache: dict[str, RobotFileParser] = {}

    def allowed(self, url: str, user_agent: str) -> bool:
        parsed = urlparse(url)
        domain_root = f"{parsed.scheme}://{parsed.netloc}"
        if domain_root not in self._cache:
            parser = RobotFileParser()
            robots_url = urljoin(domain_root, "/robots.txt")
            try:
                response = self.client.get(robots_url)
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    parser.parse([])
            except httpx.HTTPError:
                parser.parse([])
            self._cache[domain_root] = parser
        return self._cache[domain_root].can_fetch(user_agent, url)


class ArticleScraper:
    def __init__(self, client: httpx.Client | None = None) -> None:
        settings = get_settings()
        self.user_agent = settings.scraper_user_agent
        self.client = client or httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=15.0,
            follow_redirects=True,
        )
        self.rate_limiter = DomainRateLimiter(settings.scraper_domain_delay_seconds)
        self.robots_cache = RobotsCache(self.client)

    def scrape(self, url: str) -> ScrapeResult:
        domain = urlparse(url).netloc
        if not self.robots_cache.allowed(url, self.user_agent):
            return ScrapeResult("", 0, "robots", False, "Blocked by robots.txt")

        self.rate_limiter.wait(domain)

        try:
            response = self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as error:
            return ScrapeResult("", 0, "http", False, str(error))

        html = response.text
        text = self._extract_with_newspaper(url, html)
        method = "newspaper3k"
        if not text:
            text = self._extract_with_bs4(html)
            method = "beautifulsoup"

        if not text:
            return ScrapeResult("", 0, method, False, "No text extracted")
        word_count = len(text.split())
        return ScrapeResult(text, word_count, method, True, None)

    @staticmethod
    def _extract_with_newspaper(url: str, html: str) -> str:
        try:
            article = Article(url=url)
            article.set_html(html)
            article.parse()
            return normalize_whitespace(article.text)
        except Exception as error:  # pragma: no cover - third-party parser guard
            LOGGER.debug("newspaper3k failed for %s: %s", url, error)
            return ""

    @staticmethod
    def _extract_with_bs4(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        container = soup.find("article") or soup.find("main") or soup.body
        if container is None:
            return ""
        paragraphs = [normalize_whitespace(node.get_text(" ", strip=True)) for node in container.find_all("p")]
        text = " ".join(part for part in paragraphs if part)
        return normalize_whitespace(text)

