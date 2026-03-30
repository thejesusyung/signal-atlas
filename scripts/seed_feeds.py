from __future__ import annotations

from news_pipeline.ingestion.rss import RSSFeedParser


def main() -> None:
    parser = RSSFeedParser()
    for feed in parser.load_feeds():
        print(f"{feed.category:15} {feed.name:25} {feed.url}")


if __name__ == "__main__":
    main()
