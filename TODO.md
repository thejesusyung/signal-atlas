# TODO

- [x] Enforce Groq's request budget globally across the full pipeline, not just per process. The limiter in [src/news_pipeline/llm/rate_limit.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/src/news_pipeline/llm/rate_limit.py) now coordinates through shared state, paces requests across processes, and propagates Groq `retry-after` backoff signals across the queue.
