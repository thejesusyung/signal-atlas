# News Intelligence Pipeline

Portfolio-grade news intelligence pipeline built with PostgreSQL, Airflow, MLflow, Groq, and FastAPI.

## What It Does

- Polls configured RSS feeds every 2 hours.
- Deduplicates by exact URL and normalized title similarity.
- Scrapes full article text with `newspaper3k` and BeautifulSoup fallback.
- Cleans scraped article text before extraction and stores both raw and cleaned variants.
- Extracts entities and topics using Groq with JSON-only prompts.
- Stores articles, entities, topics, and extraction runs in PostgreSQL.
- Exposes a FastAPI query interface for article, entity, topic, and pipeline stats queries.

## Services

- `postgres`: application and Airflow metadata database
- `airflow-init`: runs Alembic and Airflow initialization
- `airflow-webserver`: Airflow UI at `http://localhost:8080`
- `airflow-scheduler`: scheduled DAG execution
- `mlflow`: MLflow UI at `http://localhost:5000`
- `api`: FastAPI app at `http://localhost:8000/docs`

## Quick Start

1. Copy `.env.example` to `.env`.
2. Set `GROQ_API_KEY` in `.env`.
3. Run `make up`.
4. If you are upgrading an existing local database, run `make db-upgrade`.
5. Open Airflow, MLflow, and FastAPI:
   - `http://localhost:8080`
   - `http://localhost:5000`
   - `http://localhost:8000/docs`

## Text Cleaning and Extraction Safeguards

- `raw_articles.full_text` keeps the raw scraped text.
- `raw_articles.cleaned_text` stores the deterministic extraction-ready version of that text.
- The cleaner removes common boilerplate that was polluting prompts:
  - author bio lead-ins such as `is a senior editor...`
  - digest and subscribe text
  - image and gallery markers such as `IMAGE:` and `Previous Next`
- Entity and topic extraction now prefer `cleaned_text` over `full_text`.
- Entity extraction also rejects:
  - unsupported entity types
  - zero-confidence entities
  - entities that do not appear in the cleaned prompt text
- `GET /articles/{id}` now returns both `full_text` and `cleaned_text` so you can compare what was scraped vs what was sent to the LLM.

## MLflow LLM Traces

- Extraction batch runs are logged to the `extraction_monitoring` experiment.
- Each article extraction task now creates an `article_extraction` run.
- Each Groq request inside that article run creates a nested `llm_call` run with:
  - request metadata and prompt version as params
  - attempt count, tokens, latency, and success as metrics
  - raw request/response JSON artifacts under `llm_calls/`
- On the shared filesystem, those artifacts are stored under paths like:
  - `mlruns/1/<run_id>/artifacts/llm_calls/*.json`

## Local Commands

- `make setup`: install Python dependencies locally
- `make up`: build and start all services
- `make down`: stop services
- `make test`: run the test suite
- `make lint`: run Ruff
- `make db-upgrade`: apply Alembic migrations locally
- `.venv310/bin/python scripts/test_groq_direct.py --task entity`: direct Groq smoke test outside Airflow
- `.venv310/bin/python scripts/assess_entity_extraction_batch.py --limit 20 --skip-mlflow`: run a host-side batch assessment of real entity extraction calls

Airflow is provided by the container base image. Local installs use the app/test dependencies only; if you need Airflow outside Docker, install the `airflow` extra separately.

## LLM Defaults

- Default Groq model: `llama-3.1-8b-instant`
- Client-side request budget defaults to a conservative live cap: `LLM_REQUESTS_PER_MINUTE=20`
- Shared limiter backend: `LLM_RATE_LIMIT_BACKEND=auto`

## Project Layout

- [dags/ingestion_dag.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/dags/ingestion_dag.py)
- [dags/extraction_dag.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/dags/extraction_dag.py)
- [src/news_pipeline/db/models.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/src/news_pipeline/db/models.py)
- [src/news_pipeline/ingestion/rss.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/src/news_pipeline/ingestion/rss.py)
- [src/news_pipeline/ingestion/scraper.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/src/news_pipeline/ingestion/scraper.py)
- [src/news_pipeline/extraction/entity_extractor.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/src/news_pipeline/extraction/entity_extractor.py)
- [src/news_pipeline/extraction/topic_extractor.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/src/news_pipeline/extraction/topic_extractor.py)
- [src/news_pipeline/utils.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/src/news_pipeline/utils.py)
- [src/news_pipeline/api/app.py](/Users/max/Documents/Codes/EXP_NEWS_AUTO_STRUCTURE/src/news_pipeline/api/app.py)

## Deferred Items

The repo intentionally defers the following to later phases:

- News API clients
- Claim extraction
- Gemini fallback
- HuggingFace topic classification
- Embedding-based deduplication
- Quality monitoring and prompt A/B tooling
- Offline fixture-backed demo mode
