# Signal Atlas

Portfolio-grade news intelligence pipeline with an adversarial Twitter simulation layer — built with PostgreSQL, Airflow, MLflow, Groq, and FastAPI.

## What It Does

**Intelligence pipeline** — ingests real news and extracts structured signals:

- Polls configured RSS feeds every 2 hours.
- Deduplicates by exact URL and normalized title similarity.
- Scrapes full article text with `newspaper3k` and BeautifulSoup fallback.
- Cleans scraped text before extraction; stores raw and cleaned variants.
- Extracts named entities and topics using Groq (`llama-3.1-8b-instant`) with JSON-only prompts.
- Detects emerging signals (trend spikes, entity co-occurrence, velocity) via Poisson Z-score.
- Stores everything in PostgreSQL; exposes a FastAPI query interface.

**Twitter simulation layer** — runs competing publisher agents on real stories:

- 5 writer agents each have a persona and a style prompt (e.g. `TheBreakingWire`, `SharpTake`, `FeedChronos`).
- Each simulation cycle picks the top stories from the pipeline and each writer produces tweets about them.
- 100 reader personas (across 10 archetype groups) evaluate every tweet and choose like / repost / comment / skip.
- An engagement score `(reposts×3 + comments×2 + likes×1) / (readers×3)` ranks writers each cycle.
- The bottom 2 writers each cycle have their style prompt mutated by a mutation agent that learns from the top performer.
- Every cycle is logged to MLflow: parent run with aggregate metrics, nested per-writer runs, tweet table artifact, and prompt versions registered in the MLflow Prompt Registry.
- Weekly champion tagging: after 7 cycles the best-performing writer's run is tagged `week_champion`.

## Architecture

```
RSS feeds
   │
   ▼
ingestion_dag (every 2 h)
   │  scrape → deduplicate → store raw_articles
   ▼
extraction_dag (every 2 h)
   │  clean text → extract entities/topics (Groq) → embed → cluster → detect signals
   ▼
PostgreSQL ──► FastAPI (port 8000)
   │
   ▼
simulation_dag (daily)
   seed_db
      └─► fetch_stories
              └─► create_cycle
                      └─► prepare_tweet_inputs
                              └─► generate_tweet [×N, Airflow fan-out]
                                      └─► evaluate_tweet [×N, Airflow fan-out]
                                              └─► score_and_mutate
                                                    ├─► log_to_mlflow
                                                    └─► check_weekly_reset
```

MLflow tracks every Groq extraction call (nested `llm_call` runs under `extraction_monitoring`) and every simulation cycle (nested writer runs under `simulation_cycles`).

## Services

| Service | URL | Purpose |
|---|---|---|
| `postgres` | — | application + Airflow metadata DB |
| `airflow-webserver` | `http://localhost:8080` | DAG management UI |
| `airflow-scheduler` | — | scheduled DAG execution |
| `mlflow` | `http://localhost:5000` | experiment tracking + prompt registry |
| `api` | `http://localhost:8000/docs` | FastAPI query interface |

## Quick Start

1. Copy `.env.example` to `.env`.
2. Set `GROQ_API_KEY` in `.env`.
3. Run `make up`.
4. If upgrading an existing database: `make db-upgrade`.
5. Open the UIs:
   - Airflow: `http://localhost:8080`
   - MLflow: `http://localhost:5000`
   - API docs: `http://localhost:8000/docs`

To run a simulation cycle immediately, trigger `simulation_dag` manually from the Airflow UI.

## API Endpoints

### News intelligence

| Method | Path | Description |
|---|---|---|
| `GET` | `/articles` | Paginated article list (filter by `q`, `source`, `topic`, `entity`, date range) |
| `GET` | `/articles/{id}` | Article detail with entities, topics, full and cleaned text |
| `GET` | `/entities` | Paginated entity list (filter by `entity_type`) |
| `GET` | `/entities/{id}/articles` | Articles linked to one entity |
| `GET` | `/topics` | All topics with article counts |
| `GET` | `/graph` | Entity co-occurrence graph data |
| `GET` | `/similar/{article_id}` | Semantically similar articles (vector distance) |
| `GET` | `/brief` | Latest detected signals |
| `GET` | `/stats` | Pipeline counts |

### Simulation

| Method | Path | Description |
|---|---|---|
| `GET` | `/simulation/latest` | Most recent cycle: leaderboard + tweet table + mutation log |
| `GET` | `/simulation/cycles` | Paginated cycle list with aggregate engagement stats |
| `GET` | `/simulation/cycles/{n}` | One cycle: leaderboard + every tweet with engagement breakdown |
| `GET` | `/simulation/writers` | All writers with current prompt version and all-time stats |
| `GET` | `/simulation/writers/{name}/evolution` | Full prompt version lineage for one writer |

## Simulation Design

### Writers

Defined in `config/writers.yaml`. Each writer has a name, a freeform persona description, and an initial style prompt. The style prompt evolves over time via mutation.

Example writer (`SharpTake`):
```yaml
name: SharpTake
persona_description: >
  A combative opinion columnist who treats every news story as evidence for a
  pre-existing argument. Finds mainstream consensus suspicious. Contrarian by
  instinct, never just for shock value.
initial_style_prompt: >
  Lead with the uncomfortable interpretation others are avoiding. One punchy
  sentence, no hedging.
```

### Reader personas

Defined in `config/personas.yaml`. 100 personas across 10 archetype groups (10 per group):

`political_left`, `political_right`, `gen_z`, `millennial_anxious`, `finance_pro`, `tech_enthusiast`, `casual_lurker`, `reply_guy`, `international`, `disengaged_skeptic`

Each persona is a freeform 3–5 sentence description. The reader LLM receives the full description and chooses an action (like / repost / comment / skip) with a brief reason.

### Engagement scoring

```
score = (reposts×3 + comments×2 + likes×1) / (readers_sampled × 3)
```

Score is normalized 0–1. The denominator assumes every reader could have reposted (maximum engagement).

### Mutation

After each cycle the bottom 2 writers by engagement score are mutated — unless the average score across all writers is below 0.15, in which case everyone is mutated. The mutation agent sees:

- The writer's persona description
- Their current style prompt
- Their recent engagement scores
- The top performer's name and style prompt

Instruction: *"Make exactly one meaningful change. Keep the writer's core persona intact."*

Every mutation produces a new `SimPromptVersion` row and a new entry in the MLflow Prompt Registry (`sim_{writer_name}`).

## MLflow Tracking

### Extraction experiment: `extraction_monitoring`

- One `article_extraction` run per Airflow batch task.
- Nested `llm_call` runs with: prompt version, attempt count, token counts, latency, success flag.
- Raw request/response JSON artifacts under `llm_calls/`.

### Simulation experiment: `simulation_cycles`

- One parent run per cycle (`cycle_NNNN`) with:
  - **Params**: cycle number, week number, story count, writer count, mutation count, personas per tweet.
  - **Metrics at step=cycle_number**: `avg_engagement_score`, `top_engagement_score`, `bottom_engagement_score`, `cycle_skip_rate`, `mutation_count`.
  - **Artifact**: tweet table (`cycle_tweets.json`).
- Nested writer run per writer (`{writer_name}_cNNNN`) with per-writer metrics at same step.
- Weekly champion run is tagged `week_champion=true` after 7 cycles.

### Prompt Registry

Style prompts are versioned in the MLflow Prompt Registry under names like `sim_TheBreakingWire`. Each mutation creates a new registry version with `cycle_introduced` and `triggered_by_score` tags.

## Database Schema

### Core tables (news pipeline)

- `raw_articles` — scraped articles with full/cleaned text, embeddings, processing status
- `entities` — named entities with normalization and article counts
- `topics` — topic labels
- `article_entities` — M2M link with role and confidence
- `article_topics` — M2M link with confidence and extraction method
- `signals` — detected trend spikes and anomalies

### Simulation tables

- `sim_writers` — writer name, persona, pointer to current prompt version
- `sim_prompt_versions` — versioned style prompts with parent lineage, cycle introduced, triggering score
- `sim_personas` — reader persona pool with archetype group
- `sim_cycles` — cycle number, week number, timestamps, input story IDs, MLflow run ID
- `sim_tweets` — generated tweet content linked to writer, cycle, and prompt version
- `sim_engagements` — per-persona reaction (action + reason) for each tweet
- `sim_writer_cycle_scores` — aggregate per-writer per-cycle engagement metrics

## Configuration Reference

All settings are loaded from environment variables (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model for extraction and simulation |
| `LLM_REQUESTS_PER_MINUTE` | `20` | Client-side rate limit |
| `SIM_WRITERS_CONFIG_PATH` | `config/writers.yaml` | Writer definitions |
| `SIM_PERSONAS_CONFIG_PATH` | `config/personas.yaml` | Reader persona pool |
| `SIM_PERSONAS_PER_TWEET` | `10` | How many personas evaluate each tweet |
| `SIM_CYCLE_TOP_STORIES` | `3` | Stories per simulation cycle |
| `MLFLOW_EXPERIMENT_SIMULATION` | `simulation_cycles` | MLflow experiment name |
| `API_PAGE_SIZE` | `20` | Default pagination size |
| `API_MAX_PAGE_SIZE` | `100` | Maximum pagination size |

## Project Layout

```
config/
  writers.yaml          — writer personas and initial style prompts
  personas.yaml         — 100 reader personas across 10 archetype groups
dags/
  ingestion_dag.py      — RSS scrape → deduplicate → store
  extraction_dag.py     — entity/topic extraction → embed → cluster → signals
  simulation_dag.py     — simulation cycle: write → react → score → mutate → log
src/news_pipeline/
  api/
    app.py              — FastAPI application
    simulation.py       — /simulation/* router
  db/
    models.py           — core ORM models
    session.py          — SQLAlchemy session factory
  extraction/           — entity and topic extractors
  ingestion/            — RSS poller, scraper, deduplicator
  services/
    article_service.py  — query functions for news endpoints
    signal_service.py   — signal detection and retrieval
    simulation_service.py — query functions for simulation endpoints
  simulation/
    models.py           — sim_* ORM models
    seeder.py           — idempotent writer and persona seeder
    writer.py           — TweetWriter (generates tweets via LLM)
    reader.py           — PersonaReader (evaluates tweets via LLM)
    scorer.py           — engagement score computation, mutation selection
    mutator.py          — PromptMutator (evolves style prompts via LLM)
    tracker.py          — MLflow logging and prompt registry helpers
  tracking/
    experiment.py       — MLflow experiment configuration
  config.py             — Pydantic settings
alembic/
  versions/             — database migrations
```

## Local Commands

| Command | Description |
|---|---|
| `make setup` | Install Python dependencies locally |
| `make up` | Build and start all services |
| `make down` | Stop services |
| `make test` | Run the test suite |
| `make lint` | Run Ruff |
| `make db-upgrade` | Apply Alembic migrations |

## Text Cleaning

- `raw_articles.full_text` keeps the raw scraped text.
- `raw_articles.cleaned_text` is the deterministic extraction-ready version.
- The cleaner strips author bio lead-ins, digest/subscribe boilerplate, and image markers.
- Entity and topic extraction prefer `cleaned_text` over `full_text`.
- `GET /articles/{id}` returns both so you can compare scraped vs sent-to-LLM text.
