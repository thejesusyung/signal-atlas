# TODO

- [x] Enforce Groq's request budget globally across the full pipeline, not just per process.
- [x] Fix simulation MLflow tracking — persistent per-writer runs with `step=cycle_number` for real trend lines across cycles.
- [x] Update README to match current MLflow implementation.

## MLflow Future Work

- [ ] Fix extraction prompt registry behavior in `src/news_pipeline/tracking/prompt_registry.py`. Stop treating repeated DAG runs as implicit prompt registrations, and stop hardcoding prompt loads to version `1`.
- [ ] Enrich LLM span metadata in `src/news_pipeline/llm/groq_client.py` with `article_id`, `prompt_version`, and other fields from `LLMTraceContext` so traces show stronger lineage.
- [ ] Remove hardcoded MLflow experiment names across DAGs and `tracker.py`; use settings consistently.
- [ ] Add richer MLflow artifacts for signal detection — a top-candidates table with scores, gating decisions, and supporting article counts.
