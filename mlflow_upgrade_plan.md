# MLflow Upgrade Plan: LLM Tracing & GenAI Features

## Key Finding: `mlflow.groq.autolog()` Won't Work Here

`groq_client.py` uses raw `httpx.Client` (not the `groq` Python SDK), so autolog can't patch it.
All 4 changes use the manual tracing API instead.

---

## Change 1 (High): Replace Nested LLM Runs with `@mlflow.trace` Spans

**Problem**: Each LLM call currently creates a *child run* (a heavy MLflow object) via
`log_llm_call_trace()` → `tracked_run(..., nested=True)`. The correct primitive for a single
LLM call is a **span** within a trace.

**Files**: `groq_client.py`, `experiment.py`

### Step 1a — Decorate `GroqProvider.complete()` in `groq_client.py`

```python
# Before (line 41):
def complete(self, prompt, system_prompt, temperature=0.1, max_tokens=900, trace_context=None) -> LLMResponse:

# After:
@mlflow.trace(span_type="LLM", name="groq_complete")
def complete(self, prompt, system_prompt, temperature=0.1, max_tokens=900, trace_context=None) -> LLMResponse:
```

This captures all inputs automatically and the `LLMResponse` return value as output. If no trace
is active, MLflow auto-creates one (a standalone trace). If called inside a `tracked_run`, the
span is linked to that run.

### Step 1b — Add token/latency attributes to the active span in `_complete_with_retry`

After building `llm_response` (line 190), add:

```python
span = mlflow.get_current_active_span()
if span is not None:
    span.set_attributes({
        "tokens_used": llm_response.tokens_used,
        "latency_ms": llm_response.latency_ms,
        "model": llm_response.model,
        "attempts": len(attempts),
        "operation": trace_context.operation if trace_context else "llm_call",
    })
```

### Step 1c — Remove all `_log_trace()` calls from `_complete_with_retry`

Remove calls at lines 107–114, 130–137, 156–163, 176–183, 204–217, 221–228. The span captures
success/error automatically — errors set span status to `ERROR`.

### Step 1d — Remove the `_log_trace()` helper method and its import

Remove the `_log_trace()` method (lines 275–293) and the import:
```python
from news_pipeline.tracking.experiment import log_llm_call_trace  # line 12
```
Add `import mlflow`.

### Step 1e — Remove `log_llm_call_trace()` from `experiment.py`

Remove lines 64–118 (including `_slugify`). This eliminates ~55 lines of custom tracking plumbing.

### Test impact

`test_mlflow_tracking.py` mocks `log_llm_call_trace` — rewrite these tests to assert that
`mlflow.get_current_active_span()` was called or that span attributes are set correctly. Use
`mlflow.start_span()` in test setup to provide a parent span.

---

## Change 2 (High): Instrument `signals/detector.py`

**Problem**: `generate_signals()` in `extraction_dag.py:151` calls `detect_and_persist_signals()`
with no active MLflow run. After Change 1, LLM calls inside `_generate_summary()` will create
standalone traces, but there's no run to aggregate signal-level metrics.

**Files**: `extraction_dag.py`, `detector.py`

### Step 2a — Wrap `generate_signals()` task in a `tracked_run`

```python
# extraction_dag.py, lines 150–159
@task(pool="llm_pool")
def generate_signals() -> dict:
    from news_pipeline.signals.detector import detect_and_persist_signals
    settings = get_settings()

    provider = GroqProvider()
    with session_scope() as session:
        with tracked_run(
            experiment_name=settings.mlflow_experiment_signals,
            run_name="signal_detection",
            params={"provider": provider.provider_name},
            tags={"tracking_scope": "signal_detection", "dag_id": "extraction_dag"},
        ):
            signals = detect_and_persist_signals(session, provider)
            log_metrics({"signals_generated": len(signals)})

    LOGGER.info("Generated %d signals", len(signals))
    return {"signals_generated": len(signals)}
```

### Step 2b — `trace_context` in `_generate_summary`

After Change 1, `trace_context` is no longer used for MLflow logging (it was only consumed by
`_log_trace()`). The `provider.complete()` call at `detector.py:276` still passes it — this is
fine to keep for future use or can be simplified to `trace_context=None`.

---

## Change 3 (Medium): Switch `log_dict_artifact` to `mlflow.log_table()`

**Problem**: `log_dict_artifact()` writes raw JSON files logged as opaque blobs.
`mlflow.log_table()` renders as an interactive, filterable table in the MLflow UI.

**Files**: `extraction_dag.py`, `ingestion_dag.py`

### Step 3a — Replace batch extraction results (`extraction_dag.py:64`)

```python
# Before:
log_dict_artifact({"results": processed}, "extraction_batch.json")

# After:
mlflow.log_table(
    data={
        "article_id":  [r["article_id"] for r in processed],
        "status":      [r["status"]     for r in processed],
        "entities":    [r["entities"]   for r in processed],
        "topics":      [r["topics"]     for r in processed],
        "tokens_used": [r["tokens"]     for r in processed],
    },
    artifact_file="extraction_batch.json",
)
```

### Step 3b — Per-article result artifacts (`extraction_dag.py:218–239`)

Keep these as `log_dict_artifact` — per-article details are structured differently and don't have
a consistent columnar shape across the batch. `log_table` is for tabular batch data.

### Step 3c — Ingestion stats (`ingestion_dag.py:57`)

`ingestion_stats.json` is a flat scalar dict (not a list), so `log_table` doesn't apply. Keep
as `log_dict_artifact`. The per-run metrics at lines 49–56 already cover these scalars, making
the artifact partially redundant — consider removing the artifact entirely and relying on
`log_metrics` only.

### Step 3d — Add `import mlflow` to both DAG files

Currently both files only import from `news_pipeline.tracking.experiment`.

### Step 3e — Clean up imports

Remove `log_dict_artifact` from the `extraction_dag.py` import once the batch result call is
replaced. Keep it in `ingestion_dag.py` if the ingestion stats artifact is retained.

---

## Change 4 (Medium): Prompt Registry

**Prerequisite**: Bump `pyproject.toml` lower bound from `mlflow>=2.17,<3.0` to `mlflow>=2.20,<3.0`
(prompt registry was added in 2.20). Deploy the updated dependency before running this change.

**Files**: new `tracking/prompt_registry.py`, `extraction_dag.py`, `entity_extractor.py`,
`topic_extractor.py`, `detector.py`, `pyproject.toml`

### Step 4a — Create `src/news_pipeline/tracking/prompt_registry.py`

```python
import mlflow
from news_pipeline.tracking.experiment import configure_mlflow
from news_pipeline.llm.prompts import ENTITY_EXTRACTION_PROMPT, TOPIC_CLASSIFICATION_PROMPT, PromptSpec
from jinja2 import Template


def register_all_prompts() -> None:
    configure_mlflow()
    _register_if_absent("entity_extraction", ENTITY_EXTRACTION_PROMPT)
    _register_if_absent("topic_classification", TOPIC_CLASSIFICATION_PROMPT)
    # Imported inline to avoid circular import through detector.py
    from news_pipeline.signals.detector import SIGNAL_BRIEF_PROMPT
    _register_if_absent("signal_brief", SIGNAL_BRIEF_PROMPT)


def _register_if_absent(registry_name: str, spec: PromptSpec) -> None:
    try:
        mlflow.register_prompt(
            name=registry_name,
            template=spec.user_prompt_template.source,  # Jinja2 Template.source
            commit_message=f"Register {spec.version}",
            tags={"version": spec.version},
        )
    except mlflow.exceptions.MlflowException:
        pass  # Already exists — idempotent


def get_prompt_template(registry_name: str, fallback: PromptSpec) -> Template:
    """Load from registry; fall back to the bundled constant (e.g. in tests without MLflow)."""
    try:
        configure_mlflow()
        prompt = mlflow.load_prompt(f"prompts:/{registry_name}/1")
        return Template(prompt.template)
    except Exception:
        return fallback.user_prompt_template
```

### Step 4b — Add a `register_prompts` Airflow task as the first task in `extraction_dag.py`

```python
@task
def register_prompts() -> None:
    from news_pipeline.tracking.prompt_registry import register_all_prompts
    register_all_prompts()

# Wire it first:
registered = register_prompts()
article_ids = fetch_pending_article_ids()
registered >> article_ids
```

### Step 4c — Load registered templates in `EntityExtractor` and `TopicExtractor`

In each extractor's `__init__`, replace the module-level constant with a registry-loaded version:

```python
from news_pipeline.tracking.prompt_registry import get_prompt_template

self._prompt = PromptSpec(
    name=ENTITY_EXTRACTION_PROMPT.name,
    version=ENTITY_EXTRACTION_PROMPT.version,
    system_prompt=ENTITY_EXTRACTION_PROMPT.system_prompt,
    user_prompt_template=get_prompt_template("entity_extraction", ENTITY_EXTRACTION_PROMPT),
)
```

Replace all bare `ENTITY_EXTRACTION_PROMPT` / `TOPIC_CLASSIFICATION_PROMPT` references in
`extract_for_article()` with `self._prompt`.

### Step 4d — Same pattern in `_generate_summary()` in `detector.py`

Replace `SIGNAL_BRIEF_PROMPT` with:
```python
from news_pipeline.tracking.prompt_registry import get_prompt_template
prompt = PromptSpec(
    name=SIGNAL_BRIEF_PROMPT.name,
    version=SIGNAL_BRIEF_PROMPT.version,
    system_prompt=SIGNAL_BRIEF_PROMPT.system_prompt,
    user_prompt_template=get_prompt_template("signal_brief", SIGNAL_BRIEF_PROMPT),
)
```

### Test impact

`get_prompt_template()` catches all exceptions and returns the fallback, so existing tests that
don't configure MLflow continue to work without any changes.

---

## Safe Implementation Order

```
Change 1 → Change 2 → Change 3 → Change 4
```

- Change 2 depends on Change 1 (spans from `complete()` need to be linked to the new signal run)
- Changes 3 and 4 are independent of each other but lower risk after 1 & 2 are validated
- Change 4 requires bumping the MLflow version pin before deploying

---

## File Touch Summary

| File | Change 1 | Change 2 | Change 3 | Change 4 |
|---|---|---|---|---|
| `llm/groq_client.py` | Major edit | — | — | — |
| `tracking/experiment.py` | Remove `log_llm_call_trace` | — | — | — |
| `tracking/prompt_registry.py` | — | — | — | **New file** |
| `dags/extraction_dag.py` | — | Wrap `generate_signals` | Replace `log_dict_artifact` | Add `register_prompts` task |
| `dags/ingestion_dag.py` | — | — | Minor | — |
| `signals/detector.py` | — | — | — | Load from registry |
| `extraction/entity_extractor.py` | — | — | — | Load from registry |
| `extraction/topic_extractor.py` | — | — | — | Load from registry |
| `pyproject.toml` | — | — | — | Bump `>=2.20` |
| `tests/test_llm/test_mlflow_tracking.py` | Rewrite | — | — | — |
