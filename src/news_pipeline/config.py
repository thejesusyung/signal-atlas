from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str = Field(
        default="postgresql+psycopg2://news_pipeline:news_pipeline@localhost:5432/news_pipeline",
        alias="DATABASE_URL",
    )

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_artifact_root: str = Field(default="./mlruns", alias="MLFLOW_ARTIFACT_ROOT")
    mlflow_experiment_ingestion: str = Field(
        default="ingestion_monitoring", alias="MLFLOW_EXPERIMENT_INGESTION"
    )
    mlflow_experiment_extraction: str = Field(
        default="extraction_monitoring", alias="MLFLOW_EXPERIMENT_EXTRACTION"
    )

    feeds_config_path: Path = Field(default=Path("config/feeds.yaml"), alias="FEEDS_CONFIG_PATH")
    feed_state_path: Path = Field(default=Path("data/feed_state.json"), alias="FEED_STATE_PATH")

    ingestion_batch_size: int = Field(default=100, alias="INGESTION_BATCH_SIZE")
    extraction_batch_size: int = Field(default=20, alias="EXTRACTION_BATCH_SIZE")
    dedup_recent_hours: int = Field(default=48, alias="DEDUP_RECENT_HOURS")
    dedup_title_similarity: float = Field(default=0.92, alias="DEDUP_TITLE_SIMILARITY")

    scraper_user_agent: str = Field(
        default="NewsIntelligencePipeline/0.1 (+https://localhost)", alias="SCRAPER_USER_AGENT"
    )
    scraper_domain_delay_seconds: float = Field(default=2.0, alias="SCRAPER_DOMAIN_DELAY_SECONDS")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_requests_per_minute: int = Field(default=30, alias="LLM_REQUESTS_PER_MINUTE")
    llm_rate_limit_backend: str = Field(default="auto", alias="LLM_RATE_LIMIT_BACKEND")

    topic_labels: str = Field(
        default="technology,politics,business,science,health,entertainment,sports,world,environment,finance,education,crime,climate",
        alias="TOPIC_LABELS",
    )

    api_page_size: int = Field(default=20, alias="API_PAGE_SIZE")
    api_max_page_size: int = Field(default=100, alias="API_MAX_PAGE_SIZE")

    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")
    embedding_min_cluster_size: int = Field(default=5, alias="EMBEDDING_MIN_CLUSTER_SIZE")

    signal_baseline_runs: int = Field(default=10, alias="SIGNAL_BASELINE_RUNS")
    signal_zscore_threshold: float = Field(default=1.5, alias="SIGNAL_ZSCORE_THRESHOLD")
    signal_top_n: int = Field(default=3, alias="SIGNAL_TOP_N")
    mlflow_experiment_signals: str = Field(
        default="signal_monitoring", alias="MLFLOW_EXPERIMENT_SIGNALS"
    )
    # Velocity windows for signal detection
    signal_current_window_hours: int = Field(default=24, alias="SIGNAL_CURRENT_WINDOW_HOURS")
    signal_baseline_window_hours: int = Field(default=72, alias="SIGNAL_BASELINE_WINDOW_HOURS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def topic_names(self) -> list[str]:
        return [item.strip() for item in self.topic_labels.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
