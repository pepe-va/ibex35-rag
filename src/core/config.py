"""Application settings, read from environment variables and `.env`."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Flat settings object, with fields grouped by domain."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── URLs / endpoints ────────────────────────────────────────────────────
    api_url: str = "http://localhost:8080"
    test_endpoint: str = "/api/v1/ask"
    ollama_base_url: str = "http://localhost:11434"
    redis_url: str = "redis://localhost:6379/0"
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    # Browser origins allowed to call the API cross-origin. The Chainlit frontend
    # calls the API server-side, so none are needed by default.
    cors_allow_origins: list[str] = []

    # ── Credentials ─────────────────────────────────────────────────────────
    grafana_admin_user: str
    grafana_admin_password: SecretStr
    openai_api_key: SecretStr | None = None

    # ── End-to-end test question generator ──────────────────────────────────
    agent_test_generator_backend: str = "auto"
    agent_test_generator_api_key: SecretStr | None = None

    # ── LangSmith (tracing + LLM metrics) ───────────────────────────────────
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "ibex35-rag"
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_collector_interval_seconds: int = 60

    # ── Models ──────────────────────────────────────────────────────────────
    ollama_llm_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text:latest"
    ollama_router_model: str | None = None  # None: reuse the LLM model (no model swap)
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_request_timeout: float = Field(default=300.0, gt=0)

    # ── Vector store (Qdrant) ───────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ibex35"

    # ── Cache / rate limit ──────────────────────────────────────────────────
    cache_ttl_seconds: int = 3600
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    # ── Data paths ──────────────────────────────────────────────────────────
    pdf_dir: str = "./data/raw/pdfs"
    rag_data_dir: str = "./data/processed/rag_data"

    # ── App metadata ────────────────────────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "development"
    app_name: str = "ibex35-rag"
    app_version: str = "0.1.0"

    # ── Semantic cache (Qdrant) ─────────────────────────────────────────────
    semantic_cache_collection: str = "ibex35_query_cache"
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    # ── Retrieval, chunking and evaluation ──────────────────────────────────
    similarity_top_k: int = Field(default=12, ge=1, le=20)
    rerank_top_n: int = Field(default=8, ge=1, le=10)
    chunk_size: int = Field(default=1200, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=3000)
    rag_eval_input_cost_per_1k: float = Field(default=0.0, ge=0.0)
    rag_eval_output_cost_per_1k: float = Field(default=0.0, ge=0.0)

    @property
    def effective_router_model(self) -> str:
        return self.ollama_router_model or self.ollama_llm_model


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment
