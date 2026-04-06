from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:latest"
    ollama_embed_model: str = "nomic-embed-text:latest"
    llm_temperature: float = 0.1
    llm_request_timeout: float = 300.0

    # Vector store
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ibex35"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    # Data
    pdf_dir: str = "./ibex35"
    rag_data_dir: str = "./data/rag-data"
    top_k: int = 5

    # App
    log_level: str = "INFO"
    environment: str = "development"
    app_name: str = "ibex35-rag"
    app_version: str = "0.1.0"

    # RAG
    similarity_top_k: int = Field(default=5, ge=1, le=20)
    rerank_top_n: int = Field(default=3, ge=1, le=10)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
