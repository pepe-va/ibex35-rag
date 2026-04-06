# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /usr/local/bin/uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual env (no project install yet)
RUN uv sync --frozen --no-install-project --no-dev

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Data directory + writable cache for Docling/HuggingFace model downloads
RUN mkdir -p /app/data/pdfs /app/.cache && chown -R appuser:appuser /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface" \
    DOCLING_ARTIFACTS_PATH="/app/.cache/docling"

# Pre-download models while internet is still accesible (before offline flags)
RUN python -c "from fastembed import SparseTextEmbedding; list(SparseTextEmbedding(model_name='Qdrant/bm25').embed(['test']))"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health/live').raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
