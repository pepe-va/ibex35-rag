.PHONY: help up down restart build logs shell chromadb \
        pull-models models reindex \
        frontend frontend-build frontend-logs frontend-shell \
        test test-unit test-integration \
        lint format typecheck \
        clean clean-volumes \
        status urls

# ── Config ────────────────────────────────────────────────────────────────────
COMPOSE  := docker compose
API      := $(COMPOSE) exec api

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  IBEX35 RAG — available targets"
	@echo ""
	@echo "  Stack"
	@echo "    make up              Start all services (detached)"
	@echo "    make down            Stop all services"
	@echo "    make restart         Restart the API container"
	@echo "    make build           Rebuild Docker images"
	@echo "    make logs            Follow API logs  (SVC=ollama make logs)"
	@echo "    make shell           Open a shell inside the API container"
	@echo "    make status          Show container status"
	@echo "    make urls            Print service URLs"
	@echo ""
	@echo "  Frontend"
	@echo "    make frontend        Restart Chainlit frontend"
	@echo "    make frontend-build  Rebuild frontend image"
	@echo "    make frontend-logs   Follow frontend logs"
	@echo "    make frontend-shell  Shell inside frontend container"
	@echo ""
	@echo "  Data"
	@echo "    make models        List models available in Ollama"
	@echo "    make reindex       Wipe ChromaDB and re-ingest all PDFs"
	@echo ""
	@echo "  Quality"
	@echo "    make test          Run full test suite"
	@echo "    make test-unit     Run unit tests only"
	@echo "    make test-int      Run integration tests only"
	@echo "    make lint          Run ruff linter"
	@echo "    make format        Auto-format with ruff"
	@echo "    make typecheck     Run mypy"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean         Remove __pycache__ and .pyc files"
	@echo "    make clean-volumes WARNING: deletes all Docker volumes"
	@echo ""

# ── Stack ─────────────────────────────────────────────────────────────────────
up:
	$(COMPOSE) up -d
	@$(COMPOSE) ps

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart api

build:
	$(COMPOSE) build

chromadb:
	$(COMPOSE) up -d --force-recreate chroma

SVC ?= api
logs:
	$(COMPOSE) logs -f $(SVC)

shell:
	$(API) /bin/bash

status:
	$(COMPOSE) ps

urls:
	@echo ""
	@echo "  Frontend   http://localhost:8501"
	@echo "  API        http://localhost:8080"
	@echo "  API docs   http://localhost:8080/docs"
	@echo "  ChromaDB   http://localhost:8000"
	@echo "  MLflow     http://localhost:5000"
	@echo "  Grafana    http://localhost:3000  (ver .env para credenciales)"
	@echo "  Prometheus http://localhost:9090"
	@echo "  Ollama     http://localhost:11434"
	@echo ""

# ── Frontend ──────────────────────────────────────────────────────────────────
frontend:
	$(COMPOSE) restart frontend

frontend-build:
	$(COMPOSE) build frontend

frontend-logs:
	$(COMPOSE) logs -f frontend

frontend-shell:
	$(COMPOSE) exec frontend /bin/bash

# ── Data ──────────────────────────────────────────────────────────────────────
pull-models:
	$(COMPOSE) exec ollama ollama pull llama3.2:latest
	$(COMPOSE) exec ollama ollama pull nomic-embed-text:latest

models:
	$(COMPOSE) exec ollama ollama list

reindex:
	@echo "Re-indexing: wiping ChromaDB and ingesting PDFs..."
	$(API) python scripts/ingest.py
	@echo "Done."

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-int:
	uv run pytest tests/integration/ -v

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

typecheck:
	uv run mypy src/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -not -path './.venv/*' | xargs rm -rf
	find . -name '*.pyc' -not -path './.venv/*' -delete

clean-volumes:
	@echo "WARNING: This will delete ALL Docker volumes (ChromaDB, Ollama models, MLflow data, etc.)"
	@read -p "Are you sure? [y/N] " ans && [ "$$ans" = "y" ] || exit 0
	$(COMPOSE) down -v
