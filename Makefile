# ─────────────────────────────────────────────────────────────────────────────
#  IBEX35 RAG — run `make help` to list every target
# ─────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := start
SHELL         := /bin/bash

# GPU=1 reserves an NVIDIA GPU for api and ollama; by default only when Docker has the
# NVIDIA runtime (nvidia-container-toolkit). Force it with: make start GPU=1
GPU     ?= $(shell docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && echo 1 || echo 0)
COMPOSE := docker compose -f docker-compose.yml $(if $(filter 1,$(GPU)),,-f docker-compose.cpu.yml)
OLLAMA  := $(COMPOSE) exec -T ollama ollama
API     := $(COMPOSE) exec -T api
CORE    := ollama qdrant redis api
SVC     ?= api

# Read a variable from .env with a default: $(call env,NAME,default)
env = $(or $(shell sed -n 's/^$(1)=//p' .env 2>/dev/null | tr -d '"' | head -1),$(2))

LLM_MODEL    := $(call env,OLLAMA_LLM_MODEL,qwen2.5:7b)
EMBED_MODEL  := $(call env,OLLAMA_EMBED_MODEL,nomic-embed-text:latest)
ROUTER_MODEL := $(call env,OLLAMA_ROUTER_MODEL,qwen2.5:1.5b)
COLLECTION   := $(call env,QDRANT_COLLECTION,ibex35)

# The ingestion scripts run on the host against the Dockerized Ollama and Qdrant
# Host paths override the container ones (/app/...) that .env may carry
LOCAL := OLLAMA_BASE_URL=http://localhost:11434 QDRANT_HOST=localhost \
         PDF_DIR=./data/raw/pdfs RAG_DATA_DIR=./data/processed/rag_data uv run
PDFS  := ./data/raw/pdfs
# Touched after every ingestion: a PDF newer than it has not been ingested yet
INGESTED := ./data/processed/.ingested

.PHONY: help start up stop restart build logs status preflight models verify seed \
        fetch extract ingest test test-unit test-integration smoke metrics-check agent_test \
        lint fmt langsmith-check urls

##@ Stack

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# fetch runs before up: if Docker created data/ first it would be owned by root
start: preflight fetch up models verify seed ## Start everything: latest reports, stack, models and ingestion
	@echo -e "\n✓ Stack ready"
	@$(MAKE) --no-print-directory langsmith-check urls

up: ## Start the containers and wait for the core services
	@echo "► Starting the stack..."
	@$(COMPOSE) up -d
	@$(COMPOSE) up -d --wait --wait-timeout 300 $(CORE)

stop: ## Stop and remove the containers (volumes are kept)
	$(COMPOSE) down

restart: ## Restart the API (picks up changes in src/)
	$(COMPOSE) restart api

build: ## Rebuild the API image
	$(COMPOSE) build api

logs: ## Follow the logs of a service (SVC=api)
	$(COMPOSE) logs -f $(SVC)

status: ## Container status
	$(COMPOSE) ps

preflight:
	@test -f .env || { echo "✗ .env is missing — run: cp .env.example .env"; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "✗ Docker is not running"; exit 1; }
	@[ "$(GPU)" = 1 ] || echo "~ No NVIDIA runtime in Docker: running on CPU (slower). See README → Prerequisites."

models: ## Pull the Ollama models that are missing
	@echo "► Ollama models..."
	@for m in $(LLM_MODEL) $(EMBED_MODEL) $(ROUTER_MODEL); do \
		$(OLLAMA) list | grep -qF "$$m" && echo "  $$m ✓" || $(OLLAMA) pull "$$m"; \
	done

define VERIFY_MODELS
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder
SparseTextEmbedding(model_name="Qdrant/bm25")
CrossEncoder("BAAI/bge-reranker-base")
endef
export VERIFY_MODELS

verify:
	@echo "► Models in the API image..."
	@$(API) python -c "$$VERIFY_MODELS" 2>/dev/null \
		&& echo "  Qdrant/bm25 + BAAI/bge-reranker-base ✓" \
		|| { echo "  ✗ Models missing from the image — run: make build"; exit 1; }

seed:
	@echo "► Qdrant collection '$(COLLECTION)'..."
	@n=$$(curl -sf http://localhost:6333/collections/$(COLLECTION) \
		| python3 -c "import sys, json; print(json.load(sys.stdin)['result']['points_count'] or 0)" 2>/dev/null); \
	if [ "$${n:-0}" -eq 0 ]; then echo "  Empty collection, ingesting..."; \
		$(MAKE) --no-print-directory ingest; \
	elif [ ! -f $(INGESTED) ] || [ -n "$$(find $(PDFS) -maxdepth 1 -name '*.pdf' -newer $(INGESTED))" ]; then \
		echo "  $$n vectors indexed, new reports found, ingesting..."; \
		$(MAKE) --no-print-directory ingest; \
	else echo "  $$n vectors indexed ✓"; fi

##@ Data

fetch: ## Download and verify the latest results report of each IBEX35 company
	@$(LOCAL) python src/ingestion/fetch_reports.py --pdf-dir $(PDFS)

extract: ## Phase 1: PDFs → Markdown with Docling
	$(LOCAL) python src/ingestion/preprocessor.py --pdf-dir $(PDFS)

ingest: ## Full pipeline: extraction + indexing in Qdrant
	$(LOCAL) python src/ingestion/run_ingestion.py --pdf-dir $(PDFS)
	@touch $(INGESTED)

##@ Quality

test: ## The whole test suite
	uv run pytest tests/ -v

test-unit: ## Unit tests (no services needed)
	uv run pytest tests/unit/ -v

test-integration: ## Integration tests (mocked dependencies)
	uv run pytest tests/integration/ -v

smoke: ## End-to-end smoke test (needs the running stack)
	uv run python tests/e2e/smoke_test.py

metrics-check: ## Check that RAG evaluation metrics are produced
	uv run python tests/e2e/metrics_test.py

agent_test: ## Automated QA with generated questions
	uv run python tests/e2e/agent_test.py

lint: ## Ruff lint + format check + mypy (strict)
	uv run ruff check src tests frontend
	uv run ruff format --check src tests frontend
	uv run mypy src

fmt: ## Format and auto-fix with ruff
	uv run ruff format src tests frontend
	uv run ruff check --fix src tests frontend

##@ Diagnostics

define LANGSMITH_CHECK
import os
if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
    raise SystemExit("  ~ LangSmith disabled (set LANGSMITH_TRACING=true to enable it)")
key = os.getenv("LANGSMITH_API_KEY", "")
if not key or "..." in key:
    raise SystemExit("  ✗ LANGSMITH_API_KEY is not set")
from langsmith import Client
project = os.getenv("LANGSMITH_PROJECT", "ibex35-rag")
try:
    names = {p.name for p in Client().list_projects(limit=100)}
except Exception as e:
    raise SystemExit(f"  ✗ LangSmith is not reachable: {e}")
state = "project exists" if project in names else "will be created by the first run"
print(f"  ✓ API reachable | {project}: {state}")
endef
export LANGSMITH_CHECK

langsmith-check: ## Check the LangSmith connection
	@echo "► LangSmith..."
	@uv run --env-file .env python -c "$$LANGSMITH_CHECK" || true

urls: ## Stack URLs and whether they respond
	@check() { printf "  %-13s %-32s" "$$1" "$$2"; \
		curl -sf -o /dev/null --max-time 3 "$${3:-$$2}" && echo "✓" || echo "✗"; }; \
	echo ""; \
	check "API"          http://localhost:8080/docs      http://localhost:8080/health; \
	check "Frontend"     http://localhost:8501; \
	check "Qdrant"       http://localhost:6333/dashboard http://localhost:6333/collections; \
	check "Redis UI"     http://localhost:8082; \
	check "Grafana"      http://localhost:3000           http://localhost:3000/api/health; \
	check "Prometheus"   http://localhost:9090           http://localhost:9090/-/ready; \
	check "Alertmanager" http://localhost:9093           http://localhost:9093/-/ready; \
	check "Tempo"        http://localhost:3000/explore   http://localhost:3200/ready; \
	check "Loki"         http://localhost:3000/explore   http://localhost:3100/ready; \
	check "Ollama"       http://localhost:11434          http://localhost:11434/api/tags; \
	echo ""
