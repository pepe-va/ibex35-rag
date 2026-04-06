.PHONY: start stop restart logs status ingest extract urls smoke help
.DEFAULT_GOAL := start

COMPOSE := docker compose
API     := $(COMPOSE) exec -T api
SVC    ?= api

help:
	@echo ""
	@echo "IBEX35 RAG"
	@echo ""
	@echo "  make              Levanta el stack completo"
	@echo "  make start        Levanta el stack completo"
	@echo "  make stop         Para y elimina los contenedores"
	@echo "  make restart      Reinicia la API"
	@echo "  make logs SVC=api Sigue logs de un servicio"
	@echo "  make status       Estado de los contenedores"
	@echo "  make extract      Extrae PDFs → Markdown con Docling (local)"
	@echo "  make ingest       Ingesta Markdown → Qdrant (local)"
	@echo "  make smoke        Smoke test end-to-end"
	@echo "  make urls         Muestra las URLs del stack"
	@echo ""

start:
	@test -f .env || { echo "ERROR: .env no encontrado. Copia .env.example a .env"; exit 1; }
	@docker info > /dev/null 2>&1 || { echo "ERROR: Docker no está corriendo"; exit 1; }
	@echo "► Levantando stack..."
	@$(COMPOSE) up -d
	@echo "► Esperando servicios críticos..."
	@for svc in ollama qdrant redis api; do \
		printf "  %-10s " "$$svc"; \
		for i in $$(seq 1 40); do \
			status=$$($(COMPOSE) ps --format '{{.Health}}' $$svc 2>/dev/null | head -1); \
			if [ "$$status" = "healthy" ]; then echo "healthy ✓"; break; fi; \
			if [ "$$i" = "40" ]; then echo "no respondió a tiempo"; break; fi; \
			printf "."; sleep 3; \
		done; \
	done
	@echo "► Verificando modelos Ollama..."
	@LLM=$$(grep '^OLLAMA_LLM_MODEL=' .env 2>/dev/null | cut -d= -f2-); \
	EMB=$$(grep '^OLLAMA_EMBED_MODEL=' .env 2>/dev/null | cut -d= -f2-); \
	LLM=$${LLM:-qwen2.5:7b}; \
	EMB=$${EMB:-nomic-embed-text:latest}; \
	PRESENT=$$($(COMPOSE) exec -T ollama ollama list 2>/dev/null || true); \
	echo "$$PRESENT" | grep -q "$${LLM%%:*}" \
		&& echo "  $$LLM ya disponible ✓" \
		|| { echo "  Descargando $$LLM..."; $(COMPOSE) exec -T ollama ollama pull $$LLM; }; \
	echo "$$PRESENT" | grep -q "$${EMB%%:*}" \
		&& echo "  $$EMB ya disponible ✓" \
		|| { echo "  Descargando $$EMB..."; $(COMPOSE) exec -T ollama ollama pull $$EMB; }
	@echo "► Verificando modelos de IA en la imagen..."
	@$(COMPOSE) exec -T api python -c \
		"from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')" \
		2>/dev/null && echo "  Qdrant/bm25 ✓" \
		|| { echo "  Qdrant/bm25 ✗ — ejecuta: docker compose build api"; exit 1; }
	@$(COMPOSE) exec -T api python -c \
		"from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')" \
		2>/dev/null && echo "  BAAI/bge-reranker-base ✓" \
		|| { echo "  BAAI/bge-reranker-base ✗ — ejecuta: docker compose build api"; exit 1; }
	@echo "► Verificando colección Qdrant..."
	@COLLECTION=$$(grep '^QDRANT_COLLECTION=' .env 2>/dev/null | cut -d= -f2-); \
	COLLECTION=$${COLLECTION:-ibex35}; \
	COUNT=$$(curl -s "http://localhost:6333/collections/$$COLLECTION" 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',{}).get('points_count',0))" \
		2>/dev/null || echo "0"); \
	if [ "$${COUNT:-0}" -gt 0 ] 2>/dev/null; then \
		echo "  $$COUNT vectores ya indexados ✓"; \
	else \
		echo "  Colección vacía; ejecutando ingesta inicial..."; \
		OLLAMA_BASE_URL=http://localhost:11434 QDRANT_HOST=localhost uv run python scripts/ingest.py; \
	fi
	@echo ""
	@echo "✓ Stack listo"
	@$(MAKE) --no-print-directory urls

stop:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart api

logs:
	$(COMPOSE) logs -f $(SVC)

status:
	$(COMPOSE) ps

# Fase 1: extraer PDFs → Markdown con Docling (se ejecuta en local, no en Docker)
extract:
	@echo "► Extrayendo PDFs con Docling → data/rag-data/markdown/ ..."
	OLLAMA_BASE_URL=http://localhost:11434 QDRANT_HOST=localhost \
		uv run python scripts/ingest.py --extract-only

# Fase 2: ingestar Markdown → Qdrant (se ejecuta en local, no en Docker)
ingest:
	@echo "► Ingesta de Markdown → Qdrant ..."
	OLLAMA_BASE_URL=http://localhost:11434 QDRANT_HOST=localhost \
		uv run python scripts/ingest.py

smoke:
	uv run python scripts/smoke_test.py

urls:
	@echo ""
	@echo "┌─────────────────────────────────────────────────────────────────┐"
	@echo "│                     IBEX35 RAG — URLs                           │"
	@echo "└─────────────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "  ── Aplicación ──────────────────────────────────────────────────"
	@printf "  %-12s http://localhost:8080/docs" "API Swagger"
	@STATUS=$$(curl -sf http://localhost:8080/health 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print('✓ healthy |', d.get('collection_count',0), 'vectores')" \
		2>/dev/null || echo "✗ no responde"); \
	echo "  $$STATUS"
	@printf "  %-12s http://localhost:6333/dashboard" "Qdrant"
	@STATUS=$$(curl -sf http://localhost:6333/collections 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); cols=d.get('result',{}).get('collections',[]); print('✓', len(cols), 'colección/es')" \
		2>/dev/null || echo "✗ no responde"); \
	echo "  $$STATUS"
	@printf "  %-12s http://localhost:8082          " "Redis UI"
	@KEYS=$$(docker compose exec -T redis redis-cli DBSIZE 2>/dev/null | tr -d '\r'); \
	MEM=$$(docker compose exec -T redis redis-cli INFO memory 2>/dev/null \
		| grep used_memory_human | cut -d: -f2 | tr -d '\r '); \
	if [ -n "$$KEYS" ]; then echo "✓ $$KEYS claves | $$MEM RAM"; else echo "✗ no responde"; fi
	@echo ""
	@echo "  ── Observabilidad ──────────────────────────────────────────────"
	@printf "  %-12s http://localhost:3000          " "Grafana"
	@STATUS=$$(curl -sf http://localhost:3000/api/health 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print('✓', d.get('database','?'))" \
		2>/dev/null || echo "✗ no responde"); \
	echo "  $$STATUS"
	@printf "  %-12s http://localhost:9090          " "Prometheus"
	@STATUS=$$(curl -sf 'http://localhost:9090/api/v1/targets' 2>/dev/null \
		| python3 -c "import sys,json; t=json.load(sys.stdin)['data']['activeTargets']; up=sum(1 for x in t if x['health']=='up'); print(f'✓ {up}/{len(t)} targets up')" \
		2>/dev/null || echo "✗ no responde"); \
	echo "  $$STATUS"
	@printf "  %-12s http://localhost:9093          " "Alertmanager"
	@STATUS=$$(curl -sf 'http://localhost:9093/api/v2/alerts' 2>/dev/null \
		| python3 -c "import sys,json; a=json.load(sys.stdin); firing=[x for x in a if x['status']['state']=='active']; print(f'✓ {len(firing)} alertas activas')" \
		2>/dev/null || echo "✗ no responde"); \
	echo "  $$STATUS"
	@printf "  %-12s http://localhost:3000/explore  " "Tempo (UI)"
	@STATUS=$$(curl -sf http://localhost:3200/ready 2>/dev/null \
		| grep -q "ready" && echo "✓ ready  ← ver en Grafana > Explore > Tempo" || echo "✗ no responde"); \
	echo "  $$STATUS"
	@printf "  %-12s http://localhost:3000/explore  " "Loki (UI)"
	@STATUS=$$(curl -sf http://localhost:3100/ready 2>/dev/null \
		| grep -q "ready" && echo "✓ ready  ← ver en Grafana > Explore > Loki" || echo "✗ no responde"); \
	echo "  $$STATUS"
	@echo ""
	@echo "  ── Modelos ─────────────────────────────────────────────────────"
	@printf "  %-12s http://localhost:11434         " "Ollama"
	@STATUS=$$(curl -sf http://localhost:11434/api/tags 2>/dev/null \
		| python3 -c "import sys,json; m=json.load(sys.stdin).get('models',[]); print('✓', ', '.join(x['name'] for x in m))" \
		2>/dev/null || echo "✗ no responde"); \
	echo "  $$STATUS"
	@echo ""
