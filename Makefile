.PHONY: start stop restart logs status ingest urls smoke help
.DEFAULT_GOAL := start

COMPOSE := docker compose
API := $(COMPOSE) exec -T api
SVC ?= api

help:
	@echo ""
	@echo "IBEX35 RAG"
	@echo ""
	@echo "  make        Levanta el stack completo"
	@echo "  make start  Levanta el stack completo"
	@echo "  make stop   Para y elimina los contenedores"
	@echo "  make restart Reinicia la API"
	@echo "  make logs SVC=api"
	@echo "  make status"
	@echo "  make ingest"
	@echo "  make urls"
	@echo "  make smoke  Verifica que todo el stack funciona"
	@echo ""

start:
	@test -f .env || { echo "ERROR: .env no encontrado. Copia .env.example a .env"; exit 1; }
	@docker info > /dev/null 2>&1 || { echo "ERROR: Docker no está corriendo"; exit 1; }
	@echo "► Levantando stack..."
	@$(COMPOSE) up -d
	@echo "► Esperando servicios críticos..."
	@for svc in ollama chroma redis; do \
		printf "  %-10s " "$$svc"; \
		for i in $$(seq 1 40); do \
			status=$$($(COMPOSE) ps --format '{{.Health}}' $$svc 2>/dev/null | head -1); \
			if [ "$$status" = "healthy" ]; then echo "healthy ✓"; break; fi; \
			if [ "$$i" = "40" ]; then echo "no respondió a tiempo"; break; fi; \
			printf "."; sleep 3; \
		done; \
	done
	@echo "► Verificando modelos..."
	@LLM=$$(grep '^OLLAMA_LLM_MODEL=' .env 2>/dev/null | cut -d= -f2-); \
	EMB=$$(grep '^OLLAMA_EMBED_MODEL=' .env 2>/dev/null | cut -d= -f2-); \
	LLM=$${LLM:-llama3.2:latest}; \
	EMB=$${EMB:-nomic-embed-text:latest}; \
	PRESENT=$$($(COMPOSE) exec -T ollama ollama list 2>/dev/null || true); \
	echo "$$PRESENT" | grep -q "$${LLM%%:*}" \
		&& echo "  $$LLM ya disponible ✓" \
		|| { echo "  Descargando $$LLM..."; $(COMPOSE) exec -T ollama ollama pull $$LLM; }; \
	echo "$$PRESENT" | grep -q "$${EMB%%:*}" \
		&& echo "  $$EMB ya disponible ✓" \
		|| { echo "  Descargando $$EMB..."; $(COMPOSE) exec -T ollama ollama pull $$EMB; }
	@echo "► Verificando índice..."
	@COLLECTION=$$(grep '^CHROMA_COLLECTION=' .env 2>/dev/null | cut -d= -f2-); \
	COLLECTION=$${COLLECTION:-ibex35}; \
	COUNT=$$($(COMPOSE) exec -T api python -c \
		"import chromadb; c=chromadb.HttpClient(host='chroma', port=8000); print(c.get_or_create_collection('$$COLLECTION').count())" \
		2>/dev/null || echo "0"); \
	if [ "$${COUNT:-0}" -gt 0 ] 2>/dev/null; then \
		echo "  $$COUNT chunks ya indexados ✓"; \
	else \
		echo "  Colección vacía; ejecutando ingest inicial..."; \
		$(API) python scripts/ingest.py; \
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

ingest:
	$(API) python scripts/ingest.py

smoke:
	@uv run python scripts/smoke_test.py

urls:
	@echo ""
	@echo "  Frontend   http://localhost:8501"
	@echo "  API        http://localhost:8080"
	@echo "  API docs   http://localhost:8080/docs"
	@echo "  ChromaDB   http://localhost:8000"
	@echo "  Grafana    http://localhost:3000"
	@echo "  Prometheus http://localhost:9090"
	@echo "  Ollama     http://localhost:11434"
	@echo ""
