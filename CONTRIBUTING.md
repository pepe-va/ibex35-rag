# Contributing to IBEX35 RAG

Thanks for your interest in improving this project. This guide explains how to set up a development environment, the conventions the codebase follows, and how to submit changes.

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Workflow](#workflow)
- [Code style](#code-style)
- [Testing](#testing)
- [Commit messages](#commit-messages)
- [Pull requests](#pull-requests)
- [How-to guides](#how-to-guides)
  - [Adding a company](#adding-a-company)
  - [Adding an agent tool](#adding-an-agent-tool)
  - [Adding a Prometheus metric](#adding-a-prometheus-metric)
  - [Adding a setting](#adding-a-setting)
  - [Changing retrieval or the coverage gate](#changing-retrieval-or-the-coverage-gate)
  - [Changing the prompts](#changing-the-prompts)
  - [Editing Grafana dashboards](#editing-grafana-dashboards)
- [Reporting bugs](#reporting-bugs)
- [Security issues](#security-issues)

## Ways to contribute

- Report bugs and wrong or hallucinated answers (include the question and the response).
- Improve retrieval quality, the coverage gate or the prompts.
- Add agent tools, metrics, dashboards or alert rules.
- Improve test coverage and documentation.

For larger changes (new services, changes to the public API, a different vector store or LLM provider), open an issue first so the approach can be discussed before you write the code.

## Development setup

Requirements: Docker with Compose v2, [`uv`](https://docs.astral.sh/uv/), Python 3.11, and ideally an NVIDIA GPU (see the [README prerequisites](README.md#prerequisites) for CPU-only use).

```bash
git clone git@github.com:<your-user>/ibex35-rag.git
cd ibex35-rag
cp .env.example .env          # set GRAFANA_ADMIN_PASSWORD at least
uv sync --dev                 # creates .venv with runtime + dev dependencies
```

You can then either run the full stack (`make start`) or run only the backing services in Docker and the API on the host with hot reload, as described in [Local development](README.md#local-development-api-outside-docker).

Never commit `.env`, PDFs, or anything under `data/`; they are git-ignored on purpose.

## Workflow

1. Create a branch from `main`: `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`, `refactor/<short-name>` or `test/<short-name>`.
2. Keep the change focused. Separate refactors from behaviour changes.
3. Add or update tests for the behaviour you change.
4. Run the checks below before pushing.
5. Open a pull request against `main`.

Checks to run locally:

```bash
make lint            # ruff check + ruff format --check + mypy --strict (`make fmt` fixes formatting)
make test-unit
make test-integration
```

If your change touches retrieval, generation, ingestion or infrastructure, also run the end-to-end checks against a running stack:

```bash
make start
make smoke
make metrics-check
```

## Code style

The code follows [The Zen of Python](https://peps.python.org/pep-0020/) (`python -c "import this"`), and `make lint` enforces it through the Ruff rules in `pyproject.toml`:

| Aphorism | How it is enforced |
|---|---|
| *Beautiful is better than ugly.* / *Readability counts.* | `ruff format` (line length 100), pycodestyle (`E`, `W`), naming (`N`), comprehensions (`C4`). |
| *Explicit is better than implicit.* | Imports at the top of the module (`E402`); heavy modules are imported lazily only with a comment saying why. Type-only imports go under `TYPE_CHECKING`. |
| *Simple is better than complex.* / *If the implementation is hard to explain, it's a bad idea.* | `flake8-simplify` (`SIM`) and a maximum McCabe complexity of 10 per function (`C90`). Split long functions into named steps. |
| *Flat is better than nested.* | `flake8-return` (`RET`): return early instead of nesting. |
| *Errors should never pass silently. Unless explicitly silenced.* | `flake8-bugbear` (`B`) and no `try`/`except`/`pass` (`S110`). Log the exception, or use `contextlib.suppress` with a comment when ignoring it is intended. |
| *There should be one obvious way to do it.* | `pyupgrade` (`UP`), `flake8-pie` (`PIE`), `perflint` (`PERF`), isort (`I`). Reuse the existing helper instead of writing a second version. |
| *Now is better than never.* / dead code | No commented-out code (`ERA`); delete unused code instead of keeping it "just in case". |

- **Formatting and linting**: run `make fmt` before committing and make sure `make lint` passes. There are no per-file exceptions.
- **Typing**: `src/` passes `mypy --strict` (part of `make lint`). Annotate every function, parameterize generics (`dict[str, Any]`, not `dict`), and put type-only imports under `TYPE_CHECKING`. A `# type: ignore[code]` needs a comment explaining why.
- **Python version**: target 3.11. Use built-in generics (`list[str]`, `dict[str, int]`) and `X | None` instead of `Optional[X]`.
- **Settings**: never read `os.environ` directly in application code. Add a field to `Settings` in `src/core/config.py` and inject it (see [Adding a setting](#adding-a-setting)).
- **Logging**: use `get_logger(__name__)` from `src/core/logging.py` and log events as snake_case names with keyword fields, for example `logger.info("hybrid_search", filters=filters, results=len(results))`. Do not use `print` in `src/` except in the CLI entry points.
- **Async code**: the RAG engine and the agent are synchronous. Call them from async routes with `_run_in_thread` (in `src/api/routers/query.py`) so the OpenTelemetry context is preserved and the event loop is not blocked.
- **Heavy imports**: modules that load ML models (reranker, Docling, FastEmbed) are imported lazily inside dependency factories (`src/api/dependencies.py`). Keep it that way so tests and the API start quickly.
- **Language**: code, identifiers, docstrings, comments, commit messages and documentation in English. LLM prompts and user-facing answer templates are in Spanish because the source reports and most users are Spanish-speaking; keep the answer language consistent with the existing prompts.

## Testing

| Suite | Location | Needs services | Run with |
|---|---|---|---|
| Unit | `tests/unit/` | No | `make test-unit` |
| Integration | `tests/integration/` | No (dependencies are mocked) | `make test-integration` |
| End-to-end | `tests/e2e/` | Full stack running | `make smoke`, `make metrics-check`, `make agent_test` |

Guidelines:

- Unit tests must not need Docker, network access or ML models. Mock Ollama, Qdrant and the reranker; use `fakeredis` (fixture `fake_redis` in `tests/conftest.py`) for Redis.
- Integration tests use FastAPI's `TestClient` with dependencies patched (see `tests/integration/api/test_api.py`).
- `pytest-asyncio` runs in `auto` mode, so `async def test_...` functions work without decorators.
- The `Settings` model requires `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD`; keep a `.env` in the repository root when running tests.
- Prometheus metrics are module-level globals. When asserting on them, read values from `prometheus_client.REGISTRY` and compare deltas rather than absolute values.
- For bugs, add a regression test that fails before your fix.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat(rag): guarantee minimum chunks per company after reranking
fix(api): return 409 when ingestion is already running
docs: document SSE event types
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`, `ci`. Common scopes: `api`, `rag`, `agent`, `ingestion`, `observability`, `infra`, `frontend`.

## Pull requests

A good pull request:

- Has a title in Conventional Commits format.
- Explains **what** changed and **why**, and links the related issue.
- Lists how it was tested (unit tests, `make smoke`, manual queries with example questions and answers).
- For retrieval or prompt changes, includes a few before/after answers, ideally covering a single-company question, a two-company comparison and a question that should abstain.
- Updates the documentation (`README.md`, `docs/`) and `.env.example` when behaviour or configuration changes.
- Passes Ruff, mypy and the unit and integration tests.

## How-to guides

### Adding a company

1. Add the company and its Yahoo Finance ticker to `COMPANY_TICKER_MAP` in `src/ingestion/metadata.py`. This is used for metadata, sources and the market tools.
2. Add a regex to `_COMPANY_PATTERNS` in `src/rag/retrieval.py`, including common aliases. Put specific patterns before generic ones to avoid false positives with short aliases.
3. Add it to `COMPANIES` in `frontend/app.py` so it appears in the chat filter (the frontend ships in its own image and cannot import `src`).
4. Add a unit test in `tests/unit/rag/test_retrieval.py` checking that `extract_companies` detects the name and its aliases.
5. Drop the PDF into `data/raw/pdfs/` as `<COMPANY>.pdf` and run `make ingest`.
6. Update the company list in `README.md`.

### Adding an agent tool

1. Write a plain function with type hints and a docstring in `src/agents/agent_tools.py`. Return a human-readable string and catch provider errors, returning an explanatory message instead of raising.
2. Wrap it with `StructuredTool.from_function(...)`, giving it a clear `name` and a `description` that tells the LLM when to use it.
3. Append it to `ALL_TOOLS`.
4. If it changes when the agent should be chosen, update the routing heuristics (`_MARKET_HINTS`, `_FUNDAMENTAL_HINTS`) in `src/api/routers/query.py` and the agent system prompt in `src/agents/financial_agent.py`.
5. Add unit tests with the external provider mocked.

### Adding a Prometheus metric

1. Define it in `src/observability/metrics.py`, in the section it belongs to. Use the `rag_` prefix for application metrics, `llm_` for LLM metrics, and a unit suffix such as `_seconds`, `_bytes` or `_total`.
2. Keep label cardinality bounded: never use the question text, a user ID or a thread ID as a label.
3. Update it where the event happens, and add it to `docs/observability.md`.
4. If it should be visible, add a panel to a dashboard in `infra/grafana/dashboards/` and, if relevant, an alert in `infra/prometheus/rules/rag_alerts.yml`.

### Adding a setting

1. Add a typed field with a sensible default (and `Field(..., ge=..., le=...)` bounds where it makes sense) to `Settings` in `src/core/config.py`, in the matching section.
2. Read it through `get_settings()` or the `get_settings_dep` FastAPI dependency.
3. Document it in `.env.example` and `docs/configuration.md`.

### Changing retrieval or the coverage gate

The tuning constants (`PER_COMPANY_K`, `RERANK_MIN_PER_COMPANY`, `MIN_RERANKER_SCORE`, `MIN_CHUNKS_FOR_COVERAGE`) live at the top of `src/rag/engine.py`, and the behaviour they control is explained in [docs/rag-pipeline.md](docs/rag-pipeline.md). When you change them:

- Check the effect on the `rag_eval_*` metrics and on `rag_empty_retrieval_total` in Grafana, not just on a couple of manual questions.
- Test at least a single-company question, a two-company comparison, a question for a period that is not indexed, and an out-of-domain question (e.g. about Tesla). The last two must abstain.
- `make agent_test` generates varied questions and is useful for spotting regressions.

### Changing the prompts

The RAG system prompt is in `src/rag/prompts.py`, the agent prompt in `src/agents/financial_agent.py`, and the router prompt in `src/api/routers/query.py`. The abstention rules in the RAG prompt are part of the system's safety contract: do not weaken them without discussing it in an issue. Include before/after answers in the pull request.

### Editing Grafana dashboards

Dashboards are provisioned from `infra/grafana/dashboards/*.json`. Edit a dashboard in the Grafana UI, export it as JSON ("Export for sharing externally" disabled), replace the file in the repository, and keep its `uid` unchanged so links keep working.

## Reporting bugs

Open an issue with:

- What you did (command, request body or question).
- What you expected and what happened instead (full response, status code).
- Relevant logs: `make logs SVC=api`, or the trace from Grafana Explore → Tempo.
- Your environment: OS, GPU/CPU, `OLLAMA_LLM_MODEL`, and the commit hash.

## Security issues

Do not open a public issue for security vulnerabilities. Report them privately through GitHub's "Report a vulnerability" feature (repository **Security** tab), or contact the maintainer directly.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
