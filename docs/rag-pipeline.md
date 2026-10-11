# RAG pipeline

This document explains how financial reports are ingested and how a question becomes an answer, with the constants and settings that control each step.

## Contents

- [Ingestion](#ingestion)
  - [Phase 1: extraction with Docling](#phase-1-extraction-with-docling)
  - [Metadata](#metadata)
  - [Phase 2: chunking and indexing](#phase-2-chunking-and-indexing)
  - [Deduplication and re-ingestion](#deduplication-and-re-ingestion)
- [Query pipeline](#query-pipeline)
  - [1. Intent extraction](#1-intent-extraction)
  - [2. Out-of-domain check](#2-out-of-domain-check)
  - [3. Hybrid search](#3-hybrid-search)
  - [4. Reranking](#4-reranking)
  - [5. Coverage gate](#5-coverage-gate)
  - [6. Generation](#6-generation)
  - [7. Online evaluation](#7-online-evaluation)
- [Agent](#agent)
- [Tuning reference](#tuning-reference)

## Ingestion

### Phase 1: extraction with Docling

`extract_all_pdfs` (`src/ingestion/pdf_loader.py`) processes every `*.pdf` in `PDF_DIR`. For each file, Docling converts the PDF with table structure recognition and page image generation, then the loader writes:

| Output | Path | Content |
|---|---|---|
| Markdown | `markdown/<COMPANY>/<stem>.md` | Whole document, pages separated by `<!-- page break -->`. |
| Metadata sidecar | `markdown/<COMPANY>/<stem>.meta.json` | Company, ticker, year, quarter, document type. |
| Tables | `tables/<COMPANY>/<stem>/table_<n>_page_<m>.md` | Each Markdown table with the lines of text preceding it, so the table keeps its context. |
| Page images | `images/<COMPANY>/<stem>/page_<n>.png` | Rendered pages, for visual inspection. |

All paths are relative to `RAG_DATA_DIR` (default `./data/processed/rag_data`). A failure on one PDF is logged and the rest continue.

### Metadata

Metadata is built from two sources and merged, with the document content taking precedence over the filename:

- **Filename** (`extract_metadata_from_filename`): the first word is the company. Supported patterns are `COMPANY.pdf`, `COMPANY DOCTYPE YEAR.pdf` and `COMPANY DOCTYPE QUARTER YEAR.pdf`.
- **Content** (`extract_metadata_from_content`): the first 200 lines are scanned for the document type (annual report, quarterly results, ...), the fiscal year and the quarter (`Q3`, `3T`, `tercer trimestre`, ...).

The ticker is looked up in `COMPANY_TICKER_MAP`. Metadata extraction lives in `src/ingestion/metadata.py`, which does not import Docling.

### Phase 2: chunking and indexing

`VectorStoreManager.ingest_all_files` walks every `.md` file under `RAG_DATA_DIR` with 4 worker threads. The content type is inferred from the path:

| Path contains | `content_type` | Splitting |
|---|---|---|
| `markdown` | `text` | Split on page breaks, then `RecursiveCharacterTextSplitter` (`CHUNK_SIZE`, `CHUNK_OVERLAP`, separators: paragraph, line, sentence, word). |
| `tables` | `table` | One document per table file. |
| `images_desc` | `image` | One document per file (reserved for image descriptions). |

Each chunk's payload contains the sidecar metadata plus `content_type`, `file_hash`, `source_file`, `page` and `chunk_index`.

**Per-chunk period annotation**: a single report often covers several periods (for example a 9M2025 report with FY2024 comparatives). `_annotate_chunk_period` scans each chunk and overrides `fiscal_year` with the most frequent year mentioned in it, and `fiscal_quarter` with the first period marker found (`Q1`–`Q4`, `1T`–`4T`, `9M`, `6M`, `3M`, `primer trimestre`, ...). This makes period filters work at chunk level instead of inheriting the whole document's period.

Embeddings:

- **Dense**: `OLLAMA_EMBED_MODEL` (`nomic-embed-text`, 768 dimensions) through Ollama, cosine distance.
- **Sparse**: `Qdrant/bm25` through FastEmbed, computed in-process.

On API start-up, keyword payload indexes are created on `metadata.company`, `metadata.fiscal_year` and `metadata.fiscal_quarter`.

### Deduplication and re-ingestion

Every `.md` file is hashed (SHA-256) and the hash is stored in each chunk's payload. Before ingesting, all existing hashes are loaded from Qdrant and files already present are skipped, so running ingestion again only indexes new or modified files.

A modified file gets a new hash and is indexed again, but the chunks of the old version are not removed. To rebuild from scratch, delete the collection (Qdrant dashboard, or `curl -X DELETE http://localhost:6333/collections/ibex35`) and run `make ingest`. Delete `data/processed/rag_data` as well if you want Docling to re-extract.

## Query pipeline

Implemented in `RAGEngine.query` (`src/rag/engine.py`). Each stage is traced as an OpenTelemetry span (`rag.query`, `rag.retrieve`, `rag.rerank`, `rag.generate`) and timed in `rag_eval_stage_latency_seconds`.

### 1. Intent extraction

`extract_query_intent` (`src/rag/retrieval.py`) returns a `QueryIntent`:

| Field | Detection |
|---|---|
| `companies` | `_COMPANY_PATTERNS` regexes, in order of appearance, deduplicated. Aliases include *Zara* → INDITEX, *Iberia* → IAG, *REE* / *Red Eléctrica* → REDEIA, *Merlin* → MERLINPROPERTIES, *Arcelor* → ARCELORMITTAL. |
| `fiscal_year` | First `20xx` in the question. |
| `fiscal_quarter` | `Q1`–`Q4` or `trimestre 1`–`4`, normalized to `Q<n>`. |
| `wants_annual` | "informe anual", "anual", "full year", "FY2024", "cierre del año", ... |
| `wants_latest` | "último", "más reciente", "actual", "latest", "last quarter", ... |
| `is_comparison` | Two or more companies. |
| `requires_specific_period` | A year or a quarter was given. |
| `out_of_domain_companies` | Non-IBEX35 companies (Tesla, Apple, Amazon, Alphabet, Microsoft, Meta, Nvidia, Netflix). |

If the request includes `company_filter`, it overrides the detected companies.

### 2. Out-of-domain check

If the question mentions only non-IBEX35 companies, the engine returns an abstention immediately, without searching or calling the LLM.

### 3. Hybrid search

`hybrid_search` runs dense + BM25 search in Qdrant with metadata filters built from the intent. The candidate pool is `max(SIMILARITY_TOP_K * 2, 10)`.

- **Single company or generic question**: one search with company, year and quarter filters. If it returns nothing, the year/quarter filters are dropped and the search is retried. For generic questions with no company, a final unfiltered search is attempted.
- **Several companies**: one search per company runs in parallel threads, each returning up to `PER_COMPANY_K = 10` chunks (with the same temporal fallback). The result lists are interleaved round-robin and deduplicated, so every company is represented in the candidate pool.

### 4. Reranking

`rerank_results` scores each (question, chunk) pair with the cross-encoder (`RERANKER_MODEL`, default `BAAI/bge-reranker-base`, on CUDA when available) and keeps `RERANK_TOP_N` chunks.

For comparisons, the reranker first reserves up to `RERANK_MIN_PER_COMPANY = 3` of the best chunks for each requested company, fills the remaining budget with the best of the rest, and then sorts by score. The score is stored in each chunk's metadata as `_score`.

### 5. Coverage gate

Before calling the LLM, `check_coverage` builds a `CoverageReport`:

- A chunk is **valid** if its reranker score is at least `MIN_RERANKER_SCORE = 0.05`.
- Each requested company is **covered** (≥ `MIN_CHUNKS_FOR_COVERAGE = 2` valid chunks), **partial** (1 valid chunk) or **missing** (none).
- If a year was requested, `period_ok` says whether any retrieved chunk has that `fiscal_year`, and `available_periods` lists the years found.

`decide_abstention` then returns an answer without calling the LLM when:

| Gate | Condition | Answer |
|---|---|---|
| 1 | A comparison where at least one company is missing | Names the missing companies and those that are covered. |
| 2 | Companies were requested and no chunk reaches the minimum score | Says the sources are not relevant enough. |
| 3 | A specific period was requested, it is not in the retrieved chunks, and other periods are | Lists the periods that are available. |

Gate answers include the sources that were retrieved, and are counted in `rag_eval_abstention_ratio`.

### 6. Generation

If the gate passes:

1. Chunks are reordered with `reorder_for_llm` (best first, second-best last, the rest in between) to mitigate the "lost in the middle" effect, and joined with `---` separators.
2. A **coverage block** is appended to the user message listing the requested companies, the companies present in the context, and the missing ones, with an instruction not to invent data for missing companies. If the requested year is not in the context, a **period warning** asks the model to state the real period of every figure.
3. The LLM (`OLLAMA_LLM_MODEL`, `LLM_TEMPERATURE = 0.1`, timeout `LLM_REQUEST_TIMEOUT`) is called with the system prompt in `src/rag/prompts.py`. The prompt restricts the model to the given context, forbids estimating figures or mixing periods (9M2025 ≠ FY2024 ≠ Q3 2024), and defines explicit abstention rules. It answers in the language of the question.

### 7. Online evaluation

After generation, the engine records quality metrics without a ground truth:

| Metric | How it is computed |
|---|---|
| Abstention | The answer contains phrases such as "no dispongo", "no tengo datos", "I don't have". |
| Chunk utilization / contextual precision | Binary relevance of each chunk (belongs to a requested company, or score above the median), then the share of relevant chunks and a rank-weighted precision. |
| Contextual recall | Share of question tokens that appear in the context. |
| Chunk attribution | Share of chunks that share at least 3% of the answer's tokens. |
| Answer similarity / correctness | Token-set similarity, unigram BLEU and ROUGE-L between the answer and the top chunk. |
| Context cliff | Grows with context size when the answer diverges from the top chunk. |
| Cost per inference | Estimated tokens × `RAG_EVAL_INPUT_COST_PER_1K` / `RAG_EVAL_OUTPUT_COST_PER_1K` (0 by default for local models). |
| Faithfulness / answer relevancy | RAGAS, run in a background pool of 2 threads using the same Ollama models. If RAGAS fails, token-overlap proxies are recorded instead. |

These are cheap indicators for trends on dashboards, not a substitute for an offline evaluation with a labelled dataset.

## Agent

`FinancialAgent` (`src/agents/financial_agent.py`) is built with LangChain's `create_agent` (LangGraph) and has these tools:

| Tool | Source | Purpose |
|---|---|---|
| `ibex35_financial_reports` | RAG engine | Fundamentals from the indexed reports. |
| `get_stock_price` | yfinance | Current price, daily change, 52-week range, market cap. |
| `get_price_history` | yfinance | Return, high and low over the last N days (max 365). |
| `compare_companies_price` | yfinance | Return comparison for up to 5 companies. |
| `list_ibex35_companies` | static | Companies and tickers. |

- Memory: `MemorySaver` checkpointer keyed by `thread_id`. Pass the same `thread_id` to continue a conversation; without one, each request gets a fresh, unshared memory.
- Limit: `ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS)` caps LLM calls per request (8).
- The system prompt tells the agent to use RAG for fundamentals, market tools for prices, both for complete analyses, to cite sources and to use Markdown tables for comparisons.

## Tuning reference

| Knob | Where | Default | Effect |
|---|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | env | 1200 / 150 | Chunk granularity. Requires re-ingestion. |
| `SIMILARITY_TOP_K` | env | 12 | Candidate pool is `max(2 × value, 10)` for single-company queries. |
| `RERANK_TOP_N` | env | 8 | Chunks sent to the LLM. |
| `RERANKER_MODEL` | env | `BAAI/bge-reranker-base` | Cross-encoder. Must be baked into the image (offline mode). |
| `LLM_TEMPERATURE` | env | 0.1 | Generation temperature. |
| `PER_COMPANY_K` | `engine.py` | 10 | Candidates per company in comparisons. |
| `RERANK_MIN_PER_COMPANY` | `engine.py` | 3 | Per-company floor after reranking. |
| `MIN_RERANKER_SCORE` | `engine.py` | 0.05 | Minimum score for a chunk to count as valid. |
| `MIN_CHUNKS_FOR_COVERAGE` | `engine.py` | 2 | Valid chunks needed for a company to be "covered". |
| `SEMANTIC_CACHE_THRESHOLD` | env | 0.92 | Lower values return more cached answers for paraphrases, with more risk of a wrong match. |
