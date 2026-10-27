# Exploring the vector space in Qdrant

A catalogue of queries for the Qdrant dashboard that explain how the chunks are distributed in the `ibex35` collection: what groups together, what stands apart, and whether the metadata behind the retrieval filters is right.

## Contents

- [Where to run them](#where-to-run-them)
- [Before you start](#before-you-start)
- [Payload fields](#payload-fields)
- [1. Exact counts (Console)](#1-exact-counts-console)
- [2. The whole collection](#2-the-whole-collection)
- [3. Companies and sectors](#3-companies-and-sectors)
- [4. Text versus tables](#4-text-versus-tables)
- [5. Periods](#5-periods)
- [6. Documents and position in the document](#6-documents-and-position-in-the-document)
- [7. Topics](#7-topics)
- [8. Language](#8-language)
- [9. Metadata quality](#9-metadata-quality)
- [Diagnosing an isolated cluster](#diagnosing-an-isolated-cluster)
- [What the distribution has shown so far](#what-the-distribution-has-shown-so-far)

## Where to run them

Open `http://localhost:6333/dashboard` → **Collections** → `ibex35`. The collection page has several tabs:

| Tab | Use it for |
|---|---|
| **Visualize** | A 2D projection of the vectors, coloured by any payload field. Every query in sections 2–9 goes here. |
| **Graph** | Walking the nearest-neighbour graph from one point. It colours nodes by exploration state (start, expanded, neighbour), never by payload, so it cannot segment by company. |

**Console** (in the left sidebar) runs raw REST calls such as the ones in [section 1](#1-exact-counts-console). `POST ...` syntax does not work in the Visualize or Graph editors, which only accept their own JSON.

Visualize takes three keys:

```json
{
  "limit": 1000,
  "filter": { "must": [] },
  "color_by": { "payload": "metadata.company" }
}
```

## Before you start

- **`limit` is a sample.** Visualize projects only the points it fetches. There is no "all" option: to see the whole collection, set `limit` to the total from the [count query](#1-exact-counts-console). The legend shows how many points of each colour it drew, so if the counts add up to `limit` you are not seeing everything.
- **The projection runs in the browser.** A few thousand points take seconds. Tens of thousands can freeze the tab, and very large requests have knocked Qdrant over in this setup ("Failed to fetch" on every call). Start at 1000 and go up.
- **A filter changes the map, not just the colours.** The projection is computed only from the returned points, so a filtered view is a new map. To see where a group sits *within* the whole collection, colour by the field instead of filtering on it.
- **The layout is random.** Each run places the clusters somewhere else. Compare which points group together, not where the groups are.
- **Colours repeat.** With 35 companies the palette runs out and reuses colours, including grey. Hover a point to see its payload, or filter down to a handful of values.
- **Values are case-sensitive strings.** Companies are upper case (`"INDITEX"`). `fiscal_year` is a string (`"2026"`, not `2026`).

## Payload fields

LangChain stores the chunk text in `page_content` and everything else under `metadata`:

| Field | Values | Indexed |
|---|---|---|
| `page_content` | The chunk text | No |
| `metadata.company` | One of the 35 keys in `COMPANY_TICKER_MAP` (`INDITEX`, `SANTANDER`, ...) | Yes |
| `metadata.ticker` | `ITX.MC`, `SAN.MC`, ... or `UNKNOWN` | No |
| `metadata.fiscal_year` | `"2026"`, `"2025"`, ... Taken from the chunk itself when it mentions a year, so comparatives carry the earlier year. | Yes |
| `metadata.fiscal_quarter` | The document's period (`h1`, `q3`, ...) or the one the chunk mentions (`Q1`–`Q4`, `3M`, `6M`, `9M`) | Yes |
| `metadata.doc_type` | `financial_results`, `quarterly_results`, `annual_results`, `annual_report`, `10-k`, `10-q` | No |
| `metadata.content_type` | `text`, `table`, `image` | No |
| `metadata.source_file` | Document name, e.g. `INDITEX financial_results h1 2026` | No |
| `metadata.page` | Page number in the PDF | No |
| `metadata.chunk_index` | Position of the chunk within its page, from 1 | No |
| `metadata.file_hash` | SHA of the source file | No |

Filters on non-indexed fields work, but they scan the whole collection.

## 1. Exact counts (Console)

Run these first, so you know what the samples in Visualize are drawn from.

Total points, which you can use as `limit` to see everything:

```
POST collections/ibex35/points/count
{ "exact": true }
```

Chunks per company (works on indexed fields only: `company`, `fiscal_year`, `fiscal_quarter`):

```
POST collections/ibex35/facet
{ "key": "metadata.company", "limit": 50, "exact": true }
```

```
POST collections/ibex35/facet
{ "key": "metadata.fiscal_year", "limit": 50, "exact": true }
```

```
POST collections/ibex35/facet
{ "key": "metadata.fiscal_quarter", "limit": 50, "exact": true }
```

For non-indexed fields, count with a filter, one value at a time:

```
POST collections/ibex35/points/count
{
  "exact": true,
  "filter": { "must": [{ "key": "metadata.content_type", "match": { "value": "image" } }] }
}
```

```
POST collections/ibex35/points/count
{
  "exact": true,
  "filter": { "must": [{ "key": "metadata.ticker", "match": { "value": "UNKNOWN" } }] }
}
```

**What to look for:** companies with far more chunks than the rest (they dominate any unfiltered sample), companies missing altogether, and an `image` count of 0, which means the image descriptions are not being indexed.

## 2. The whole collection

The same sample coloured five ways. Run them one after another to see which field explains the big clusters.

By company:

```json
{
  "limit": 2000,
  "color_by": { "payload": "metadata.company" }
}
```

By content type:

```json
{
  "limit": 2000,
  "color_by": { "payload": "metadata.content_type" }
}
```

By document type:

```json
{
  "limit": 2000,
  "color_by": { "payload": "metadata.doc_type" }
}
```

By year:

```json
{
  "limit": 2000,
  "color_by": { "payload": "metadata.fiscal_year" }
}
```

By period:

```json
{
  "limit": 2000,
  "color_by": { "payload": "metadata.fiscal_quarter" }
}
```

**What to look for:** a field whose colours line up with the clusters explains them. If none of them does, the clusters come from the text itself: topic, format or [language](#8-language).

## 3. Companies and sectors

One company at a time, coloured by document. This shows whether a company's chunks form one region or spread over the whole map:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.company", "match": { "value": "INDITEX" } }]
  },
  "color_by": { "payload": "metadata.source_file" }
}
```

Two companies from different sectors. If they barely overlap, sector vocabulary is strong enough to separate them:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.company", "match": { "any": ["INDITEX", "SANTANDER"] } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Two direct competitors. Heavy overlap is expected, and it is exactly the case where retrieval relies on the company filter:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.company", "match": { "any": ["IBERDROLA", "ENDESA"] } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Banks:

```json
{
  "limit": 3000,
  "filter": {
    "must": [{ "key": "metadata.company", "match": { "any": ["SANTANDER", "BBVA", "CAIXABANK", "SABADELL", "BANKINTER", "UNICAJA"] } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Energy and utilities:

```json
{
  "limit": 3000,
  "filter": {
    "must": [{ "key": "metadata.company", "match": { "any": ["IBERDROLA", "ENDESA", "NATURGY", "REDEIA", "ENAGAS", "ACCIONAENERGIA", "SOLARIA", "REPSOL"] } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Construction, infrastructure and real estate:

```json
{
  "limit": 3000,
  "filter": {
    "must": [{ "key": "metadata.company", "match": { "any": ["ACS", "FERROVIAL", "SACYR", "ACCIONA", "AENA", "COLONIAL", "MERLINPROPERTIES"] } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Banks and insurer against everyone else. Financial companies report with a different vocabulary (net interest income, NPLs, CET1), so they should form their own region:

```json
{
  "limit": 2000,
  "filter": {
    "must_not": [{ "key": "metadata.company", "match": { "any": ["SANTANDER", "BBVA", "CAIXABANK", "SABADELL", "BANKINTER", "UNICAJA", "MAPFRE"] } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Run it once with `must_not` and once with `must` and compare how compact each map is.

## 4. Text versus tables

Tables only, by company. A Markdown table looks more like another table than like its own company's text, so expect less separation by company than in prose:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.content_type", "match": { "value": "table" } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Text only, by company:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.content_type", "match": { "value": "text" } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Image descriptions, if any were indexed:

```json
{
  "limit": 1000,
  "filter": {
    "must": [{ "key": "metadata.content_type", "match": { "value": "image" } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

One company's text and tables together:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.company", "match": { "value": "IBERDROLA" } }]
  },
  "color_by": { "payload": "metadata.content_type" }
}
```

## 5. Periods

Chunks take their year and quarter from their own text when it mentions one, so a single report holds several periods (current figures plus comparatives).

Current year against comparatives, by year:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.fiscal_year", "match": { "any": ["2025", "2026"] } }]
  },
  "color_by": { "payload": "metadata.fiscal_year" }
}
```

One period, by company. With the period fixed, whatever separation remains comes from the company or the topic:

```json
{
  "limit": 2000,
  "filter": {
    "must": [
      { "key": "metadata.fiscal_year", "match": { "value": "2026" } },
      { "key": "metadata.fiscal_quarter", "match": { "value": "h1" } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

One company across periods:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.company", "match": { "value": "SANTANDER" } }]
  },
  "color_by": { "payload": "metadata.fiscal_quarter" }
}
```

Cumulative periods (`3M`, `6M`, `9M`) against quarters:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.fiscal_quarter", "match": { "any": ["3M", "6M", "9M", "Q1", "Q2", "Q3", "Q4"] } }]
  },
  "color_by": { "payload": "metadata.fiscal_quarter" }
}
```

**What to look for:** if periods do not separate (and they usually do not, because the same metric for two periods reads almost the same), retrieval depends on the year and quarter filters, and they must be right.

## 6. Documents and position in the document

Every document in a sample, by file. A file that forms its own island points to a format or language difference in that PDF:

```json
{
  "limit": 2000,
  "color_by": { "payload": "metadata.source_file" }
}
```

Opening pages (cover, highlights, key figures), by company:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.page", "range": { "lte": 3 } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Back pages (notes, legal disclaimers, glossaries). Disclaimers are near-identical across companies and tend to form one tight, mixed cluster:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.page", "range": { "gte": 40 } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

First chunk of each page, by company. These often carry the running header or section title:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "metadata.chunk_index", "match": { "value": 1 } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

## 7. Topics

`match.text` on a field without a full-text index (such as `page_content`) is a case-sensitive substring match. Use it to see where a theme lives and which companies talk about it.

Profitability:

```json
{
  "limit": 2000,
  "filter": {
    "must": [{ "key": "page_content", "match": { "text": "EBITDA" } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Dividends and shareholder remuneration:

```json
{
  "limit": 2000,
  "filter": {
    "should": [
      { "key": "page_content", "match": { "text": "dividend" } },
      { "key": "page_content", "match": { "text": "dividendo" } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Debt:

```json
{
  "limit": 2000,
  "filter": {
    "should": [
      { "key": "page_content", "match": { "text": "deuda neta" } },
      { "key": "page_content", "match": { "text": "net debt" } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Bank capital and asset quality:

```json
{
  "limit": 2000,
  "filter": {
    "should": [
      { "key": "page_content", "match": { "text": "CET1" } },
      { "key": "page_content", "match": { "text": "morosidad" } },
      { "key": "page_content", "match": { "text": "NPL" } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Statements and their format, by content type:

```json
{
  "limit": 2000,
  "filter": {
    "should": [
      { "key": "page_content", "match": { "text": "Balance" } },
      { "key": "page_content", "match": { "text": "Cuenta de resultados" } },
      { "key": "page_content", "match": { "text": "Income statement" } }
    ]
  },
  "color_by": { "payload": "metadata.content_type" }
}
```

Boilerplate (forward-looking statements, legal notices):

```json
{
  "limit": 2000,
  "filter": {
    "should": [
      { "key": "page_content", "match": { "text": "forward-looking" } },
      { "key": "page_content", "match": { "text": "Aviso legal" } },
      { "key": "page_content", "match": { "text": "Disclaimer" } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

**What to look for:** a topic that forms one mixed cluster is described the same way by every company, so the company filter matters most there. A topic split into single-colour clumps uses company-specific wording.

## 8. Language

Embedding models separate languages even when the content is the same, so a cluster mixing companies and content types often turns out to be the reports in the other language. Substring matches on common words make a rough language detector.

Chunks that read as English, by company. The legend counts tell you which companies publish in English:

```json
{
  "limit": 2000,
  "filter": {
    "must": [
      { "key": "page_content", "match": { "text": " the " } },
      { "key": "page_content", "match": { "text": " and " } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Chunks that read as Spanish:

```json
{
  "limit": 2000,
  "filter": {
    "must": [
      { "key": "page_content", "match": { "text": " de " } },
      { "key": "page_content", "match": { "text": " los " } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Both languages in one map. Neither filter can be coloured directly, so run the unfiltered company view from [section 2](#2-the-whole-collection) and hover the points of each cluster to read their `page_content`.

## 9. Metadata quality

Chunks whose company is not in the ticker map:

```json
{
  "limit": 1000,
  "filter": {
    "must": [{ "key": "metadata.ticker", "match": { "value": "UNKNOWN" } }]
  },
  "color_by": { "payload": "metadata.source_file" }
}
```

Chunks with no year:

```json
{
  "limit": 1000,
  "filter": {
    "should": [
      { "is_null": { "key": "metadata.fiscal_year" } },
      { "is_empty": { "key": "metadata.fiscal_year" } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Chunks with no period:

```json
{
  "limit": 1000,
  "filter": {
    "should": [
      { "is_null": { "key": "metadata.fiscal_quarter" } },
      { "is_empty": { "key": "metadata.fiscal_quarter" } }
    ]
  },
  "color_by": { "payload": "metadata.company" }
}
```

Years outside the expected range, which usually come from a chunk that mentions a historic or target year (e.g. "2030 strategic plan"):

```json
{
  "limit": 1000,
  "filter": {
    "must_not": [{ "key": "metadata.fiscal_year", "match": { "any": ["2024", "2025", "2026"] } }]
  },
  "color_by": { "payload": "metadata.fiscal_year" }
}
```

Documents of the generic type, where neither the filename nor the content gave a more specific one:

```json
{
  "limit": 1000,
  "filter": {
    "must": [{ "key": "metadata.doc_type", "match": { "value": "financial_results" } }]
  },
  "color_by": { "payload": "metadata.company" }
}
```

**What to look for:** any of these with many points means a retrieval filter will miss chunks, because a question filtered on `fiscal_year = "2026"` never sees a chunk whose year is missing or wrong.

## Diagnosing an isolated cluster

When a cluster sits apart from the main cloud:

1. Colour the same sample by `metadata.company`, `metadata.content_type`, `metadata.doc_type`, `metadata.fiscal_year` and `metadata.source_file` in turn ([section 2](#2-the-whole-collection), [section 6](#6-documents-and-position-in-the-document)). The first field whose colours match the cluster explains it.
2. If none does, hover five or six of its points and read `page_content`. Look for a different language, boilerplate, tables of figures only, or OCR noise.
3. Confirm the hypothesis with a filter from [section 7](#7-topics) or [section 8](#8-language), and check its size with a count from [section 1](#1-exact-counts-console).

## What the distribution has shown so far

Observed on a 2000-point sample of the collection in October 2026:

- **Chunks group by topic, not by company.** The main cloud mixes every company. A paragraph about one company's EBITDA is closer to another company's EBITDA paragraph than to its own governance section. Dense search alone does not separate companies, which is why retrieval filters on `metadata.company`.
- **Tables form their own region** inside the main cloud. The BM25 half of the hybrid search helps retrieve them when a question names a figure or line item.
- **One island sits apart** with mixed companies, text and tables. Content type does not explain it, and language is the leading suspect ([section 8](#8-language)). If confirmed, Spanish questions will retrieve English-language reports less well, and a multilingual embedding model or query translation would fix it.
- **No `image` chunks appeared in the sample.** Check with the count in [section 1](#1-exact-counts-console) whether image descriptions are being indexed at all.
