# Feature: Data ingestion (Phase 1)

Status: implemented · Code: `src/landscape/clients.py`, `src/landscape/ingest.py`, `src/landscape/store.py`

## What it does

Fetches paper metadata from OpenAlex (primary) and Semantic Scholar (S2, for abstract gaps and
citation contexts), and fills further abstract gaps from arXiv copies of journal papers. The
raw responses are cached and then parsed into DuckDB tables.

## Sources and calls

| Need | Call | Where |
| --- | --- | --- |
| Papers by DOI, 100 per call | `/works?filter=doi:a\|b…&per_page=100&select=…` | `OpenAlex.works_by_dois` |
| Papers by OpenAlex id, 100 per call | `/works?filter=openalex:W1\|W2…` | `OpenAlex.works_by_ids` |
| Forward citations, author works, topic works | `/works?filter=…&cursor=*` (cursor paging) | `OpenAlex.list_works` |
| Missing abstracts | `POST /graph/v1/paper/batch` (400 ids per call) | `SemanticScholar.batch_papers` |
| Citation contexts / intents / isInfluential | `GET /graph/v1/paper/{id}/references` | `SemanticScholar.references` |

Every OpenAlex call passes the API key, `per_page=100` and the `select=` list from the recipe.
Author IDs come from `authorships`; the pipeline never calls `/authors?search=`.

## Budget, rate limits and backoff

- **Budget stop.** `Budget` adds up `meta.cost_usd` from every OpenAlex response and raises
  `BudgetExceeded` once the run's `metadata_spend_cap_usd` (default $2.00) is reached. Expansion
  catches this, saves the provenance gathered so far, and stops cleanly.
- **Call log.** Every call is written to `api_calls` (run_id, source, endpoint, cost, status, ts),
  and the run's running total is stored in `runs.spend_usd.openalex`.
- **Rate.** OpenAlex: at most 16 concurrent requests and 50 req/s (the API limit is 100 req/s).
  S2: calls are serialized at ≤1 req/s.
- **Backoff.** Exponential backoff with jitter on 429/5xx and transport errors (8 tries for
  OpenAlex, 6 for S2).
- **S2 circuit breaker.** After 3 consecutive S2 failures the client disables itself for the
  rest of the run. Without an S2 key the shared pool returns many 429s, and the pipeline keeps
  going on OpenAlex data alone.

## Caching and idempotency

Raw JSON is stored gzipped under `data/raw/{source}/{kind}/{key}.json.gz`, a local stand-in for
object storage:

- `openalex/work/W….json.gz`: one file per work, written by every call that returns works.
- `openalex/{cites,author_works,topic_works}/<sha1>.json.gz`: whole paged listings, keyed by a
  hash of filter + sort + cap.
- `openalex/work_missing/`: ids OpenAlex no longer serves (merged or deleted works), so they are
  not requested again.
- `s2/paper/`, `s2/refs/`: S2 batch records and reference lists.

An entry is reused while its file is younger than the run's `freshness_days` (default 30), so
an interrupted or repeated run costs almost nothing.

## Tables written

`papers`, `authors`, `authorships` (with `institution_ids[]`), `institutions`, and `refs`. The
recipe's `references` table is called `refs` because `references` is an SQL keyword. Notes:

- Abstracts are rebuilt from `abstract_inverted_index`; `abstract_source` is `openalex` or `s2`.
  An S2-filled abstract is not overwritten by a later OpenAlex re-fetch that still lacks one.
- `arxiv_id` is taken from an arXiv DOI (`10.48550/arxiv.*`) or the primary location URL.
- `topics` (with field and subfield) is kept as JSON. It feeds the semantic route and the
  per-field limitation null rate.
- `refs` rows come from OpenAlex `referenced_works`. The S2 step fills `contexts`, `intents`
  and `is_influential` on matching edges, matched by DOI with the arXiv id as fallback.

## Abstracts from arXiv copies (`preprints.py`)

Many journal records, Elsevier especially, come without an abstract in both OpenAlex and S2
for licensing reasons. S2 also queries by DOI first, so it never looks at a preprint. A third
fill step therefore runs after the S2 fill (inside `expand`, or alone with
`landscape preprints`):

1. For corpus papers still missing an abstract, fetch only their OpenAlex `locations`
   (`select=id,locations`, 100 per call, cached under `raw/openalex/locations`). Extract an
   arXiv id from any location that has an arXiv DOI or an arxiv.org URL. The id is also saved
   to `papers.arxiv_id`.
2. If a work with that arXiv id is already in `papers` with an abstract, copy it. This covers
   a journal record and its preprint stored as separate works, like the pilot seed's
   `W2963450292` / `W1712265143`.
3. Otherwise, fetch the abstract from the arXiv API (`export.arxiv.org/api/query?id_list=…`).
   This is free, takes 50 ids per request, and runs at one request every 3.1 s. Responses are
   cached under `raw/arxiv/abs`.

Filled abstracts get `abstract_source = 'arxiv'`, so it stays visible that the text came from
the preprint, which can differ slightly from the published version. `landscape extract` then
re-extracts papers previously marked `title_only` that now have an abstract.

On the pilot, 56 of 926 missing papers (6%) had an arXiv copy, and all 56 were filled. The
remaining gap is mostly Elsevier operations-research and energy journals whose authors do not
post to arXiv.

Not built: matching by title and year against a separate OpenAlex preprint record. It would
reach preprints that OpenAlex has not linked through `locations`, but it needs search calls,
which are billed at OpenAlex's search rate, and a strict match rule to avoid false merges.
Given the 6% hit rate through `locations`, it is unlikely to close the gap.

## Not done yet / known gaps

- The OpenAlex S3 snapshot path for corpora above ~200k papers is not implemented.
- Cross-checking author papers against S2 (`/author/{id}/papers`) is not implemented.
- `.env.local` currently has no `S2_API_KEY`. Unauthenticated S2 is slow (about one successful
  call per 30–60 s) and unreliable, so abstract fill and citation intents are best-effort until a
  key is added.
