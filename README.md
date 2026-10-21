# Research Landscape Engine

The engine takes a seed paper and builds an evidence-backed map of its field. The full design
is in [docs/build_recipe.md](docs/build_recipe.md). This repository implements **Phases 1–5**:
ingestion, corpus expansion, LLM facet extraction, embeddings, and the core analytical
objects. It also implements **Phase 6** signals (direction, emergence, niches, bridges with an
LLM judge, influence), a **Phase 6b** explorer prototype that shows each signal as a layer
(with corpus, facet and cluster review views), and a **Phase 8** backtest on the existing
corpus (see [08-backtest.md](docs/features/08-backtest.md)). Phase 7 (integrity) is not built. Feature docs are in
[docs/features/](docs/features/).

The [Phase 8c research lab](docs/features/08c-research-lab.md) audits sampling-sensitive
directions, compares broader bridge lists and semantic ablations, and exposes underconnected
communities and topic-specific people roles. Open **Research lab** in the explorer for the results.

## Setup

```bash
uv sync
```

Keys are read from `.env.local`. Values there take precedence over the shell environment.

| Variable | Used for |
| --- | --- |
| `OPENALEX_API_KEY` | OpenAlex (required) |
| `OPENAI_API_KEY` | OpenRouter key (`sk-or-…`) for extraction and labels (required) |
| `S2_API_KEY` | Semantic Scholar (optional; without it S2 is slow and best-effort) |

Data lives in `data/`: the DuckDB file `landscape.duckdb`, the raw response cache `raw/`, and
the vector store `vectors/`. Set `LANDSCAPE_DATA` to move it.

## Pipeline

```bash
uv run landscape new-run --run-id dro-pilot --seed-doi 10.1007/s10107-017-1172-1 --params configs/dro-pilot.json
uv run landscape expand   --run-id dro-pilot   # Phase 1-2: seed, backward, forward, author routes; S2 + arXiv abstract fill
uv run landscape extract  --run-id dro-pilot   # Phase 3: facets for the corpus so far
uv run landscape semantic --run-id dro-pilot   # Phase 2 route 5 (embeds abstracts first)
uv run landscape extract  --run-id dro-pilot   # Phase 3 for the semantic additions (incremental)
uv run landscape embed    --run-id dro-pilot   # Phase 4 + tag normalization
uv run landscape core     --run-id dro-pilot   # Phase 5
uv run landscape viz      --run-id dro-pilot   # Phase 6b: fixed UMAP layouts for the explorer
uv run landscape signals  --run-id dro-pilot   # Phase 6: signals, LLM judge, explorer layer tables
uv run landscape backtest --run-id dro-pilot   # Phase 8: cutoffs 2020-2023, precision@k vs baselines
uv run python scripts/phase8c/prepare_cutoffs.py --run-id dro-pilot --cutoffs 2019
uv run landscape research-lab --run-id dro-pilot  # offline diagnostics and experiments, including a complete chronological test
uv run landscape status   --run-id dro-pilot
uv run landscape explorer                      # http://localhost:8501 (close it before pipeline steps)
```

Every step is incremental and cached, so re-running a step only does the missing work.

## Layout

| Module | Phase |
| --- | --- |
| `clients.py`, `ingest.py`, `store.py` | 1: OpenAlex / S2 clients, parsing, DuckDB schema, raw cache |
| `expand.py` | 2: routes and provenance |
| `llm.py`, `extract.py` | 3: OpenRouter client, extraction, QA |
| `embed.py` | 4: EmbeddingGemma 2, Parquet + FAISS store, gloss blend, tag vocabulary |
| `core.py` | 5: clusters, matrix, time slices, citation flows, author profiles |
| `signals.py` | 6: direction, emergence, niches, bridges + judge, influence |
| `viz.py`, `app.py` | 6b: layout precompute and the Streamlit explorer |
| `backtest.py`, `scripts/phase8/` | 8: cutoff runs, outcomes, precision@k, analysis scripts |
| `cli.py` | one sub-command per phase |
