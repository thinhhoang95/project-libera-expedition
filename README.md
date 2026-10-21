# Libera Expedition

<img src="web/public/logo.png" width="72" alt="">

Libera Expedition is a web app on top of the Research Landscape Engine.

The engine takes a seed paper and builds an evidence-backed map of its field. The full design
is in [docs/build_recipe.md](docs/build_recipe.md). This repository implements **Phases 1–5**:
ingestion, corpus expansion, LLM facet extraction, embeddings, and the core analytical
objects. It also implements **Phase 6** signals (direction, emergence, niches, bridges with an
LLM judge, influence), a **Phase 6b** explorer prototype that shows each signal as a layer
(with corpus, facet and cluster review views), and a **Phase 8** backtest on the existing
corpus (see [08-backtest.md](docs/features/08-backtest.md)). Phase 7 (integrity) is not built. Feature docs are in
[docs/features/](docs/features/).

The explorer opens on **Start here** (orientation in the field) and **My paper** (which communities are
most likely to be warm to a paper, and the opportunities around it): see
[09-portfolio.md](docs/features/09-portfolio.md). What each view is backed by is in the
[findings portfolio](docs/features/findings-portfolio.md); every experiment is in the Experiments view.

## Portfolio products

Each product is a result from the experiments, placed in the explorer by its evidence. **Use** beat
its baselines on the pilot corpus; **Use with care** has known weaknesses and shows a warning;
**Describe** helps understanding but does not predict. Details, numbers and the dropped results are
in the [findings portfolio](docs/features/findings-portfolio.md).

| Product | What it gives you | Verdict | Where |
| --- | --- | --- | --- |
| Papers to bring in (and hidden gems) | Papers a community is likely to adopt, including ones cited once or never | Use | Bridges; My paper; Start here |
| Communities warm to your paper | The communities most likely to cite a given paper (top 3 hit for 69% of papers vs 25% at random) | Use | My paper |
| Borrowed ideas | Works imported from outside the field now, and where the field borrows from | Use | Bridges › From outside the field; Start here |
| Second waves | Established ideas reaching new audiences, as a watchlist | Use | Start here; My paper |
| Early work | Recent papers the field is likely to read | Use | People › Early work; Start here › Read now |
| Read this first | A reading path: classics, current method sources and quiet foundations | Use | Start here; My paper |
| Survey gap | Communities where a survey is missing and would be taken up | Use | Start here; My paper |
| Tools to learn | Stable methods that keep their share and that newcomers stay with | Use | Start here |
| Cluster pairs | Limitation → innovation links between clusters, judge-rated | Use with care | Bridges › Cluster pairs |
| Method transfer | Empty problem × method cells that are likely to fill | Use with care | Grid |
| Untried applications | Problem × application cells not yet tried | Use with care | My paper |
| Rising pockets | Local momentum in the map | Use with care | Map layer |
| Cluster direction | Growing, emerging, niche or dormant badges (off by default) | Use with care | Map; Ranking |
| Crowding gauge | Whether crowding marks attention or demand for a paper | Describe | My paper |
| Novelty, people, entry routes | Which papers are different, who plays which role, how newcomers got in | Describe | Map; People |

The [Phase 8c research lab](docs/features/08c-research-lab.md) audits sampling-sensitive
directions, compares broader bridge lists and semantic ablations, and exposes underconnected
communities and topic-specific people roles. Open **Research lab** in the explorer for the results.

## Web app

```bash
uv sync
cd web && npm install && npm run build && cd ..
uv run landscape serve                         # http://127.0.0.1:8600
```

The app is organized by projects (one field each). Every facet (problem, method, innovation, result,
limitations) has browsable categories, each with a details sheet: its papers, its trend, and how it relates
to the other facets in any time window. It shows every view of the explorer, and its job
queue runs the whole pipeline with progress you can pause and resume. Pages stay usable while a step
runs, because the app reads a copy of the database published after each step. See
[11-web-app.md](docs/features/11-web-app.md), and [11b-web-app-ux.md](docs/features/11b-web-app-ux.md) for the usability
pass (shared time window, readable map labels, CSV on every table, job notices).

## Setup

Keys are read from `.env.local`. Values there take precedence over the shell environment.

| Variable | Used for |
| --- | --- |
| `OPENALEX_API_KEY` | OpenAlex (required) |
| `OPENAI_API_KEY` | OpenRouter key (`sk-or-…`) for extraction and labels (required) |
| `S2_API_KEY` | Semantic Scholar (optional; without it S2 is slow and best-effort) |

Data lives in `data/`. Set `LANDSCAPE_DATA` to move it.
- `shared/` holds what every project reuses: the DuckDB file `landscape.duckdb`, the raw response cache
  `raw/` and the paper vector store `vectors/`.
- `projects/<id>/` holds everything one project builds: portfolio, external works, backtest, research
  lab, snapshots, run-level vectors and step logs.
- `app/` holds the web app's queue and its read copies of the database.

A data folder from before projects existed is moved into this layout with `landscape migrate-layout`
(`--dry-run` lists the moves first).

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
uv run landscape external --run-id dro-pilot   # external works the field cites (~$1 of LLM extraction, once)
uv run landscape portfolio --run-id dro-pilot  # Start here / My paper artefacts (import map, imports now, warmth)
uv run landscape my-paper --run-id dro-pilot --doi <DOI>  # any paper: communities warm to it, opportunities
uv run landscape field-report --run-id dro-pilot --model claude  # AI field intelligence report (or --model gpt)
uv run landscape score-snapshot --run-id dro-pilot --date 2026-10-21 --months 12  # the frozen prospective test
uv run python scripts/phase8c/prepare_cutoffs.py --run-id dro-pilot --cutoffs 2019
uv run landscape research-lab --run-id dro-pilot  # offline diagnostics and experiments, including a complete chronological test
uv run python scripts/run_experiments.py --step experiments-8b --run-id dro-pilot  # every experiment of a phase: 8, 8b or 10
uv run python scripts/phase8b/fit_models.py --run-id dro-pilot  # optional: this run's own production models
uv run python scripts/phase8b/freeze_snapshot.py --run-id dro-pilot  # freeze today's lists for a prospective test
uv run landscape status   --run-id dro-pilot
uv run landscape explorer                      # legacy Streamlit explorer (close it before pipeline steps)
```

Every step is incremental and cached, so re-running a step only does the missing work. To run the
whole pipeline as one resumable job instead, without the web app:

```bash
uv run landscape new-run --run-id my-field --seed-doi <DOI> --name "My field"
uv run landscape queue --run-id my-field --template map    # or full: + backtest, research lab, experiments 1-28
uv run landscape worker                                    # Ctrl-C interrupts; the next start resumes
uv run landscape jobs
```

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
| `projects.py`, `jobs.py`, `replica.py` | project layout and migration, the job queue and worker, the database read copy |
| `web/` (Python), `web/` (repo root, React) | the Libera Expedition API and frontend |
