# Handoff: resume work on another machine

Last updated 2026-10-21. Read this first, then `README.md` and `docs/features/`.

## Where things stand

| Phase | State | Doc |
| --- | --- | --- |
| 1 Ingestion (OpenAlex, S2, arXiv preprint abstracts) | done | `docs/features/01-data-ingestion.md` |
| 2 Corpus expansion (5 routes, calibrated semantic threshold) | done | `02-corpus-expansion.md` |
| 3 LLM facet extraction (DeepSeek v4.1 flash, fp8+ cheapest provider) | done | `03-facet-extraction.md` |
| 4 Embeddings (EmbeddingGemma 2, CPU) + tag vocabulary | done | `04-embeddings.md` |
| 5 Core objects (Leiden clusters with stable / **hub** / unstable kinds, matrix, flows, author profiles) | done | `05-core-objects.md` |
| 6 Signals + LLM judge | done, partly reworked by the backtest | `06-signals.md` |
| 6b Streamlit explorer (every signal as a layer; corpus, facet and cluster review) | prototype | `06b-explorer.md` |
| 7 Integrity module | **not started** (out of scope so far) | — |
| 8 Backtest on the existing corpus (cutoffs 2020–2023) | done | `08-backtest.md` |
| 8b Signal-improvement experiments | **in progress** (see below) | to write: `08b-signal-experiments.md` |

The pilot run is `dro-pilot` (seed: Esfahani & Kuhn, Wasserstein DRO, DOI
10.1007/s10107-017-1172-1): 5,410 papers, total spend about $0.65 so far. The user confirmed DRO
as the pilot field.

## Decisions the user made (keep them)

- **DRO is the pilot field.**
- **Hub clusters** (unstable but large and coherent, like the DRO core) are **kept and reported
  separately**, never dropped.
- The backtest uses **only papers already in the database** (no new downloads) unless the user
  says otherwise.
- **Commit to `main` before starting a new batch of work.**
- **For each completed feature batch, write the feature doc in `docs/features/`.**
- **Keep tests minimal**; validate by running the pipeline and the backtest.
- The user invited **imaginative experiments** to improve signal strength. Adopt only what wins
  **leave-one-cutoff-out** against baselines, and report both the recipe's and our hit
  definitions.

## In-progress work: Phase 8b signal experiments

Scripts are in `scripts/phase8b/`. Results so far (leave-one-cutoff-out, cutoffs 2020–2023):

| Experiment | Script | Result | Decision |
| --- | --- | --- | --- |
| 1. Paper-level bridges: which not-yet-cited innovation papers a problem cluster will cite | `exp1_paper_bridges.py`, `exp1_evaluate.py` | logistic model p@10 per cluster 0.234 vs popularity 0.186, random 0.017; AUC 0.84; **long-tail AUC 0.84** (popularity 0.55) | **adopted**: signals `bridge_paper` (top 10 per cluster) and `bridge_paper_gem` (top 5 papers cited ≤ 1×) |
| 2. Early-work detector for papers from year T | `exp2_early_work.py` | AUC 0.815, p@50 0.67 vs citations-so-far 0.62, coupling 0.50, base 0.19; author fame left out on purpose (adds ~nothing) | **adopted**: signal `early_work` replaces `recent_paper_coupling`; books, chapters and reference entries are excluded |
| 3a. Burstiness micro-topics | `exp3_micro_topics.py` | AUC 0.46 (mean-reverting) | dropped |
| 3b. Local momentum pockets | same script | ρ +0.13, p@50 0.17 vs 0.085 random | **adopted lightly**: `paper_scores` metric `local_momentum`. The explorer layer "Rising pockets" is **not built yet** |
| 4. Method transfer as matrix completion | `exp4_method_transfer.py` | cluster level: item CF AUC 0.70 (≥1 paper) / 0.80 (≥2) vs recipe 0.67 / 0.66, popularity 0.51 / 0.62. Tag level: popularity wins | **adopted**: `SignalParams.transfer_method = "item_cf"` |

The production models are fitted on all four cutoffs by `scripts/phase8b/fit_models.py` and
saved to `src/landscape/models/{paper_bridges,early_work}.json` (committed). `signals.py` loads
them through `model_score()`.

### State at handoff

- The code for all adopted items is in `src/landscape/signals.py` and `config.py`.
- The last `landscape signals --run-id dro-pilot` run (with hidden gems and the book filter for
  early work) **was still running at handoff and is unverified**. Re-run it first:
  `uv run landscape signals --run-id dro-pilot`. The judge costs about $0.06.
- An earlier run of the same code without those two fixes worked and produced `bridge_paper`
  (250 rows), `early_work` (50) and item-CF method transfers (174, judge top scores 4–5).

### Next steps (in order)

1. **Re-run signals** and check the output:
   - `bridge_paper_gem` rows exist.
   - `early_work` has no old textbooks at the top.
2. **Explorer:**
   - In the Bridges tab, add "Papers to bring in" (likely and hidden-gem lists per cluster, with
     the matched limitation as evidence).
   - Rename the People tab's recent-paper list to "Early work".
   - Add a **Rising pockets** layer (rings on recent papers in the top 5% of `local_momentum`).
3. **Docs:**
   - Write `docs/features/08b-signal-experiments.md` from the tables above.
   - Update `06-signals.md` (item CF transfer, paper bridges, early work, local momentum) and
     `06b-explorer.md`.
4. **Optionally add the experiments to `landscape backtest`** so they re-run with one command.
   The scripts currently depend on the cutoff runs `dro-pilot@T2020..2023`, which `backtest`
   builds.
5. **Commit.**

### Further ideas not yet tried

- An LLM pair-checker on paper bridges ("does innovation i address limitation l?"). It must be
  off, or use a pre-cutoff model, in backtests because of leakage.
- An unbiased per-year field sample (needs downloads). Cluster-trend conclusions depend on the
  expansion routes (see `08-backtest.md`, section 1).
- An off-field audit of the corpus: how many papers, and which routes brought them in.

## Setting up the new machine

1. **Code:** this repo has **no git remote**. Copy the folder (or add a remote and push from the
   old machine). Everything except `data/`, `.venv/` and `.env.local` is in git on `main`.
2. **Keys:** copy `.env.local`. It holds `OPENAI_API_KEY`, which is really the OpenRouter key
   (sk-or-…), and `OPENALEX_API_KEY`. Values in `.env.local` override the shell environment on
   purpose. No `S2_API_KEY` yet; adding one would help.
3. **Environment:** `uv sync` (Python 3.12, CPU torch). The first embedding run downloads
   EmbeddingGemma 2 (about 1.5 GB) from Hugging Face.
4. **Data:** `data/` is **not in git** (about 2 GB). It holds the DuckDB file
   `data/landscape.duckdb`, the raw API cache `data/raw/`, vectors in `data/vectors/`, and backtest
   outputs in `data/backtest/`.
   - **Best option:** copy the whole `data/` folder. Everything resumes as is.
   - **Otherwise, rebuild.** The raw cache makes this cheap if `data/raw/` is copied; without it,
     it re-fetches (OpenAlex about $0.02, extraction about $0.27, labels about $0.08, judge about
     $0.06). Embedding takes about 30 minutes on CPU:
     ```
     uv run landscape new-run --run-id dro-pilot --seed-doi 10.1007/s10107-017-1172-1 --params configs/dro-pilot.json
     uv run landscape expand   --run-id dro-pilot
     uv run landscape extract  --run-id dro-pilot
     uv run landscape semantic --run-id dro-pilot
     uv run landscape extract  --run-id dro-pilot
     uv run landscape embed    --run-id dro-pilot
     uv run landscape core     --run-id dro-pilot
     uv run landscape viz      --run-id dro-pilot
     uv run landscape signals  --run-id dro-pilot
     uv run landscape backtest --run-id dro-pilot
     ```
     After a rebuild, the run params in the DB must match the tuned ones. Check `runs.params` for
     `clusters.resolutions = {coarse: 1.0, medium: 2.0, fine: 6.0}` and
     `expansion.semantic_threshold = "auto"`. The defaults in `config.py` already have these.
5. **Explorer:** `uv run landscape explorer` serves http://localhost:8501. DuckDB allows one
   process per file: **close the explorer before running pipeline steps**.

## Machine-specific notes

- **Git safe directory:** on the old machine, git needed `-c safe.directory=<repo path>`, because
  that drive doesn't record file ownership. The user's global git config was left untouched.
- **Git user name:** the user's git `user.name` ends in a stray backslash (`Thinh Hoang\`); worth
  fixing.
- **Windows:** use `PYTHONIOENCODING=utf-8` for scripts that print non-ASCII text.
- **SSL interception:** Cloudflare WARP once caused a transient SSL "self-signed certificate"
  error. It cleared by itself.
