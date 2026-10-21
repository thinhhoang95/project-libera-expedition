# Feature: Product portfolio (Start here, My paper)

Status: implemented · Code: `src/landscape/portfolio.py`, `external.py`, `portfolio_view.py` · CLI: `landscape external`, `landscape portfolio`, `landscape my-paper` · Models: `src/landscape/models/community_warmth.json` · Evidence: [findings-portfolio.md](findings-portfolio.md), [08b-signal-experiments.md](08b-signal-experiments.md)

This phase turns the backtested results into a product for someone who wants to understand a field
and find opportunities for their own work. Each result's place in the product follows its evidence:
- **Prominent:** the results that beat their baselines.
- **Description, with a warning:** the ones that describe the field but do not predict.
- **Experiments view only:** the ones that failed.

## Run

```bash
uv run landscape external  --run-id dro-pilot   # external works cited >= 5 times: fetch, facets, embeddings (~$1 LLM once)
uv run landscape portfolio --run-id dro-pilot   # live artefacts: import map, imports now, community warmth (seconds)
uv run landscape my-paper  --run-id dro-pilot --doi 10.1007/s00440-020-00993-8   # any paper (< $0.001)
uv run landscape score-snapshot --run-id dro-pilot --date 2026-10-21 --months 12   # the prospective test
uv run landscape explorer                       # Start here and My paper are the first two views
```

Run `external` and `portfolio` after `signals`. `external` resumes where it stopped and costs
nothing on a re-run. `my-paper` makes one LLM call for a paper that is not in the map (none for a
paper already in it). Close the explorer before these steps: DuckDB allows one process per file.

## Artefacts

| File | Contents |
| --- | --- |
| `data/projects/<run>/external/` | `edges` (every citation from a corpus paper to an outside work), `works` (metadata, abstracts, subfields, yearly global citations), `facets`, `vectors_<facet>` |
| `data/projects/<run>/portfolio/import_map.parquet` | External citations by the cited work's subfield, three years six years ago vs the last three |
| `data/projects/<run>/portfolio/live_imports.parquet` | External works with ≥ 3 citing papers in the last three complete years, scored by the experiment-6 model (fitted on the cutoffs at build time); `outside_field` = the third farthest from the corpus's abstracts |
| `data/projects/<run>/portfolio/warmth.parquet` | For every corpus paper with an innovation statement and every live problem community (α 0.5, medium): the warmth features, score and rank; `home` and `already_citing` flags |
| `data/projects/<run>/portfolio/second_waves.parquet` | The second-waves watchlist (below) |
| `data/projects/<run>/portfolio/papers/<id>.json` | One analysed paper: metadata, facets, vectors, warmth rows |
| `data/projects/<run>/portfolio/summary.json` | As-of year, counts, share of references outside the map |

## Community warmth (experiment 10)

**Question.** Which communities will take up *this* paper? Among the communities that do not cite a
paper yet, rank them by:
- **coupling:** references shared with the community's papers;
- **recent closeness:** how close the community's last three years of methods and innovations are
  to the paper's innovation;
- **problem and limitation similarity;**
- **community size.**

**Backtest** (leave-one-cutoff-out, 2,592 papers with at least one warm community):

| Ranking | AUC within a paper | hit@3 | MRR |
| --- | --- | --- | --- |
| random | 0.50 | 0.25 | 0.25 |
| community size | 0.64 | 0.39 | 0.32 |
| coupling | 0.77 | 0.59 | 0.48 |
| semantic (problem, limitation, recent closeness) | 0.77 | 0.60 | 0.50 |
| **all (production)** | **0.81** | **0.69** | **0.57** |

- **Citation outcome:** for 69% of papers, one of the top three communities cites them within three
  years (25% at random).
- **Content outcome** (the community's later work moves towards the idea, cited or not): problem
  similarity alone is the best guide (hit@3 0.60 vs 0.40 at random). The view shows it as
  **content fit** next to warmth.

**Training and live computation.** The production model is fitted on all four cutoffs. Live
features are computed exactly as in training:
- coupling and the 768-d similarities as in experiment 1;
- recent closeness on 256-d vectors as in experiment 9;
- every paper assigned to the nearest live medium-community centroid.

For a paper outside the map, the *anchor community* (used for "papers to bring in" and "methods new
to your problem") is the community that cites it most. If nobody cites it, the anchor is the
community closest to its problem.

## Second waves: ideas reaching new audiences

A watchlist of **diffusion already under way**: established ideas that communities other than their
own began citing (or, for methods, using) in the last three complete years. It is not a ranking of
importance and not advance warning (see the evidence below). Three kinds:

| Kind | Idea | New audience | Likely next (model) |
| --- | --- | --- | --- |
| paper | a paper in the map, at least 3 years old | communities that first cited it in the last 3 years | top-3 warm communities not citing it yet (experiment 10) |
| external | a work outside the map from the last 15 years, in the live import list (experiment 6) | communities that first cited it in the last 3 years | the closest community |
| method | a method community | problem communities where it first appeared in the last 3 years (≥ 2 papers) | method-transfer candidates (experiment 4) |

For each idea:
- **Citing papers**, the last three years against the three before. Only ideas gaining use are
  listed, and only research works (textbooks are cited by every community).
- **Quiet uptake:** communities whose recent methods or innovations moved towards the idea without
  citing it. The definition is experiment 9's: closest statement ≥ 0.894, and closer than three to
  five years before.
- **Supporting papers:** the recent papers in the new audiences.

Pilot: 109 papers, 166 external works, 6 methods. Examples:
- *Differentiable Convex Optimization Layers* and *Finite-Sample Guarantees for Wasserstein DRO*
  reaching 6 new communities each;
- normalizing flows arriving from AI.

**Evidence:**
- The import model ordered growing imports modestly better (+0.025 AUC) but did not sharpen the top
  of the list (experiment 6).
- Method transfer: AUC 0.70–0.80 (experiment 4).
- Uncited convergence is real but no better foreseen by semantics than by persistence (experiment 9).

Hence a watchlist with evidence, not a "hotness" ranking.

## Views

### Start here (default view)

1. **Headline numbers:** papers, publication years (95% span), communities and hubs, the share of
   references pointing outside the map (88% on the pilot), external works tracked.
2. **What this map covers, and what it cannot tell you:** how the map was sampled from the seed;
   what is reliable, what needs care, and what is not covered.
3. **The communities:** size, papers in the last three years, share and change. Described as the
   map as sampled, not a forecast.
4. **Where this field borrows from:** the import map (experiment 6).
5. **Being imported now:** recent outside-the-field works the field cites more and more.
   **Read now:** early work (experiment 2).
6. **Opportunities to inspect:**
   - hidden gems (experiment 1);
   - limitation → innovation (Phase 8, with the judge's rationale);
   - method transfer (experiment 4);
   - underconnected communities (Phase 8c).
7. **Second waves:** the watchlist above, in three tabs. Selecting an idea shows its new
   audiences, quiet uptake, likely next audiences and supporting papers.
8. **Coverage gaps:** works the field cites heavily that are missing from the map. On the pilot,
   these include *Conditional value-at-risk for general loss distributions*, *Data-driven robust
   optimization* and *Adjustable robust solutions of uncertain linear programs*. They are inputs
   for a new seed or a corpus expansion.
9. **How far to trust each view:** each view's verdict and the evidence behind it.

### My paper

Pick a paper in the map (title search), or one analysed with `landscape my-paper`. The view shows:
1. **The communities most likely to be warm to it:**
   - a warmth bar (relative within the paper), content fit, and the two strongest reasons;
   - the communities already citing it, listed separately;
   - selecting a community lists its recent papers closest to yours: who to read, cite or contact.
2. **Whose stated problems it may answer:** stated limitations in other abstracts closest to the
   paper's innovation. They are candidates to read, since matching wording does not mean
   compatible assumptions.
3. **Ideas from outside the field that fit:** current imports closest to its problem, excluding
   works it already cites.
4. **Papers to bring in:** the anchor community's likely and hidden-gem lists, closest to the
   paper's problem first, excluding works it already cites.
5. **Methods new to your problem:** method transfers for the anchor community.
6. **Early work nearby.**
7. **Second waves near you:** watchlist ideas whose new audiences, quiet uptake or likely next
   audiences include the paper's home or top-3 warm communities.
8. **Active in your warmest communities:** authors with the most recent papers there, labelled as
   descriptive.
   - Each author has an **insider** marker: at least 2 earlier papers in the map. Insiders'
     papers draw more citations from others (experiments 5 and 11).
   - Big names stepping somewhere new do not start a follow-up wave (experiment 11).

Review buttons (useful / obvious / wrong) write to `review_actions` (`view = 'my_paper'`).

## Scoring the prospective snapshot

```bash
uv run landscape score-snapshot --run-id dro-pilot --date 2026-10-21 --months 12
```

1. **Integrity:** checks every frozen file against the SHA-256 hashes committed in
   `docs/snapshots/`.
2. **New papers:** takes corpus papers published after the snapshot date, within the horizon, and
   assigns them to the frozen stable and hub communities.
3. **Scores:** scores each list next to its frozen baseline, using the pre-registered hit
   definitions:
   - paper bridges and hidden gems: precision per community;
   - community warmth: whether a citing community was among the listed 3;
   - early work: top quarter of its cohort;
   - external imports: rate growth;
   - method transfer: at least 2 new papers in the cell.
   - The cluster-pair, pocket, emerging and birth lists are reported as manual.

Collect new papers first with the run's routes (`expand`, `extract`, `embed`). Today the command
reports that there are none yet.

The code paths were checked with a throwaway simulation (a snapshot dated 2025-12-31, scored on the
215 papers from 2026). In that simulation, the lists that exclude works already cited score zero by
construction, because the lists were built with the 2026 citations in view. A real snapshot freezes
its lists before the new papers exist.

## What is deliberately not in the product

- **Community birth regions** (experiment 8): not demonstrated. They are frozen in the
  prospective snapshot only.
- **Global-success prediction** (experiment 5): nothing beats the bibliometric base.
- **Direction labels** stay off by default, with a sampling warning.

All result tables, including these, are in the Experiments view.

## Checked

- `landscape external` resumes with no new spend; it now remembers works OpenAlex no longer
  serves.
- `landscape portfolio` builds in about 5 seconds and reproduces experiment 6's live list (1,538
  works, 301 outside the field). Warmth covers 113,300 paper-community rows.
- `landscape my-paper` on *All adapted topologies are equal* (not in the map): about 27 seconds,
  $0.0002. Warmest: mean-field PDEs, distribution shift, function spaces, portfolio optimization.
- In the browser at 1600×1000:
  - Start here: every section renders.
  - My paper: an in-map paper and a DOI paper; all seven sections render, and community
    selection shows evidence.
  - Bridges › From outside the field reads the portfolio files.
  - The server log shows no errors.

## Limits

- **Uptake by existing communities:** warmth and the opportunity lists are backtested only for
  communities that exist at the cutoff. They cannot show a community that does not exist yet.
- **Sampling:** the map comes from one seed's citation graph. The import map covers external works
  cited at least 5 times (about a quarter of external citations).
- **Outside papers:** a paper outside the map gets facets from one LLM call and the corpus
  embedding model. Its warmth uses its OpenAlex reference list; papers with few listed references
  lose most of the coupling signal.
