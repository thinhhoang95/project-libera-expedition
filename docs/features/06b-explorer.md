# Feature: Visual explorer prototype (Phase 6b, first part)

Status: prototype · Code: `src/landscape/app.py` (Streamlit + Plotly), `src/landscape/viz.py` (precompute) · CLI: `landscape viz`, `landscape explorer`

This is the prototype from the recipe: Streamlit with Plotly, reading DuckDB directly. It now
shows every Phase 6 signal as a layer or panel, as the recipe's "definition of done" requires
(see [06-signals.md](06-signals.md)). Signal layers are greyed out until `landscape signals`
has run.

## Run it

```bash
uv run landscape viz --run-id dro-pilot   # once per run, after `core` (UMAP layouts, ~5 min on CPU)
uv run landscape explorer                 # http://localhost:8501
```

DuckDB allows only one process per database file, so **close the explorer before running
any pipeline step** and reopen it afterwards.

## Precomputed tables (`landscape viz`)

| Table | Contents |
| --- | --- |
| `viz_layout(run_id, facet, alpha, paper_id, x, y)` | One UMAP fit per facet and α (cosine, 20 neighbours, `random_state=42`) on the 256-d vectors. Facets: abstract, problem and method (α = 0, 0.3, 0.5, 0.8), innovation, result, limitations. The fit is stored once, so the map never moves between sessions or years |
| `viz_cluster_layout(run_id, cluster_id, x, y, color)` | Median position of a cluster's papers on its own facet/α layout |
| `viz_cluster_year(run_id, cluster_id, year, n, share, status, scores)` | Yearly cluster rows. `status` and `scores` are filled by Phase 6 |

New papers are not placed with `UMAP.transform` yet. Re-running `viz` refits the layout.

## Views

### Map & grid

There is one map, one grid and one time axis, as in the recipe.

- **Layer panel** (sidebar): facet, gloss weight α (snaps to 0 / 0.3 / 0.5 / 0.8; disabled for
  facets without a gloss), resolution, and the layer toggles. A hint appears when more than
  three analytic layers are on, and a slider sets how many arcs/arrows are shown (default 30).
  The layers:
  - *Papers*, *Clusters*.
  - *Activity shift*, with "compare with previous window".
  - *Status badges*: ▲ emerging · ◆ niche · ○ dormant at the playhead year; hubs are drawn as
    diamonds and labelled "◇hub".
  - *Citation flows*: curved arcs between cluster centroids. Width is citations in the window;
    green or red shows the change per year against the previous window; top N only.
  - *Author migration*: arrows, with width set by the number of authors whose dominant cluster
    moved within the window.
  - *Novel papers*: open rings on the top 5% novelty papers in the window (novelty of the map's
    facet).
  - *Rising pockets*: green rings on recent papers in the top 5% of `local_momentum`, which
    marks neighbourhoods filling up fastest. Only papers from the last two years have it, so
    the window must reach the end. It is weak (p@50 about 2× random) and meant as a pointer,
    not a forecast.
  - *Bridges*: accent-coloured dashed arcs for the top limitation → innovation and
    semantic-not-cited candidates. Drawn on problem, α = 0.5, medium, where they are computed.
  - *People*: pick a seeder or broker to see their yearly path on the map, ending at the
    playhead.
- **Map**: papers in the window are coloured by cluster, papers before the window fade to
  grey, and later papers are hidden. Each cluster has a marker; the 12 largest in the window
  (plus the selection) get a text label sized by window share, and the rest show their label
  on hover. *Activity shift* draws density contours for the window. In compare mode it draws
  the density difference against the window of the same length just before (red = gain,
  blue = loss), computed in the browser session rather than read from a stored `viz_density`.
  Clicking a paper opens its card and selects its cluster.
- **Time axis**: corpus papers per year, the window outlined in the accent colour, and the
  selected cluster's share line. A range slider sets the window, and **▶ Play** steps
  year by year.
- **Problem × method grid**: medium clusters at the map's α (0.3 for facets without a
  gloss). Cell colour is log(papers in the window). Cells whose first nonzero year falls in
  the last 2 years of the window are outlined in the accent colour ("new combinations",
  recipe 6.2). The selected cluster's row or column is outlined.
- **Detail**: a cluster card (label, description, size, stability, papers in the window,
  c-TF-IDF terms, the 10 central papers as evidence) and a paper card (title with a DOI link,
  year, venue, entry route, authors, every facet and gloss, and the raw abstract with its
  source).
- **Ranking**: separate tables for stable and hub clusters, sortable by size, papers in the
  window, window share, Δ share, and the as-of scores (growth g, emergence E, niche N, net
  migration, status). A toggle switches between the whole corpus and the corpus without the
  semantic route. Selecting a row selects the cluster everywhere.
- **Bridges**: two modes.
  - *Cluster pairs*: candidates by type, with the judge's 1–5 rating, rationale and "obvious
    reason". The detail shows the limitation ↔ innovation pairs (facet text, year, cosine) or
    the evidence papers.
  - *Papers to bring in*: pick a problem cluster (largest first; defaults to the selected
    cluster) to see two lists: the 10 innovation papers it is most likely to start citing,
    and up to 5 hidden gems cited at most once. Selecting a paper shows its innovation text,
    the cluster's closest limitation, the model features, and the paper card in Detail.
  - In both modes, reviewers rate each candidate *worth pursuing / obvious / wrong*, which is
    written to `review_actions` (`view = 'bridges'`).
- **People**: seeders, brokers and institutions, labelled as internal decision support, then
  **Early work**: the recent papers the backtested model expects to be top-cited for their
  year. Selecting one opens its card in Detail.
- **Grid**: at α = 0.5 the top 25 method-transfer candidates are outlined with accent diamonds
  (new combinations stay as squares).

The accent colour (`#e8590c`) is used only for the selection, the window/playhead and new
grid cells.

### Corpus view (Phase 2)

Shows papers, missing-abstract rate and spend, plus papers per year stacked by entry route
(years before 2010 grouped). Tables list routes and hops (missing abstracts, S2/arXiv fills,
year span) and the primary OpenAlex topics split into citation routes vs the semantic route,
which is the seed-bias check. You can browse papers by route with their paper cards. **Mark
composition checked** writes `composition_checked` with your notes.

### Facet review (Phase 3)

- **Gold set**: a fixed sample of 200 papers, the first 200 by `md5(run_id|paper_id)` among
  successfully extracted papers, so every reviewer sees the same set. Two other pools are
  available: QA-flagged papers and papers with a non-null limitation.
- **Layout**: the abstract sits beside the six faithful facets, each rated *supported /
  partly supported / unsupported*, and the two glosses, each rated *correct variant / variant
  unclear (ok) / wrong variant*.
- **No default rating.** Every field must be rated before saving, because a preselected
  "supported" would bias the gold set upwards.
- **Gold-set acceptance** is computed live across all reviewers against the recipe targets:
  ≥ 90% faithful facets supported, < 5% unsupported limitations, ≥ 85% correct gloss variant.

### Cluster review (Phase 5)

Pick a facet, α and resolution to list clusters of 5 or more papers, including unstable
ones. A cluster's card shows its label, description, labelling model, parent, stability,
terms, share over time, and the 12 central papers with their facet text. Actions are
*approve / rename / merge into / split / flag*, each with a note.

## Review actions

Every judgment is stored in `review_actions(run_id, reviewer, view, subject, action, note,
alpha, ts)`. `subject` is JSON (for example `{"paper_id", "facet", "model_id",
"prompt_version", "gold_set"}` or `{"cluster_id", "facet", "resolution", "merge_into"}`), and
`alpha` records the gloss weights in effect. A reviewer name is entered once in the sidebar.

## Supporting change in Phase 5

`paper_cluster.raw_cluster_id` now keeps every paper's raw Leiden cluster, including unstable
ones, whose `cluster_id` stays NULL for the signals. Without it, the cluster review could not
show an unstable cluster's members.

## Checked

The four views were exercised in the browser pane at 1600×1000:

- selecting a cluster through the Detail picker,
- the grid's new-combination outlines,
- the corpus view,
- facet review on the gold set,
- a cluster-review save, confirmed in DuckDB and then deleted.

Not exercised: the Play animation and the activity-shift compare mode.

The Phase 8b additions were checked the same way: *Papers to bring in* (cluster picker, both
lists, limitation evidence), the *Early work* list opening a paper card, and the *Rising
pockets* layer (29 rings at 2018–2026).

## Not built yet

- A stored `viz_density` table (density is computed live), and edge bundling for arcs.
- PNG/SVG snapshot and animation export, edge bundling, and the production React + deck.gl app.
