# Feature: Corpus expansion (Phase 2)

Status: implemented · Code: `src/landscape/expand.py` · CLI: `landscape expand`, `landscape semantic`

## What it does

Expansion grows one to five seed papers, plus optionally the references of a deep research
report, into a corpus of a few thousand papers. Each route is capped, and every paper gets a
`provenance` row (run_id, paper_id, route, hop, via_id, score). Routes are filled in this
order until the paper budget is reached:

| Route | `route` value | How papers are chosen |
| --- | --- | --- |
| 1. Seeds | `seed` | DOIs or OpenAlex ids given to `new-run` |
| 1b. AI report | `ai_report` | DOIs found in the attached report and resolved on OpenAlex (see below) |
| 2a. Backward hop 1 | `back`, hop 1 | All references of the seeds |
| 2b. Backward hop 2 | `back`, hop 2 | For each hop-1 paper, up to `hop2_keep_per_paper` (15) references. Substantive edges only (S2 `isInfluential` or `methodology` intent) when that paper has S2 data; otherwise the top-k by co-citation count inside the corpus, ties broken by `cited_by_count`. `score` = co-citation count |
| 3. Forward | `forward`, hop 1 | Papers citing a seed (cap `forward_per_seed` = 1000) or a hop-1 paper (cap `forward_per_paper` = 100) in the last `forward_years` (5), fetched by citation count. Candidates are ranked by `score` = (# seed/hop-1 papers cited) + 10 × (# seeds cited), so papers citing the seed come first |
| 4. Author fan-out | `author`, hop 1 | Authors of the seeds, then authors of substantively cited hop-1 papers (S2), or, when S2 data is missing, of the 15 hop-1 papers most co-cited in the corpus. Up to `max_authors`; each author's newest `works_per_author` (50) works from the last `author_years` (5). `via_id` = author id |
| 5. Semantic | `semantic`, hop 1 | Runs after a first abstract-embedding pass (see below). `via_id` = OpenAlex topic id, `score` = cosine to the nearest centroid |

`Corpus.add` enforces both the global `paper_budget` and the per-route cap in
`route_caps`, so a large route such as author fan-out cannot crowd out the others.

### Semantic route (counter to seed bias)

`landscape semantic` does the following:

1. Embeds the abstracts of the corpus built so far (Phase 4 embedder).
2. Takes the `semantic_topics` (8) most common primary OpenAlex topics in the non-semantic corpus.
3. Pulls up to `semantic_candidates_per_topic` (250) recent works for each topic, newest five
   years, sorted by citations, abstract required.
4. Fits KMeans centroids on the corpus abstract vectors (256-d; k = n/150, clipped to [10, 60]).
   Phase 5 clusters do not exist yet at this point, so these centroids stand in for them.
5. Keeps candidates whose cosine to the nearest centroid is at least the threshold, best
   first, up to the route cap. Each run of `semantic` clears and refills the route, so a new
   threshold takes effect without re-fetching anything.

**Threshold calibration (deviation from the recipe's fixed 0.75).** With EmbeddingGemma 2,
cosine scores sit high. On the pilot, even plainly off-topic candidates (group theory,
Liouville quantum gravity, a GIS healthcare study) scored about 0.876, so 0.75 let everything
through. The default is therefore `semantic_threshold = "auto"`. It uses the
`semantic_auto_quantile` (0.10) of the existing corpus's own nearest-centroid similarity:
a candidate must sit at least as close to the field as the closest 90% of papers already in
it. This works for any embedding model. A number can still be passed to override it. The
command prints the chosen threshold and the candidates' similarity quantiles.

### AI research report seed

Pass `--ai-report <file>` to `new-run`. DOIs are extracted from the file with a regex and
resolved on OpenAlex. Anything that does not resolve is dropped. The report file is copied to
`data/runs/<run_id>/`. Resolved papers enter as route `ai_report` and are then treated like
seeds. The report text itself is never passed to any LLM step.

Limitation: matching by exact title plus year is not implemented, because it needs OpenAlex
search calls, which are billed at search rates. References without a DOI are dropped.

## Defaults

Defaults follow the recipe (`ExpansionParams` in `config.py`). Override any of them per run
with a JSON file: `landscape new-run --params configs/<file>.json`.

| Parameter | Default |
| --- | --- |
| `max_hops` | 2 |
| `hop2_keep_per_paper` | 15 |
| `forward_years` / `author_years` | 5 / 5 |
| `works_per_author` | 50 |
| `semantic_threshold` | `"auto"` (10th percentile of corpus own-centroid similarity; 0.897 on the pilot) |
| `paper_budget` | 20,000 |
| `route_caps` | back 4000, forward 6000, author 6000, semantic 4000 |
| `metadata_spend_cap_usd` | 2.00 |
| `s2_contexts_for` | `seed_hop1` (S2 contexts for seed and hop-1 papers only) |

After routes 1–4, S2 fills missing abstracts for the whole corpus, best-effort.

## Usage

```bash
uv run landscape new-run --run-id dro-pilot --seed-doi 10.1007/s10107-017-1172-1 --params configs/dro-pilot.json
uv run landscape expand --run-id dro-pilot
uv run landscape semantic --run-id dro-pilot
uv run landscape status --run-id dro-pilot
```

Re-running `expand` is cheap, because every OpenAlex and S2 response is cached.

## Pilot run results

See [pilot-run-dro.md](pilot-run-dro.md).

## Known gaps

- The corpus view (Phase 6b inspection view) is not built yet. For now `landscape status`
  prints counts by route and hop, and the missing-abstract rate.
- Signals with and without the semantic route (Phase 6) can filter on `provenance.route`. This
  is not wired up yet, because Phase 6 is out of scope.
