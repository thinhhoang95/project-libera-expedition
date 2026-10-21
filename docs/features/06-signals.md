# Feature: Signals and scoring (Phase 6)

Status: implemented · Code: `src/landscape/signals.py` · CLI: `landscape signals [--skip-judge]` · Params: `SignalParams` in `config.py`

Every signal writes ranked rows to `signals(run_id, signal_type, subject, score, rank, features,
evidence_paper_ids, judge_score, judge_rationale, cutoff_year, variant, group_kind)`. Each
row carries the evidence papers a reviewer needs. All code takes a cutoff year T and uses only
papers and citations up to T, so the Phase 8 backtest can call the same functions. The
weights are the recipe's starting values; the backtest is meant to tune them.

## Run

```bash
uv run landscape signals --run-id dro-pilot     # ~3 min on the pilot, ~$0.06 for the judge
```

Run it after `core` and `viz`. It also writes the explorer's `viz_cluster_year` statuses,
`viz_flow_year` (citation and migration flows), `viz_author_year` (author paths), and
`paper_scores` (novelty per facet, CD index).

## Hub clusters (reported separately, not dropped)

Phase 5 now gives each cluster a `kind`:

- **`stable`**: stability ≥ 0.6.
- **`hub`**: unstable, but at least 30 papers (and at least 1% of the facet's papers) and at
  least as coherent as the 25th percentile of the stable clusters in the same clustering.
- **`unstable`**: everything else.

Hubs keep their papers, get LLM labels and enter every signal. Each ranked signal list is then
split by `group_kind` (`stable` / `hub`), so a hub is never mixed into, or lost from, the
stable rankings. `clusters.centrality` (the z-score of a centroid's mean cosine to the other
centroids) identifies the central hubs.

On the pilot, the DRO core (`problem:0.3:medium:008`, 246 papers, stability 0.50) is a hub
labelled "Distributionally Robust Optimization". Papers assigned to a cluster rose from 78–91%
to 92–99% per medium clustering. Most hubs are coherent topics whose boundary moves between
seeds (energy management, OPF, EV charging), not noise. At α = 0.5 the DRO core is a *stable*
cluster (361 papers): blending in the glosses steadies it.

## Clusterings used

| Signals | Clustering |
| --- | --- |
| 6.1–6.3 (cluster level) | medium; problem and method at α = 0.3, innovation and result at α = 0 (`cluster_sets`) |
| 6.2 paper novelty | α = 0 vectors for problem, method, innovation, abstract |
| 6.2 new combinations | problem × method at α = 0.3 (`grid_alpha`, same as the grid) |
| 6.4 bridges | problem and method clusters at α = 0.5 (recipe α table) |
| 6.5 seeders / brokers | problem, α = 0.3 |

## Cutoffs and corpus variants

- Cluster signals are computed at **every cutoff year** in which that year and the two before it
  each have ≥ 100 papers (pilot: 2020–2026). This drives the map's statuses at the playhead.
- At the latest cutoff they are ranked twice: `variant = 'all'`, and `variant = 'no_semantic'`
  with the semantic-route papers removed. This keeps seed bias visible.
- Bridges and influence are computed at the latest year only, as the recipe allows for v1.

## 6.1 Direction

| Feature | Definition |
| --- | --- |
| `g` | slope of log share over three buckets ending at T (1-year buckets, or 2-year buckets for clusters with `time_bucket_years = 2`); share = (n + 0.5) / N |
| `dg` | `g(T) − g(T − bucket)` (acceleration) |
| `mig_in`, `mig_out`, `mig_net` | authors whose dominant cluster was i in t−1 and j ≠ i in t, summed over T−2…T |
| `ref_age` | median age of references cited by the cluster's papers in T−2…T (only references whose year is known, i.e. works we fetched) |

- **Signals:** `direction_growing` (E′ > 0, ranked by E′) and `direction_shrinking` (E′ < 0,
  ranked by −E′), for clusters with at least 20 papers. The recipe ranks by g, which the backtest
  found below random (mean reversion); the backtested E′ beats every baseline. g is still stored
  as a feature. Smaller clusters gave noisy slopes; a 5-paper "AR display" cluster topped the
  first run.
- **Baseline shift:** not computed. S2 intents cover only 51 papers on the pilot (no S2 key),
  far too few. The edge-level intents are in `refs` for when they become available.

## 6.2 Emerging topics

- **`novel_paper`**: `nov_f(x) = 1 − max cos(e_f(x), e_f(y))` over papers from earlier years
  (exact FAISS search). Each facet's top 50 from the last 3 years is ranked, and all values are
  in `paper_scores`.
- **`new_combination`**: problem × method cells whose first nonzero year is within the last 2
  years.
- **`emerging_cluster`**: ranked by **E′ = z(momentum) + z(dg) − z(new-author share)** for
  clusters with ≥ 10 papers (`emergence_score = "backtested"`).
  - The recipe's `E = z(g) + z(dg) + z(new-author share) − z(ref age) + z(mean novelty)` is still
    computed and stored as `E_recipe`. Set `emergence_score = "recipe"` to rank by it.
  - The Phase 8 backtest found the recipe's E anti-predictive on the pilot (ρ = −0.22 with the
    future share change); see [08-backtest.md](08-backtest.md).
  - Momentum is the share change from T−1 to T. New-author share is the share of window authors
    with no earlier paper in the cluster.
- **Status** `emerging`: E′ in the top decile and momentum > 0. Backtested as weak (about at
  momentum's level), so treat it as a prompt for review, not a prediction.

## 6.3 Niches

`N = z(coherence) − z(external inflow)`:

- **Coherence** is mean pairwise cosine within the cluster (500 sampled pairs, computed in
  Phase 5).
- **External inflow** is the share of the cluster's received corpus citations that come from
  other clusters.
- **Status:** `niche` if N ≥ 1 and the cluster has ≥ 10 papers in the last 3 years; `dormant`
  if N ≥ 1 and it has fewer.

Pilot at 2026 (variant all): 10 emerging, 11 niche, 4 dormant across the four clusterings.

## 6.4 Missing bridges and opportunities

### (a) Limitation → innovation (`bridge_lim_inn`)

The recipe's formula, `mean_top5 cos(e_lim, e_inn) · (1 − F[X,Y] / (max_Y F[X,Y] + 1))^p`, over
limitation papers in problem cluster X and innovation papers in Y ≠ X published no earlier.

The recipe uses p = 1. The default is now **p = 0.5** (`lim_flow_penalty_power`): in the
backtest it ranks better (p@50 0.255 vs 0.215) while the top candidates still have a median of
only 1.5 existing citations between the clusters. This is the best-validated signal: about
2.4× random, and it beats every baseline.

Deviation: the **20% most generic limitation statements** (closest to the mean limitation
vector) are dropped first. Statements like "future work is needed" otherwise match every
innovation.

Evidence is the top-5 limitation ↔ innovation pairs, stored in `features.pairs`.

### (b) Method transfer (`bridge_method_transfer`)

`T(P, m) = Σ sim(P, P′)·1[M(P′, m) > 0] / Σ sim(P, P′)` over the 10 nearest problem clusters,
for cells with M(P, m) = 0.

The indicator saturates at 1 for large method clusters, so the score is broken by
`neighbor_usage_share`: how much of the neighbours' output uses m. `transfer_score` keeps the
recipe value.

### (c) Unresolved limitations (`bridge_unresolved_limitation`)

1. Limitations within each problem cluster are grouped by average-linkage agglomerative
   clustering. The cosine threshold is the 90th percentile of limitation–limitation similarity.
2. Groups with at least 5 papers are kept.
3. A group is *resolved* if at least half of its statements have a later innovation at or
   above the match threshold.

Deviation: the recipe's fixed 0.7 match threshold is replaced by `"auto"`, the 99th percentile
of limitation→innovation similarity. On this model even unrelated texts score about 0.78, so
0.7 would mark everything resolved.

Matching is done per statement, not on the group centroid, because averaging inflates
similarity. Pilot: 1 open-problem group, which the judge rated 2/5 ("no consistent, specific
limitation").

### (d) Semantic but not cited (`bridge_semantic_not_cited`)

Pairs of problem clusters in the top 5% of centroid similarity whose size-normalized two-way
citation flow is at or below the 10th percentile.

**Pilot: empty, and this is a genuine result.** The 15 most similar pairs (cosine ≥ 0.973)
all cite each other heavily. The 62 pairs with zero flow are not semantically close.

### LLM judge

- **Inputs:** the top 50 candidates of each bridge type go to `deepseek/deepseek-v4-pro` (fp8+,
  price-sorted, $2 cap) with the evidence papers' problem, method, innovation and limitation
  facets. The deep-research report text never reaches it.
- **Output:** plausibility 1–5, a two-sentence rationale, and an "obvious reason it hasn't been
  done" (stored in `features.judge_obvious_reason`).
- **Reordering:** judged candidates are re-ranked by (judge score, score), and the original
  rank is kept in `features.pre_judge_rank`. The judge never adds candidates.
- **Pilot:** 101 candidates judged for ~$0.06. Mean plausibility was 2.4 for lim → inn and 2.1
  for method transfer. Top-rated examples:
  - Energy-scheduling overfitting ↔ adversarial / OOD-generalization methods (4/5).
  - Wasserstein DRO → mean-field PDE limits (5/5).

## 6.5 Influence

- **`disruptive_paper`**: the CD index over corpus papers that cite p or any reference of p (and
  are not older than p). It needs at least 5 corpus citers and at least 5 known references of p.
  Ranked only for papers from 5 years before the first cutoff onward (pilot: 2015+), because
  with corpus-only citers, textbooks and classics trivially score CD ≈ 1. All CD values are in
  `paper_scores`.
Backtest verdict for this section: seeders, brokers and recent-paper coupling **do not beat
popularity** as predictors of future citations, so they are descriptive roles, not predictions
(see [08-backtest.md](08-backtest.md)). Paper novelty is likewise anti-predictive of 3-year
citations.

- **`seeder_author`**: count of an author's papers among the earliest 10% of clusters whose g is
  above the median.
- **`broker_author`**: participation coefficient `1 − Σ (k_ac / k_a)²` over clusters, for
  authors with at least 3 papers. Pilot top 5: Iyengar, Kuhn, Calafiore, Natarajan, Blanchet.
- **`recent_paper_coupling`**: for papers from the last 2 years, the number of corpus papers
  sharing at least 2 references (bibliographic coupling). Surveys are excluded, because they
  couple with almost everything.
- **`institution_seeder` / `institution_broker`**: author scores aggregated through authorships,
  weighted by 1 / (number of institutions on the authorship). Pilot seeders: MIT, Stanford,
  Meta, UC Berkeley, KU Leuven.
- **Substantive citations:** counted (S2 influential or methodology intent), but with 51
  S2-annotated papers they are too sparse to rank.

These lists are internal decision support, not a public ranking of people (recipe, Scope).

## Known limits

- Citations come only from corpus papers. Influence and flows describe the field as sampled,
  not all of science.
- Clusters are fitted on all years. That is fine for the live run, but the Phase 8 backtest
  must re-fit at each cutoff, as the recipe requires.
- The judge knows the future. Backtests must switch it off (`--skip-judge`) or use a model with
  an earlier training cutoff.
