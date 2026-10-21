# Feature: Core analytical objects (Phase 5)

Status: implemented · Code: `src/landscape/core.py` · CLI: `landscape core [--skip-labels]`

Builds the five objects every Phase 6 signal reads, once per run, keyed by `run_id`.

## 5.1 Facet clusters

For each facet in `ClusterParams.facets` (`problem`, `method`, `innovation`, `result`,
`limitations`), and for `problem` / `method` at each gloss weight α ∈ {0, 0.3, 0.5, 0.8}, the
job does the following:

1. **kNN graph.** Builds a graph on the 256-d (truncated and re-normalized) vectors with k = 20
   and exact cosine search (FAISS `IndexFlatIP`). Edges are made undirected, keeping the
   larger weight, and edges below the 5th weight percentile are dropped.
2. **Leiden** (`leidenalg.RBConfigurationVertexPartition`, modularity with a resolution
   parameter) at three resolutions: `coarse`, `medium` and `fine`. Defaults are
   `ClusterParams.resolutions`.
3. **Stability and kind.** The clustering is repeated with 5 seeds (`ClusterParams.seed_values`
   overrides them). For each cluster of the reference run (seed 0), stability is the mean over
   the other runs of its best Jaccard overlap. A cluster is kept (`is_stable = true`) when
   stability ≥ 0.6 and it has at least 5 papers. Unstable clusters stay in `clusters` for
   review, but their papers get `cluster_id = NULL` in `paper_cluster`, so no signal is built
   on them.
   Each cluster also gets a `kind`. **`hub`** clusters are unstable but large (at least 30
   papers and 1% of the facet's papers) and at least as coherent as the 25th percentile of the
   stable clusters. They keep their papers and are reported separately by Phase 6 (see
   [06-signals.md](06-signals.md)). `clusters.coherence` (mean pairwise cosine) and
   `clusters.centrality` (the z-score of a centroid's mean similarity to the other centroids)
   are stored. `paper_cluster.raw_cluster_id` keeps the raw membership of every cluster,
   including unstable ones.
4. **All years at once.** Clusters are fitted on the whole corpus and never re-clustered per
   year.
5. **Hierarchy.** `parent_cluster_id` links each medium cluster to the coarse cluster it
   overlaps most, and each fine cluster to its medium cluster.
6. **Labels.** Each cluster gets the top 15 c-TF-IDF terms (1–2-grams over the facet text,
   class-based TF-IDF). For stable coarse and medium clusters, an LLM (`LabelParams.model`,
   default `deepseek/deepseek-v4-pro`, same fp8+ price routing) receives the facet text of the
   10 papers closest to the centroid plus those terms. It returns a label of at most 6 words and
   a one-line description. Fine clusters keep a c-TF-IDF label unless
   `LabelParams.label_fine` is set, which keeps the stronger model to the recipe's "hundreds of
   calls per run".
7. **Colors.** One color per cluster, assigned once per run (golden-angle hues), stored in
   `clusters.color`.

8. **Consolidation (off by default).** The extraction model sees one abstract at a time and the
   clustering never compares categories by meaning, so the finished catalogue can hold the same
   category twice, or categories too vague to be useful. With `ClusterParams.consolidate`, each
   medium and fine clustering (`consolidate_levels`) gets one review call to
   `LabelParams.consolidate_model` (Claude Sonnet 5.5). The call sees the whole catalogue: every
   stable or hub cluster with its size, c-TF-IDF terms, the 5 statements closest to its centroid,
   and the run's seed titles as the corpus's central topic. It returns:
   - **merge** groups. A group merges into its largest member, and the absorbed cluster is kept as
     `kind = 'merged'` with `merged_into`. A group is skipped when the reviewer names a difference
     that would keep its members apart. A member joins only if its centroid cosine to the survivor
     is in the top 25% of sibling pairs (`merge_pair_quantile`).
   - **generic** flags: boilerplate, a genre without a subject, or a catch-all. Flagged clusters get
     `is_generic = true` and keep their papers. A flag is skipped when the reviewer names a shared
     subject. Only with `drop_generic` do they become `kind = 'generic'` with their papers unassigned.

   The reviewer's reason goes to `consolidation_note`. Cluster ids follow the raw partition, so a
   cluster keeps its id with or without consolidation. `drop_boilerplate` separately removes
   limitation statements such as "no specific limitation is stated" or a short "future research is
   discussed" before the graph is built (32 on the pilot).

   Pilot review (2026-10-21, ~$0.25 for 15 calls). The cheaper `deepseek-v4-flash/pro` merged
   categories while describing how they differed in the same answer, so merging uses Sonnet. Sonnet
   found no true topic duplicates, including the two "Distributionally Robust Optimization" problem
   hubs, which it judged distinct. It proposed merges only among boilerplate clusters ("simulation
   results show effectiveness" + "case studies show effectiveness"). It flagged 3–17 clusters per
   facet as generic: about a third of result papers are validation or "outperforms state-of-the-art"
   boilerplate, plus surveys spanning unrelated topics, a textbook and crystal-structure deposits.
   Reviewing coarse clusters flagged whole umbrellas of real work, which is why coarse is excluded.
   Without the seed titles, it flagged the corpus's own DRO core as "generic".

   `dro-pilot` runs with `consolidate = true` since 2026-10-21 ($0.59 for the build). Against a
   rebuild from the same papers and settings without it, the run has 6 merges: result-facet
   validation boilerplate at medium and fine, plus two Wasserstein-DRO method clusters at α 0.5 fine.
   It also has 67 clusters flagged generic. Medium assignments are identical except in the result
   facet, where 6.8% of papers moved. Signals changed only where they use result clusters: the two
   absorbed validation clusters had ranked #2–3 in shrinking directions and #4 in emerging clusters.
   Signals do not read `is_generic` yet, so flagged clusters still take 2–6 of the top-10 slots in
   the emerging and direction lists.

Cluster ids are readable: `{facet}:{alpha}:{resolution}:{rank}`, for example
`problem:0.3:medium:004`, where the rank is by cluster size.

Outputs:

- `clusters(run_id, cluster_id, facet, resolution, alpha, label, description, top_terms,
  stability, is_stable, kind, coherence, centrality, size, parent_cluster_id, color,
  time_bucket_years, label_model, merged_into, is_generic, consolidation_note)`
- `paper_cluster(run_id, paper_id, facet, resolution, alpha, cluster_id, sim_to_centroid)`
- `data/projects/<run>/vectors/cluster_centroids_{run_id}.parquet` (768-d)

## 5.2 Problem × method matrix

`pm_matrix(run_id, alpha, problem_cluster, method_cluster, year, count, first_year)` crosses
medium problem clusters with medium method clusters at the same α. `first_year` is the first
year the cell became nonzero. Totals over all years are summed at query time.

An alternative axis is the canonical tags: `pm_tag_matrix(run_id, problem_class,
method_family, year, count, first_year)`, built on `tag_vocab` canonicals.

## 5.3 Time slices

`cluster_year(run_id, facet, resolution, alpha, cluster_id, year, n, n_total, share)` holds
yearly counts with `share = n / n_total`. Here `n_total` is the number of corpus papers that
have that facet in that year, so the denominator is the same population that was clustered.
`clusters.time_bucket_years` is 2 when a cluster averages fewer than 20 papers per active
year; consumers then sum pairs of years. `year_totals(run_id, facet, year, n)` stores the
corpus total (`facet = 'corpus'`) and the per-facet totals.

## 5.4 Citation flows

`citation_flows(run_id, facet, alpha, year, from_cluster, to_cluster, count, row_share)` has
one row per citation between two corpus papers in medium clusters of the same clustering. The
row is keyed by the citing paper's year. `row_share` is the row-normalized version (share of
cluster i's outgoing citations in year t).

## 5.5 Author profiles

`author_year(run_id, author_id, year, facet, alpha, n_papers, cluster_dist, dominant_cluster,
institution_ids, centroid_ref)` gives each author and year:

- The distribution of their corpus papers over medium clusters (JSON), and the dominant
  cluster, which Phase 6.1 migration will use.
- The institutions from their authorships that year.

Profiles are built at α = 0 for every facet and also at α = 0.3 for problem and method.
`centroid_ref` points to the author-year centroid row in
`data/projects/<run>/vectors/author_year_{facet}_{run_id}.parquet`.

## Tuning notes

- The resolutions are modularity resolution parameters, not cluster counts. A sweep on the
  pilot set the defaults to **coarse 1.0, medium 2.0, fine 6.0**, which gives about 10, 20–25
  and 55–75 real clusters. The first guesses (0.15 / 0.6 / 2.0) produced one giant coarse
  cluster. Re-check the defaults for a new field or corpus size, and override them per run in
  the params JSON.
- Every clustering also contains 20–30 *fragments* of fewer than 5 papers. These are mostly
  near-duplicate records (journal and preprint versions) that form their own kNN components.
  They are never stable, so their papers stay unassigned. The log reports real clusters and
  fragments separately.
- `alpha` is a column wherever it matters, so the explorer's α slider can snap to the
  precomputed values.

## Pilot run results

See [pilot-run-dro.md](pilot-run-dro.md).
