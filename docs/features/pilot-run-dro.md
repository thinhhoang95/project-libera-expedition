# Pilot run: `dro-pilot` (Phases 1–5)

Run on 2026-10-21 · Params: `configs/dro-pilot.json` · Seed: Esfahani & Kuhn, *Data-driven
distributionally robust optimization using the Wasserstein metric* (DOI
10.1007/s10107-017-1172-1). OpenAlex holds two records for it, `W2963450292` (journal) and
`W1712265143` (preprint), and both are seeds.

The pilot field was an open question in the recipe. Distributionally robust optimization (DRO)
was picked as a mid-sized, fast-moving field that connects operations research, energy and
machine learning. Change the seed with `new-run --seed-doi`.

## Spend

| Item | USD |
| --- | --- |
| OpenAlex metadata (cap $2.00) | 0.020 |
| Facet extraction, DeepSeek v4.1 flash via Baidu fp8 (4,533 calls) | 0.270 |
| Cluster labels, DeepSeek v4-pro (292 calls) | 0.058 |
| **Total** | **0.35** |

Embeddings ran locally on the CPU. Abstract embedding took about 25 minutes. Facet embedding
(about 30k short texts) took a few minutes, because facet texts are short.

## Corpus (Phases 1–2)

| Route | Hop | Papers | Missing abstract |
| --- | --- | --- | --- |
| seed | 0 | 2 | 0% |
| back | 1 | 55 | 26% |
| back | 2 | 307 | 41% |
| forward | 1 | 3,500 (route cap) | 19% |
| author | 1 | 411 (13 authors) | 14% |
| semantic | 1 | 1,135 | 1% (abstract required) |
| **Total** | | **5,410** | **16.2%** |

- **Abstract fill.** S2 filled 498 abstracts and arXiv copies another 56. Most of what remains
  is Elsevier journals (EJOR, Applied Energy, Energy, Transportation Research), where OpenAlex
  and S2 both withhold the abstract and authors rarely post preprints. The M1 target of under
  10% missing is **not met** for this field. The affected papers are flagged `title_only`.
- **Time coverage.** Years run 2018 to 2026 at 215–790 papers a year. With the default 5-year
  forward window, 2018–2020 were almost empty, because the seed is from 2017. The pilot
  therefore uses `forward_years = 8`. Volume falls after 2020 because forward citers are taken
  by citation count, which favors older papers. Phase 6 shares divide by yearly totals, but this
  selection effect should be kept in mind for the direction signals.
- **S2 citation intents.** Without an S2 key, 51 papers got S2 reference annotations (seed and
  hop-1) within the 5-minute time budget. Hop-2 therefore used S2 substantive edges where they
  were available and co-citation elsewhere, and the author fan-out found only 13 authors.
- **Semantic route.** The threshold was auto-calibrated to 0.897 (see the corpus expansion
  doc). The topics pulled were Risk and Portfolio Optimization, Electric Power System
  Optimization, Supply Chain and Inventory, Geometric Analysis and Curvature Flows, Advanced
  Optimization Algorithms, Facility Location, Optimal Power Flow, and Integrated Energy Systems.
  "Geometric Analysis" adds pure-math optimal-transport and PDE papers. They are in the corpus
  because the forward route already brought in 137 such papers through Wasserstein citations.
  **A reviewer should decide whether that neighbourhood belongs in the field.**

## Facets (Phase 3)

- 4,532 papers extracted, 877 title-only, and 1 failure (logged in `facet_status`).
- Contribution types: method 3,060, theory 795, survey 469, application 90, other 54, and fewer
  than 30 each for benchmark, position and dataset.
- `limitations` is non-null for 345 papers (8%). The null rate is about 93% in every major
  field, so there is no sign of invented limitations.
- QA flags: `copied` fires on 27% of `result` fields and is lower on the others. That is
  expected, since faithful fields reuse the abstract's wording. `too_long` fires on 109
  `result` fields, and `tag_too_long` on 59 tags.
- Tag vocabulary: 2,056 → 1,261 problem classes and 2,460 → 1,621 method families. The first
  attempt used connected components at cosine 0.9; it chained unrelated tags together and
  merged "systematic literature review" into "distributionally robust optimization". It was
  replaced by greedy merging with a specialization guard (see the embeddings doc). Merges at
  the edges, such as "distributed optimization" → DRO, still need human review.

## Clusters (Phase 5)

- 33 clusterings (5 facets, 3 resolutions, plus 4 α values for problem and method), 681 stable
  clusters, and 292 LLM labels.
- **Resolutions.** The initial guesses (0.15 / 0.6 / 2.0) produced one giant cluster at the
  coarse level. A resolution sweep led to **coarse 1.0, medium 2.0, fine 6.0**, which gives
  about 10, 20–25 and 55–75 real clusters. Every clustering also has about 20–29 *fragments*
  (fewer than 5 papers, mostly near-duplicate records that form their own kNN components).
  Fragments are never marked stable.
- **Assignment.** Stable clusters cover 85–99% of papers at the coarse and medium levels for
  problem and method, and 50–87% at the fine level. The innovation and limitations facets are
  less stable: 50–67% at medium.
- The default landscape view (problem, α = 0.3, medium) has 19 stable clusters, for example:
  Uncertainty-aware energy system optimization (524 papers), Domain generalization and
  adaptation (331), Nonlinear PDE solution analysis (299), Portfolio optimization under
  uncertainty (279), Distributionally robust stochastic MPC (252), Empirical optimal transport
  statistics (239), Multistage stochastic programming (116), Robust risk-sensitive RL (91).
- Limitation clusters separate generic statements ("Vague limitation statements",
  "Future research directions") from concrete ones ("Computational scalability",
  "Intractability of exact solutions"). Phase 6.4 should down-weight the generic ones.
- Problem × method matrix: 1,013 cells over all α. At α = 0.3, 11 cells first appeared in
  2025 or later. Citation flows: 32,849 citation edges inside the corpus. Author profiles:
  65,127 author-year rows.

## Found in the explorer

- **The DRO core is an unstable cluster** (now handled: it is kept as a *hub* cluster and
  reported separately, see [06-signals.md](06-signals.md)). For the problem facet at α = 0.3, medium
  resolution, the cluster whose central papers are the core DRO work ("An introduction to
  distributionally robust optimization", "Kernel DRO", "Bayesian DRO", …; 246 papers,
  including both seed records) has stability 0.50. That is below the 0.6 threshold, so its
  papers are unassigned for the signals and it only has a c-TF-IDF label ("decision,
  uncertainty, decisions"). These abstracts state the problem generically ("decide under
  uncertainty"), so the papers sit at the hub of the kNN graph and split differently across
  Leiden seeds. **Before Phase 6 direction signals**, decide how to handle this: for example,
  check the core at other α values or resolutions, or report hub clusters separately instead
  of dropping them.
- **Some papers are off-field.** The gold-set sample includes a fake-news detection paper,
  and the corpus has a few others like it. How many, and which routes brought them in, has
  not been measured yet. The corpus view's route browser is where to check.

## Signals (Phase 6)

- **Cost:** ~$0.06 for the judge (101 candidates judged).
- **Statuses at 2026:** 10 emerging, 11 niche, 4 dormant across the four signal clusterings.
- **Bridges:**
  - 200 limitation → innovation candidates (mean judge score 2.4).
  - 166 method transfers (2.1).
  - 1 unresolved limitation group (judge 2/5).
  - 0 semantic-but-not-cited pairs: the most similar problem clusters all cite each other.
- **Recognizable outputs:**
  - Brokers: Iyengar, Kuhn, Calafiore, Natarajan, Blanchet.
  - Seeders: Bertsimas, Boyd, Sim.
  - Seeding institutions: MIT, Stanford, Meta, UC Berkeley, KU Leuven.
- **Hubs:** the DRO core is a hub at α = 0 and 0.3 and a stable cluster at α = 0.5.

## M1 exit criteria check

| Criterion | Result |
| --- | --- |
| 5,000+ paper corpus | ✅ 5,410 |
| < 10% missing abstracts after S2 fill | ❌ 16.2% (publisher gaps; see above) |
| Under the metadata spend cap | ✅ $0.02 of $2.00 |
| Reviewer checked composition in the corpus view | ⏳ Corpus view is built (`landscape explorer`); a reviewer still has to check it |
