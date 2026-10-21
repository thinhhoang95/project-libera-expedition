# Phase 8c: what is missing from the signals?

2026-10-21 · Pilot: `dro-pilot` · Existing corpus only · No downloads or paid calls

**Finding:** the main bottlenecks are the sampled population, the specificity of the
evidence, and the objective being optimized. Adding more semantic features did not improve
the main bridge-ranking result. Better evidence hygiene and a broader review list help,
but do not turn citation prediction into opportunity discovery.

Code: `src/landscape/research_lab.py`, `research_lab_view.py`, `identity.py`.
The explorer now has a **Research lab** view with dynamics, paper bridges, underconnected
communities, people by topic, and experiment results. Every candidate opens its papers.
Bridge and community-pair judgments can be saved to `review_actions`.

## 1. The apparent direction depends heavily on the sampling mix

The earlier reports correctly identified route bias. This experiment measures how much it
changes the story, rather than trying another growth formula.

Compare 2020–2022 with 2023–2025, excluding the incomplete publication year 2026. Use one
record per conservatively identified work and the live problem clustering at α = 0.3.
Keep routes represented by at least 20 assigned works in each three-year period. These are
author, forward, and semantic. The assigned-work route proportions change as follows:

| Route | 2020–2022 | 2023–2025 |
| --- | ---: | ---: |
| Author | 1.54% | 19.27% |
| Forward | 71.33% | 72.86% |
| Semantic | 27.12% | 7.87% |

For topic c, let w be each route's fraction of the sample and q its within-route topic
share. Decompose the observed share change exactly:

`Δshare = Σ mean(w_before,w_after)·Δq + Σ mean(q_before,q_after)·Δw`

The first term is change within routes; the second is the contribution of the route mix.
**Six of 15 communities with at least 20 works in both periods reverse sign.** Including
small communities gives seven of 22; one of those has only one versus three works and
should not be used for a direction claim.

| Topic | Observed change, percentage points | Within-route change | Route-mix contribution |
| --- | ---: | ---: | ---: |
| Uncertainty-aware renewable energy optimization | −3.08 | +1.27 | −4.35 |
| Mathematical optimization | −1.41 | +1.19 | −2.60 |
| Domain generalization / distribution shift | +0.91 | −2.14 | +3.05 |
| Empirical optimal transport | +0.55 | −0.31 | +0.86 |
| EV charging station optimization | −0.07 | +1.25 | −1.32 |
| Energy trading and bidding | +1.11 | −0.74 | +1.85 |

This does **not** establish the true trends. Within-route selection also changes, first-entry
provenance is not a sampling probability, and the author route starts from only 13 authors.
The live clustering also uses all years: this is a descriptive sensitivity analysis, not a
historical forecasting test. The explorer shows each route's trajectory and denominator so
small samples remain visible.

**Consequence:** field-direction claims need a stable sampling frame. Reweighting this
convenience sample cannot create one. An independently sampled, year-balanced field corpus
is the next data experiment; no new papers were downloaded in this batch.

## 2. Some “missing bridges” were record-identity errors

Conservative identity matching finds **82 alias groups and 92 redundant records**, leaving
5,318 works from 5,410 records. The stricter count differs from the earlier approximate
title-only count: identical substantial titles also require a shared author; equal normalized
DOIs can establish identity directly. Short generic titles are not merged. Representatives
are chosen from records available at the cutoff, by earliest year then paper ID, without
future citation counts.

Before the fix, **10 of 250 production paper-bridge recommendations were already cited by
the target community under an alias**. For example, the preprint of the seed could be
recommended to a community already citing the journal version. The list contained only
90 distinct paper IDs, and individual papers appeared in up to 11 communities.

The production paper-bridge code now:

- excludes a work if any known version is already cited by, or belongs to, the target community;
- emits at most one version per list and records the alias IDs in its evidence features;
- counts distinct citing works across versions when measuring popularity and hidden-gem eligibility;
- excludes references between versions of the same work from citation counts and experimental adoption outcomes.

The refreshed production lists contain 250 likely adoptions and 125 gems, **zero known
already-cited aliases**, and 94 distinct papers in the likely-adoption list. The 1,192 other
signal rows were fingerprinted before and after and are identical. Original bridge rows and
the pre-fix audit are saved under the run's lab directory.

This is evidence hygiene, not a claim of improved predictive accuracy. The existing model
weights remain unchanged. Coupling, embeddings, clustering, novelty and the original rising
pockets still use stored records; this is not full-pipeline deduplication.

## 3. Semantic specificity and list diversity: controlled experiments

Rebuilt paper-bridge candidates at 2019–2023 using only cutoff-available papers, reference
lists, geometry means, and limitation filtering. Candidate identities, cited-alias exclusion,
and outcomes count works rather than duplicated records. There are 264,950 candidate pairs
and 4,723 subsequent-adoption hits across the five cutoffs. All compared rankings use the
same candidate pool.

Experiments:

- **Existing feature recipe, refitted:** log citations, log coupling, problem similarity,
  limitation similarity, age. This is the baseline model in the tables, not an evaluation
  of the already fitted production model on its own training data.
- **Contrastive features:** subtract each facet's past-only mean and normalize; add centered
  problem and limitation similarities plus CSLS-style local-density correction. This tests
  whether generic semantic directions and retrieval hubs obscure useful matches.
- **Broader methods:** retain the baseline model but cap a top-ten list at three papers from
  one method community. Unfilled slots count as misses. No outcome was used to tune this cap.

Random-pair cosine medians are around 0.8, while centering spreads similarities around zero.
That makes fixed absolute cosine thresholds hard to interpret. It is **not** evidence that
centering will improve rankings: raw problem similarity performs better than centered
similarity alone in these experiments.

### A chronological test with a complete outcome window

The previous leave-one-cutoff-out protocol can train on outcomes from the future of the test
cutoff. It tests retrospective transport between cutoffs; it does not simulate deployable
historical training. The new protocol trains only on cutoffs S where **S + 3 ≤ T**.

An unlabeled 2019 core run was added without downloads or LLM calls. This permits training
on 2019, scoring 2022, and measuring adoption in **2023–2025**. There are 22 communities and
220 recommendation slots. “Complete” means the calendar window elapsed, not that the
retrospectively sampled corpus captures all papers or citations.

| Ranking | Precision@10 per community | Hits / 220 slots | Mean method communities / list | AUC |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 13.64% | 30 | 6.23 | 0.678 |
| Existing feature recipe, refitted | 19.55% | 43 | 5.86 | 0.846 |
| Add contrastive features | 18.64% | 41 | 6.14 | 0.855 |
| Broader-method baseline list | 20.00% | 44 | 6.18 | 0.846 |

The broader list changes one hit and gains approximately seven method-community appearances
across the 22 lists. This is a modest coverage tradeoff, not evidence of a significant
predictive improvement. Its AUC is the unchanged underlying model's AUC; a list quota does
not define a new scalar ranking of every candidate.

### Other protocols, reported separately

| Protocol | Existing feature model p@10 | Contrastive model | Broader-method list |
| --- | ---: | ---: | ---: |
| LOCO retrospective, mean over 2019–2023 | 28.79% | 27.79% | 28.26% |
| Purged chronological, score 2023 | 13.75% | 17.08% | 15.00% |

The 2023 test trains on 2019/2020, but its outcome includes partial 2026. It is provisional;
it is not pooled with the complete-window test. The contrastive model wins on that cutoff
and loses on the complete-window test and the LOCO mean. **It is not adopted.** The broader
list remains an optional review view. Historical paper sampling and model-training leakage
identified in the earlier reports remain; the temporal split cannot remove them.

### Follow-up: how much information comes from semantics itself?

`scripts/phase8c/semantic_ablation.py` separates the two raw semantic similarities from
received citations, age and bibliographic coupling, on the same 2019 -> 2022 chronological
test. Outcomes are subsequent community adoption in 2023–2025. All models are trained on
the full 2019 candidate population; the uncited subsets are evaluation subsets, not separate
training fits. Uncited refers to received citations within this corpus.

| Features | AUC, all candidates | AUC, uncited candidates | p@10, all | p@10, cited ≤1 |
| --- | ---: | ---: | ---: | ---: |
| Citations + age | 0.7090 | 0.6632 | 0.1364 | 0.0273 |
| Citations + coupling + age | 0.8362 | 0.8361 | 0.1955 | 0.0636 |
| Problem and limitation similarity only | 0.7170 | 0.7600 | 0.0636 | 0.0591 |
| Bibliometric base + semantic similarities | 0.8464 | 0.8507 | 0.1955 | 0.0864 |

Semantics carries information before a candidate has received corpus citations. Its
increment beyond the bibliometric base is modest: +0.0102 AUC overall and +0.0146 on
uncited candidates. Among candidates cited at most once, top-ten precision rises from
0.0636 to 0.0864. Overall top-ten precision does not change. These are point estimates
from one mature temporal split, not significance claims.

Bibliographic coupling uses references already listed by a paper; it does not require
received citations. Neither this test nor the earlier paper-bridge experiments tests the
birth of a community: they predict uptake by already established cutoff communities.

Reproduce with `uv run python scripts/phase8c/semantic_ablation.py --run-id dro-pilot`.
Results are saved to `data/projects/dro-pilot/research_lab/semantic_ablation.csv`.

## 4. Opportunities should be a disagreement to investigate

The old semantic-but-not-cited signal intersects two extreme cutoffs and returns nothing.
This batch instead displays citation deficits relative to exposure:

`expected(X→Y) = outgoing(X) × incoming(Y) / total_citations`

Use distinct-work citations made in 2023–2025, retain within-community citations when fitting
the degree null, and center problem vectors on cutoff-available papers. Review pairs in the
top semantic quartile with at least five expected citations and a positive deficit. Order by
`positive_standardized_deficit × positive_semantic_similarity`. This produces **38 directed
review candidates**, including both directions of some community pairs. It is a descriptive
null-model comparison, not a calibrated probability, significance test or validated forecast.

For illustration, robust financial risk measures → heavy-tail estimation has **0 observed
citations versus 12.6 expected**. Its evidence includes:

- [Robust Actuarial Risk Analysis](https://doi.org/10.1080/10920277.2018.1504686): robust estimates
  of actuarial quantities under nonparametric model misspecification.
- [Robust covariance estimation for approximate factor models](https://doi.org/10.1016/j.jeconom.2018.09.003):
  an adaptive Huber-loss approach under bounded fourth moments.

The review hypothesis is whether robust moment estimation could inform some actuarial
uncertainty models. Neither similarity nor a citation deficit establishes that transfer is
valid. Differences in assumptions may explain the missing edge. Evidence is restricted to
research articles/preprints among each community's central papers, and same-work pairs are
excluded; otherwise a generic textbook chapter can become a misleading “bridge.”

EV charging → location/routing has 2 citations versus 9.5 expected. That is another
community pair to inspect, not proof of an untouched niche. These views expose actual,
expected and semantic values together, and allow useful/non-obvious, obvious, wrong or
uncertain judgments with an explanation.

## 5. People need roles and scope, and opportunities need richer evidence

The new people view is scoped to one topic and separates recent fractional contribution,
recent external adoption, and cross-community participation. Aliases are collapsed; author
credit is divided by team size; adoption excludes citing teams sharing any author with the
cited work. First activity in the topic and supporting papers are shown. This is descriptive
internal decision support, not another global fame ranking or validated influence forecast.
Recent citation counts are sparse and the author-route sample is narrow.

Only **345 papers** have an explicit limitation, out of 4,532 extracted abstracts. The engine
therefore has little direct evidence for “unresolved problem” assertions. A generic limitation
vector and an innovation vector can share vocabulary while making incompatible claims. More
weight tuning cannot supply missing premises.

The next experiments should be prioritized as follows:

1. **Stable sampling and inclusion records.** Test directions against an independent field
   sample by year; preserve every discovery route, not only the first. Keep partial years
   separate and compare both the local seed neighborhood and a broader field scope.
2. **Structured compatibility evidence.** Extend extraction, with an explicit unknown state,
   to assumptions, supported regimes and claimed capabilities: finite-sample/asymptotic,
   convex/nonconvex, distribution shift, data requirements, computation and application
   constraints. Distinguish stated limitations from inferred research questions. A candidate
   needs a specific need, a claimed capability, compatible assumptions and a reason it might
   be missing. Do not claim that abstract silence means unresolved.
3. **Expert usefulness as a separate endpoint.** Use the new review forms to label paired
   baseline and broader lists for useful/non-obvious, obvious, wrong and uncertain. Evaluate
   yield of distinct useful ideas per review session as well as citation adoption. The
   recipe's 30% useful/non-obvious target remains unmeasured.
4. **Prospective and stronger holdouts.** Keep the complete chronological test, add independent
   seeds/fields and held-out communities, and save future candidate snapshots before outcomes
   occur. One mature temporal split on this selected corpus cannot justify a major model change.

## Reproduce and inspect

On this macOS machine the existing environment is `~/.venvs/libera-expedition`:

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/libera-expedition
uv run python scripts/phase8c/prepare_cutoffs.py --run-id dro-pilot --cutoffs 2019
uv run landscape research-lab --run-id dro-pilot --as-of 2026-10-21
uv run landscape signals --run-id dro-pilot --paper-bridges-only
uv run landscape explorer
```

The original 2020–2023 cutoff objects come from `landscape backtest`. On a new corpus, the
offline preparation script can build all requested cutoff core objects. Preparation and the
narrow signal refresh write derived DB tables; `research-lab` opens DuckDB read-only and
writes only `data/projects/<run>/research_lab/`. Close the explorer before DB-writing steps.

Artifacts include `summary.json`, route counts and weights, decomposed trends and time series,
geometry diagnostics, all historical candidates, per-cutoff metrics and comparisons, live
bridge candidates, community pairs, and contextual people. The data directory is ignored by
Git; this document and the source code preserve the method and measured results.

Validation: five focused identity regression tests; full lab execution and finite-feature,
single-work and exact-decomposition invariants; production bridge checks for cited aliases,
duplicate picks and hidden-gem counts; unchanged unrelated-signal fingerprints; Streamlit
application and control-interaction checks.
