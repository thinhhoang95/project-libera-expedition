# Feature: Signal-improvement experiments (Phase 8b)

Status: done (four experiments; three adopted, one dropped) · Scripts: `scripts/phase8b/` · Production models: `src/landscape/models/{paper_bridges,early_work}.json` (fitted by `scripts/phase8b/fit_models.py`) · Signals: `bridge_paper`, `bridge_paper_gem`, `early_work`, `paper_scores.local_momentum`, item-CF `bridge_method_transfer`

The Phase 8 backtest ([08-backtest.md](08-backtest.md)) showed which recipe signals carry
information and which do not. Phase 8b tried new ways to make the signals stronger. The rule
for adoption: a new signal must beat its baselines **leave-one-cutoff-out** (fit on three of the
cutoffs 2020–2023, score the fourth, repeat), on the existing corpus only (no new downloads).

## Run

All scripts take `--run-id` (default `dro-pilot`) and read the cutoff runs `<run>@T2020..2023`,
which `landscape backtest --run-id <run>` builds. Each runs in seconds to minutes and costs nothing
(no LLM calls). Intermediate tables go to `data/projects/<run>/backtest/`.

**Every result table is saved, adopted or not**, to `data/projects/<run>/backtest/experiments/` (with an
`index.json` of titles and how-to-read notes) by `src/landscape/experiments.py`. The explorer's
**Experiments** view shows them all for the selected run, together with the Phase 8 backtest
summary. These results describe one corpus: another field may give different answers, so a
signal dropped on the pilot stays visible and re-runnable.

```bash
uv run python scripts/phase8b/exp1_paper_bridges.py   # writes data/projects/<run>/backtest/exp1_pairs.parquet
uv run python scripts/phase8b/exp1_evaluate.py
uv run python scripts/phase8b/exp2_early_work.py      # writes data/projects/<run>/backtest/exp2_papers.parquet
uv run python scripts/phase8b/exp3_micro_topics.py
uv run python scripts/phase8b/exp4_method_transfer.py
uv run python scripts/phase8b/fit_models.py           # refits the two production models on all cutoffs
uv run python scripts/phase8b/fetch_counts_by_year.py # exp5 only: OpenAlex yearly counts (~$0.02, cached)
uv run python scripts/phase8b/exp5_features.py        # writes data/projects/<run>/backtest/exp5_papers.parquet
uv run python scripts/phase8b/exp5_evaluate.py [logit|hgb] [y_...]
uv run landscape external --run-id dro-pilot            # exp6 data (~$1.10 LLM once; data/projects/<run>/external/)
uv run python scripts/phase8b/exp6_imports.py         # before exp5_features.py, for the borrowing family
uv run python scripts/phase8b/exp7_coupling.py        # coupling vs semantics by stratum
uv run python scripts/phase8b/exp8_birth.py           # community birth (needs exp6 for borrowing features)
uv run python scripts/phase8b/exp9_realized.py && uv run python scripts/phase8b/exp9_within_candidate.py
uv run python scripts/phase8b/exp10_warmth.py         # community warmth; writes models/community_warmth.json
uv run python scripts/phase8b/exp11_leaders.py        # big labs: follow-up waves and community entries
uv run python scripts/phase8b/freeze_snapshot.py      # prospective snapshot (date defaults to today)
```

`LANDSCAPE_DB=<copy of data/shared/landscape.duckdb>` lets the scripts run while the explorer holds the lock.

The production models are fitted on all four cutoffs. The honest accuracy estimates are the
leave-one-cutoff-out numbers below, not the in-sample fit.

## Summary

| Experiment | Result (leave-one-cutoff-out, cutoffs 2020–2023) | Decision |
| --- | --- | --- |
| 1. Paper-level bridges | p@10 per cluster 0.234 vs popularity 0.186, random 0.017; AUC 0.84; long-tail AUC 0.84 (popularity 0.55) | **adopted**: `bridge_paper`, `bridge_paper_gem` |
| 2. Early-work detector | AUC 0.815, p@50 0.67 vs citations-so-far 0.62, coupling 0.50, base rate 0.19 | **adopted**: `early_work` replaces `recent_paper_coupling` |
| 3a. Burstiness micro-topics | AUC 0.46 (mean-reverting) | dropped |
| 3b. Local momentum pockets | ρ +0.13, AUC 0.58, p@50 0.17 vs 0.084 random | **adopted lightly**: `paper_scores` metric `local_momentum`, explorer layer *Rising pockets* |
| 4. Method transfer as matrix completion | cluster level, item CF AUC 0.70 (≥ 1 paper) / 0.80 (≥ 2) vs recipe 0.67 / 0.66, popularity 0.51 / 0.62 | **adopted**: `SignalParams.transfer_method = "item_cf"` |
| 5. Topic, question, credibility and imports beyond rich-get-richer | in-field, peer-normalized, no self-citations: rising imports +0.022 AUC, venue +0.017, author +0.012, topic/question +0.007 (not significant); global field-normalized: only global author credibility, +0.012 (forward route) | findings; not adopted yet |
| 6. Ideas borrowed from outside the field | imports: semantic fit to where the field's questions are moving +0.020 AUC, global trend +0.013 (strict hit, work-level bootstrap). Papers: borrowing what trends globally, from nearby, +0.016 in-field; nothing globally | **adopted as a review list**: Bridges › From outside the field |
| 7. Coupling vs semantics | coupling needs no received citations (fresh papers: 0.65 → 0.84) but defines the reachable set (91% of adoptions share references); semantics adds +0.08 to +0.12 on fresh / uncited papers | findings |
| 8. Community birth | best AUC 0.58–0.61; p@50 at the base rate; births are offshoots within one community, not crossovers | **not demonstrated**; live list frozen as a low-confidence prediction |
| 9. Realized opportunities (content) | 95% of realized uptake is uncited; for a given idea, semantics ≈ coupling ≈ persistence (about 0.64) | better outcome measure; no special foresight |
| 10. Community warmth (which communities take up a paper) | within-paper AUC 0.81, hit@3 0.69 vs 0.25 random, MRR 0.57; on the content outcome problem similarity alone is best (hit@3 0.60 vs 0.40) | **adopted**: My paper view ([09-portfolio.md](09-portfolio.md)) |
| 11. Big labs steering the field | no follow-up wave by others after big authors or labs step in (gain −0.004 to +0.001); a small citation gain that comes from insiders, not big-lab steps; entry of big names does not predict community growth | **not supported** on this corpus |

## 1. Paper-level bridges

**Question.** The recipe's limitation → innovation bridge links two *clusters*. A reviewer can
act more easily on a list of *papers*: which innovation papers will problem cluster X start
citing?

- **Candidates at T:** pairs (X, i), where X is a problem cluster (α = 0.5, medium) and i is an
  innovation paper published by T, outside X, and not cited by any X paper by T.
- **Hit:** an X paper published in T+1…T+3 cites i. This is the paper-level form of the recipe's
  limitation → innovation hit ("a T+1…T+3 paper in X cites a matched innovation"), so a
  second definition is not needed.
- **Data:** 244,794 pairs and 3,757 hits (base rate 1.5%; 0.9% among papers cited at most once).

| Ranking | p@10 per cluster | AUC | AUC, long tail (cited ≤ 1×) |
| --- | --- | --- | --- |
| random | 0.017 | 0.50 | 0.50 |
| popularity (log citations so far) | 0.186 | 0.68 | 0.55 |
| bibliographic coupling with X | 0.115 | 0.77 | 0.81 |
| similarity to X's problem centroid | 0.083 | 0.71 | 0.77 |
| similarity to X's limitations (mean of top 3) | 0.047 | 0.66 | 0.67 |
| recency (−age) | 0.054 | 0.55 | 0.66 |
| **logistic, all features** | **0.234** | **0.84** | **0.84** |
| logistic, no popularity | 0.105 | 0.80 | 0.83 |

- **Coefficients** (standardized, fold without 2023): coupling 0.85, popularity 0.66, problem
  similarity 0.47, limitation similarity 0.20, age 0.13.
- **Long tail:** popularity is no better than chance among papers cited at most once, while
  the model keeps AUC 0.84. This is why there is a second list, the hidden gems.
- **Subset check:** the script also scores "clusters with stated limitations only". Every
  candidate cluster has at least one limitation, so the subset equals the whole set and its
  numbers are identical.

**In production** (`signals.paper_bridges`), each problem cluster gets two lists:

- `bridge_paper`: the top 10 (`paper_bridges_per_cluster`).
- `bridge_paper_gem`: the top 5 among papers cited at most once (`paper_bridges_gems_per_cluster`).

Each row's evidence is the innovation paper plus the best-matching limitation in X; the 20%
most generic limitations are dropped first, as in 6.4(a). Pilot: 250 + 125 rows.

## 2. Early-work detector

**Question.** Which papers published in year T will be in the top quartile of their year for
corpus citations received in T+1…T+3? This uses only information available at T.

- **Data:** 2,556 papers, base rate 0.19.
- **Features:** citations so far, bibliographic coupling (papers sharing ≥ 2 references),
  number of references, the cluster's E′, centrality in its cluster, novelty, author count,
  and contribution-type flags.

| Ranking | AUC | p@50 |
| --- | --- | --- |
| citations so far | 0.68 | 0.62 |
| author prominence | 0.68 | 0.35 |
| bibliographic coupling (the old `recent_paper_coupling`) | 0.74 | 0.50 |
| centrality in cluster | 0.66 | 0.34 |
| cluster E′ | 0.57 | 0.33 |
| novelty | 0.29 | 0.02 |
| logistic, all features | 0.82 | 0.70 |
| **logistic, no author fame** (adopted) | **0.815** | **0.67** |

- **No author fame:** it adds about 0.002 AUC, and leaving it out keeps the list from
  rewarding famous names.
- **Novelty is strongly anti-predictive** (AUC 0.29). Novel papers in this corpus collect fewer
  citations, matching the Phase 8 finding. The model gives it a negative weight.
- **Selection leak, and its robustness check.** The forward and semantic expansion routes picked
  papers by their 2026 citation counts, which leaks the future into who is in the corpus. On
  author-route papers only (no citation filter), the model keeps AUC 0.74. That subset has
  only 84 papers and about 2 hits, so it is a sanity check, not an estimate.
- **Production fix:** OpenAlex dates new editions of old books as new works. A 2025
  "Convex Optimization Theory" chapter, typed `other`, topped the first production run.
  `early_work` now keeps only `article`, `preprint`, `conference-paper` and
  `proceedings-article`. After the fix, the top 50 are 48 articles and 2 preprints.

## 3. Micro-topics

**Question.** Do dense pockets of recent papers keep attracting work?

- **Setup:** for each paper x from T−1…T, take its k = 15 nearest earlier neighbours (abstract
  vectors, 256-d). The ball around x has the radius of the 15th neighbour's similarity.
- **Hit:** the share of T+1…T+3 papers inside the ball is at least 1.5 × the share of T−2…T
  papers. 5,388 papers, hit rate 0.084.

| Ranking | ρ with local growth | AUC | p@50 |
| --- | --- | --- | --- |
| random | 0.00 | 0.50 | 0.084 |
| 3a. burstiness (share of neighbours from T−1…T) | −0.05 | 0.46 | 0.08 |
| cluster E′ of the paper's cluster | +0.11 | 0.55 | 0.08 |
| citations so far | −0.02 | 0.49 | 0.09 |
| **3b. local momentum** | **+0.13** | **0.58** | **0.17** |

- **3a, burstiness:** mean-reverting (AUC below 0.5). Dropped.
- **3b, local momentum:** the log ratio of the share of year-T papers to the share of
  year-(T−1) papers inside the ball. It is about 2× random at p@50, but weak overall (AUC
  0.58). It is stored for papers from T−1…T as `paper_scores(metric = 'local_momentum')` and
  drawn as a map layer (*Rising pockets*), not ranked as a signal.
- **Known weakness:** about 2% of the corpus is duplicate OpenAlex records of the same paper
  (preprint and article versions; 101 title groups, 217 records). Duplicates fill each other's
  neighbourhoods. 5 of the 31 top-5% pockets on the pilot fall in such groups.

## 4. Method transfer as matrix completion

**Question.** Which empty problem × method cells will be filled? This treats the grid as a
recommender problem.

- **Hit definitions:** the cell gets ≥ 1 paper in T+1…T+3 (ours), or ≥ 2 papers (the
  recipe's).
- **Methods:** all are unsupervised, so nothing is fitted on the test cutoff.

Cluster level (α = 0.5, medium; about 120 empty cells per cutoff):

| Ranking | AUC ≥ 1 | p@50 ≥ 1 | AUC ≥ 2 | p@50 ≥ 2 |
| --- | --- | --- | --- | --- |
| base rate | — | 0.178 | — | 0.036 |
| popularity (row × column totals) | 0.51 | 0.195 | 0.62 | 0.045 |
| recipe neighbour share | 0.67 | **0.270** | 0.66 | 0.065 |
| **item-based CF** (adopted) | **0.70** | 0.255 | **0.80** | 0.060 |
| SVD rank 5 | 0.70 | 0.290 | 0.75 | 0.065 |
| hybrid popularity + item CF | 0.61 | 0.235 | 0.73 | 0.060 |

- **Which recipe score:** "recipe neighbour share" here finds the 10 nearest problems by
  *method profile*. The production recipe score finds them by *centroid* similarity, and
  Phase 8 could not tell it apart from random (p@50 0.035 vs base 0.036 for ≥ 2 papers;
  [08-backtest.md](08-backtest.md), section 4).
- **The trade-off:** item CF has the best AUC under both definitions, which is what decided
  the adoption. At p@50 it is not better than the recipe: 0.255 vs 0.270 for ≥ 1 paper, and
  0.060 vs 0.065 for ≥ 2. With about 4–20 hits per cutoff these p@50 gaps are one hit or less.
  SVD rank 5 is close, but it is sensitive to the rank (ranks 10–20 fall to chance).
- **Tag level** (`problem_class` × `method_family`, about 40,000 empty cells): popularity wins
  (AUC 0.81 for ≥ 1, 0.84 for ≥ 2). Item CF does not help there, so tags keep no transfer
  signal.

In production, `bridge_method_transfer` uses item CF by default
(`SignalParams.transfer_method = "item_cf"`; `"recipe"` restores the neighbour share). Pilot:
174 candidates; the judge's top scores are 4–5.

## 5. Topic, question, credibility and imports beyond rich-get-richer

**Question.** Do *what* a paper is about, the question it asks, who wrote it and what it borrows
predict its success once the rich-get-richer effect is controlled for?

**Population.** 2,103 papers published in T = 2020–2023, article and preprint types. 113
duplicate records were collapsed by normalized title first.

### Method

- **Base model:** the bibliometric and rich-get-richer features. These are corpus and global
  citations so far, coupling, reference count, team size, preprint, and expansion route.
- **Feature families:** each family is added to the base, and its AUC gain is reported with a
  paired bootstrap 90% interval, leave-one-cutoff-out.
- **Populations:** all papers; the forward route only; cold start (no corpus citations at T);
  global cold start (no citations anywhere at T, 624 papers).
- **Models:** logistic. Gradient boosting gave the same or slightly worse results.
- **Inputs available at T only.** Global counts come from OpenAlex `counts_by_year`, fetched by
  `fetch_counts_by_year.py` for 5,410 works and 11,547 authors ($0.017). Only years ≤ T feed the
  features.

### Outcomes

All are top quartile within T. They differ in what a paper is compared with.

| Outcome | Definition | Why |
| --- | --- | --- |
| `y_corpus` | corpus citations in T+1…T+3 (as in exp2) | in-field uptake, raw |
| `y_corpus_peer_ns` | the same without self-citations, relative to the 15 most similar same-year papers | in-field uptake, compared with like papers. Self-citations are 16% of corpus citations |
| `y_g3` | global citations received in T+1…T+3 | raw global success |
| `y_g3_field` | the same relative to the median of its OpenAlex subfield and year | removes field citation norms |
| `y_g3_peer` | the same relative to the 15 most similar same-year papers | removes field and topic norms |

In-field and global success are almost unrelated: `y_corpus` vs `y_global` r = −0.02, and
`y_corpus_peer_ns` vs `y_g3_peer` r = 0.22.

### Feature families (all as of T)

| Family | Features |
| --- | --- |
| topic | semantic neighbourhood among earlier papers (abstract and problem vectors): in-field and global *heat* (neighbours' citation percentile by T, and in year T alone), typicality, crowding, local momentum, drift alignment |
| question | *reach* (how many problems posed in T−2…T the paper's innovation matches), breadth, match to stated open limitations, typicality of the problem, atypicality of the method chosen for it |
| author | in-field track record and topical expertise (corpus only), number of prior corpus papers |
| author_global | OpenAlex citations received by T and in T−2…T, works by T, citations per work, team total |
| venue | the venue's earlier papers' citation percentile, in-field and global (shrunk), and its prior count |
| imports | share of references outside the corpus; *rising imports* (external works whose citations from corpus papers in T−1…T outpace T−4…T−2): the maximum and the count cited; *fresh imports* (external works cited at most twice before) |

### Results: AUC gain over the base

| Family | In-field, peer, no self-cites: all | forward route | cold start | Global field-normalized: all | forward route | Global peer: forward route |
| --- | --- | --- | --- | --- | --- | --- |
| topic | +0.007 (n.s.) | **+0.012** | +0.006 (n.s.) | −0.002 | −0.003 | −0.002 |
| question | 0.000 | −0.004 | +0.002 | −0.001 | −0.005 | −0.004 |
| author (corpus) | **+0.012** | +0.009 (n.s.) | **+0.018** | −0.004 | −0.001 | −0.001 |
| author_global | +0.005 (n.s.) | +0.004 (n.s.) | +0.008 (n.s.) | +0.004 (n.s.) | **+0.012** | +0.008 (n.s.) |
| venue | **+0.017** | +0.008 (n.s.) | **+0.027** | −0.003 | +0.003 | **+0.011** |
| imports | **+0.022** | **+0.021** | **+0.029** | −0.001 | +0.004 | −0.002 |
| all | **+0.035** | **+0.028** | **+0.039** | −0.002 | +0.004 | +0.003 |

Bold marks a 90% interval above zero. Base AUC is 0.74 (in-field) and 0.83–0.84 (global).

### What it means

1. **Topic and question predict the audience, not the winner.** Against *raw* in-field
   citations, topic and question add +0.03 to +0.045 AUC; the strongest feature is typicality
   (posing the field's mainstream questions). Against same-topic peers, that gain mostly
   disappears. Dense regions of the corpus have more potential citers. That is real and useful
   for "which topics draw attention", but it does not pick the better paper within a topic.
2. **Credibility shows its hand only like-for-like, and modestly.**
   - **Author (corpus):** about half of the raw effect was self-citation; +0.012 survives.
   - **Venue:** +0.017 in-field and +0.011 globally (forward route).
   - **Global author credibility:** adds +0.012 for global field-normalized success (forward
     route), the one family that helps there. Citations per work is its best feature.
3. **Rising imports are the most robust new in-field signal.** They hold in every population,
   including the forward route, where other families fade.
   - **What helps:** citing external works that the field is importing at an increasing rate.
   - **What hurts:** relying mostly on external references, or on brand-new imports (in-field).
   - In short: ride an incoming wave; being the first importer pays off less in-field.
4. **Global success stays mostly unexplained beyond the base.** After field normalization,
   the base (dominated by global citations so far, reference count and route) is not
   improved by any content family.
5. **Selection on the outcome limits the global tests.** The forward and semantic routes
   fetched papers in order of 2026 citations. Route alone has AUC 0.80 for global success
   among papers uncited at T. An unbiased per-year sample of the field is needed before the
   global conclusions can be trusted.

### Coverage gap: external ideas

80–94% of each year's references point outside the corpus, and 6,624 external works are cited
at least 5 times by corpus papers. Reference hopping fetches the seed's foundations, not
current imports. The import signal above uses only the external *ids* in `refs`. The next step
is to fetch and embed those works, to see which fields DRO imports from and whether an
external trend that matches DRO's problems predicts its import.

## 6. Ideas borrowed from outside the field

**Question.** Ideas often come from neighbouring fields and from what is trending there. Do
the field's imports follow external trends and its own changing questions, and does it pay a
paper to borrow from them?

**Data** (`external_mine.py`, about $1.10 in LLM extraction and $0.01 in OpenAlex calls):
- 80–94% of each year's references point outside the corpus: 252,053 citations to 148,992
  external works.
- The 6,624 works cited by at least 5 corpus papers were fetched (6,523 still served), with
  abstracts, subfields and yearly global citations. They cover about a quarter of external
  citations; about 75% of each year's external citations go to works cited fewer times.
- 4,094 works with an abstract got the corpus facet prompt (problem, method, innovation, …).
  All 6,523 were embedded with the corpus model.
- Everything stays in `data/projects/<run>/backtest/external/`, apart from the corpus tables.

### A. Import map

Among the fetched works, the share of external citations by subfield, 2017–19 vs 2023–25:
- **Rising:** Management Science and Operations Research, 16% → 22%; Artificial Intelligence,
  12% → 14%.
- **Falling:** Statistics and Probability, 30% → 7%; Electrical and Electronic Engineering,
  18% → 11%.
- **Caveat:** recent external works have had less time to reach 5 corpus citers, so the newest
  imports are under-represented.

### B. What predicts more borrowing (lead indicator)

- **Candidates at T:** external works already cited by at least 5 corpus papers *by T*. The
  fetched set means "cited ≥ 5 times ever", which would select on the outcome; this rule
  keeps the selection to data up to T.
- **Strict hit:** the share of corpus papers citing the work in T+1…T+3, relative to T−2…T, is
  in the top quarter for T, *and* at least 3 corpus papers cite it after T. A ratio from a
  near-zero rate is mostly noise and mean reversion.
- **Data:** 17,348 candidate-cutoff rows, strict base rate 0.14.
- **Bootstrap:** the 90% interval resamples whole works, because a work appears at several
  cutoffs.

| Model (strict hit) | AUC | Gain over base | 90% CI | p@50 |
| --- | --- | --- | --- | --- |
| base (in-field rate and momentum, global citations by T, age) | 0.685 | — | — | 0.350 |
| + external trend (global citation momentum at T) | 0.698 | +0.013 | [+0.009, +0.017] | 0.335 |
| + semantic fit (to the problems posed now, shift since T−4…T−2, limitation match) | 0.705 | **+0.020** | [+0.015, +0.027] | 0.355 |
| + externality (subfield rare in the corpus) | 0.685 | 0.000 | n.s. | 0.345 |
| all | 0.710 | **+0.025** | [+0.019, +0.031] | 0.345 |

- **Best single feature:** `fit_shift` (AUC 0.65 alone). The field imports more of what sits
  where its questions are *moving*.
- **Second:** the work's global trend (0.62). External trending ideas do get pulled in more.
- **Not predictive:** how foreign the subfield is.
- **Where the gain shows:** overall ordering, not a sharper top of the list (p@50 unchanged).
  With the ratio-only hit the gain is +0.032, but that definition is noisier.

### C. Does it pay a paper to borrow? (exp5 "borrowing" family)

For each corpus paper, the external works it cites that were already imported by its year
(≥ 5 corpus citers then) are described by:
- their global trend at the time;
- their semantic distance from the paper;
- how foreign their subfields are.

| Outcome | All papers | Forward route | Cold start |
| --- | --- | --- | --- |
| In-field, peer-normalized, no self-cites | **+0.016** [+0.006, +0.024] | **+0.014** [+0.005, +0.024] | **+0.018** [+0.002, +0.035] |
| Global, field-normalized | −0.003 | −0.002 | −0.004 |

- **What helps:** citing a work that is trending globally (`b_trend_max`, within-decile AUC
  0.57–0.59).
- **What hurts:** borrowing from semantically far away (0.43) or from foreign subfields (0.47).
- **In short:** *borrow what is trending, from nearby.* This agrees with the import result in
  section 5 (ride the wave, don't be the first importer). Like every other family, it does
  not explain global success.

### D. Live list (explorer: Bridges › From outside the field)

- **What it is:** the strict model fitted on all cutoffs scores the external works with at
  least 3 corpus citers in 2023–25 (1,538 works, as of 2025).
- **Outside the field:** the third of the list whose *abstracts* are farthest from the corpus.
  OpenAlex subfields mislabel many in-field works, and title-only records are excluded.
- **Current examples:** causal and adapted optimal transport, elicitable risk measures,
  high-dimensional probability, multi-marginal optimal transport, robust Kalman filtering.
- **Coverage gap:** the rest of the list is often *core papers of this field that the corpus
  never fetched*, such as "Data-driven robust optimization" and "Tight bounds for some risk
  measures". That is a coverage gap, and a direct input for corpus expansion.

## 7. Coupling vs semantics: what needs the citation network?

**Question.** Bibliographic coupling (shared references) is the strongest bridge feature. Does its
strength depend on received citations, and what is left for semantics where the network is blind?

- **Setup:** the experiment 1 pairs (community X, innovation paper i not yet cited by X). Models
  are trained on all pairs and scored inside strata where the network gives less and less.
- **Protocols:** leave-one-cutoff-out, and chronological (train 2020, test 2023; outcomes include
  partial 2026). Both agree.

AUC, leave-one-cutoff-out (chronological in brackets):

| Stratum | Pairs / adoptions (LOCO) | Citations + age | + coupling | Semantic only | All |
| --- | --- | --- | --- | --- | --- |
| Fresh (published in T) | 42,720 / 858 | 0.65 (0.64) | **0.84** (0.83) | 0.75 (0.75) | **0.86** (0.85) |
| Uncited at T | 117,794 / 936 | 0.66 (0.70) | 0.81 (0.82) | 0.78 (0.78) | 0.84 (0.85) |
| Zero coupling | 76,424 / 327 | 0.77 (0.80) | — | 0.63 (0.61) | 0.85 (0.84) |
| Uncited and zero coupling | 35,758 / 35 | 0.54 (0.69) | — | 0.66 (0.76) | 0.64 (0.75) |

1. **Coupling does not need received citations.** It is strongest on papers published in the
   cutoff year itself: adding it to citations + age lifts AUC from 0.65 to 0.84 (chronological:
   0.64 to 0.83). A reference
   list exists on the day a paper appears.
2. **Coupling defines what is reachable.** 91% of adoptions involve a candidate that shares
   references with the adopting community. Without shared references, adoption is rare (0.4%
   vs 1.5%), and semantics alone is weak (0.61–0.63), weaker than popularity.
3. **Semantics helps most on fresh and uncited papers:**
   - alone, +0.08 to +0.12 AUC over citations + age (90% CI above zero in both protocols);
   - on top of coupling, +0.02 to +0.03.
4. **Truly network-blind adoption** (uncited and no shared references) is too rare to measure:
   35 adoptions in four cutoffs (7 in the chronological test). In that stratum, models trained on
   all pairs differ mainly in the sign of the age coefficient (citations and coupling are both
   zero), so their differences are artefacts.

## 8. Community birth: seeing new work before its community exists

**Question.** Every earlier test predicts uptake by communities that already exist at the
cutoff. Can signals at T show where a new community will form?

**Births.** Leiden communities are too coarse to show births (new work is absorbed), so births
are detected in embedding space:
- **Frontier papers:** papers from T+1…T+3 in the top quarter of distance from everything
  published up to T.
- **Births:** groups of at least 5 frontier papers close to each other. Cutoffs 2019–2022 give
  104 births.
- **Seeds:** papers from T−2…T nearest to several members of a birth (337 of 8,163 recent
  papers).

**Test.** Signals at T rank the seeds against all other recent papers. Because seeds of new work
tend to sit at the edge of existing work, the base model includes isolation, novelty, citations
and coupling.

| Population | Base AUC | All signals | Gain (90% CI) | p@50 (base rate 0.04) |
| --- | --- | --- | --- | --- |
| All recent papers | 0.55 | 0.58 | +0.029 [+0.004, +0.055] | 0.04 |
| Forward route only | 0.57 | 0.61 | +0.049 [+0.019, +0.080] | 0.085 |

- **Not demonstrated.** The best model reaches AUC 0.58–0.61, and the top 50 are close to the
  base rate. Local momentum, reference-pair novelty and borrowing add nothing significant.
- **Births are offshoots, not crossovers.** Straddling several communities' literatures points
  *away* from births (bridging entropy AUC 0.38–0.43). Being concentrated in one community's
  literature points towards them (0.57–0.61). Nearby stated limitations help a little in the
  forward route (+0.039).
- **Sampling shows through.** Many births consist mostly of author-route papers (seed authors'
  newest work drifting into new topics). Even restricted to the forward route, the live list's
  top is mostly off-field outliers, such as function theory and ship-engine diagnostics. The
  signals mainly point at the corpus's edges.
- **Live list:** frozen in the prospective snapshot as a pre-registered, low-confidence
  prediction (`exp8_regions_live.parquet`). It is not shown in the product.

## 9. Realized opportunities: content, not citations

**Question.** Citation adoption is a social act. Did the opportunity *happen*, whether or not
anyone cited it?

**Realized.** A pair (X, i) counts as realized when, in T+1…T+3:
- an X paper's method or innovation is close to i's innovation (cosine ≥ 0.894, the 99th
  percentile of random pairs), and
- it is closer than any of X's papers from T−2…T were.

| Outcome (LOCO) | Positives | Citations + age | + coupling | Semantic only | All |
| --- | --- | --- | --- | --- | --- |
| Cited | 3,757 | 0.71 / p@10 0.18 | 0.82 / 0.22 | 0.72 / 0.08 | 0.84 / 0.23 |
| Realized | 19,122 | 0.61 / 0.17 | 0.73 / 0.22 | 0.76 / 0.31 | 0.79 / 0.27 |
| Realized without citation | 18,128 | 0.60 / 0.11 | 0.72 / 0.15 | 0.76 / 0.28 | 0.78 / 0.23 |

- **95% of realized uptake happens without citing the idea** (994 of 19,122 realized pairs were
  cited). Measured by citations alone, uptake of ideas looks about five times smaller than it is.
- **Control for generic ideas** (`exp9_within_candidate.py`). Rank communities *for each idea*,
  so that an idea close to everything cannot score by being generic. Then semantics (0.64),
  coupling (0.63) and plain persistence (the community was already close: 0.65) are about
  equal. Most of semantics' apparent advantage above came from generic ideas.
- **Conclusion:** the content outcome is a better *measurement* of uptake, but it does not show
  that semantics has special foresight about who will take an idea up.

## 10. Community warmth

**Question.** The reverse of experiment 1: for a given paper, which communities will take it up?
For each candidate paper at cutoff T, the communities that do not cite it yet are ranked, and a
community is warm if it cites the paper in T+1…T+3 (or realizes it, experiment 9). The scores are
computed within each paper (`exp10_warmth.py`).

| Ranking (cited) | AUC within a paper | hit@3 | MRR |
| --- | --- | --- | --- |
| random | 0.50 | 0.25 | 0.25 |
| community size | 0.64 | 0.39 | 0.32 |
| coupling | 0.77 | 0.59 | 0.48 |
| problem similarity | 0.71 | 0.52 | 0.45 |
| recent closeness | 0.75 | 0.57 | 0.47 |
| semantic (problem, limitation, recent) | 0.77 | 0.60 | 0.50 |
| **all** | **0.81** | **0.69** | **0.57** |

- **Content outcome:** problem similarity alone is best (hit@3 0.60 vs 0.40 at random); the full
  model is not better there.
- **Production:** the model fitted on all cutoffs is saved to `models/community_warmth.json` and
  served by `landscape portfolio` / `my-paper` ([09-portfolio.md](09-portfolio.md)).

## 11. Do big labs steer the field?

**Question.** Conversations look steered by big labs. When a prominent team steps somewhere new, does
a follow-up wave by *other* people come after?

**Prominence, as of the step year T only** (OpenAlex citations received by each author up to T):
- **big author:** the team's most-cited author;
- **big lab:** the team's institution whose authors in the map have the most citations up to T;
- **big** = the top 10% of that year's papers.

**A step**, in two versions:
- **novel for the field:** the paper's novelty is in the top quarter;
- **new for the lead author:** distance from that author's own earlier papers in the map is in the
  top quarter.

**Outcomes** (3,440 papers from 2018–2022; outcomes through 2025):
- **Follow-up wave:** the share of papers by other authors in the paper's neighbourhood (its
  15-nearest-earlier-paper ball) in T+1…T+3, against T−2…T. Hit = top quarter and at least 3 such
  papers.
- **Citations from others:** citations from papers sharing no author with it.

**Base model:** novelty, local density and momentum, early global citations, team size, references,
route.

| Outcome (gain over base) | Big author | Big lab | Big author × novel | Big lab × novel | Big author × new for them |
| --- | --- | --- | --- | --- | --- |
| Follow-up wave (all) | −0.001 | −0.001 | −0.004 | −0.001 | −0.000 |
| Follow-up wave (forward route) | −0.002 | −0.001 | −0.002 | +0.001 | −0.003 |
| Citations from others (all) | +0.003 | 0.000 | +0.007 [+0.001, +0.012] | +0.005 | +0.010 [+0.003, +0.018] |
| Citations from others (forward route) | +0.001 | +0.001 | −0.001 | +0.002 | +0.013 [+0.004, +0.021] |

**Community level** (cutoffs 2019–2022, 76 community-cutoff cases): the number or prominence of big
names entering a community does not predict its growth once momentum, size and the number of
entrants are held fixed (partial ρ about 0, 90% CI −0.21 to +0.25).

What it says:
1. **No measurable wave after big names step in.** A big author's novel paper is followed by more
   papers nearby (median 18 vs 10), but its neighbourhood was already busier. The growth is no
   different: big teams step into places that are already active.
2. **Big authors mostly work in mature areas.** Outside their novel papers, their papers'
   neighbourhoods grow *less* afterwards (wave hit rate 14–19% vs 26–33%). For novel papers there
   is no difference (24% vs 25%).
3. **They get cited, which is not steering.** Novel papers by big labs are cited by others about
   twice as often in raw rates (17.5% vs 7.5%, 80 papers), but the controlled gain is small.
   The significant +0.010 to +0.013 for "new for the lead author" comes from having a track record
   in the map at all: the raw rates of big and other authors moving somewhere new are equal (25.7%
   vs 25.3%). That is the insider and credibility effect of experiment 5.

**Limits:**
- one field;
- prominence from OpenAlex author profiles (disambiguation errors), a proxy for "big lab";
- the "new for the lead author" step needs history inside the map (35 big-author steps);
- three-year horizon;
- a corpus sampled through the seed's citations.

Steering by big labs may well happen through channels this corpus cannot see: industry papers
outside it, benchmarks, software, and longer horizons.

## Prospective snapshot (frozen 2026-10-21)

`freeze_snapshot.py` froze 19 lists, each recommendation list next to a baseline frozen at the
same moment:
- paper bridges and gems; early work;
- method transfer; limitation → innovation;
- external imports; rising pockets; emerging clusters;
- the birth regions;
- the corpus ids and the community centroids.

The files are in `data/projects/dro-pilot/snapshots/2026-10-21/`. The manifest, with SHA-256 hashes of
every file and the pre-registered scoring protocol (hit definitions; horizons of 6, 12 and 24
months; comparison with the baselines), is versioned in
`docs/snapshots/dro-pilot-2026-10-21.json`. It is the only test free of hindsight, model
leakage and corpus selection. Score it by collecting papers published after the snapshot date
with the same expansion routes.

## Explorer

See [06b-explorer.md](06b-explorer.md):

- Bridges tab → *Papers to bring in*, with the likely and hidden-gem lists per cluster and the
  matched limitation as evidence.
- People tab → *Early work*.
- Map layer *Rising pockets*.

## Not tried yet

- **An LLM pair-checker on paper bridges** ("does innovation i address limitation l?"). It must
  be off, or use a model with an earlier training cutoff, in backtests, because of leakage.
- **An unbiased per-year field sample** (needs downloads). Cluster-trend conclusions depend on
  the expansion routes ([08-backtest.md](08-backtest.md), section 1).
- **An off-field audit of the corpus**, and deduplication of preprint/article twins.
- **Adopting exp5 features in `early_work`.** Against raw corpus citations (its target), all
  families together add +0.042 AUC (90% CI +0.030 to +0.055).
- **External works cited fewer than 5 times** (about 75% of external citations) are not fetched
  yet; the newest imports are under-represented.
- **Folding these experiments into `landscape backtest`**, so they re-run with one command.
