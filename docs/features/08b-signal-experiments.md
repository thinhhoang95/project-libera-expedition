# Feature: Signal-improvement experiments (Phase 8b)

Status: done (four experiments; three adopted, one dropped) · Scripts: `scripts/phase8b/` · Production models: `src/landscape/models/{paper_bridges,early_work}.json` (fitted by `scripts/phase8b/fit_models.py`) · Signals: `bridge_paper`, `bridge_paper_gem`, `early_work`, `paper_scores.local_momentum`, item-CF `bridge_method_transfer`

The Phase 8 backtest ([08-backtest.md](08-backtest.md)) showed which recipe signals carry
information and which do not. Phase 8b tried new ways to make the signals stronger. The rule
for adoption: a new signal must beat its baselines **leave-one-cutoff-out** (fit on three of the
cutoffs 2020–2023, score the fourth, repeat), on the existing corpus only (no new downloads).

## Run

All scripts take `--run-id` (default `dro-pilot`) and read the cutoff runs `<run>@T2020..2023`,
which `landscape backtest --run-id <run>` builds. Each runs in seconds to minutes and costs nothing
(no LLM calls). Intermediate tables go to `data/backtest/<run>/`.

**Every result table is saved, adopted or not**, to `data/backtest/<run>/experiments/` (with an
`index.json` of titles and how-to-read notes) by `src/landscape/experiments.py`. The explorer's
**Experiments** view shows them all for the selected run, together with the Phase 8 backtest
summary. These results describe one corpus: another field may give different answers, so a
signal dropped on the pilot stays visible and re-runnable.

```bash
uv run python scripts/phase8b/exp1_paper_bridges.py   # writes data/backtest/exp1_pairs.parquet
uv run python scripts/phase8b/exp1_evaluate.py
uv run python scripts/phase8b/exp2_early_work.py      # writes data/backtest/exp2_papers.parquet
uv run python scripts/phase8b/exp3_micro_topics.py
uv run python scripts/phase8b/exp4_method_transfer.py
uv run python scripts/phase8b/fit_models.py           # refits the two production models on all cutoffs
uv run python scripts/phase8b/fetch_counts_by_year.py # exp5 only: OpenAlex yearly counts (~$0.02, cached)
uv run python scripts/phase8b/exp5_features.py        # writes data/backtest/exp5_papers.parquet
uv run python scripts/phase8b/exp5_evaluate.py [logit|hgb] [y_...]
```

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
- **Metadata and embeddings for external imports** (see section 5).
- **Folding these experiments into `landscape backtest`**, so they re-run with one command.
