# Feature: Backtest and evaluation (Phase 8)

Status: implemented on the existing corpus (no new downloads) · Code: `src/landscape/backtest.py`, analysis scripts in `scripts/phase8/` · CLI: `landscape backtest --run-id dro-pilot [--cutoffs 2020,2021,2022,2023] [--rebuild]`

Output: `data/projects/<run>/backtest/results.parquet` (every cutoff × signal × ranking) and `summary.csv`
(averaged over cutoffs).

## Protocol as built

1. **Cutoffs T = 2020, 2021, 2022, 2023.** The recipe's 2018–2021 cannot work: the corpus has
   almost nothing before 2018, and T+3 must stay ≤ 2026 (2026 is a partial year, which is fine
   for shares).
2. **Cutoff runs.** For each T, a derived run `dro-pilot@T<T>` holds only the base run's papers
   from T or earlier, so only citations made up to T are visible. Phase 5 is re-fitted on that
   data (problem/method at α 0.3 and 0.5, all other facets; new cluster ids, never the full
   run's), and the Phase 6 signals are computed at T with the LLM judge off.
3. **Outcomes from T+1…T+3.** Every paper, before and after T, is assigned to the cutoff's medium
   clusters by nearest centroid, so shares are measured the same way on both sides of T.
4. **Scoring.** Precision@k averaged over the four cutoffs. Cluster lists are short (about 17–20
   per clustering), so cluster signals use p@5/10/20, both per clustering and *pooled* over the
   four clusterings (scores z-normalized within each). Other signals use p@10/20/50.
5. **Baselines.**
   - Random: reported as the base rate.
   - Popularity: size, or citations up to T.
   - Momentum: last-year share change.
   - Citation-only versions: received-citation growth for clusters; citations from P's papers to
     m-papers for method transfer; works cited by both X and Y (common neighbours) for bridges.

### Hit definitions

| Signal | Recipe hit | Smoothed hit (added) |
| --- | --- | --- |
| Emerging | share(T+3) ≥ 1.5 × share(T) | mean share T+1…T+3 ≥ 1.5 × mean share T−2…T |
| Growing (added; the recipe has none) | mean share T+1…T+3 > share(T) | mean share after > mean share before |
| Moving away | share falls in T+1 and again in T+2 | mean share after < before and falls T+1 → T+2 |
| Method transfer (P, m) | ≥ 2 papers in the empty cell in T+1…T+3 | — |
| Limitation → innovation (X → Y) | flow X→Y per year at least triples (and ≥ 1/yr when it was 0), or a T+1…T+3 paper in X cites a matched innovation | — |
| Seeder / broker | author's corpus citations received in T+1…T+3 in the top decile | — |
| Novel paper (extra; not in the recipe's table) | the paper's T+1…T+3 citations in the top quartile of its year | — |

**Why a smoothed hit.** The recipe's cluster hits compare one year T with later years. A cluster
that a momentum signal ranks high has, by construction, an unusually high share in year T, so
regression to the mean counts it as a miss. Both definitions are reported side by side.

## What we learned

### 1. The recipe's emergence score is anti-predictive on this corpus

Spearman correlation between each cluster feature at T and the smoothed future share change
(log of mean share after over mean share before), averaged over cutoffs and clusterings:

| Feature | All routes | Outcomes on forward-route papers only | Outcomes on non-forward papers |
| --- | --- | --- | --- |
| **E (recipe)** | **−0.22** | −0.13 | −0.07 |
| novelty | −0.19 | −0.46 | +0.11 |
| new-author share | −0.17 | −0.10 | −0.13 |
| g (recent growth) | −0.12 | +0.02 | −0.07 |
| dg (acceleration) | +0.15 | +0.10 | +0.14 |
| momentum | +0.25 | +0.14 | +0.28 |
| reference age | +0.30 | +0.03 | +0.23 |
| external inflow | +0.26 | +0.15 | +0.05 |
| coherence | +0.20 | +0.33 | 0.00 |
| citation growth | −0.04 | +0.31 | −0.19 |

Three findings:

- **Every ingredient of E except acceleration points the wrong way.** This includes novelty,
  new-author share and recent growth. Young, novel clusters with many newcomers tend to be fringe
  arrivals that fade. Clusters that are integrated into the field (momentum, rising acceleration,
  established reference base) keep growing.
- **Several features flip sign depending on which papers measure the outcome.** The corpus's
  composition over time is set by the expansion routes: the forward route samples papers that
  cite the seed's references, ranked by citations. So cluster "trends" are partly a sampling
  artefact of how the corpus was built.
- **Only a few features keep their sign under all three outcome samples:** momentum (+),
  acceleration (+) and new-author share (−).

### 2. Reworked cluster score (adopted)

`E′ = z(momentum) + z(dg) − z(new-author share)` replaces the recipe's E for emerging clusters
and also ranks growing (E′ > 0) and moving away (E′ < 0), for clusters with at least 20 papers.

- **How the features were chosen.** A feature was kept only if its sign was consistent (|ρ| >
  0.05) across all three outcome samples. Under leave-one-cutoff-out, momentum was selected in
  4/4 folds and dg and new-author share in 3/4.
- **Configuration.** `SignalParams.emergence_score = "backtested"`, with weights in
  `emergence_weights`. Set it to `"recipe"` to restore the original. The recipe's E is still
  stored as `E_recipe`.

**Out-of-sample (leave-one-cutoff-out, features chosen without the held-out cutoff), per clustering:**

| Ranking | ρ with future | Emerging p@3 (smoothed) | Growing p@5 (smoothed) | Moving away p@5 (smoothed) |
| --- | --- | --- | --- | --- |
| Recipe E | −0.22 | 0.000 | 0.488 | 0.175 |
| Random | — | 0.083 | 0.525 | 0.350 |
| Size | 0.01 | 0.062 | 0.500 | 0.288 |
| **E′** | **+0.25** | **0.125** | **0.575** | **0.400** |
| Base rate | | 0.071 | 0.486 | 0.307 |

Against the recipe's single-year hits, E′ is still about at the base rate out of sample
(emerging 0.10 vs 0.12, growing 0.45 vs 0.49, moving away 0.25 vs 0.22).

**Full backtest after the change (pooled, p@10).** This is partly in-sample, because the
features were picked on these cutoffs:

| Signal | E′ | Recipe | Best baseline | Random |
| --- | --- | --- | --- | --- |
| Growing (smoothed) | 0.70 | 0.45 | 0.50 (momentum) | 0.48 |
| Moving away (smoothed) | 0.55 | 0.18 | 0.35 (momentum) | 0.31 |
| Moving away (recipe hit) | 0.35 | 0.08 | 0.23 (size) | 0.22 |
| Emerging (smoothed) | 0.10 | 0.03 | 0.08 (momentum) | 0.07 |
| Emerging (recipe hit) | 0.05 | 0.05 | 0.08 (citation growth) | 0.12 |

**Verdict.**

- **Direction (growing / moving away):** keep E′. It beats every baseline under smoothed hits,
  and beats them for moving away under the recipe's hit too.
- **Emerging:** weak. About 5 hits in 70 candidates per cutoff is too few to tell apart from
  momentum. Keep it as a review prompt, not a prediction.

### 3. Limitation → innovation: the one strong signal

| Ranking | p@10 | p@20 | p@50 |
| --- | --- | --- | --- |
| **Signal (flow-penalty power 0.5)** | **0.275** | **0.262** | **0.255** |
| Signal with the recipe's penalty (power 1) | 0.225 | 0.225 | 0.215 |
| Popularity (size X × size Y) | 0.100 | 0.075 | 0.115 |
| Citation-only (common cited works) | 0.050 | 0.075 | 0.105 |
| Random | 0.105 | 0.105 | 0.105 |

It beats random about 2.4× and every baseline, with the recipe's own formula as well.

**Ablation** (`lim_flow_penalty.py` and an earlier variant run):

| Variant | p@50 |
| --- | --- |
| Generic-limitation filter: drop 0% | 0.185 |
| Generic-limitation filter: drop 20% (as built) | 0.215 |
| Generic-limitation filter: drop 40% | 0.145 |
| top-1 match | 0.110 |
| top-5 match (as built) | 0.215 |
| top-10 match | 0.225 |
| No flow penalty | 0.275, but the top-50 median existing flow rises to 3.1 citations |
| Flow-penalty power 0.5 (adopted) | 0.255, median existing flow 1.5 citations |

Removing the flow penalty ranks best, but the candidates stop being *missing* bridges. Power 0.5
(`SignalParams.lim_flow_penalty_power`) keeps them near-missing while recovering most of the
gain. The differences between 0.215, 0.255 and 0.275 are small on four cutoffs.

### 4. Method transfer: cannot be validated here

Only about 4 of 121 empty cells per cutoff get ≥ 2 papers within three years. With so few hits,
the signal (0.035), the baselines (0.04–0.045) and random (0.036) cannot be told apart. This
needs a larger or longer corpus (later cutoffs with more future years, or a 20k-paper run), not
tuning.

### 5. Seeders, brokers, novel papers: descriptive, not predictive

- **Seeders and brokers lose to popularity.** Popularity wins at 1.00 (seeders score 0.54 and
  brokers 0.83 at p@50). Within popularity deciles, the broker score's AUC swings between 0.49
  and 0.79 across cutoffs (mean 0.59), so there is no robust information beyond "already cited".
  Future corpus citations are almost determined by past ones, which partly reflects how the
  corpus was sampled.
- **Novelty is consistently anti-predictive of 3-year citations** (AUC 0.31–0.39 for every facet,
  and 0.39 even for papers central to their cluster). The semantic route is not the cause: after
  2020 it supplies almost none of the most novel papers. This matches the delayed recognition
  of novel work: novel papers are cited less in short windows.
- **Decision:** these lists stay in the explorer as descriptive roles (who connects communities,
  who wrote the early papers of growing areas, what is unlike earlier work), labelled as not
  predictive. Novelty is no longer an ingredient of emergence.

## Threats to validity

- **Expansion-route sampling.** The corpus was built in 2026. The forward route takes citers by
  citation count, so the papers up to T are partly selected by their *future* popularity, and
  later years are sampled through citations to older work. Section 1 shows that this changes
  cluster-level conclusions. An unbiased check needs a sample collected independently of the
  seed's citation graph (for example a per-year random OpenAlex sample of the field's topics),
  which needs new downloads.
- **Model leakage.** The extraction LLM and the embedding model were trained on later papers.
  The judge is switched off in the backtest.
- **Small samples.** There are 4 cutoffs and about 70 pooled clusters each; cluster-level
  precision differences of ±0.05 are noise.
- **Tuned on the test period.** The adopted choices (E′ features, flow power 0.5) were made on
  these four cutoffs. The leave-one-cutoff-out numbers are the honest ones for E′. There is no
  held-out estimate for the flow power.
