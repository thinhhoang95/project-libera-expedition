# Findings portfolio: what the experiments say, and where it lives in the product

2026-10-21 · Pilot: `dro-pilot` (Wasserstein DRO, 5,410 papers) · Sources: [08-backtest.md](08-backtest.md),
[08b-signal-experiments.md](08b-signal-experiments.md), [08c-research-lab.md](08c-research-lab.md)

This page collects every experiment so far into one list and links each finding to a product
surface. Everything below was measured on **one corpus**. That corpus was sampled from the seed's
citation graph, partly in citation order. Another field can give different answers, so the
explorer's **Experiments** view keeps every result table open and re-runnable (`--run-id`).

## Verdicts

- **Use:** backed by a backtest against baselines; shown in the product as a recommendation.
- **Use with care:** some evidence, with known weaknesses; shown with a warning.
- **Describe:** useful for understanding the field, but not predictive; never shown as a forecast.
- **Dropped:** lost to the baselines on the pilot; tables stay in the Experiments view.
- **Open:** cannot be tested properly with the current data.

## The portfolio

| # | Signal or question | Evidence (protocol) | Verdict | Where in the product |
| --- | --- | --- | --- | --- |
| 1 | **Papers a community will adopt** (paper bridges) | exp1: p@10 0.23 vs popularity 0.19, AUC 0.84 (leave-one-cutoff-out). 8c chronological test (train 2019, test 2022, outcomes 2023–25): 19.5% vs 13.6% | Use | Map & grid › Bridges › Papers to bring in; My paper › Papers to bring in; Start here |
| 2 | **Hidden gems** (the same, cited ≤ 1×) | exp1: long-tail AUC 0.84 vs popularity 0.55. 8c: p@10 among papers cited ≤ 1×: 0.086 with semantics vs 0.064 without | Use: this is where popularity is blind | Papers to bring in › Hidden gems |
| 3 | **Limitation → innovation between clusters** | Phase 8: about 2.4× random, the best-validated recipe signal (flow penalty power 0.5). 8c: only 345 papers state a limitation; shared vocabulary ≠ compatible assumptions | Use with care | Bridges › Cluster pairs (judge-rated) |
| 4 | **Method transfer** (empty problem × method cells) | exp4: item CF AUC 0.70 / 0.80 vs popularity 0.51 / 0.62; p@50 level with a neighbour share; only 4–20 hits per cutoff | Use with care | Grid: accent diamonds |
| 5 | **Underconnected communities** (cite each other less than expected) | 8c: 38 review hypotheses from a degree null model; not validated | Describe, for expert review | Research lab › Underconnected communities |
| 6 | **Early work** (recent papers the field will read) | exp2: AUC 0.815, p@50 0.67 vs citations so far 0.62. exp5: predicts in-field uptake, partly audience size | Use as a reading list, not a quality ranking | People › Early work; Start here › Read now; My paper |
| 7 | **Borrowed ideas** (imports from outside the field) | exp5: riding rising imports +0.022 AUC in-field (peer-normalized), in every population; the first importer does worse. exp6: borrowing what trends globally, from nearby, +0.016; the field imports more of what fits where its questions are moving (+0.020) and what trends globally (+0.013) | Use, as a review list | Bridges › From outside the field; Start here › Being imported now, Where this field borrows from, Coverage gaps; My paper |
| 8 | **Credibility** (venue, author record) | exp5: venue +0.017, author +0.012 in-field (half the raw author effect was self-citation); global author record +0.012 for global success (forward route) | Describe, as context on a paper | not ranked; visible on paper cards (venue, authors) |
| 9 | **Topic and question semantics** (heat, typicality, question reach) | exp5: +0.03 to +0.045 on raw in-field citations, about 0 against same-topic peers: they predict the audience, not the winner. 8c: semantics adds +0.01 AUC to paper bridges, mostly for uncited papers | Describe | Map, clusters, grid |
| 10 | **Novelty** | Phase 8 and exp2: anti-predictive of 3-year citations (AUC 0.29–0.39); novel work is recognized late | Describe: "different", not "promising" | Map › Novel papers layer |
| 11 | **Cluster direction** (growing, emerging, niche, dormant) | Phase 8: the recipe's E is anti-predictive; E′ is weak. 8c: 6 of 15 topics reverse direction once route-mix changes are separated out | Use with care | Status badges, **off by default**, with a warning; Ranking tab |
| 12 | **Rising pockets** (local momentum) | exp3: ρ +0.13, p@50 about 2× random; duplicate records can make false pockets | Use with care, as a pointer | Map › Rising pockets layer |
| 13 | Burstiness micro-topics | exp3: AUC 0.46 (mean-reverting) | Dropped | Experiments view only |
| 14 | Seeders, brokers, institutions | Phase 8: lose to popularity at predicting citations. 8c: more informative within one topic and role | Describe | People tab; Research lab › People by topic |
| 15 | Semantic but not cited (recipe) | Phase 8: empty on the pilot | Dropped; replaced by #5 | — |
| 16 | Contrastive (centered) embeddings for bridges | 8c: lower on the complete chronological test (18.6% vs 19.5%) | Dropped | — |
| 17 | **Global success** (citations from all of science) | exp5: no content, question, credibility or import family adds beyond the base after field normalization; the corpus is selected on this outcome | Open: needs an unbiased field sample | — |
| 18 | **Where the citation network is blind** (no shared references, uncited) | exp7: coupling needs no received citations but defines the reachable set (91% of adoptions share references); semantics adds +0.08 to +0.12 AUC on fresh / uncited papers; network-blind adoption is too rare to measure (35 cases) | Use semantics for fresh and uncited papers; network-blind discovery is open | Papers to bring in, Hidden gems |
| 19 | **Seeing a community before it forms** | exp8: best AUC 0.58–0.61, top 50 at the base rate; births are offshoots within one community, not crossovers; signals mostly point at the corpus's edges | Not demonstrated; frozen for the prospective test | — |
| 20 | **Uptake without citation** (the content moves towards an idea) | exp9: 95% of realized uptake is uncited; for a given idea, semantics ≈ coupling ≈ persistence (about 0.64) | A better outcome measure; no special foresight | — |
| 21 | **Prospective snapshot** (2026-10-21) | 19 frozen lists with baselines; SHA-256 manifest and protocol in docs/snapshots/ | Running: score at 6, 12 and 24 months | `landscape score-snapshot --run-id dro-pilot --date 2026-10-21 --months 12` |
| 22 | **Communities warm to a paper** | exp10: a top-3 community cites the paper within 3 years for 69% of papers (25% at random); within-paper AUC 0.81 | Use | My paper › Communities most likely to be warm to it |
| 23 | **Big labs steering the field** (a follow-up wave after prominent teams step in) | exp11: no wave by others in the neighbourhood (gain about 0, both "novel for the field" and "new for the team"); big authors' non-novel papers sit in mature areas that grow less; their small citation edge is the insider effect; big names entering a community do not predict its growth | Not supported here; steering may run through channels outside the corpus | — |
| 24 | **Second waves** (established ideas reaching new audiences) | combines exp6 (growing imports, +0.025 AUC), exp4 (method transfer, AUC 0.70–0.80), exp9 (uncited convergence) and exp10 (likely next audiences); diffusion already under way, not advance warning | Use, as a watchlist with evidence | Start here › Second waves; My paper › Second waves near you |
| 25 | **How newcomers successfully entered a community** | exp12 (6,585 entries, like for like): anchored in the community's literature and with an insider co-author does better; an unusual method (−5.4 pts uptake) or a new problem (−8.2 pts uptake, −3.4 pts staying) does worse in the first 3 years | Describe (what happened here, not a guarantee) | `entry_routes.parquet` (orientation.py): How people got in |
| 26 | **What to read first** | exp13 (76 community-cutoffs): newcomers *cite* the community's most cited papers (coverage 0.23) but *build on* the sources of its current methods (method lineage 0.22 vs 0.07); alternating both covers most of each; surveys cover neither | Use, as a reading path | `reading_paths.parquet` (orientation.py): Read this first |
| 27 | **Durable ideas** (still used 5+ years later) | exp14: among equally generic ideas, breadth at publication (already close to several communities) lasts (AUC 0.58); novelty fades (0.39); early citations say nothing (0.51) | Describe, as a marker | `lasting.parquet` (orientation.py): Lasting marker |
| 28 | **Community lifecycle and the survey gap** (what kind of contribution is wanted) | exp15: a community's own mix adds little to field drift, except generic → applied methods; "theory wanted" gaps do not attract theory; a survey written where surveys are scarce is taken up far more (47% vs 10%, +0.38 [+0.16, +0.54]; dose-response across cuts), cited rather than built on | Survey gap: Use (pending a product slot); lifecycle mix: Describe; theory wanted: Dropped | — (portfolio files busy with another agent's refactor) |
| 29 | **Long-standing open questions** | exp16: whether a question was answered cannot be read from abstracts (outside proposals are closer to a question than the askers' own work); neither "standing" nor the breadth of askers predicts a question being asked again beyond recent activity (+0.011 AUC, n.s.) | Not demonstrated; recurring-question list kept for inspection only | — |

## Borrowed ideas (exp6)

- **Scale:** 80–94% of references point outside the corpus. The 6,523 external works cited at
  least 5 times were fetched, faceted and embedded; they cover about a quarter of external
  citations.
- **What gets imported more:** works whose content fits where the field's questions are
  *moving* (the best single feature), and works trending across all of science. How foreign
  the subfield is does not matter. The gain is in overall ordering, not in a sharper top 50.
- **Who benefits:** papers that borrow globally trending work from nearby out-draw their peers
  in-field. Borrowing from far away does worse in the short run. Nothing here explains global
  success.
- **Product:** Bridges › *From outside the field* lists current imports, as of 2025,
  scored by this model. Its other scope, *in-field, missing from the corpus*, shows core papers
  the corpus never fetched: a coverage gap to feed back into corpus expansion.
- **Not covered yet:** external works cited fewer than 5 times (about 75% of external
  citations), where the newest imports sit.

## What the evidence says about how attention works (pilot)

1. **Short-term attention is relational.** Papers are cited by those whose references they
   share (bibliographic coupling is the strongest predictor everywhere). Popularity is a strong
   baseline but beats nothing in the long tail.
2. **Content pays late, if at all, in citations.** Novelty is penalized over three years;
   topic and question features only tell you where the readers are.
3. **Ideas move in waves.** The field imports more of what fits where its questions are moving
   and what trends across science. Following an import the field is taking up pays off
   in-field; being the first to import, or borrowing from far away, does not.
4. **Success depends on the audience.** In-field and global success are almost unrelated
   (r ≈ −0.02).
5. **Ideas spread far more than citations show.** When a community's later work moves towards an
   idea, it cites that idea only about 5% of the time. Citation-based evaluation sees a small,
   socially filtered part of uptake.
6. **New communities grow out of one community's literature, not between two.** Bridging
   signals point away from births; nothing tested sees a birth coming with useful precision.
7. **Measurement creates much of the signal.** Field citation norms, audience size,
   self-citation, duplicate records, sampling routes and leave-one-cutoff-out training (which
   can learn from the test period's future) each inflated something on the pilot.

## Implications for the product

- **Lead with discovery, not prediction:** papers to bring in, hidden gems, borrowed ideas,
  method transfers and cluster bridges, each with its evidence and a review action.
- **Show direction and people as description,** with warnings, never as forecasts or rankings
  of merit.
- **Measure usefulness directly.** Citation adoption is a proxy that favours incremental,
  well-connected work. The review buttons (worth pursuing / obvious / wrong) are the endpoint
  that matters; the recipe's 30% useful-and-non-obvious target is still unmeasured.
- **Before claiming trends,** collect an independent, year-balanced field sample.

## Protocols, for reading the numbers

- **Leave-one-cutoff-out** (Phase 8, exp1–6): fit on three cutoffs, score the fourth. It can
  learn from outcomes after the test cutoff, so it is retrospective and may be optimistic.
- **Chronological** (8c): train only on cutoffs whose outcomes were complete before the test
  cutoff. Stricter; available for paper bridges.
- **Like-for-like outcomes** (exp5): compare a paper with its field or its semantic peers, and
  remove self-citations, before crediting a signal.
