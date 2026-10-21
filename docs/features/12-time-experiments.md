# Feature: Time experiments (opportunity age, still open)

Status: experiments run on the pilot; product changes **made** (below) · Code: `scripts/phase11/` (`common.py`, `exp29_opportunity_age.py`, `exp30_still_open.py`) · Step: `experiments-11` (no spend) · Results: Experiments view, `exp29_opportunity_age__*`, `exp30_still_open__*` · Pilot: `dro-pilot`, 2026-10-21

## The question

Every opportunity list rests on dated evidence: a limitation stated in some year, a method a neighbouring community
picked up in some year, a paper published in some year. The earlier backtests score a list at cutoff T against what
happens in T+1..T+3, so they tell us whether a list works **on average**. They never asked two things a user needs:

1. **Age (experiment 29).** Does old evidence convert as well as fresh evidence, once the list's score is taken into
   account?
2. **Still open (experiment 30).** When the list is made, is the gap still open, already being taken, or already
   closed?

## Run

```bash
uv run python scripts/run_experiments.py --step experiments-11 --run-id dro-pilot   # or the queue step experiments-11
```

Experiment 29 needs the cutoff runs 2019–2023 (backtest and prepare-cutoffs) and experiment 1's candidate pairs
(`experiments-8b`). Experiment 30 reads experiment 29's candidate tables (`backtest/exp29_*.parquet`). They take a few
minutes each and make no downloads and no LLM calls.

## Protocol

**The three lists and their hits** (the lists' own definitions, 3-year horizon):

| List | Candidates | Hit |
| --- | --- | --- |
| Method transfer (exp4, cluster level) | empty problem × method cells with item CF > 0, cutoffs 2019–2023 (581 cells, 31 hits) | ≥ 2 papers in the cell |
| Limitation → innovation | the cutoff runs' `bridge_lim_inn` lists, 200 per cutoff 2020–2023 | *strict*: X cites a listed innovation; *recipe*: or the flow triples (Phase 8) |
| Paper bridges and hidden gems (exp1) | community × not-yet-cited paper, cutoffs 2020–2023 (244,794 pairs) | X cites the paper |

**Observed / expected (O/E).** "Expected" is the hit rate of the candidate's score quintile at its cutoff. A band with
O/E below 1 converts worse than its score promises, which is an age or status effect the score does not see. The 90%
intervals come from resampling communities.

**Model test.** Logistic models on the score's own features, with and without the time or status features, scored
two ways:
- *LOCO*: fit on the other cutoffs;
- *chronological*: train only on cutoffs whose outcomes were complete by the test cutoff (T′ + 3 ≤ T).

**Time features (experiment 29):**
- *Exposure*: years since one of the problem's 10 most similar communities (by method profile) first used the method.
- *Method age*: T − median year of the method's papers.
- *Momentum* of the method and of the problem community: log of papers in T−2..T over T−5..T−3.
- *Ages* of the limitations, of the innovations and of the paper.

**Status (experiment 30)**, from evidence up to T only:
- **closed**: X already cites a listed work.
- **being taken**: not closed, and X's papers of T−2..T reach the idea, closer than X's papers of T−5..T−3. "Reach"
  means a method or innovation statement within 0.894 of the idea's innovation statement, which is experiment 9's
  threshold.
- **open**: neither.

Method-transfer cells are empty by construction. Their status is how close the problem community's recent methods are
to the method's centroid (tertile at the cutoff) and whether they moved towards it.

## Experiment 29: does age matter?

### Paper bridges and hidden gems: an uncited paper's pull fades within five years

O/E by the paper's age at the cutoff (score without its age term):

| Age | 0 y | 1 y | 2 y | 3 y | 4 y | 5 y | 6–10 y | 11+ y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All candidates | 1.23 | 1.11 | 0.90 | **0.73** | **0.59** | **0.67** | 1.60 | 1.57 |
| Hidden gems (cited ≤ 1×) | 1.33 | 1.11 | 0.78 | **0.54** | **0.42** | **0.49** | 0 of 169 | 1.23 (5 hits) |

- **The fade is steady and holds at every cutoff with data.**
  - All candidates aged 3–5 years: O/E 0.70, 0.64 and 0.65 at cutoffs 2021–2023.
  - Gems aged 3–5 years: 0.50, 0.50 and 0.51.
  - The 2020 cutoff is the exception: its 3–5-year-old papers are the field's founding works of 2015–17.
- **It holds within each status:**
  - papers already being taken: hit rate 5.9% at age 0, 1.6% at ages 4–5;
  - open papers: 1.2% at age 0, 0.5% at ages 4–5.
- **Old works (6+ years) convert *above* their score.** These are established papers reaching new communities: the
  second waves.
- **Ranking barely changes.** Age is already a linear feature of the production model (coefficient +0.13). Age bands
  move AUC by +0.002 and p@500 not at all. For hidden gems, a linear age term lifts p@500 from 0.106 to 0.118 under
  LOCO, but only from 0.050 to 0.052 under the chronological test.

**Verdict: describe.** The age effect is real and stable, but it calibrates the list rather than reordering it. Show
the paper's age on gems. On a gem uncited for 3 or more years, say that such papers are picked up about half as often
as their score suggests.

### Limitation → innovation: fresh limitations look worse, but only because old ones had time to close

- Limitations 0–1 year old at the cutoff convert at O/E 0.47 [0.24, 0.76] (strict hit).
- That is mostly closure: 5% of fresh items are already closed at T, against 11–16% of items 2–5 years old, and
  closed items are re-cited.
- Among items not yet closed, there is no gradient. The strict hit rate is 4.8%, 7.6%, 2.7% and 10% by age band
  (n = 208, 383, 110 and 20).
- Time features lower AUC: from 0.697 to 0.672 (LOCO), and from 0.688 to 0.609 (chronological).

**Verdict: not adopted.** Use the status (experiment 30) instead of age.

### Method transfer: too few hits to see an age effect; the 3-year horizon is too short

- **31 hits in 581 cells** is too few for age bands to be informative. The intervals span 1 everywhere except one
  band: exposure 3–5 years, O/E 1.61 [1.01, 2.41].
- **Rising methods carry 28 of the 31 hits**, but the item CF score already sees this (O/E ≈ 1.0).
- **Time features hurt:** AUC falls from 0.736 to 0.656 (LOCO) and from 0.812 to 0.759 (chronological). The plain item
  CF score is best: 0.762 under LOCO, 0.922 under the chronological test (on 2 hits per cutoff).
- **Horizon:** cutoffs 2019 and 2020 have complete 5-year outcomes, and over 5 years about twice as many cells fill
  as over 3:

  | Exposure | Cells | Within 1 y | Within 3 y | Within 5 y | Share of 5-y hits after year 3 |
  | --- | --- | --- | --- | --- | --- |
  | 0–2 y (a neighbour just started) | 69 | 0% | 7.2% | 20.3% | 64% |
  | 9+ y | 128 | 2.3% | 10.9% | 21.1% | 48% |

**Verdict:**
- **Describe:** method transfers take about five years to fill, longest when the exposure is fresh.
- **Score at 5 years:** future backtests of this list should use a 5-year horizon (cutoffs 2019–2020) next to the
  3-year one.
- **No time re-ranking.**

## Experiment 30: is it still open?

### How stale are the lists?

| List | Scope | Closed | Being taken | Open |
| --- | --- | --- | --- | --- |
| Limitation → innovation | all 200 per cutoff | 10% | 54% | 36% |
| Limitation → innovation | top 20 | 5% | 62% | 33% |
| Paper bridges | all candidates | — | 14% | 86% |
| Paper bridges | top 10 per community | — | 44% | 56% |
| Hidden gems | top 10 per community | — | **78%** | 22% |

*Closed* cannot occur for paper bridges: they exclude works the community already cites.

### What happens next, by status

| List | Status | Citation hit | O/E [90%] | Content uptake |
| --- | --- | --- | --- | --- |
| Limitation → innovation | closed | 43% (34 of 79) | 4.16 [2.72, 5.48] | 65% |
| | being taken | 9.0% | 0.88 [0.60, 1.13] | 48% |
| | open | **1.7%** | **0.20 [0.08, 0.34]** | 25% |
| Paper bridges | being taken | 4.7% | 1.32 [1.11, 1.51] | 22% |
| | open | 1.0% | 0.86 [0.74, 0.96] | 5.6% |
| Hidden gems | being taken | 3.5% | 1.49 [1.23, 1.80] | 22% |
| | open | 0.5% | 0.77 [0.64, 0.89] | 5.2% |

1. **Limitation → innovation: closed items carry the backtest.**
   - Closed items are 10% of the list but supply 44% of its strict hits (34 of 78): the community cites again what it
     already cited.
   - With them removed (the next items move up), top-20 precision falls from 0.150 to 0.112 (strict) and from 0.262
     to 0.225 (recipe).
   - Genuinely open items are almost never cited within three years: O/E 0.20. A quarter of them still see content
     uptake.
   - A model with the status features reaches AUC 0.815 (LOCO) and 0.833 (chronological). That gain is the closed
     items being re-cited, not foresight.
2. **Paper bridges and hidden gems: "being taken" is a real, new signal.**
   - A community whose recent papers already say what a paper says, without citing it, cites it four to seven times
     as often later.
   - Adding the status (being taken, closeness) to the gem model raises p@500:
     - from 0.102 (production) or 0.118 (refit without status) to **0.138** under LOCO;
     - from 0.046 or 0.052 to **0.072** under the chronological test.
   - AUC rises by 0.014 (LOCO) and 0.010 (chronological).
   - For all paper bridges the gain is small: p@500 from 0.202 to 0.208 (LOCO), unchanged under the chronological
     test.
   - So the gems list mostly shows ideas a community is already converging on: an *act soon* list.
3. **Method transfer: proximity adds nothing.**
   - Cells in the nearest tertile fill more often (8.3% vs 2.6%), but item CF already knows this (O/E 1.10 vs 0.87).
   - The status model lowers AUC from 0.736 to 0.705 (LOCO) and leaves it at 0.812 (chronological).

**Verdicts:**
- Limitation → innovation: **adopt (pending)**. Drop closed items, or move them to *already connected*. Show *being
  taken* or *open*. Quote the open-only precision (0.11 strict) as the list's honest number.
- Hidden gems: **adopt (pending)**. Refit the gem model with the status features (it wins both protocols) and label
  *being taken: the community is moving towards it without citing it*.
- Method-transfer status: **not adopted**.

## Caveats

- **The threshold is loose for limitation → innovation.** The *being taken* threshold (0.894, experiment 9) labels
  54% of that list as being taken. Its limitations and innovations are close by construction, so a community's
  recent work often reaches the innovation. The label discriminates better on paper bridges.
- **Hits are citations inside the corpus**, and the corpus is sampled through the seed's citation graph (see
  findings, point 7).
- **The chronological test has few cutoffs:** one for limitation → innovation and paper bridges (2023), two for
  method transfer (2022, 2023). LOCO can learn from the test period's future.
- **Untried applications (exp19) are left out:** 24 filled cells are too few to split by age or status.

## In the product (2026-10-21)

1. **Limitation → innovation** (`signals.bridges`):
   - Innovations the limitation's community already cites are excluded. On the pilot, 18 of the previous 200 items
     were closed.
   - Each item carries `status` and `closeness`.
   - Pilot after the change: 79 items *being taken*, 121 *open*.
2. **Hidden gems** (`signals.paper_bridges`, `models/hidden_gems.json`, fitted by `scripts/phase8b/fit_models.py`
   from `exp30_paper_bridges.parquet`):
   - They are ranked by the gem model with the status. Pilot coefficients: closeness +0.61, age −0.57; being taken
     ≈ 0 once closeness is in.
   - Only 17 of the previous 125 gems remain. The new list is younger (median age 2 years) and 80% *being taken*.
   - *Papers to bring in* keeps its model and ranking (250 of 250 unchanged); its rows carry the status too.
3. **Web app** (`Timing.tsx`: `StatusPill`, `FadePill`):
   - Columns:
     - *Status* on Start here › Opportunities (hidden gems, limitation → innovation);
     - *Status* on Map & grid › Bridges (limitation → innovation, both paper lists);
     - *Status* on My paper › Papers to bring in.
   - An amber *uncited N y* or *1 cite in N y* badge marks papers cited at most once and 3 or more years old.
   - The timing help text separates these backtested markers from the descriptive timing.
4. **Signals run:** re-run on the pilot through the queue (`signals`, then `portfolio`; judge $0.067).

**Still to do:**
- Method transfer: show when the exposure started, and say such cells typically fill over about five years.
- Score method transfer at 5 years in future backtests.
- The frozen 2026-10-21 snapshot keeps the old lists, as frozen.
