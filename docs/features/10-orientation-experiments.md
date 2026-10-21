# Experiments: orientation for choosing a direction and what to read (Phase 10)

Status: experiments 12-16 done (no product change yet) · Scripts: `scripts/phase10/` · Results: Experiments view (`data/projects/<run>/backtest/experiments/`) · Pilot: `dro-pilot`

These experiments ask what helps someone **orient**: how to enter a community, what to read first, and which ideas
last. None is an outcome-prediction model. Each tests a simple, readable rule against simple alternatives, on
outcomes that matter in the long run:
- did a newcomer stay;
- did others take up their work;
- what did newcomers actually build on;
- is an idea still being used years later, cited or not.

## Run

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/libera-expedition
cd scripts/phase10
uv run --no-sync python exp12_entry.py    [--run-id dro-pilot]   # ~20 s
uv run --no-sync python exp13_reading.py  [--run-id dro-pilot]   # ~35 s
uv run --no-sync python exp14_durable.py  [--run-id dro-pilot]   # ~10 s
uv run --no-sync python exp15_lifecycle.py [--run-id dro-pilot]   # ~8 s
uv run --no-sync python exp16_open_questions.py [--run-id dro-pilot]   # ~20 s
```

- **Self-contained:** the scripts use `scripts/phase10/common.py`, not the `landscape` package, and read the
  migrated layout (`data/shared/`, `data/projects/<run>/`).
- **Database lock:** `LANDSCAPE_DB=<copy of data/shared/landscape.duckdb>` lets them run while an app holds
  the lock.
- **Communities:** the problem communities (α 0.5, medium) of each cutoff run. Every paper is assigned to the
  nearest stable or hub centroid, as in the backtest.
- **Reuse threshold:** "close" means cosine ≥ 0.894 between an innovation and a method or innovation statement
  (256-d), the 99th percentile of random pairs (experiment 9).

## 12. How people successfully entered a community

**Entries.** For each cutoff year T (2019-2022), an entry is an author's first paper in a problem community,
published in T (6,585 entries). The strategy is read from the paper's facets against the community's earlier
papers:

| Strategy | Definition | Entries |
| --- | --- | --- |
| core incremental (reference) | none of the below | 3,477 |
| method import | its method is closest to another community's methods *and* unusual for this one (below the 25th percentile of members) | 1,354 |
| new problem for the community | its problem is in the top quarter of distance from the community's core | 1,139 |
| bridge | its references span several communities' literatures (top quarter of entropy, ≥ 3 references in the map) | 615 |

**Outcomes (next 3 years):**
- **stayed:** another paper in the community;
- **taken up by others:** non-self citations in the top quarter of its year;
- **the community moved towards it:** experiment 9's measure.

**Comparison.** Adjusted differences against core incremental, holding fixed: an insider co-author, team size,
community size, newcomer to the whole map, route, the number of references in the map, and year. The 90%
intervals resample communities.

| Strategy or trait | Stayed | Taken up by others | Community moved towards it |
| --- | --- | --- | --- |
| method import | −2.4 pts [−5.1, +0.3] | **−5.4** [−8.7, −1.1] | **−7.6** [−12.2, −3.4] |
| new problem for the community | **−3.4** [−6.2, −0.7] | **−8.2** [−12.0, −4.3] | −4.0 [−8.6, +0.8] |
| bridge | −2.6 [−5.0, +0.1] | **+7.5** [+2.8, +12.9] | +5.9 [−1.5, +12.8] |
| insider co-author (≥ 2 earlier papers in the community) | **+3.5** [+0.3, +6.2] | **+5.8** [+1.8, +10.3] | −4.4 [−9.2, −0.4] |

Base rates for core incremental: 12% stayed, 22% taken up, 32% moved towards.

**What it says:**
1. **Entries anchored in the community's own literature do better.**
   - Bridges that draw on several communities' literature are taken up more.
   - Without the reference-count control the bridge gain was +24 points; most of it was simply citing the field's
     literature.
2. **Bringing an unusual method, or opening a new problem, does worse in the first three years.** Neither is taken
   up as much, and newcomers who open a new problem leave more often. This agrees with the short-run novelty penalty
   (experiments 5 and 14). It says *nothing* about long-run value.
3. **Joining with an insider co-author helps:**
   - newcomers stay more and are taken up more;
   - the community's later work moves towards their paper slightly *less* than for others (−4.4 points).

   It is the insider effect seen in experiments 5 and 11.

**Limits:**
- "Stayed" only sees later papers that are in the map, which the seed's citation graph reached.
- The corpus is sampled through citations.
- Three-year horizon.

## 13. "Read this first": what newcomers actually needed

**Setup.**
- **Needs:** for each community and cutoff (76 community-cutoff pairs, 2019-2022), the authors who joined in
  T+1…T+3, and what they needed from papers up to T:
  - **cited:** the earlier papers they cite;
  - **reused:** the earlier papers that are the earliest close source of their methods and innovations, cited or
    not.
- **Lists:** candidate reading lists are built from data up to T only.
- **Score:** coverage = share of the needed set the list contains, weighted by how many newcomers needed each paper.

| List (top 10) | Coverage of what they cited | Hit (share of newcomers) | Coverage of what they built on | Hit |
| --- | --- | --- | --- | --- |
| most cited in the community | **0.229** | 0.67 | 0.065 | 0.11 |
| lineage + popular (alternating) | 0.174 | 0.61 | 0.185 | 0.29 |
| most cited overall | 0.138 | 0.52 | 0.042 | 0.07 |
| recent hits | 0.059 | 0.27 | 0.037 | 0.06 |
| **method lineage** | 0.051 | 0.29 | **0.217** | 0.34 |
| central to the community | 0.045 | 0.21 | 0.049 | 0.08 |
| surveys | 0.032 | 0.16 | 0.017 | 0.03 |

All differences against "most cited in the community" have 90% intervals clear of zero. The top-20 lists show the
same pattern.

**What it says:**
1. **Newcomers *cite* the community's classics but *build on* the original sources of its current methods.**
   - The method-lineage list covers 3.3× more of what they build on.
   - Caveat: the list and this outcome use the same kind of measurement (earliest close source), so part of the
     gap is built in. What the result does show is that the sources a community draws on persist from one cohort
     to the next.
2. **Alternating the two lists covers most of both:** about three quarters of the citation coverage and 85% of the
   reuse coverage. That is the basis for a "read this first" path: the classics everyone cites, interleaved with
   the sources the current methods come from.
3. **Surveys are neither cited nor built on by newcomers here.** Central papers and recent hits do not help either.

## 14. Durable ideas vs flashes

**Setup.**
- **Ideas:** the innovation statements of 2018-2019 papers (1,307).
- **Use:** papers by other authors whose method or innovation is close to the idea, cited or not, per paper
  published, in the two years after publication and in 2023-2025.
- **Outcomes:** among ideas used early (824), **durable** = used at least as much in 2023-2025 (32%). **Late
  bloomers** = not used early, used later (11 papers, too few to read).

**The artefact.** **Genericness** (closeness to everything) is the strongest trait (AUC 0.69). Generic ideas look
"used" everywhere and keep looking so; it is a measurement artefact, not durability. So the traits are compared
among equally generic ideas (genericness quintiles):

| Trait at publication | AUC (durable vs faded), within genericness quintiles | 90% CI |
| --- | --- | --- |
| **breadth**: how many communities' earlier work is already close to the idea | **0.577** | [0.538, 0.614] |
| generality of the problem | 0.546 | [0.510, 0.583] |
| **novelty** | **0.393** | [0.359, 0.431] |
| early citations by others | 0.508 | [0.469, 0.546] |
| contribution type (method / theory / application), stated limitation, quantitative result | 0.49-0.52 | around 0.5 |

**What it says:**
1. **Ideas that already speak to several communities when published keep being used.** It is the clearest durable
   trait.
2. **Novel ideas fade faster in use, not just in citations.**
3. **Early attention does not predict durability.** Being cited early and still being used later are different
   things.

**Limits:**
- Only two publication years; the corpus is thin before 2018.
- Use is measured by closeness of wording within the map.
- Five to seven years is still short for the slow-burning ideas the novelty finding may be missing.

## 15. Community lifecycle: what kind of contribution is wanted now

**Question.** Fields are often said to move from theory to methods to applications. Where is each community in
that cycle, and is the kind of work it lacks taken up more?

**Kinds of contribution**, from each paper's extracted facets:

| Kind | Definition | Papers |
| --- | --- | --- |
| theory | contribution type *theory* | 795 |
| applied method | method (or application, dataset, benchmark) naming an application domain or data | 2,579 |
| generic method | method naming no domain or data | 615 |
| survey | contribution type *survey* | 469 |

About 20% of papers also report a quantitative result (*empirical*). A manual read of 36 abstracts (stratified by
label) agreed with the label in about 89% of cases. "Application" alone is rare here, because applied engineering
papers are labelled *method*; hence the applied / generic split by the named domain.

**Setup.** Problem communities (α 0.5, medium) of the cutoff runs, 2019–2022. Mixes over T−2…T ("before") and
T+1…T+3 ("after"), for communities with at least 10 papers in each window (75 community-cutoffs).

### A. Does a community's stage predict its next mix?

Each kind's share after was predicted three ways:
- **persistence:** no change;
- **field drift:** the community moves like the whole field;
- **lifecycle:** the change depends on the community's own mix (ridge, leave-one-cutoff-out).

| Kind | MAE persistence | MAE field drift | MAE lifecycle | Lifecycle − field drift (90% CI) |
| --- | --- | --- | --- | --- |
| theory | 0.043 | 0.046 | 0.048 | +0.002 [+0.000, +0.004] (worse) |
| applied method | 0.059 | 0.056 | 0.056 | −0.000 [−0.004, +0.003] |
| generic method | 0.060 | 0.056 | 0.051 | **−0.005 [−0.008, −0.002]** |
| survey | 0.039 | 0.038 | 0.037 | −0.002 [−0.004, +0.000] |
| all kinds | 0.050 | 0.049 | 0.048 | −0.001 [−0.003, +0.000] |

- **Weak overall:** a community's own stage adds little to the field-wide drift.
- **One transition holds.** Communities heavy in **generic methods** gain **applied methods** afterwards (coefficient
  +0.09) and lose generic methods (−0.19). Theory-heavy communities gain generic methods (+0.09).
- **The sequence:** theory → generic method → applied method, visible but small.

### B. Do "theory wanted" gaps get theory?

Method communities whose use is growing and empirical but which have few theory papers (top quarter of that gap;
21 of 77 method community-cutoffs) were compared with method communities of the same theory share and size. Their
theory share changed by −0.006 [−0.020, +0.008] relative to those. **Not supported:** theory does not flow to where
it looks missing, at least not within three years.

### C. Is the scarce kind of contribution taken up more?

A paper's kind is **scarce** in its community when the community's share of that kind (T−2…T) is under half the
field's. Scarce papers are compared with papers of the same kind where it is common, on three outcomes:
- **taken up by others:** non-self citations in the top quarter of year-T papers;
- **peer-normalized uptake:** the same, relative to the paper's 15 nearest same-year papers;
- **the community moving towards it:** experiment 9's measure.

Intervals resample communities.

| Scarce kind | Papers | Taken up: scarce vs common | Peer-normalized | Community moved towards it |
| --- | --- | --- | --- | --- |
| **survey** | 19 of 249 | **47% vs 10%, +0.38 [+0.16, +0.54]** | **+0.31 [+0.07, +0.49]** | −0.06 [−0.24, +0.12] |
| theory | 18 of 385 | 6% vs 19%, −0.13 [−0.24, +0.03] | −0.15 [−0.29, +0.15] | −0.10 [−0.22, +0.03] |
| generic method | 33 of 314 | −0.03 [−0.17, +0.09] | −0.03 [−0.15, +0.07] | +0.09 [+0.005, +0.16] |
| applied method | 40 of 1,480 | −0.02 [−0.20, +0.12] | −0.08 [−0.20, +0.01] | −0.08 [−0.27, +0.03] |

**The survey gap is robust.** It holds at every cut of "scarce" and grows the scarcer surveys are:

| Scarce if the community's survey share is below | Scarce surveys | Taken up: difference (90% CI) |
| --- | --- | --- |
| 1.0× the field's | 79 | +0.11 [+0.01, +0.21] |
| 0.75× | 38 | +0.26 [+0.11, +0.40] |
| 0.5× | 19 | +0.38 [+0.16, +0.54] |

What it says:
1. **A survey written where a community lacks one is read much more**, by others and against its nearest peers. It
   is cited, not built on: the community's methods do not move towards it. That fits experiment 13, where newcomers
   did not *build on* surveys.
2. **Scarcity does not help other kinds.** A rare theory paper in an applied community fares no better, perhaps
   worse (few cases).
3. **Generic methods** in communities that have few of them draw a little more convergence from the community's
   later work.

**In the product:** a *survey gap* status per community (`survey_gaps.parquet`, below). It is the one
"what is missing here" signal with like-for-like, dose-response evidence. The lifecycle mix itself is descriptive,
with the one transition above, and is not shown.

## 16. Long-standing open questions

**Question.** Which questions keep being asked without an answer that ends them?

**Question groups.** For each cutoff T (2019–2022), using statements up to T only:
- problem statements (problem facet, 256-d) from 2010…T, one record per normalized title, grouped by average
  linkage at cosine 0.93 (just above the 99.9th percentile of random pairs, 0.923);
- each group needs at least 3 statements, at least 2 distinct years, and at least 2 papers with no author in common;
- the most generic fifth is dropped.

Groups read as one question asked by different people: unit commitment under wind uncertainty, the newsvendor with
partial demand information, composite convex minimization. 211 question groups are active (asked in T−2…T)
across the four cutoffs. **Still asked** means a statement in T+1…T+3 joins the group (mean cosine to its members
≥ 0.93).

### Whether a question was answered cannot be read from abstracts

The test looked at innovation statements from outside the group (by other teams) against the askers' own:
- for the median question, the closest outside innovation (cosine 0.926) is *closer* to the question than the
  askers' own innovations (0.906);
- 64% of questions have an outside "attempt" above the 99.99th percentile of random pairs.

Closeness of wording does not certify an answer, so open versus closed cannot be classified this way.

### What does predict a question being asked again

| Label | Labelled | Still asked if labelled | Otherwise | Matched difference (same size and recent activity) |
| --- | --- | --- | --- | --- |
| standing (first asked 4+ years before, by 3+ unrelated teams) | 34 of 211 | 50% | 55% | +0.04 [−0.16, +0.25] |
| asked by more unrelated teams than the median | 169 of 211 | 58% | 38% | +0.11 [−0.01, +0.27] |

| Model (leave-one-cutoff-out) | AUC | Gain over base (90% CI) |
| --- | --- | --- |
| age + size + recent recurrence | 0.644 | — |
| + number of unrelated teams asking | 0.655 | +0.011 [−0.006, +0.028] |
| + standing label | 0.599 | −0.045 [−0.105, +0.016] |

- **Not demonstrated.** Recent activity and size predict whether a question is asked again (AUC 0.64). Neither the
  "standing" label nor the breadth of askers adds significantly, though breadth leans positive.
- **The recurring-question list** (53 questions asked by 3+ unrelated teams, as of 2025) is written to
  `data/projects/<run>/backtest/exp16_recurring_questions.parquet` for inspection. It is **not used by the
  product**.

Scripts: `exp15_lifecycle.py` (~8 s), `exp16_open_questions.py` (~20 s).

## 17. Quiet foundations: papers people build on but rarely cite

**Question.** Experiment 9 found most real uptake is uncited. Which papers are the hidden sources of later work, and
do they stay that way?

**Setup.**
- **Candidates:** for each cutoff T (2019-2022), papers published up to T−2 (5,051 paper-cutoffs).
- **Quiet reuse:** papers by other authors, up to T, whose method or innovation is within 0.894 of the paper's
  innovation, and which do not cite it.
- **Attributed:** stricter. The paper is credited only when it is the *earliest* close source of a later statement,
  as in the reading paths' method lineage.
- **Outcomes in T+1…T+3:**
  - still reused by at least 3 papers (base 44%);
  - still the earliest source of at least 2 (base 14%).

**Comparison.** Rankings are compared among papers of similar genericness (generic ideas look used everywhere),
similar prior density (how busy the area already was: busy areas stay busy) and the same citation bucket.

| Ranking | Still reused by ≥ 3 | Still the earliest source of ≥ 2 |
| --- | --- | --- |
| quiet reuse | **0.91** [0.90, 0.92] | 0.75 [0.73, 0.78] |
| attributed, uncited | 0.64 [0.62, 0.67] | **0.83** [0.80, 0.85] |
| attributed (all) | 0.65 | **0.84** [0.82, 0.86] |
| citations by others | 0.52 [0.50, 0.55] | 0.51 [0.49, 0.55] |

**Quiet foundations** = top tenth of quiet reuse among equally generic papers, with at least 3 quiet reuses (508
paper-cutoffs):
- 89% are still built on in the next three years, against 41% of papers with the same genericness and citations;
- **40% are not among their community's 50 most cited papers.**

**What it says:**
1. **Being the uncited source of later work is a stable property, and citations do not show it.** Among papers with
   the same citation count, citations do not separate future sources from the rest; attributed reuse does (0.83).
2. **A reading list ranked by citations misses about two in five of the papers the field actually keeps building
   on.** That is a strong case for adding quiet foundations to *Read this first*.

**Limits:**
- "Reuse" is closeness of wording within the map; the genericness and density controls remove the obvious
  artefacts, but not every one.
- Three-year outcomes.

## 18. Structural twins: same mathematical structure, different application

**Question.** Do communities whose problems share the same textbook *structure* (problem glosses), but serve
different applications (domain facet), converge in content more than their topic similarity predicts?

**Setup.** Pairs of problem communities at each cutoff (2019-2022; 693 pair-cutoffs). Outcome: the share of each
community's papers in T+1…T+3 whose method or innovation is within 0.894 of the other's innovations, top quarter
= hit. Controls: topic similarity, application similarity, existing citations, prior convergence, size.

| Measure | Value | 90% CI |
| --- | --- | --- |
| partial ρ, structure with later convergence | +0.04 | [−0.02, +0.11] |
| partial ρ, topic | **+0.25** | [+0.18, +0.31] |
| partial ρ, application | **−0.09** | [−0.16, −0.03] |
| AUC gain of structure over everything else (leave one cutoff out) | +0.002 | [−0.002, +0.007] |

**What it says:**
- **Not supported at the community level.** Shared structure adds nothing beyond topic; DRO's communities share much
  of their mathematical vocabulary, so their glosses barely differ.
- **One real nuance:** for the same topic, communities with *different* applications converge slightly more. "Same
  problem, other application areas" is a pattern; a more abstract "same structure" is not, here.
- Only 21 twin pairs (top-quarter structure, below-median application): too few to read on their own.

## 19. Applications not yet tried

**Question.** Experiment 4's recipe on problem community × application domain (15 clusters of the domain
statements up to T). Which empty cells get at least 2 papers in the next three years?

| Ranking (no fitting on outcomes) | AUC | p@20 (base 5.6%) |
| --- | --- | --- |
| **semantic fit** (problem community centroid vs domain centroid) | **0.81** | **15%** |
| item CF + semantic fit (rank average) | 0.77 | 10% |
| item CF (experiment 4's recipe) | 0.66 | 10% |
| neighbour share | 0.55 | 11% |
| popularity | 0.53 | 7.5% |

**What it says:**
- Plain semantic fit between a problem and an application finds the cells that fill, better than co-usage (unlike
  methods, experiment 4).
- **Small sample:** 24 filled cells over four cutoffs (11, 8, 4, 1). Read as promising, not settled.

## 20. Tools to learn vs methods in flux

**Traits** of each method community at T (2019-2022; 77 community-cutoffs):
- **breadth:** problem communities using it;
- **dispersion:** how varied its recent variants are;
- **inward:** the share of its papers innovating on the method itself.

The **stable score** is breadth minus dispersion minus inward (standardized).

| Trait | Its share sustained in T+1…T+3 (ρ) | Spread to new problems (ρ) |
| --- | --- | --- |
| breadth | **+0.19** [+0.02, +0.37] | −0.18 [−0.36, +0.01] |
| dispersion | −0.14 [−0.33, +0.05] | −0.04 |
| inward | −0.08 | −0.19 [−0.37, 0.00] |
| momentum | **+0.32** [+0.12, +0.49] | +0.13 |
| **stable score** | **+0.22** [+0.04, +0.39] | +0.04 |

**Entries** (experiment 12's 6,585 entries, with its controls), comparing a stable tool (75th percentile of the
score) with a less stable one (25th):
- newcomers stayed more: **+3.0 points** [+1.9, +4.2];
- they were taken up more: **+6.2 points** [+3.1, +8.8].

**What it says:** broadly used, settled tools keep their place, and newcomers who entered with them fared better.
This is a reasonable "learn these" list.

## 21. Gateway reads: papers whose readers join their community

**Setup.**
- **Outsider citer:** an author citing a paper from a community they have not published in. They *entered* if they
  publish there later.
- **Gateway score:** a paper's outsider citers' entry rate in 2018-2020 (known in 2021).
- **Test:** new outsider citers in 2021-2022, with entries through 2025; 16,999 citers, 10.5% entered.

| Score | Paper level: ρ with new citers' entry rate (629 papers) | Reader level: AUC for "this reader entered" |
| --- | --- | --- |
| gateway | +0.12 [+0.05, +0.18] | 0.63 |
| popularity | **+0.20** [+0.14, +0.26] | 0.42 |
| community size | +0.17 | 0.57 |
| gateway, within the same community and popularity tercile | — | **0.535** [0.51, 0.55] |

**What it says:** weak. Readers of a high-gateway paper join more often (0.63), but most of that comes from *which*
community and how popular the paper is. Like for like, the gateway effect is small (0.53). Readers of very popular
papers join *less*, because everyone reads them. Not product-grade on this corpus.

## 22. How the question is being asked now

**Measure.** For an entry (experiment 12's entries, 2020-2022), **currency** = how much closer its problem statement is to
the community's formulation of the last two years than to that of three to four years earlier.

**Result.** No effect on any outcome:
- stayed: −0.4 points [−2.0, +1.1];
- taken up: +0.8 [−0.9, +2.6];
- community moved towards it: +1.7 [−1.1, +3.7].

Asking the question "the current way" does not help on this corpus. (A first run printed zeros because the measure's
tiny scale was shrunk away by the regularized model; it is standardized now.)

## 23. Crowding vs demand

For each entry, counting papers by other authors:
- **same question:** papers from T−2…T posing the same question;
- **near duplicate:** of those, the ones also using the same method (42% of entries have one);
- **demand:** stated limitations the entry's innovation answers (55% have one).

The thresholds are the 99th percentiles of random pairs.

| Measure (adjusted, experiment 12's controls) | Stayed | Taken up by others | Community moved towards it |
| --- | --- | --- | --- |
| same question, 75th vs 25th percentile | −0.3 [−1.1, +0.3] | **+2.7** [+1.9, +3.6] | +1.7 [0.0, +3.2] |
| near duplicate present | +1.7 [−0.7, +4.1] | **+9.5** [+6.3, +12.5] | **+4.8** [+0.6, +8.5] |
| demand present | +0.3 [−2.1, +2.8] | +1.4 [−1.0, +4.3] | **+5.4** [+1.3, +9.0] |

**What it says:**
- **Crowding is not a penalty here; it marks attention.** Entries into areas where others are posing the same
  question, even with the same method, are taken up more.
- **Answering a stated limitation pays off in content, not citations:** the community's later work moves towards the
  entry.
- The "too crowded" worry is not supported over three years. A long-run scooping effect cannot be measured here.

## In the product portfolio

`src/landscape/orientation.py` turns the results into artefacts, written next to the other portfolio files
in `data/projects/<run>/portfolio/`:

```bash
uv run --no-sync python src/landscape/orientation.py --run-id dro-pilot   # ~3 s; LANDSCAPE_DB may point at a copy
```

| File | Feature | Backed by | Pilot |
| --- | --- | --- | --- |
| `reading_paths.parquet` | **Read this first**, per community: up to 20 papers alternating *classic* (most cited by the community) and *source of current methods* (the earliest close source of the methods and innovations of its last 3 years), with how many community papers cite each and how many current methods it is the source of | exp13 | 24 communities |
| `entry_routes.parquet`, `entry_effects.json` | **How people got in**, per community and strategy: entries of the latest year with three complete years of outcomes (2022 → 2023-25), their stayed and taken-up rates, insider co-author share, up to 3 example entries taken up by others, and the strategy mix of the latest year; plus the pilot's like-for-like effects | exp12 | 1,209 entries with outcomes |
| `lasting.parquet` | **Lasting marker**, per paper from 2018 on: breadth at publication (how many communities' earlier work is already close to its idea), ranked among equally generic ideas; *lasting* = top third and breadth ≥ 2. Older papers are *not assessed* (the map is too thin before 2018) | exp14 | 1,383 of 4,311 assessed |
| `survey_gaps.parquet` | **Where a survey is missing**, per community: its share of surveys over the last three complete years against the field's; *survey gap* (< 0.5×), *thin* (< 0.75×), *covered*; communities with fewer than 10 recent papers are not judged; with its latest survey and the survey it cites most | exp15 | 4 gaps, 3 thin of 25 communities (2023–2025) |

**How to present them:**
- **Read this first:** say which role each paper plays. The classics are what newcomers end up citing; the sources
  of current methods are what they end up building on. Surveys are not the default, because newcomers here neither
  cited nor built on them.
- **How people got in:** describe what happened on this map ("entries anchored in the community's literature, and
  with an insider co-author, were taken up more; unusual methods and new problems less, over three years"), with
  examples. It is not advice that guarantees an outcome.
- **Where a survey is missing:** a status with its evidence. A survey written where surveys are scarce has been
  taken up by others far more often (experiment 15), cited rather than built on. It is a gap to fill, not a guarantee.
- **Lasting:** a small marker with its evidence. Breadth at publication is the clearest trait of ideas still used
  5+ years later; novel ideas fade faster, and early citations say nothing.

The web app renders them (Start here and My paper); the Streamlit explorer is being replaced by it.
