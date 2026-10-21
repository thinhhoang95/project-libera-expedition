# Experiments: orientation for choosing a direction and what to read (Phase 10)

Status: experiments 12-14 done (no product change yet) · Scripts: `scripts/phase10/` · Results: Experiments view (`data/projects/<run>/backtest/experiments/`) · Pilot: `dro-pilot`

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

## What could go into the product (not built yet)

- **A "read this first" path per community:** the most cited classics interleaved with the method-lineage sources,
  backed by experiment 13. It replaces surveys as the default reading suggestion.
- **"How people got in" on My paper and Start here:**
  - entry routes that worked: anchored in the community's literature, with an insider co-author;
  - routes that cost: unusual methods, new problems.

  It would be phrased as a description of what happened here, not advice that guarantees anything (experiment 12).
- **A "lasting" marker on reading lists:** the idea is already close to work in several communities (breadth). It
  would be shown with genericness controlled, because breadth alone partly measures how generic the wording is
  (experiment 14).

All three need the web app's data layout to settle; the scripts already read it.
