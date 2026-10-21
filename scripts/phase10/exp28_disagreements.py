"""Experiment 28: open disagreements. Where do papers on the same question report conflicting results, and do
disagreements attract follow-up work?

Pairs: two papers whose problem statements are close (cosine >= 0.93, the question groups of experiment 16; not
>= 0.99, which is the same work), from teams with no author in common, both with a result statement.

Stage `classify` (LLM, about $0.10, cached): the run's extraction model reads both papers' problem, method and result
statements and labels the pair *conflict* (incompatible claims about the same question: opposite rankings of methods,
an impossibility against an achievement, opposite effects), *agree* (findings point the same way) or *different*
(results about different aspects, no comparison possible), with the point of conflict and the quoted words.
Written to data/projects/<run>/backtest/exp28_pairs.parquet.

Stage `analyse`: a pair becomes visible in T = the later paper's year (2016-2022, outcomes through 2025). Follow-up
work = papers of T+1..T+3, by neither team, whose problem statement is close (>= 0.93) to the pair's mean question.
Conflict vs agree, matched on T and on how many papers asked the question in T-2..T; 90% intervals resample pairs.
Usage: uv run --no-sync python exp28_disagreements.py [classify] [analyse] [--run-id R] [--cap 0.5]
"""
import argparse
import asyncio
import json
import sys

import numpy as np
import pandas as pd

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("stages", nargs="*", default=["classify", "analyse"])
ap.add_argument("--run-id", default="dro-pilot")
ap.add_argument("--cap", type=float, default=0.5)
a = ap.parse_args()
RUN = a.run_id
OUT = C.project(RUN, "backtest") / "exp28_pairs.parquet"
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
title = papers.set_index("paper_id").title
fac = con.execute("""SELECT paper_id, facet, text FROM facets WHERE facet IN ('problem', 'method', 'result')""").df() \
    .pivot_table(index="paper_id", columns="facet", values="text", aggfunc="first")
has_res = set(fac.index[fac.result.fillna("").str.len() > 20])
PP, PV = C.vectors("problem", ids)
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["relation", "point", "quote_a", "quote_b"],
          "properties": {"relation": {"type": "string", "enum": ["conflict", "agree", "different"]},
                         "point": {"type": ["string", "null"], "description": "one sentence: what they disagree or agree on"},
                         "quote_a": {"type": ["string", "null"]}, "quote_b": {"type": ["string", "null"]}}}
PROMPT = """Two papers address the same research question. Compare what their RESULTS claim.
- "conflict": incompatible claims about the same question or measure (method X beats Y in one and Y beats X in the
  other; one shows something is impossible or ineffective, the other achieves it or finds it effective; opposite
  effects or conclusions).
- "agree": their findings point the same way or support each other.
- "different": the results concern different aspects, settings or measures, so they cannot be compared.
Be strict: different settings or merely different methods are "different", not "conflict". Quote the result words.

Paper A ({ya}). Problem: {pa}
Method: {ma}
Result: {ra}

Paper B ({yb}). Problem: {pb}
Method: {mb}
Result: {rb}"""


def candidate_pairs():
    pid = [x for x in PP if x in has_res]
    V = PV[[PP[x] for x in pid]]
    S = V @ V.T
    np.fill_diagonal(S, -1)
    out = []
    for i in range(len(pid)):
        for j in np.where((S[i] >= 0.93) & (S[i] < 0.99))[0]:
            if j > i and not (team.get(pid[i], frozenset()) & team.get(pid[j], frozenset())):
                a_, b_ = sorted((pid[i], pid[j]), key=lambda x: year.get(x, 0))
                out.append({"a": a_, "b": b_, "sim": float(S[i, j])})
    return pd.DataFrame(out)


async def classify():
    sys.path.insert(0, str(C.ROOT / "src"))
    from landscape.llm import LLMBudgetExceeded, OpenRouter

    P = candidate_pairs()
    have = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["a", "b"])
    done = set(zip(have.a, have.b))
    P = P[[(x, y) not in done for x, y in zip(P.a, P.b)]]
    ep = json.loads(con.execute("SELECT params FROM runs WHERE run_id = ?", [RUN]).fetchone()[0])["extraction"]
    client = OpenRouter(ep["model"], ep["quantizations"], 32, a.cap)
    rows = []

    async def one(r):
        g = lambda x, k: str(fac[k].get(x) or "")  # noqa: E731
        prompt = PROMPT.format(ya=year.get(r.a), pa=g(r.a, "problem"), ma=g(r.a, "method"), ra=g(r.a, "result"),
                               yb=year.get(r.b), pb=g(r.b, "problem"), mb=g(r.b, "method"), rb=g(r.b, "result"))
        try:
            obj, _, err = await client.chat_json(prompt, SCHEMA, "pair", max_tokens=300)
        except LLMBudgetExceeded:
            raise
        except Exception as ex:
            obj, err = None, repr(ex)[:200]
        rows.append({"a": r.a, "b": r.b, "sim": r.sim, **(obj or {"relation": "error", "point": None, "quote_a": None,
                                                                   "quote_b": None}), "error": err})

    try:
        await asyncio.gather(*(one(r) for r in P.itertuples()))
    except LLMBudgetExceeded as ex:
        print("stopped:", ex)
    finally:
        await client.close()
    pd.concat([have, pd.DataFrame(rows)]).to_parquet(OUT, index=False)
    print(f"classified {len(rows)}, spend ${client.spent:.4f}")
    print(f"@@spend llm_experiments={client.spent:.6f}")  # scripts/run_experiments.py adds it to the run's spend


def analyse():
    D = pd.read_parquet(OUT)
    D = D[D.relation.isin(["conflict", "agree", "different"])].copy()
    print(D.relation.value_counts().to_dict())
    D["ya"], D["yb"] = D.a.map(year), D.b.map(year)
    D["T"] = D.yb
    D = D[D["T"].between(2016, 2022)].reset_index(drop=True)
    yrs = np.array([year[x] for x in PP])
    allid = list(PP)
    rows = []
    for r in D.itertuples():
        q = C.unit(PV[PP[r.a]] + PV[PP[r.b]])
        s = PV @ q
        tm = team.get(r.a, frozenset()) | team.get(r.b, frozenset())
        near = np.where(s >= 0.93)[0]
        after = [allid[k] for k in near if r.T < yrs[k] <= r.T + 3 and not (team.get(allid[k], frozenset()) & tm)]
        before = [allid[k] for k in near if r.T - 2 <= yrs[k] <= r.T and allid[k] not in (r.a, r.b)]
        rows.append({"followups": len(after), "density_before": len(before)})
    D = pd.concat([D, pd.DataFrame(rows)], axis=1)
    D["any_followup"] = D.followups > 0
    D["dens_bin"] = pd.cut(D.density_before, [-1, 0, 2, 5, 1000], labels=["0", "1-2", "3-5", "6+"])
    sub = D[D.relation.isin(["conflict", "agree"])]

    def matched(d, y):
        diffs, w = [], []
        for _, g in d.groupby(["T", "dens_bin"], observed=True):
            c_, a_ = g[g.relation == "conflict"], g[g.relation == "agree"]
            if len(c_) and len(a_):
                diffs.append(c_[y].mean() - a_[y].mean())
                w.append(len(c_))
        return np.average(diffs, weights=w) if w else np.nan

    rng = np.random.default_rng(0)
    res = []
    for y in ("any_followup", "followups"):
        est = matched(sub, y)
        bs = [matched(sub.iloc[rng.integers(0, len(sub), len(sub))], y) for _ in range(500)]
        res.append({"outcome": y, "conflict pairs": int((sub.relation == "conflict").sum()),
                    "agree pairs": int((sub.relation == "agree").sum()),
                    "mean, conflict": float(sub[sub.relation == "conflict"][y].mean()),
                    "mean, agree": float(sub[sub.relation == "agree"][y].mean()),
                    "matched difference (same year, same prior activity)": float(est),
                    "90% CI": f"[{np.nanpercentile(bs, 5):+.3f}, {np.nanpercentile(bs, 95):+.3f}]"})
    R = pd.DataFrame(res)
    counts = D.relation.value_counts().rename("pairs").reset_index()
    pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False))
    C.save(RUN, "exp28_disagreements", "counts", counts, title="Disagreements: how often same-question papers conflict",
           how_to_read="Pairs of papers asking the same question (problem statements >= 0.93 apart, unrelated teams), "
           "labelled by the LLM from their result statements: conflict, agree, or different aspects. See the hand check "
           "in the doc before trusting the labels.")
    C.save(RUN, "exp28_disagreements", "followups", R, title="Disagreements: do conflicts attract follow-up work?",
           how_to_read="Follow-up work = later papers (T+1..T+3, by neither team) asking the same question, where T is the "
           "later paper's year. Conflicting against agreeing pairs, matched on the year and on how many papers asked the "
           "question in T-2..T; intervals resample pairs.")
    D.to_parquet(C.project(RUN, "backtest") / "exp28_pairs_analysed.parquet", index=False)


for s in a.stages:
    asyncio.run(classify()) if s == "classify" else analyse()
