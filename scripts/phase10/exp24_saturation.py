"""Experiment 24: is a direction saturating? Diminishing returns in the gains papers report.

Stage `extract` (LLM, about $0.10, cached): for each paper whose abstract reports a quantitative result, the abstract and
its result statement go to the run's extraction model, which returns the headline gain over the best baseline the
abstract names, as a percentage, and its kind (performance improvement, error reduction, cost reduction, speed-up,
other). Abstracts that state no comparable gain return none. Written to data/projects/<run>/backtest/exp24_gains.parquet.

Stage `analyse`: per community (problem communities of the cutoff-T run, alpha 0.5, medium) and cutoff T (2019-2022),
the median log-gain of its papers in T-3..T-1 against T-6..T-4 within the same kind of gain. A community is
*saturating* when that median gain shrank. Outcomes in T+1..T+3: the community's share of new papers (relative to
T-2..T) and its share of entries (authors' first papers there), against communities with the same size and gain level.
Usage: uv run --no-sync python exp24_saturation.py [extract] [analyse] [--run-id R] [--cap 0.3]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("stages", nargs="*", default=["extract", "analyse"])
ap.add_argument("--run-id", default="dro-pilot")
ap.add_argument("--cap", type=float, default=0.30)
a = ap.parse_args()
RUN = a.run_id
OUT = C.project(RUN, "backtest") / "exp24_gains.parquet"
con = C.connect()

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["kind", "gain_percent", "metric", "quote"],
          "properties": {
              "kind": {"type": "string", "enum": ["performance improvement", "error reduction", "cost reduction",
                                                  "speed-up", "other", "none"]},
              "gain_percent": {"type": ["number", "null"],
                               "description": "headline relative gain over the best named baseline, in percent "
                                              "(a 3x speed-up = 200; a 15% cost reduction = 15); null if none"},
              "metric": {"type": ["string", "null"], "description": "what is measured, in a few words"},
              "quote": {"type": ["string", "null"], "description": "the exact words of the abstract stating the gain"}}}
PROMPT = """From this abstract, extract the single headline quantitative gain the paper reports over the best baseline,
state of the art or previous method it names.
- gain_percent: the relative gain in percent. Convert multiples to percent (2x faster = 100). Use the largest headline
  gain if several are listed. If the abstract gives only absolute numbers without a comparison, or no comparison to a
  baseline, return kind "none" and null.
- Never compute a gain the abstract does not state or directly imply. Copy the words that state it into quote.

Title: {title}
Result statement: {result}
Abstract: {abstract}"""


async def extract():
    sys.path.insert(0, str(C.ROOT / "src"))
    from landscape.llm import LLMBudgetExceeded, OpenRouter

    todo = con.execute("""
        SELECT p.paper_id, p.title, p.abstract_raw AS abstract, r.text AS result FROM papers p
        JOIN provenance v ON v.paper_id = p.paper_id AND v.run_id = ?
        JOIN facets r ON r.paper_id = p.paper_id AND r.facet = 'result'
        JOIN facets q ON q.paper_id = p.paper_id AND q.facet = 'has_quantitative_result' AND q.text = 'true'
        WHERE p.abstract_raw IS NOT NULL""", [RUN]).df().drop_duplicates("paper_id")
    have = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["paper_id"])
    todo = todo[~todo.paper_id.isin(set(have.paper_id))]
    ep = json.loads(con.execute("SELECT params FROM runs WHERE run_id = ?", [RUN]).fetchone()[0])["extraction"]
    client = OpenRouter(ep["model"], ep["quantizations"], 32, a.cap)
    rows = []

    async def one(r):
        prompt = PROMPT.format(title=r.title or "", result=r.result or "", abstract=r.abstract)
        try:
            obj, cost, err = await client.chat_json(prompt, SCHEMA, "gain", max_tokens=300)
        except LLMBudgetExceeded:
            raise
        except Exception as ex:  # one bad record must not stop the batch
            obj, err = None, repr(ex)[:200]
        rows.append({"paper_id": r.paper_id, **(obj or {"kind": "error", "gain_percent": None, "metric": None,
                                                        "quote": None}), "error": err})

    try:
        await asyncio.gather(*(one(r) for r in todo.itertuples()))
    except LLMBudgetExceeded as ex:
        print("stopped:", ex)
    finally:
        await client.close()
    new = pd.DataFrame(rows)
    pd.concat([have, new]).to_parquet(OUT, index=False)
    print(f"extracted {len(new)} (total {len(have) + len(new)}), spend ${client.spent:.4f}")
    print(f"@@spend llm_experiments={client.spent:.6f}")  # scripts/run_experiments.py adds it to the run's spend


def analyse():
    papers, au, refs, year, team = C.base_tables(con, RUN)
    ids = papers.paper_id.tolist()
    G = pd.read_parquet(OUT)
    G = G[(G.kind != "none") & (G.kind != "error") & G.gain_percent.notna() & (G.gain_percent > 0)].copy()
    G["year"] = G.paper_id.map(year)
    G["lg"] = np.log1p(G.gain_percent / 100)
    print("usable gains:", len(G), G.kind.value_counts().to_dict())
    rows = []
    for T in range(2019, 2023):
        lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
        yr = lab.index.map(year)
        tot_b = ((yr >= T - 2) & (yr <= T)).sum()
        tot_a = ((yr > T) & (yr <= T + 3)).sum()
        a2 = au.assign(X=au.paper_id.map(lab)).dropna(subset=["X"]).sort_values("year")
        first = a2.groupby(["author_id", "X"]).head(1)
        ent_b = first[first.year.between(T - 2, T)].groupby("X").size()
        ent_a = first[first.year.between(T + 1, T + 3)].groupby("X").size()
        n_ent_b, n_ent_a = first.year.between(T - 2, T).sum(), first.year.between(T + 1, T + 3).sum()
        g = G.assign(X=G.paper_id.map(lab)).dropna(subset=["X"])
        for X in sorted(set(lab)):
            gx = g[g.X == X]
            recent, older = gx[gx.year.between(T - 3, T - 1)], gx[gx.year.between(T - 6, T - 4)]
            # compare within kinds: median of (recent - older) per kind present in both, weighted by count
            diffs, w = [], []
            for k in set(recent.kind) & set(older.kind):
                r, o = recent[recent.kind == k].lg, older[older.kind == k].lg
                if len(r) >= 2 and len(o) >= 2:
                    diffs.append(r.median() - o.median())
                    w.append(min(len(r), len(o)))
            if not w:
                continue
            nb = int(((yr >= T - 2) & (yr <= T) & (lab.values == X)).sum())
            na = int(((yr > T) & (yr <= T + 3) & (lab.values == X)).sum())
            rows.append({"T": T, "X": X, "gain_change": float(np.average(diffs, weights=w)), "n_gains": int(sum(w)),
                         "gain_level": float(recent.lg.median()), "size": nb,
                         "share_growth": np.log((na / tot_a + 1e-4) / (nb / tot_b + 1e-4)),
                         "entry_growth": np.log((ent_a.get(X, 0) / max(n_ent_a, 1) + 1e-4) / (ent_b.get(X, 0) / max(n_ent_b, 1) + 1e-4))})
    D = pd.DataFrame(rows)
    print("community-cutoffs with comparable gains:", len(D))
    if len(D) < 10:
        print("too few to test")
        R = pd.DataFrame([{"community-cutoffs": len(D), "note": "too few communities with comparable gains in both windows"}])
    else:
        D["saturating"] = D.gain_change < 0
        Z = np.column_stack([np.ones(len(D)), np.log(D["size"] + 1), D.gain_level])
        resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
        from scipy.stats import spearmanr
        rng = np.random.default_rng(0)
        out = []
        for y in ("share_growth", "entry_growth"):
            rho = spearmanr(resid(D.gain_change.to_numpy()), resid(D[y].to_numpy())).statistic
            bs = [spearmanr(resid(D.gain_change.to_numpy())[i], resid(D[y].to_numpy())[i]).statistic
                  for i in (rng.integers(0, len(D), len(D)) for _ in range(500))]
            out.append({"outcome": y, "community-cutoffs": len(D), "saturating": int(D.saturating.sum()),
                        "mean outcome, saturating": float(D[D.saturating][y].mean()),
                        "mean outcome, not saturating": float(D[~D.saturating][y].mean()),
                        "partial rho (gain change vs outcome; size, gain level held)": float(rho),
                        "90% CI": f"[{np.nanpercentile(bs, 5):+.3f}, {np.nanpercentile(bs, 95):+.3f}]"})
        R = pd.DataFrame(out)
    cov = pd.DataFrame([{"papers with a quantitative result": int(con.execute(
        "SELECT count(*) FROM facets WHERE facet = 'has_quantitative_result' AND text = 'true'").fetchone()[0]),
        "extracted": len(pd.read_parquet(OUT)), "usable relative gains": len(G),
        **{f"kind: {k}": int(v) for k, v in G.kind.value_counts().items()}}])
    pd.set_option("display.width", 220)
    print(cov.T.to_string())
    print(R.round(3).to_string(index=False))
    C.save(RUN, "exp24_saturation", "coverage", cov, title="Saturation: how many abstracts state a comparable gain",
           how_to_read="Papers whose abstract reports a quantitative result, how many the extraction could turn into a "
           "relative gain over a named baseline, and of which kind.")
    C.save(RUN, "exp24_saturation", "effects", R, title="Saturation: do shrinking reported gains predict a direction slowing?",
           how_to_read="Per community and cutoff, the change in the median reported gain (same kind of gain, papers of "
           "T-3..T-1 against T-6..T-4). A negative change = *saturating*. Outcomes: the community's share of new papers "
           "and of first entries in T+1..T+3 against T-2..T. The partial rank correlation holds size and the current gain "
           "level fixed; a positive rho means shrinking gains go with shrinking activity.")


for s in a.stages:
    asyncio.run(extract()) if s == "extract" else analyse()
