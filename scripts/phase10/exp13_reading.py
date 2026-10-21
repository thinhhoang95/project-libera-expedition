"""Experiment 13: "read this first" -- which reading list covers what newcomers to a community actually needed?

For each cutoff T (2019-2022) and problem community X (cutoff-T run, alpha 0.5, medium), the newcomers are the
authors whose first paper in X appears in T+1..T+3. What they needed, from papers published up to T (in the map):
    cited    the earlier papers their X papers in T+1..T+3 cite
    reused   the earlier papers whose innovation is the closest earlier source (>= 0.894, experiment 9) of a method
             or innovation in their X papers -- what their work builds on, cited or not
Candidate reading lists of k papers, built from data up to T only:
    popular in X       most cited by X's papers up to T
    popular overall    most cited by the map up to T
    central            closest to X's problem centroid (members up to T)
    surveys            surveys closest to X's centroid (filled with "popular in X" if fewer than k)
    recent hits        X papers from T-2..T most cited by X up to T
    method lineage     the earliest close sources (innovation >= 0.894) of the methods and innovations of X's
                       papers from T-2..T, ranked by how many of them they are the source of
    lineage + popular  alternating the two
Score: coverage@k = share of the needed set (weighted by how many newcomers needed each paper) that the list
contains; hit@k = share of newcomers needing at least one listed paper. Averages over communities, 90% intervals
resampling communities. Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
from collections import Counter

import numpy as np
import pandas as pd

import common as C

RUN = C.run_arg()
K = (10, 20)
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
PP, PV = C.vectors("problem", ids)
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
ctype = dict(con.execute("SELECT paper_id, text FROM facets WHERE facet = 'contribution_type'").fetchall())
inn_ids = [x for x in IP]
inn_year = np.array([year.get(x, 9999) for x in inn_ids])


def sources(vecs: np.ndarray, before_year: int) -> list[str]:
    """For each method/innovation vector, the earliest paper (published before `before_year`) whose innovation is
    >= THR close; none if no such paper."""
    ok = np.where(inn_year < before_year)[0]
    if not len(ok) or not len(vecs):
        return []
    S = vecs @ IV[ok].T
    out = []
    for row in S:
        hit = ok[row >= C.THR_REUSE]
        if len(hit):
            out.append(inn_ids[hit[np.argmin(inn_year[hit])]])
    return out


def stack(pids):
    a = [MV[MP[x]] for x in pids if x in MP] + [IV[IP[x]] for x in pids if x in IP]
    return np.vstack(a) if a else np.zeros((0, MV.shape[1]), np.float32)


rows = []
for T in range(2019, 2023):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    pre = set(papers[papers.year <= T].paper_id)
    rT = refs[refs.cy <= T]
    pop_all = rT[rT.cited_id.isin(pre)].groupby("cited_id").size()
    a2 = au.assign(X=au.paper_id.map(lab)).dropna(subset=["X"])
    first = a2.sort_values("year").groupby(["author_id", "X"]).head(1)
    for X in sorted(set(lab)):
        mem = lab.index[lab.values == X]
        mem_pre = [x for x in mem if year[x] <= T]
        if len(mem_pre) < 10:
            continue
        newc = first[(first.X == X) & first.year.between(T + 1, T + 3)].author_id.unique()
        if len(newc) < 5:
            continue
        # what each newcomer needed
        need_cited, need_reused = {}, {}
        for a in newc:
            their = a2[(a2.author_id == a) & (a2.X == X) & a2.year.between(T + 1, T + 3)].paper_id.unique()
            need_cited[a] = set(refs[refs.citing_id.isin(their)].cited_id) & pre
            need_reused[a] = set(sources(stack(their), T + 1))
        # candidate lists
        inX = rT[rT.citing_id.isin(set(mem_pre)) & rT.cited_id.isin(pre)].groupby("cited_id").size()
        cent = C.unit(np.mean([PV[PP[x]] for x in mem_pre if x in PP], axis=0))
        prob_ids = [x for x in pre if x in PP]
        sim = pd.Series(PV[[PP[x] for x in prob_ids]] @ cent, index=prob_ids).sort_values(ascending=False)
        recent = [x for x in mem_pre if year[x] >= T - 2]
        lin = Counter(sources(stack(recent), T - 1))
        lists = {
            "popular in X": inX.sort_values(ascending=False).index.tolist(),
            "popular overall": pop_all.sort_values(ascending=False).index.tolist(),
            "central": sim.index.tolist(),
            "surveys": [x for x in sim.index if ctype.get(x) == "survey"],
            "recent hits": inX.reindex(recent).fillna(0).sort_values(ascending=False).index.tolist(),
            "method lineage": [x for x, _ in lin.most_common()],
        }
        lists["surveys"] = lists["surveys"][:max(K)] + [x for x in lists["popular in X"] if x not in lists["surveys"]]
        lists["lineage + popular"] = [x for pair in zip(lists["method lineage"], lists["popular in X"]) for x in pair]
        for name, L in lists.items():
            L = list(dict.fromkeys(L))
            for k in K:
                top = set(L[:k])
                for kind, need in (("cited", need_cited), ("reused", need_reused)):
                    w = Counter(p for s in need.values() for p in s)
                    total = sum(w.values())
                    if total == 0:
                        continue
                    rows.append({"T": T, "X": X, "list": name, "k": k, "needed": kind,
                                 "coverage": sum(w[p] for p in top) / total,
                                 "hit": np.mean([bool(s & top) for s in need.values() if s]),
                                 "newcomers": len(newc)})
    print(T, "communities", len({r["X"] for r in rows if r["T"] == T}), flush=True)

D = pd.DataFrame(rows)
D.to_parquet(C.project(RUN, "backtest") / "exp13_reading.parquet", index=False)
res = []
for (kind, k), g in D.groupby(["needed", "k"]):
    groups = list(g.groupby(["T", "X"]).indices.values())
    base = g[g.list == "popular in X"].set_index(["T", "X"]).coverage
    for name, h in g.groupby("list"):
        diff = (h.set_index(["T", "X"]).coverage - base).dropna()
        rng = np.random.default_rng(0)
        bs = [diff.sample(len(diff), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(500)]
        res.append({"needed": kind, "k": k, "list": name, "coverage": h.coverage.mean(), "hit": h.hit.mean(),
                    "coverage vs popular in X": diff.mean(), "90% CI": f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]",
                    "communities x cutoffs": len(h)})
R = pd.DataFrame(res).sort_values(["needed", "k", "coverage"], ascending=[True, True, False])
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
C.save(RUN, "exp13_reading", "lists", R, title="Read this first: which list covers what newcomers needed",
       how_to_read="For each community and cutoff T (2019-2022), the authors who joined it in T+1..T+3 and what they "
       "built on from papers up to T: the earlier papers they **cited**, and the earlier papers whose innovation is the "
       "closest source of their methods (**reused**, cited or not). Each candidate list is built from data up to T "
       "only. **coverage** is the share of what newcomers needed that the list contains; **hit** the share of "
       "newcomers needing at least one listed paper. Differences are against 'popular in X' (the most cited papers "
       "in the community), resampling communities.")
