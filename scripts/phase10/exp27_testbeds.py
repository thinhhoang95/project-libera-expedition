"""Experiment 27: how a community evaluates. Does using its standard testbeds pay?

For each community X (problem communities of the cutoff-T run, alpha 0.5, medium) and cutoff T (2019-2022):

    testbed groups    X's data / domain statements (domain_data facet, 256-d) from T-3..T-1, grouped by average
                      linkage at cosine 0.94 (the 99.9th percentile of random pairs); a *standard testbed* is a group of
                      >= 3 papers from >= 2 unrelated teams
    year-T papers     research papers of X published in T, labelled:
                        standard testbed   their data statement is close (cosine >= 0.94) to a member of a standard
                                           group; generic groups ("real data", "simulated data": the fifth of all groups
                                           closest to the mean data statement) do not count
                        own testbed        a data statement that matches no standard group
                        no named data      no data statement

Outcomes in T+1..T+3: taken up by others (non-self citations, top quarter of year-T papers), and X's later work moved
towards the paper (experiment 9). Standard vs own testbed compared like for like (logistic: team size, references in
the map, distance from X's core problem, empirical result, kind of contribution, community size, year), 90% intervals
resampling communities. The live testbeds (latest complete years) go to
data/projects/<run>/backtest/exp27_testbeds_live.parquet for inspection.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.linear_model import LogisticRegression

import common as C

RUN = C.run_arg()
COS = 0.94
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
ptype = papers.set_index("paper_id").type
DP, DV = C.vectors("domain_data", ids)
PP, PV = C.vectors("problem", ids)
DMEAN = C.unit(DV.mean(axis=0))
GEN_CUT = None  # set below: genericness of the most generic fifth of testbed groups
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
f = con.execute("""SELECT paper_id, facet, text FROM facets WHERE facet IN ('contribution_type', 'has_quantitative_result',
                   'domain_data')""").df().pivot_table(index="paper_id", columns="facet", values="text", aggfunc="first")
dtext = f.domain_data
nonself = refs.assign(is_self=[bool(team.get(a, frozenset()) & team.get(b, frozenset()))
                               for a, b in zip(refs.citing_id, refs.cited_id)])


def stack(pids):
    a = [MV[MP[x]] for x in pids if x in MP] + [IV[IP[x]] for x in pids if x in IP]
    return np.vstack(a) if a else np.zeros((0, MV.shape[1]), np.float32)


def unrelated_teams(mem):
    seen, n = set(), 0
    for x in mem:
        t = team.get(x, frozenset())
        if not (t & seen):
            n += 1
        seen |= t
    return n


def testbeds(mem, drop_generic=True):
    """Standard testbed groups among a community's papers: list of (member ids, vectors), generic ones dropped."""
    mem = [x for x in mem if x in DP]
    if len(mem) < 3:
        return []
    V = DV[[DP[x] for x in mem]]
    lab = fcluster(linkage(V, method="average", metric="cosine"), t=1 - COS, criterion="distance")
    out = []
    for g in np.unique(lab):
        k = np.where(lab == g)[0]
        m = [mem[i] for i in k]
        if len(m) >= 3 and unrelated_teams(m) >= 2:
            if drop_generic and GEN_CUT is not None and float(C.unit(V[k].mean(axis=0)) @ DMEAN) > GEN_CUT:
                continue
            out.append((m, V[k]))
    return out


# genericness cut: over every candidate group of every community and cutoff, the most generic fifth
_gen = []
for _T in range(2019, 2023):
    _lab = C.assign(con, RUN, f"{RUN}@T{_T}", ids)
    for _X in set(_lab):
        _win = [x for x in _lab.index[_lab.values == _X] if _T - 3 <= year[x] <= _T - 1]
        _gen += [float(C.unit(Vg.mean(axis=0)) @ DMEAN) for _, Vg in testbeds(_win, drop_generic=False)]
GEN_CUT = float(np.quantile(_gen, 0.8)) if _gen else None
print(f"candidate testbed groups: {len(_gen)}; genericness cut {GEN_CUT:.3f}")


rows, sample = [], []
for T in range(2019, 2023):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    yr = lab.index.map(year)
    fut = nonself[(nonself.cy > T) & (nonself.cy <= T + 3) & ~nonself.is_self].groupby("cited_id").size()
    yT = [x for x in lab.index[yr == T] if ptype.get(x) in C.RESEARCH]
    up_cut = pd.Series([fut.get(x, 0) for x in yT]).quantile(0.75)
    for X in sorted(set(lab)):
        mem = lab.index[lab.values == X]
        win = [x for x in mem if T - 3 <= year[x] <= T - 1]
        if len(win) < 10:
            continue
        G = testbeds(win)
        if len(sample) < 12:
            for m, _ in G[:2]:
                sample.append({"T": T, "X": X, "examples": " | ".join(str(dtext.get(x))[:80] for x in m[:3])})
        prev = [x for x in mem if x in PP and year[x] <= T - 1]
        if len(prev) < 5:
            continue
        core = C.unit(PV[[PP[x] for x in prev]].mean(axis=0))
        later = [x for x in mem if T < year[x] <= T + 3]
        recent = [x for x in mem if T - 2 <= year[x] <= T]
        for x in [p for p in mem if year[p] == T and ptype.get(p) in C.RESEARCH]:
            if x in DP:
                v = DV[DP[x]]
                standard = any(float((Vg @ v).max()) >= COS for _, Vg in G)
                use = "standard testbed" if standard else "own testbed"
            else:
                use = "no named data"
            moved = np.nan
            if x in IP:
                tm = team.get(x, frozenset())
                Fo = stack([y for y in later if not (team.get(y, frozenset()) & tm)])
                Rw = stack([y for y in recent if y != x])
                if len(Fo):
                    fmax = float((Fo @ IV[IP[x]]).max())
                    rmax = float((Rw @ IV[IP[x]]).max()) if len(Rw) else -1.0
                    moved = float((fmax >= C.THR_REUSE) and (fmax > rmax))
            rows.append({"T": T, "X": X, "paper_id": x, "use": use, "n_testbeds": len(G),
                         "taken_up": fut.get(x, 0) > up_cut, "moved_towards": moved,
                         "team": len(team.get(x, frozenset())), "n_refs": int((refs.citing_id == x).sum()),
                         "core_distance": 1 - float(PV[PP[x]] @ core) if x in PP else np.nan,
                         "empirical": f.has_quantitative_result.get(x) == "true",
                         "kind": f.contribution_type.get(x), "size": len(prev)})
    print(T, len(rows), flush=True)
D = pd.DataFrame(rows)
D["core_distance"] = D.core_distance.fillna(D.core_distance.median())
cov = D.groupby("use").size().rename("papers").reset_index()
cov["share"] = cov.papers / cov.papers.sum()
print(cov.to_string(index=False))
print(pd.DataFrame(sample).to_string(index=False))


def design(d):
    X = pd.DataFrame({"standard testbed": (d.use == "standard testbed").astype(float)})
    X["log team"] = np.log1p(d.team)
    X["log refs"] = np.log1p(d.n_refs)
    X["core distance"] = d.core_distance
    X["empirical"] = d.empirical.astype(float)
    for k in ("theory", "survey"):
        X[k] = (d.kind == k).astype(float)
    X["log community size"] = np.log(d["size"])
    for T in sorted(d["T"].unique())[1:]:
        X[f"T{T}"] = (d["T"] == T).astype(float)
    return X


def adjusted(d, y):
    X, yy = design(d), d[y].astype(int)
    m = LogisticRegression(max_iter=3000).fit(X, yy)
    Z1, Z0 = X.copy(), X.copy()
    Z1["standard testbed"], Z0["standard testbed"] = 1, 0
    return m.predict_proba(Z1)[:, 1].mean() - m.predict_proba(Z0)[:, 1].mean()


res = []
named = D[D.use != "no named data"]
for y in ("taken_up", "moved_towards"):
    d = named.dropna(subset=[y]).reset_index(drop=True)
    est = adjusted(d, y)
    groups = list(d.groupby(["T", "X"]).indices.values())
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(300):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        try:
            bs.append(adjusted(d.iloc[i].reset_index(drop=True), y))
        except ValueError:
            pass
    st = d.use == "standard testbed"
    res.append({"outcome": y, "papers with named data": len(d), "on a standard testbed": int(st.sum()),
                "raw rate, standard": float(d[st][y].mean()), "raw rate, own testbed": float(d[~st][y].mean()),
                "raw rate, no named data": float(D[D.use == "no named data"][y].dropna().mean()),
                "adjusted difference (standard - own)": est, "90% CI": f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]"})
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
how = ("Year-T research papers (2019-2022) of each community, by the data they evaluate on: a **standard testbed** (close "
       "to a data / domain description used by 3+ of the community's papers from 2+ unrelated teams in the three years "
       "before), their **own testbed** (a description matching none), or **no named data**. Outcomes: taken up by others "
       "(non-self citations, top quarter of year-T papers) and the community moving towards the paper (experiment 9). "
       "The adjusted difference compares standard with own testbeds like for like (team, references, distance from the "
       "core problem, empirical result, kind, community size, year); intervals resample communities.")
C.save(RUN, "exp27_testbeds", "effects", R, title="Testbeds: does using a community's standard testbed pay?", how_to_read=how)
C.save(RUN, "exp27_testbeds", "coverage", cov, title="Testbeds: how papers evaluate",
       how_to_read="Share of year-T research papers on a standard testbed, on their own, or naming no data.")
C.save(RUN, "exp27_testbeds", "examples", pd.DataFrame(sample), title="Testbeds: examples of standard testbed groups",
       how_to_read="Three data / domain statements from each of a few standard testbed groups, to judge face validity.")

# ---------------------------------------------------------------- live testbeds
T = int(papers.year.max())
lab = C.assign(con, RUN, RUN, ids)
labels = dict(con.execute("SELECT cluster_id, label FROM clusters WHERE run_id = ?", [RUN]).fetchall())
live = []
for X in sorted(set(lab)):
    win = [x for x in lab.index[lab.values == X] if T - 3 <= year[x] <= T - 1]
    for m, Vg in testbeds(win):
        k = int(np.argmax(Vg @ C.unit(Vg.mean(axis=0))))
        live.append({"cluster_id": X, "label": labels.get(X, X), "testbed": dtext.get(m[k]), "papers": len(m),
                     "teams": unrelated_teams(m), "examples": m[:5], "window": f"{T - 3}-{T - 1}"})
pd.DataFrame(live).to_parquet(C.project(RUN, "backtest") / "exp27_testbeds_live.parquet", index=False)
print("live testbeds:", len(live))
