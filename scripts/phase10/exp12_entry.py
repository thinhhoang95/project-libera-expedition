"""Experiment 12: how did people successfully enter a community?

Entry events: for each cutoff T (2019-2022), an author's first paper in problem community X (communities of the
cutoff-T run, alpha 0.5, medium) published in year T (research types). Strategies, from the entry paper's facets
against X's papers up to T-1 (256-d vectors):

    method import         the paper's method is closest to another community's methods and unusual for X (below
                          the 25th percentile of X's own members' similarity to X's method centroid)
    new problem for X     not an import, and its problem is far from X's core (top quarter of distance)
    bridge                neither, and its references spread across several communities' literatures (top quarter
                          of entropy, at least 3 references inside the map)
    core incremental      none of the above (the reference group)

Outcomes in T+1..T+3:
    stayed                the author has another paper in X (author level)
    taken up by others    non-self citations in the top quarter of year-T papers (paper level)
    X moved towards it    a paper of X by other authors has a method or innovation closer than 0.894 to the entry's
                          innovation, and closer than any of X's papers from T-2..T (paper level; experiment 9)

Like-for-like comparison: logistic models with the strategy and controls (insider co-author with >= 2 earlier
papers in X, team size, community size, newcomer to the whole map vs mover, forward route, number of references
inside the map, year); adjusted
differences vs core incremental, with 90% intervals resampling communities.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ptype = papers.set_index("paper_id").type
route = papers.set_index("paper_id").route
ids = papers.paper_id.tolist()
PP, PV = C.vectors("problem", ids)
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
inn_or_meth = {}


def stack(pids):
    a = [MV[MP[x]] for x in pids if x in MP] + [IV[IP[x]] for x in pids if x in IP]
    return np.vstack(a) if a else np.zeros((0, MV.shape[1]), np.float32)


rows = []
for T in range(2019, 2023):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    comm = sorted(set(lab))
    mem_pre = {c: [x for x in lab.index[lab.values == c] if year[x] <= T - 1] for c in comm}
    core = {c: C.unit(np.mean([PV[PP[x]] for x in m if x in PP], axis=0)) for c, m in mem_pre.items()
            if sum(x in PP for x in m) >= 5}
    mcent = {c: C.unit(np.mean([MV[MP[x]] for x in m if x in MP], axis=0)) for c, m in mem_pre.items()
             if sum(x in MP for x in m) >= 5}
    mc_ids = list(mcent)
    # how close a community's own members' methods are to its method centroid: an import must sit below the
    # 25th percentile of that, i.e. be unusual for the community, and closest to another community's methods
    own_q = {c: np.quantile([float(MV[MP[x]] @ mcent[c]) for x in mem_pre[c] if x in MP], 0.25) for c in mcent}
    MC = np.vstack([mcent[c] for c in mc_ids])
    a2 = au.assign(X=au.paper_id.map(lab)).dropna(subset=["X"])
    first = a2[a2.year <= T].sort_values("year").groupby(["author_id", "X"]).head(1)
    ent = first[(first.year == T) & first.paper_id.map(ptype).isin(C.RESEARCH) & first.X.isin(core)]
    earlier_any = set(au[au.year < T].author_id)
    n_in_X_before = a2[a2.year < T].groupby(["author_id", "X"]).paper_id.nunique()
    later = a2[a2.year.between(T + 1, T + 3)].groupby(["author_id", "X"]).paper_id.nunique()
    # paper-level features of each entry paper
    pfeat = {}
    for x, X in ent[["paper_id", "X"]].drop_duplicates().itertuples(index=False):
        if x not in PP or x not in MP:
            continue
        d_core = 1 - float(PV[PP[x]] @ core[X])
        origin = mc_ids[int(np.argmax(MC @ MV[MP[x]]))]
        cited = [lab.get(r) for r in refs[refs.citing_id == x].cited_id if year.get(r, 9999) <= T - 1]
        cited = [c for c in cited if isinstance(c, str)]
        if len(cited) >= 3:
            p = pd.Series(cited).value_counts(normalize=True).to_numpy()
            ent_ = float(-(p * np.log(p)).sum())
        else:
            ent_ = np.nan
        tm = team.get(x, frozenset())
        insider = any(n_in_X_before.get((a, X), 0) >= 2 for a in tm)
        unusual = X in mcent and float(MV[MP[x]] @ mcent[X]) < own_q[X]
        pfeat[(x, X)] = {"core_distance": d_core, "import": (origin != X) and unusual, "ref_entropy": ent_, "n_inmap_refs": len(cited), "insider": insider,
                         "team": len(tm), "size": len(mem_pre[X]), "forward": route.get(x) == "forward"}
    # outcomes: uptake by others (paper level) and X moving towards it (paper level)
    yT = papers[(papers.year == T) & papers.type.isin(C.RESEARCH)].paper_id
    nonself = refs[refs.cy.between(T + 1, T + 3) & refs.cited_id.isin(set(yT))]
    nonself = nonself[[not (team.get(a, frozenset()) & team.get(b, frozenset())) for a, b in
                       zip(nonself.citing_id, nonself.cited_id)]]
    n_ns = nonself.groupby("cited_id").size().reindex(yT).fillna(0)
    up_cut = n_ns.quantile(0.75)
    for (x, X), f in pfeat.items():
        fut = [p for p in lab.index[lab.values == X] if T + 1 <= year[p] <= T + 3 and not (team.get(p, frozenset()) & team.get(x, frozenset()))]
        rec = [p for p in lab.index[lab.values == X] if T - 2 <= year[p] <= T and p != x]
        if x in IP:
            v = IV[IP[x]]
            F, R = stack(fut), stack(rec)
            fmax = float((F @ v).max()) if len(F) else -1
            rmax = float((R @ v).max()) if len(R) else -1
            f["moved_towards"] = (fmax >= C.THR_REUSE) and (fmax > rmax)
        else:
            f["moved_towards"] = np.nan
        f["taken_up"] = n_ns.get(x, 0) > up_cut
    for r in ent.itertuples():
        f = pfeat.get((r.paper_id, r.X))
        if f is None:
            continue
        rows.append({"T": T, "author_id": r.author_id, "paper_id": r.paper_id, "X": r.X,
                     "newcomer": r.author_id not in earlier_any, "stayed": later.get((r.author_id, r.X), 0) >= 1, **f})
    print(T, "entries", len(ent), "with features", sum(1 for r in rows if r["T"] == T), flush=True)

D = pd.DataFrame(rows)
D["far"] = D.core_distance >= D.groupby("T").core_distance.transform(lambda s: s.quantile(0.75))
ecut = D.groupby("T").ref_entropy.transform(lambda s: s.quantile(0.75))
D["bridge_ref"] = D.ref_entropy.notna() & (D.ref_entropy >= ecut)
D["strategy"] = np.select([D["import"], D.far, D.bridge_ref], ["method import", "new problem for X", "bridge"],
                          "core incremental")
D.to_parquet(C.project(RUN, "backtest") / "exp12_entries.parquet", index=False)
STRATS = ["method import", "new problem for X", "bridge"]


def design(d):
    X = pd.DataFrame({s: (d.strategy == s).astype(float) for s in STRATS})
    X["insider co-author"] = d.insider.astype(float)
    X["newcomer to the map"] = d.newcomer.astype(float)
    X["log team"] = np.log1p(d.team)
    X["log community size"] = np.log(d["size"])
    X["forward route"] = d.forward.astype(float)
    X["log in-map references"] = np.log1p(d.n_inmap_refs)  # bridges need references in the map: hold that fixed
    for T in sorted(d["T"].unique())[1:]:
        X[f"T{T}"] = (d["T"] == T).astype(float)
    return X


def adjusted(d, y):
    """Average predicted rate if every entry used strategy s, minus if every entry were core incremental."""
    d = d.dropna(subset=[y])
    X, yy = design(d), d[y].astype(int)
    m = LogisticRegression(max_iter=3000, C=1.0).fit(X, yy)
    out = {}
    base = X.copy()
    base[STRATS] = 0
    p0 = m.predict_proba(base)[:, 1].mean()
    for s in STRATS + ["insider co-author"]:
        Z = base.copy()
        if s in STRATS:
            Z[s] = 1
        else:
            Z = X.copy()
            Z[s] = 1
            Z0 = X.copy()
            Z0[s] = 0
            out[s] = m.predict_proba(Z)[:, 1].mean() - m.predict_proba(Z0)[:, 1].mean()
            continue
        out[s] = m.predict_proba(Z)[:, 1].mean() - p0
    return out, p0


res = []
OUT = {"stayed": ("author", D), "taken_up": ("paper", D.drop_duplicates("paper_id")),
       "moved_towards": ("paper", D.drop_duplicates("paper_id"))}
for y, (unit, d) in OUT.items():
    d = d.dropna(subset=[y]).reset_index(drop=True)
    est, p0 = adjusted(d, y)
    groups = list(d.groupby(["T", "X"]).indices.values())
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        try:
            boots.append(adjusted(d.iloc[i].reset_index(drop=True), y)[0])
        except ValueError:
            pass
    for s in STRATS + ["insider co-author"]:
        b = np.array([x[s] for x in boots])
        raw = d[d.strategy == s][y].mean() if s in STRATS else d[d.insider][y].mean()
        res.append({"outcome": y, "unit": unit, "strategy / trait": s, "entries": int((d.strategy == s).sum()) if s in STRATS
                    else int(d.insider.sum()), "raw rate": raw,
                    "raw rate, core incremental" if s in STRATS else "raw rate, without": d[d.strategy == "core incremental"][y].mean()
                    if s in STRATS else d[~d.insider][y].mean(),
                    "adjusted difference": est[s], "90% CI": f"[{np.percentile(b, 5):+.3f}, {np.percentile(b, 95):+.3f}]",
                    "adjusted rate, core incremental": p0})
R = pd.DataFrame(res)
R = R.rename(columns={"raw rate, core incremental": "raw rate, reference"})
if "raw rate, without" in R:
    R["raw rate, reference"] = R["raw rate, reference"].fillna(R.pop("raw rate, without"))
mix = D.groupby(["strategy"]).agg(entries=("paper_id", "size"), newcomers=("newcomer", "mean"),
                                   insider_coauthor=("insider", "mean"), stayed=("stayed", "mean"),
                                   taken_up=("taken_up", "mean"), moved_towards=("moved_towards", "mean")).reset_index()
pd.set_option("display.width", 220)
print(mix.round(3).to_string(index=False))
print(R.round(3).to_string(index=False))
how = ("Entries are an author's first paper in a problem community (communities as they stood at the cutoff year T, "
       "2019-2022). **Strategy** is read from the paper's facets against the community's earlier papers: its method "
       "comes from another community (*method import*), its problem is far from the community's core (*new problem*), "
       "its references span several communities (*bridge*), or none (*core incremental*, the reference). Outcomes in "
       "the next three years: the author **stayed** (another paper in the community), the paper was **taken up** by "
       "others (non-self citations, top quarter of its year), the community's later work **moved towards** it "
       "(experiment 9). The **adjusted difference** compares like with like (insider co-author, team and community "
       "size, newcomer to the map, route, number of references inside the map, year); intervals resample communities. The insider row compares entries "
       "with and without a co-author who already had 2+ papers in the community.")
C.save(RUN, "exp12_entry", "strategies", R, title="Entering a community: which strategies worked", how_to_read=how)
C.save(RUN, "exp12_entry", "mix", mix, title="Entering a community: raw rates by strategy",
       how_to_read="Raw shares, without controls: how many entries used each strategy, how many were newcomers to the "
       "whole map, had an insider co-author, and each outcome's raw rate.")
