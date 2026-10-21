"""Experiment 20: tools to learn vs methods in flux.

Method communities of each cutoff run (2019-2022; method, alpha 0.5, medium, stable and hub), every paper assigned to
the nearest centroid on alpha-0.5 blended method vectors. Traits of method community m at T (its papers from T-2..T):
    breadth        problem communities (cutoff-T run) with >= 2 of its papers
    dispersion     1 - mean cosine of its papers' method statements to their centroid (how varied the variants are)
    inward         share of its papers whose innovation is about the method itself (innovation within 0.85 of the
                   method community's centroid)
    momentum       its share of all papers in T vs T-2..T-1
Outcomes in T+1..T+3:
    sustained      its share of papers in T+1..T+3 >= its share in T-2..T
    spread         problem communities newly reached (>= 2 papers, none in T-2..T)
    stayed         for newcomers who entered a problem community in T with a paper using m (experiment 12's entries):
                   did they publish there again (adjusted for experiment 12's controls)
Rank correlations across method communities x cutoffs, and the entry-level effect of a "stable tool" score
(breadth high, dispersion and inward low).
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
E = pd.read_parquet(C.project(RUN, "backtest") / "exp12_entries.parquet")


def assign_method(cutoff_run):
    cents = pd.read_parquet(C.project(RUN, "vectors") / f"cluster_centroids_{cutoff_run}.parquet")
    keep = set(con.execute("""SELECT cluster_id FROM clusters WHERE run_id = ? AND facet = 'method' AND alpha = 0.5
                              AND resolution = 'medium' AND kind IN ('stable', 'hub')""", [cutoff_run]).df().cluster_id)
    c = cents[cents.cluster_id.isin(keep)]
    Cm = np.stack([np.asarray(x, dtype=np.float32) for x in c.centroid])
    pos, v = C.blended("method", "method_gloss", 0.5, ids)
    return pd.Series(np.array(c.cluster_id)[np.argmax(v @ Cm.T, axis=1)], index=list(pos))


rows, ent_rows = [], []
for T in range(2019, 2023):
    mlab = assign_method(f"{RUN}@T{T}")
    plab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    yr = pd.Series({x: year[x] for x in mlab.index})
    n_win = lambda a, b: ((yr >= a) & (yr <= b)).sum()  # noqa: E731
    for m in sorted(set(mlab)):
        mem = mlab.index[mlab.values == m]
        rec = [x for x in mem if T - 2 <= year[x] <= T]
        if len(rec) < 10:
            continue
        mv = MV[[MP[x] for x in rec if x in MP]]
        cen = C.unit(mv.mean(axis=0))
        inn = [IV[IP[x]] @ cen for x in rec if x in IP]
        pc_rec = pd.Series([plab.get(x) for x in rec]).value_counts()
        fut = [x for x in mem if T + 1 <= year[x] <= T + 3]
        pc_fut = pd.Series([plab.get(x) for x in fut]).value_counts()
        share_rec = len(rec) / n_win(T - 2, T)
        share_fut = len(fut) / max(n_win(T + 1, T + 3), 1)
        rows.append({"T": T, "m": m, "papers_recent": len(rec), "breadth": int((pc_rec >= 2).sum()),
                     "dispersion": float(1 - (mv @ cen).mean()), "inward": float(np.mean(np.array(inn) >= 0.85)) if inn else np.nan,
                     "momentum": np.log(((yr[mem] == T).sum() / max(n_win(T, T), 1) + 1e-4) /
                                        (((yr[mem] >= T - 2) & (yr[mem] <= T - 1)).sum() / max(n_win(T - 2, T - 1), 1) + 1e-4)),
                     "sustained": share_fut >= share_rec,
                     "spread": int(sum(1 for p, n in pc_fut.items() if n >= 2 and pc_rec.get(p, 0) == 0))})
    e = E[E["T"] == T].copy()
    e["m"] = e.paper_id.map(mlab)
    ent_rows.append(e)
    print(T, "method communities", sum(1 for r in rows if r["T"] == T), flush=True)

D = pd.DataFrame(rows)
for c in ("breadth", "dispersion", "inward"):
    D[c + "_z"] = D.groupby("T")[c].transform(lambda s: (s - s.mean()) / (s.std() or 1))
D["stable_score"] = D.breadth_z - D.dispersion_z - D.inward_z.fillna(0)
D.to_parquet(C.project(RUN, "backtest") / "exp20_tools.parquet", index=False)
res = []
for t in ("breadth", "dispersion", "inward", "momentum", "stable_score"):
    for y in ("sustained", "spread"):
        rs = np.random.default_rng(0)
        d = D.dropna(subset=[t])[[t, y]].astype(float)
        bs = [spearmanr(*d.sample(len(d), replace=True, random_state=int(rs.integers(1_000_000_000))).to_numpy().T).statistic
              for _ in range(500)]
        res.append({"trait": t, "outcome": y, "rank correlation": spearmanr(d[t], d[y]).statistic,
                    "90% CI": f"[{np.nanpercentile(bs, 5):+.3f}, {np.nanpercentile(bs, 95):+.3f}]",
                    "method communities x cutoffs": len(d)})
# entries: did newcomers who entered with a stable tool stay?
EN = pd.concat(ent_rows).merge(D[["T", "m", "stable_score"]], on=["T", "m"], how="left").dropna(subset=["stable_score"])
X = pd.DataFrame({"stable tool": EN.stable_score, "insider": EN.insider.astype(float), "newcomer": EN.newcomer.astype(float),
                  "log team": np.log1p(EN.team), "log size": np.log(EN["size"]), "forward": EN.forward.astype(float),
                  "log refs": np.log1p(EN.n_inmap_refs)})
for T in sorted(EN["T"].unique())[1:]:
    X[f"T{T}"] = (EN["T"] == T).astype(float)


def effect(Xd, yd):
    m = LogisticRegression(max_iter=3000).fit(Xd, yd)
    hi, lo = Xd.copy(), Xd.copy()
    q = Xd["stable tool"].quantile([0.25, 0.75]).to_numpy()
    hi["stable tool"], lo["stable tool"] = q[1], q[0]
    return m.predict_proba(hi)[:, 1].mean() - m.predict_proba(lo)[:, 1].mean()


for y in ("stayed", "taken_up"):
    yy = EN[y].astype(int)
    est = effect(X, yy)
    groups = list(EN.reset_index(drop=True).groupby(["T", "X"]).indices.values())
    rs = np.random.default_rng(0)
    bs = []
    for _ in range(300):
        i = np.concatenate([groups[j] for j in rs.integers(0, len(groups), len(groups))])
        bs.append(effect(X.iloc[i].reset_index(drop=True), yy.iloc[i].reset_index(drop=True)))
    res.append({"trait": "stable tool score (entries, 75th vs 25th percentile, adjusted)", "outcome": y,
                "rank correlation": est, "90% CI": f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]",
                "method communities x cutoffs": len(EN)})
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
C.save(RUN, "exp20_tools", "results", R, title="Tools to learn vs methods in flux",
       how_to_read="Method communities at each cutoff (2019-2022). **breadth** = problem communities using the method; "
       "**dispersion** = how varied its recent variants are; **inward** = share of its papers innovating on the method "
       "itself; **stable_score** = breadth minus dispersion minus inward (standardized). Outcomes: its share of papers is "
       "**sustained** in the next three years; **spread** = problem communities newly reached. The last rows use "
       "experiment 12's entries: the adjusted difference in staying / being taken up between newcomers whose entry paper "
       "used a stable tool (75th percentile) and a less stable one (25th), with experiment 12's controls.")
