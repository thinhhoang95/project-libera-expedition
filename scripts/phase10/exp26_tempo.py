"""Experiment 26: community tempo. How recent are the ideas that best match a community's current questions?

For each problem statement of community X published in year t, the closest innovation statement published before t
(any community, not the same paper; cosine >= 0.880, the 99th percentile of random innovation-problem pairs, 256-d) is
its *matching idea*; its age is t minus its year. A community's **tempo** at T is the share of the matching ideas of
its problems from T-2..T that are 3 or more years old (high = slow: its questions are still best matched by older
ideas). The median age is uninformative here: the map is densest in recent years, so most matches are a year old.

1. Stability: is tempo a trait? Rank correlation of tempo at T with tempo at T+3, against the field-wide change.
2. Payoff: experiment 12's entries (an author's first paper in X in T). The *source age* of an entry is the age of the
   closest earlier innovation to its method statement (cosine >= 0.894). Does building on recent sources pay more in
   fast communities than in slow ones? Logistic models of stayed / taken up / moved towards with source recency, tempo
   and their interaction, plus experiment 12's controls; 90% intervals resampling communities.

Communities are those of the cutoff-T run (2019-2022). Ages are bounded by the map (thin before 2015), so tempo is
compared across communities within a cutoff, not read as years.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

import common as C

RUN = C.run_arg()
MATCH = 0.880
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
PP, PV = C.vectors("problem", ids)
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
inn_ids = list(IP)
inn_year = np.array([year.get(x, 9999) for x in inn_ids])
E = pd.read_parquet(C.project(RUN, "backtest") / "exp12_entries.parquet")


def ages(vecs, pids, t_of, thr):
    """Age of the closest earlier innovation (not the same paper) for each vector; NaN if none above thr."""
    out = []
    for start in range(0, len(pids), 400):
        S = vecs[start:start + 400] @ IV.T
        for k, x in enumerate(pids[start:start + 400]):
            t = t_of(x)
            ok = (inn_year < t)
            s = np.where(ok, S[k], -2)
            j = int(np.argmax(s))
            out.append(t - inn_year[j] if s[j] >= thr and inn_ids[j] != x else np.nan)
    return np.array(out, dtype=float)


tempo_rows, ent_rows = [], []
for T in range(2019, 2023):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    for when, (y0, y1) in (("now", (T - 2, T)), ("later", (T + 1, T + 3))):
        pids = [x for x in lab.index if x in PP and y0 <= year[x] <= y1]
        a = ages(PV[[PP[x] for x in pids]], pids, lambda x: year[x], MATCH)
        d = pd.DataFrame({"X": lab.reindex(pids).values, "age": a}).dropna()
        for X, g in d.groupby("X"):
            if len(g) >= 8:
                tempo_rows.append({"T": T, "X": X, "when": when, "tempo": float((g.age >= 3).mean()), "n": len(g)})
    e = E[E["T"] == T]
    pids = [x for x in e.paper_id.unique() if x in MP]
    a = ages(MV[[MP[x] for x in pids]], pids, lambda x: T, C.THR_REUSE)
    src = dict(zip(pids, a))
    for r in e.itertuples():
        ent_rows.append({**r._asdict(), "source_age": src.get(r.paper_id, np.nan)})
    print(T, flush=True)
TP = pd.DataFrame(tempo_rows)
W = TP.pivot_table(index=["T", "X"], columns="when", values="tempo").dropna().reset_index()
rho = [spearmanr(g.now, g.later).statistic for _, g in W.groupby("T") if len(g) >= 5]
rng = np.random.default_rng(0)
bs = []
for _ in range(500):
    s = W.iloc[rng.integers(0, len(W), len(W))]
    r_ = [spearmanr(g.now, g.later).statistic for _, g in s.groupby("T") if len(g) >= 5 and g.now.nunique() > 1]
    if r_:
        bs.append(np.nanmean(r_))
stab = pd.DataFrame([{"community-cutoffs": len(W), "mean rank correlation, tempo now vs 3 years later": float(np.mean(rho)),
                      "90% CI": f"[{np.nanpercentile(bs, 5):+.3f}, {np.nanpercentile(bs, 95):+.3f}]",
                      "median share of 3+ year-old matches, now": float(W.now.median()),
                      "median share, 3 years later": float(W.later.median()),
                      "spread across communities (IQR)": float(W.now.quantile(0.75) - W.now.quantile(0.25))}])

D = pd.DataFrame(ent_rows).drop(columns=["Index"])
tnow = TP[TP.when == "now"].set_index(["T", "X"]).tempo
D["tempo"] = [tnow.get((t, x), np.nan) for t, x in zip(D["T"], D.X)]
D = D.dropna(subset=["source_age", "tempo"]).reset_index(drop=True)
D["fast"] = D.tempo <= D.groupby("T").tempo.transform("median")
D["recent_source"] = D.source_age <= D.groupby("T").source_age.transform("median")
STRATS = ["method import", "new problem for X", "bridge"]


def design(d):
    X = pd.DataFrame({s: (d.strategy == s).astype(float) for s in STRATS})
    X["recent source"] = d.recent_source.astype(float)
    X["fast community"] = d.fast.astype(float)
    X["recent x fast"] = X["recent source"] * X["fast community"]
    X["insider co-author"] = d.insider.astype(float)
    X["newcomer to the map"] = d.newcomer.astype(float)
    X["log team"] = np.log1p(d.team)
    X["log community size"] = np.log(d["size"])
    X["forward route"] = d.forward.astype(float)
    X["log in-map references"] = np.log1p(d.n_inmap_refs)
    for T in sorted(d["T"].unique())[1:]:
        X[f"T{T}"] = (d["T"] == T).astype(float)
    return X


def effects(d, y):
    X, yy = design(d), d[y].astype(int)
    m = LogisticRegression(max_iter=3000).fit(X, yy)
    out = {}
    for fast in (1.0, 0.0):
        Z1, Z0 = X.copy(), X.copy()
        Z1["fast community"] = Z0["fast community"] = fast
        Z1["recent source"], Z0["recent source"] = 1.0, 0.0
        Z1["recent x fast"], Z0["recent x fast"] = fast, 0.0
        out["fast" if fast else "slow"] = m.predict_proba(Z1)[:, 1].mean() - m.predict_proba(Z0)[:, 1].mean()
    out["fast - slow"] = out["fast"] - out["slow"]
    return out


res = []
for y, d in (("stayed", D), ("taken_up", D.drop_duplicates("paper_id")), ("moved_towards", D.drop_duplicates("paper_id"))):
    d = d.dropna(subset=[y]).reset_index(drop=True)
    est = effects(d, y)
    groups = list(d.groupby(["T", "X"]).indices.values())
    bsx = []
    for _ in range(300):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        try:
            bsx.append(effects(d.iloc[i].reset_index(drop=True), y))
        except ValueError:
            pass
    for k in ("fast", "slow", "fast - slow"):
        b = np.array([x[k] for x in bsx])
        res.append({"outcome": y, "effect of a recent source": k, "entries": len(d), "adjusted difference": est[k],
                    "90% CI": f"[{np.percentile(b, 5):+.3f}, {np.percentile(b, 95):+.3f}]"})
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(stab.round(3).T.to_string())
print(R.round(3).to_string(index=False))
C.save(RUN, "exp26_tempo", "stability", stab, title="Tempo: is a community's pace a stable trait?",
       how_to_read="Tempo = median age of the closest earlier idea (innovation statement) to each of a community's "
       "problems from the last three years: young = its questions are best matched by recent ideas. A positive rank "
       "correlation between tempo now and three years later, with an interval above zero, means pace is a trait worth "
       "showing. Ages are bounded by the map, so compare communities rather than reading them as years.")
C.save(RUN, "exp26_tempo", "payoff", R, title="Tempo: does building on recent work pay more in fast communities?",
       how_to_read="Entries (experiment 12) whose method's closest earlier source is recent (below the median age) "
       "against those building on older sources, in fast (tempo below the median) and slow communities. Adjusted "
       "differences hold experiment 12's controls fixed; *fast - slow* is the interaction. Intervals resample communities.")
