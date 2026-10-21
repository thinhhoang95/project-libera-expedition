"""Experiments 22 and 23: the context of an entry into a community (uses experiment 12's entries and controls).

22. How the question is asked now. For an entry x into community X at T (2020-2022): **currency** = cosine of x's
    problem statement to X's *recent* formulation (centroid of X's other papers from T-1..T) minus its cosine to X's
    *earlier* formulation (centroid of T-4..T-2). Does asking the question the current way predict the outcomes?
23. Crowding vs demand. For an entry x at T (2019-2022), counting papers by other authors:
        same question   papers from T-2..T whose problem is within the 99th percentile of random problem pairs of x's
        near duplicate  of those, papers whose method is also within the 99th percentile of random method pairs
        demand          stated limitations (papers up to T) within the 99th percentile of random innovation-limitation
                        pairs of x's innovation (what the field says it is missing)
Outcomes and controls as experiment 12 (stayed; taken up by others; the community moved towards it). Effects are the
adjusted difference between the 75th and the 25th percentile of each measure (or present vs absent for counts that
are mostly zero), with 90% intervals resampling communities.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
PP, PV = C.vectors("problem", ids)
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
LP, LV = C.vectors("limitations", ids)
E = pd.read_parquet(C.project(RUN, "backtest") / "exp12_entries.parquet")
rng = np.random.default_rng(0)


def q99(A, B, n=4000):
    i, j = rng.integers(0, len(A), n), rng.integers(0, len(B), n)
    return float(np.quantile(np.sum(A[i] * B[j], axis=1), 0.99))


QP, QM, QIL = q99(PV, PV), q99(MV, MV), q99(IV, LV)
p_ids = [x for x in PP]
p_year = np.array([year.get(x, 9999) for x in p_ids])
l_ids = [x for x in LP]
l_year = np.array([year.get(x, 9999) for x in l_ids])

feat = {}
for T in sorted(E["T"].unique()):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    e = E[E["T"] == T].drop_duplicates("paper_id")
    win = np.where((p_year >= T - 2) & (p_year <= T))[0]
    lim = np.where(l_year <= T)[0]
    for x, X in zip(e.paper_id, e.X):
        if x not in PP:
            continue
        tm = team.get(x, frozenset())
        sp = PV[win] @ PV[PP[x]]
        same = [p_ids[k] for k, s in zip(win, sp) if s >= QP and p_ids[k] != x and not (team.get(p_ids[k], frozenset()) & tm)]
        dup = [p for p in same if p in MP and x in MP and float(MV[MP[p]] @ MV[MP[x]]) >= QM]
        demand = int((LV[lim] @ IV[IP[x]] >= QIL).sum()) if x in IP and len(lim) else 0
        mem = lab.index[lab.values == X]
        now = [p for p in mem if T - 1 <= year[p] <= T and p != x and p in PP]
        then = [p for p in mem if T - 4 <= year[p] <= T - 2 and p in PP]
        cur = np.nan
        if len(now) >= 5 and len(then) >= 5:
            cn, ct = C.unit(PV[[PP[p] for p in now]].mean(0)), C.unit(PV[[PP[p] for p in then]].mean(0))
            cur = float(PV[PP[x]] @ cn - PV[PP[x]] @ ct)
        feat[(T, x)] = {"same_question": len(same), "near_duplicate": len(dup), "demand": demand, "currency": cur}
    print(T, "entries", len(e), flush=True)

F = pd.DataFrame([{"T": T, "paper_id": x, **v} for (T, x), v in feat.items()])
D = E.merge(F, on=["T", "paper_id"])
D.to_parquet(C.project(RUN, "backtest") / "exp22_23_entries.parquet", index=False)


def controls(d):
    X = pd.DataFrame({"insider": d.insider.astype(float), "newcomer": d.newcomer.astype(float), "log team": np.log1p(d.team),
                      "log size": np.log(d["size"]), "forward": d.forward.astype(float), "log refs": np.log1p(d.n_inmap_refs)})
    for s in ("method import", "new problem for X", "bridge"):
        X[s] = (d.strategy == s).astype(float)
    for T in sorted(d["T"].unique())[1:]:
        X[f"T{T}"] = (d["T"] == T).astype(float)
    return X


def effect(d, y, f, lo, hi):
    """Adjusted difference between hi and lo of measure f. The measure is standardized before fitting: on its raw
    scale (currency differences are ~0.01) the regularized model would shrink its coefficient to nothing."""
    X = controls(d)
    mu, sd = float(d[f].mean()), float(d[f].std() or 1.0)
    X[f] = (d[f].astype(float) - mu) / sd
    m = LogisticRegression(max_iter=3000).fit(X, d[y].astype(int))
    a, b = X.copy(), X.copy()
    a[f], b[f] = (hi - mu) / sd, (lo - mu) / sd
    return m.predict_proba(a)[:, 1].mean() - m.predict_proba(b)[:, 1].mean()


res = []
MEAS = {"currency": "quartiles", "same_question": "quartiles", "near_duplicate": "any", "demand": "any"}
for f, how in MEAS.items():
    for y, unit in (("stayed", "author"), ("taken_up", "paper"), ("moved_towards", "paper")):
        d = D.dropna(subset=[f, y])
        d = d if unit == "author" else d.drop_duplicates("paper_id")
        d = d.reset_index(drop=True)
        if how == "quartiles":
            lo, hi = d[f].quantile([0.25, 0.75]).to_numpy()
            label = "75th vs 25th percentile"
        else:
            d[f] = (d[f] > 0).astype(float)
            lo, hi, label = 0.0, 1.0, "present vs absent"
        est = effect(d, y, f, lo, hi)
        groups = list(d.groupby(["T", "X"]).indices.values())
        rs = np.random.default_rng(0)
        bs = []
        for _ in range(300):
            i = np.concatenate([groups[j] for j in rs.integers(0, len(groups), len(groups))])
            try:
                bs.append(effect(d.iloc[i].reset_index(drop=True), y, f, lo, hi))
            except ValueError:
                pass
        res.append({"experiment": "22 currency" if f == "currency" else "23 crowding / demand", "measure": f,
                    "comparison": label, "outcome": y, "adjusted difference": est,
                    "90% CI": f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]",
                    "entries": len(d), "share with measure > 0": float((D[f] > 0).mean()) if how == "any" else np.nan})
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
C.save(RUN, "exp22_23_entry_context", "results", R, title="Entering a community: current formulation, crowding and demand",
       how_to_read="Experiment 12's entries, with its controls and strategies held fixed. **currency** (experiment 22): "
       "how much closer the entry's problem statement is to the community's formulation of the last two years than to "
       "its formulation of three to four years before. **same_question** (experiment 23): recent papers by others posing "
       "the same question; **near_duplicate**: of those, the ones using the same method too; **demand**: stated "
       "limitations the entry's innovation answers. Adjusted differences in the outcome rates (stayed; taken up by "
       "others; the community moved towards it), with 90% intervals resampling communities.")
