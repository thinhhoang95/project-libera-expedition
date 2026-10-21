"""Experiment 10: which communities will be warm to a paper?

The reverse of experiment 1. For each candidate paper i at cutoff T, rank the problem communities X
that do not cite it yet (alpha 0.5, medium) and check which ones cite it in T+1..T+3 (and, as a second
outcome, which ones *realize* it: their later work moves towards i's innovation, exp9).

Signals per (i, X), all available at T: coupling (shared references), problem similarity (i's
innovation vs X's problems), limitation similarity, recent closeness (X's recent methods and
innovations vs i's innovation), community size. A logistic model of all of them is fitted
leave-one-cutoff-out. Scored *within each paper*: mean AUC across communities, hit@3 (a top-3
community cites it) and the reciprocal rank of the first warm community, over papers with at least one
warm community. Fits the production model on all cutoffs to data/projects/<run>/backtest/community_warmth_model.json;
scripts/phase8b/fit_models.py puts it into production for the run (pipeline step fit-models).
Needs exp9_realized.py output. Usage: --run-id (default dro-pilot).
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from landscape import experiments

RUN = experiments.run_arg()
d = pd.read_parquet(experiments.work_dir(RUN) / "exp9_pairs.parquet")
d["l_size"] = np.log(d.X_size)
d["sim_recent"] = d.sim_recent.clip(lower=-1)
SIGNALS = {"community size": ["l_size"], "coupling": ["lcoup"], "problem similarity": ["prob_sim"],
           "limitation similarity": ["lim_sim3_f"], "recent closeness": ["sim_recent"],
           "semantic (problem + limitation + recent)": ["prob_sim", "lim_sim3_f", "sim_recent"],
           "all": ["l_size", "lcoup", "prob_sim", "lim_sim3_f", "sim_recent"]}
FEATS = SIGNALS["all"]


def loco(feats, y):
    out = pd.Series(np.nan, index=d.index)
    for T in sorted(d["T"].unique()):
        tr, te = d["T"] != T, d["T"] == T
        s = StandardScaler().fit(d.loc[tr, feats])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s.transform(d.loc[tr, feats]), d.loc[tr, y])
        out[te] = m.decision_function(s.transform(d.loc[te, feats]))
    return out


def within(score, y):
    auc, hit3, rr = [], [], []
    for _, g in d.assign(_s=score).groupby(["T", "i"]):
        if not 0 < g[y].sum() < len(g):
            continue
        auc.append(roc_auc_score(g[y], g._s))
        r = g.sort_values("_s", ascending=False)[y].to_numpy()
        hit3.append(r[:3].max())
        rr.append(1 / (np.argmax(r) + 1))
    return np.mean(auc), np.mean(hit3), np.mean(rr), len(auc)


rows = []
rng = np.random.default_rng(0)
for y in ("hit", "realized"):
    a, h, r, n = within(pd.Series(rng.random(len(d)), index=d.index), y)
    rows.append({"outcome": y, "ranking": "random", "mean AUC within paper": a, "hit@3": h, "MRR": r, "papers": n})
    for name, f in SIGNALS.items():
        s = loco(f, y) if len(f) > 1 or name == "all" else d[f[0]]
        a, h, r, n = within(s, y)
        rows.append({"outcome": y, "ranking": name, "mean AUC within paper": a, "hit@3": h, "MRR": r, "papers": n})
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
experiments.save(RUN, "exp10_warmth", "rankings", R, title="Community warmth: which communities take up a paper",
                 how_to_read="For each paper at cutoff T, the communities that do not cite it yet are ranked; a "
                 "community is **warm** if it cites the paper in T+1..T+3 (*hit*) or its later work moves towards the "
                 "paper's innovation (*realized*, exp9). Scored within each paper with at least one warm community: "
                 "**mean AUC** across communities, **hit@3** (a top-3 community is warm) and **MRR** (1 / rank of the "
                 "first warm one). Multi-feature rankings are logistic models fitted leave-one-cutoff-out.")

# production model: fitted on all cutoffs, citation outcome
s = StandardScaler().fit(d[FEATS])
m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s.transform(d[FEATS]), d.hit)
best = R[(R.outcome == "hit") & (R.ranking == "all")].iloc[0]
spec = {"name": "community_warmth", "features": FEATS, "mean": s.mean_.round(6).tolist(),
        "scale": s.scale_.round(6).tolist(), "coef": m.coef_[0].round(6).tolist(), "intercept": float(m.intercept_[0]),
        "trained_on": sorted(int(t) for t in d["T"].unique()), "n_rows": int(len(d)), "n_positive": int(d.hit.sum()),
        "note": "P(problem community X cites paper i within 3 years | not citing it yet), ranked within a paper. "
                f"Leave-one-cutoff-out: mean within-paper AUC {best['mean AUC within paper']:.3f}, hit@3 "
                f"{best['hit@3']:.3f}, MRR {best['MRR']:.3f} (experiment 10).",
        "feature_definitions": {
            "l_size": "log(number of papers in X)",
            "lcoup": "log1p(100 * shared references between i and X's papers / size of X)",
            "prob_sim": "cosine(i innovation, X's mean problem vector), 256-d",
            "lim_sim3_f": "mean of the 3 highest cosines(i innovation, X's non-generic limitations), else prob_sim",
            "sim_recent": "max cosine(i innovation, method/innovation of X's papers from the last 3 years)"}}
spec["run"] = RUN
out = experiments.work_dir(RUN) / "community_warmth_model.json"  # staged; fit_models.py puts it in production
out.write_text(json.dumps(spec, indent=2) + "\n")
print("saved", out, dict(zip(FEATS, spec["coef"])))
