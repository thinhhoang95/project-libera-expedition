"""Experiment 9: did the opportunity happen, whether or not anyone cited it?

Citation adoption (exp1, exp7) measures a social act. Here the outcome is the content itself: for a
candidate pair (problem community X, innovation paper i) from experiment 1, the opportunity counts as
**realized** when a paper of X published in T+1..T+3 has a method or innovation statement close to i's
innovation (above the 99th percentile of similarity between random innovation-method pairs) *and*
closer than any of X's papers from T-2..T were (the community moved towards i). **Realized without
citation** additionally requires that no X paper cites i in T+1..T+3.

The models of experiment 7 (citations, coupling, semantic) are scored leave-one-cutoff-out against
the citation hit, the realized outcome and the realized-without-citation outcome.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a copy of the database.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from landscape import embed, experiments
from landscape.backtest import Assigner
from landscape.cli import _params

RUN = experiments.run_arg()
WORK = experiments.work_dir(RUN)
con = experiments.connect_ro()
P = _params(con, RUN)
VER = P.embedding.version
d = pd.read_parquet(WORK / "exp1_pairs.parquet")
papers = con.execute("""SELECT v.paper_id, p.year FROM provenance v JOIN papers p USING (paper_id)
                        WHERE v.run_id = ? AND p.year IS NOT NULL""", [RUN]).df()
year = papers.set_index("paper_id").year


def load(f):
    ids, v = embed.load_vectors(f, ids=papers.paper_id.tolist(), version=VER)
    return {x: k for k, x in enumerate(ids)}, embed.truncate(v, 256)


IN, ME = load("innovation"), load("method")
rng = np.random.default_rng(0)
a = rng.choice(len(IN[1]), 4000)
b = rng.choice(len(ME[1]), 4000)
THR = float(np.quantile(np.sum(IN[1][a] * ME[1][b], axis=1), 0.99))
print(f"realization threshold (99th pct of random innovation-method cosine): {THR:.3f}")


def stack(ids):
    """Method and innovation vectors of a set of papers, stacked."""
    out = [V[[P_[x] for x in ids if x in P_]] for P_, V in ((ME[0], ME[1]), (IN[0], IN[1]))]
    return np.vstack(out) if sum(len(o) for o in out) else np.zeros((0, 256), np.float32)


parts = []
for T, g in d.groupby("T"):
    A = Assigner(con, f"{RUN}@T{T}", VER)
    lab = A.assign("problem", 0.5, papers.paper_id.tolist())
    yr = lab.index.map(year)
    fut = lab[(yr > T) & (yr <= T + 3)]
    rec = lab[(yr >= T - 2) & (yr <= T)]
    g = g.copy()
    g["sim_future"], g["sim_recent"] = np.nan, np.nan
    for X, gx in g.groupby("X"):
        F, Rv = stack(fut.index[fut.values == X]), stack(rec.index[rec.values == X])
        cand = [i for i in gx.i if i in IN[0]]
        if not cand or not len(F):
            continue
        C = IN[1][[IN[0][i] for i in cand]]
        sf = pd.Series((C @ F.T).max(axis=1), index=cand)
        sr = pd.Series((C @ Rv.T).max(axis=1) if len(Rv) else np.full(len(cand), -1.0), index=cand)
        g.loc[gx.index, "sim_future"] = gx.i.map(sf).to_numpy()
        g.loc[gx.index, "sim_recent"] = gx.i.map(sr).to_numpy()
    parts.append(g)
    print(T, len(g), flush=True)
d = pd.concat(parts)
d = d.dropna(subset=["sim_future"])
d["realized"] = ((d.sim_future >= THR) & (d.sim_future > d.sim_recent)).astype(int)
d["realized_uncited"] = (d.realized.astype(bool) & ~d.hit.astype(bool)).astype(int)
d["hit"] = d.hit.astype(int)
d["lpop"], d["age"] = np.log1p(d["pop"]), d["T"] - d.i_year
d["lcoup"] = np.log1p(d.coupling * 100)
d["lim_sim3_f"] = d.lim_sim3.fillna(d.prob_sim)
d.to_parquet(WORK / "exp9_pairs.parquet", index=False)

MODELS = {"citations + age": ["lpop", "age"], "+ coupling": ["lpop", "age", "lcoup"],
          "semantic only": ["prob_sim", "lim_sim3_f"], "citations + age + semantic": ["lpop", "age", "prob_sim", "lim_sim3_f"],
          "all": ["lpop", "age", "lcoup", "prob_sim", "lim_sim3_f"]}


def loco(feats, y):
    out = pd.Series(np.nan, index=d.index)
    for T in sorted(d["T"].unique()):
        tr, te = d["T"] != T, d["T"] == T
        s = StandardScaler().fit(d.loc[tr, feats])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s.transform(d.loc[tr, feats]), d.loc[tr, y])
        out[te] = m.decision_function(s.transform(d.loc[te, feats]))
    return out


def p_at(s, y, k=10):
    v = [g.nlargest(k, "_s")[y].mean() for _, g in d.assign(_s=s).groupby(["T", "X"]) if g[y].sum()]
    return np.mean(v) if v else np.nan


rows = []
for y in ("hit", "realized", "realized_uncited"):
    for k, f in MODELS.items():
        s = loco(f, y)
        rows.append({"outcome": y, "model": k, "AUC": roc_auc_score(d[y], s), "p@10": p_at(s, y),
                     "positives": int(d[y].sum()), "base_rate": d[y].mean()})
R = pd.DataFrame(rows)
overlap = pd.crosstab(d.hit, d.realized)
print(overlap)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
experiments.save(RUN, "exp9_realized", "models", R, title="Realized opportunities: content, not citations",
                 how_to_read=f"Candidate pairs from experiment 1 (community X, innovation paper i not yet cited by X). "
                 f"**hit**: an X paper cites i in T+1..T+3 (as before). **realized**: a later X paper's method or "
                 f"innovation is close to i's innovation (cosine >= {THR:.3f}, the 99th percentile of random pairs) "
                 f"and closer than any of X's papers from T-2..T: the community moved towards the idea. "
                 f"**realized_uncited**: realized, but no X paper cites i. Leave-one-cutoff-out; p@10 per community "
                 f"with at least one positive. If coupling only predicts citing, it should fade on the realized outcomes.")
experiments.save(RUN, "exp9_realized", "overlap", overlap.reset_index().rename(columns={0: "not realized", 1: "realized"}),
                 title="Realized opportunities: overlap with citation adoption",
                 how_to_read="Rows: cited in T+1..T+3 (hit 0/1). Columns: realized (0/1). How often the two outcomes agree.")
