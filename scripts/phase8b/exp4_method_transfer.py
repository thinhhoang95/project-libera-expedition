"""Experiment 4: method transfer as matrix completion (problem x method), at tag and cluster level.

Empty cells at T; hit = the cell gets >= 1 (and, stricter, >= 2) papers in T+1..T+3. Scores:
recipe-style neighbour indicator, item-based collaborative filtering, low-rank SVD completion,
and a popularity baseline. Unsupervised, so no fitting on the test cutoff.
Usage: --run-id (default dro-pilot); tables are saved to data/projects/<run>/backtest/experiments/.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from landscape import experiments, store
from landscape.backtest import Assigner
from landscape.cli import _params

BASE = experiments.run_arg()
con = experiments.connect_ro()
p = _params(con, BASE)
papers = con.execute("""SELECT v.paper_id, p.year FROM provenance v JOIN papers p USING (paper_id)
                        WHERE v.run_id = ?""", [BASE]).df()
year = papers.set_index("paper_id").year
tags = con.execute("""
    SELECT f.paper_id, f.facet, coalesce(v.canonical, lower(trim(f.text))) AS tag FROM facets f
    JOIN provenance pv ON pv.paper_id = f.paper_id AND pv.run_id = ?
    LEFT JOIN tag_vocab v ON v.run_id = pv.run_id AND v.facet = f.facet AND v.tag = lower(trim(f.text))
    WHERE f.facet IN ('problem_class', 'method_family')""", [BASE]).df()
tags = tags.pivot_table(index="paper_id", columns="facet", values="tag", aggfunc="first").dropna()
tags["year"] = tags.index.map(year)


def cosine_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1
    X = M / n
    return X @ X.T


def scores_for(M):
    """M: problems x methods counts (<= T). Returns dict of score matrices."""
    L = np.log1p(M)
    out = {}
    out["popularity (row x col)"] = np.outer(M.sum(1), M.sum(0))
    # recipe-style: share of the 10 most similar problems (by method profile) that use m
    Sp = cosine_rows(L)
    np.fill_diagonal(Sp, -np.inf)
    nn = np.argsort(-Sp, axis=1)[:, :10]
    w = np.clip(np.take_along_axis(Sp, nn, axis=1), 0, None)
    use = (M[nn] > 0)                                   # problems x 10 x methods
    out["recipe neighbour share"] = (w[:, :, None] * use).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
    # item-based CF: methods similar to the ones this problem already uses
    Sm = cosine_rows(L.T)
    np.fill_diagonal(Sm, 0)
    out["item CF"] = L @ Sm
    # low-rank completion
    for r in (5, 10, 20):
        U, s, Vt = np.linalg.svd(L, full_matrices=False)
        r = min(r, len(s))
        out[f"SVD rank {r}"] = (U[:, :r] * s[:r]) @ Vt[:r]
    # hybrid: average of rank-normalised popularity and item CF over empty cells
    from scipy.stats import rankdata
    empty = M == 0
    for a, b, name in (("popularity (row x col)", "item CF", "hybrid: popularity + item CF"),
                       ("popularity (row x col)", "SVD rank 5", "hybrid: popularity + SVD 5")):
        H = np.full(M.shape, -np.inf)
        H[empty] = rankdata(out[a][empty]) + rankdata(out[b][empty])
        out[name] = H
    return out


def evaluate(level, rows_by_T):
    res = []
    for T, (M, fut, min_hit) in rows_by_T.items():
        sc = scores_for(M)
        empty = M == 0
        for strict in (1, 2):
            y = (fut >= strict)[empty]
            if y.sum() == 0:
                continue
            for name, S in sc.items():
                s = S[empty]
                top = np.argsort(-s)[:50]
                res.append({"level": level, "T": T, "hit": f">= {strict} paper", "ranking": name,
                            "AUC": roc_auc_score(y, s), "p@50": y[top].mean(), "base": y.mean(), "n_hits": int(y.sum()),
                            "n_empty": int(empty.sum())})
    return pd.DataFrame(res)


tag_data, cl_data = {}, {}
for T in (2020, 2021, 2022, 2023):
    past = tags[tags.year <= T]
    fut = tags[(tags.year > T) & (tags.year <= T + 3)]
    pc = past.problem_class.value_counts()
    mf = past.method_family.value_counts()
    P, Mth = pc[pc >= 3].index.tolist(), mf[mf >= 3].index.tolist()
    M = pd.crosstab(past.problem_class, past.method_family).reindex(index=P, columns=Mth, fill_value=0).values.astype(float)
    F = pd.crosstab(fut.problem_class, fut.method_family).reindex(index=P, columns=Mth, fill_value=0).values
    tag_data[T] = (M, F, 1)
    A = Assigner(con, f"{BASE}@T{T}", p.embedding.version)
    pa = A.assign("problem", 0.5, papers.paper_id.tolist())
    ma = A.assign("method", 0.5, papers.paper_id.tolist())
    both = pd.DataFrame({"P": pa, "m": ma}).dropna()
    both["year"] = both.index.map(year)
    Pc, mc = sorted(both.P.unique()), sorted(both.m.unique())
    Mc = pd.crosstab(both[both.year <= T].P, both[both.year <= T].m).reindex(index=Pc, columns=mc, fill_value=0).values.astype(float)
    fb = both[(both.year > T) & (both.year <= T + 3)]
    Fc = pd.crosstab(fb.P, fb.m).reindex(index=Pc, columns=mc, fill_value=0).values
    cl_data[T] = (Mc, Fc, 2)
r = pd.concat([evaluate("tags", tag_data), evaluate("clusters (alpha 0.5)", cl_data)])
pd.set_option("display.width", 200)
print(r.groupby(["level", "hit", "ranking"])[["AUC", "p@50", "base", "n_hits", "n_empty"]].mean().round(3).to_string())
r.to_parquet(experiments.work_dir(BASE) / "exp4_transfer.parquet")
experiments.save(BASE, "exp4_method_transfer", "rankings",
                 r.groupby(["level", "hit", "ranking"])[["AUC", "p@50", "base", "n_hits", "n_empty"]].mean().reset_index(),
                 title="Method transfer as matrix completion",
                 how_to_read="Empty problem x method cells at cutoff T; a hit is the cell receiving at least 1 (or, "
                 "the recipe's definition, at least 2) papers in T+1..T+3. Numbers are averaged over cutoffs. "
                 "**base** is the hit rate among empty cells, the p@50 of a random ranking; **n_hits** shows how "
                 "few hits there are, so p@50 gaps of a few points are one hit or less. Cluster level uses the "
                 "alpha 0.5 medium clusters; tag level the extracted problem_class x method_family tags.")
