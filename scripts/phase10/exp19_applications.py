"""Experiment 19: applications not yet tried -- which empty problem x application cells get filled?

Experiment 4's recipe on problem community x application domain. For each cutoff T (2019-2022): problem communities
of the cutoff-T run (alpha 0.5, medium); application domains = k-means (k=15) on the domain_data vectors of papers up
to T (papers whose abstract states no application have none). Every paper with a domain statement is placed in
both. Cells empty at T; hit = at least 2 papers in T+1..T+3 (outcome papers placed with the same centroids).
Rankings (no fitting on outcomes): item CF on the problem x domain matrix (domains used by the same problems), the
neighbour share (domains used by the 10 problems with the most similar domain profile), popularity (row x column
totals), semantic fit (problem community centroid vs domain centroid), and a rank average of item CF and fit.
AUC and p@20 among empty cells, averaged over cutoffs.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import common as C

RUN = C.run_arg()
K_DOM = 15
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
DP, DV = C.vectors("domain_data", ids)
PP, PV = C.vectors("problem", ids)
dom_ids = [x for x in DP if x in year.index]
dom_year = np.array([year[x] for x in dom_ids])
dlabels = dict(con.execute("SELECT paper_id, text FROM facets WHERE facet = 'domain_data'").fetchall())

rows, examples = [], []
for T in range(2019, 2023):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    pre = np.where(dom_year <= T)[0]
    km = KMeans(K_DOM, n_init=10, random_state=0).fit(DV[pre])
    cent = C.unit(km.cluster_centers_)
    dom_of = pd.Series(np.argmax(DV @ cent.T, axis=1), index=dom_ids)
    P = sorted(set(lab))
    pix = {p: k for k, p in enumerate(P)}
    M = np.zeros((len(P), K_DOM))
    F = np.zeros((len(P), K_DOM))
    for x in dom_ids:
        if x in lab.index:
            if year[x] <= T:
                M[pix[lab[x]], dom_of[x]] += 1
            elif year[x] <= T + 3:
                F[pix[lab[x]], dom_of[x]] += 1
    L = np.log1p(M)
    Ln = L / np.maximum(np.linalg.norm(L, axis=0, keepdims=True), 1e-9)
    Sd = Ln.T @ Ln
    np.fill_diagonal(Sd, 0)
    cf = L @ Sd
    Lp = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-9)
    Sp = Lp @ Lp.T
    np.fill_diagonal(Sp, -np.inf)
    nn = np.argsort(-Sp, axis=1)[:, :10]
    neigh = np.array([[(M[nn[i], d] > 0).mean() for d in range(K_DOM)] for i in range(len(P))])
    pop = np.outer(M.sum(1), M.sum(0))
    pc = np.vstack([C.unit(np.mean([PV[PP[x]] for x in lab.index[lab.values == p] if x in PP and year[x] <= T], axis=0))
                    for p in P])
    fit = pc @ cent.T
    empty = M == 0
    y = (F >= 2)[empty]
    if y.sum() == 0:
        continue
    scores = {"item CF": cf, "neighbour share": neigh, "popularity": pop, "semantic fit": fit}
    scores["item CF + semantic fit (rank average)"] = np.zeros_like(cf)
    scores["item CF + semantic fit (rank average)"][empty] = rankdata(cf[empty]) + rankdata(fit[empty])
    for name, S in scores.items():
        s = S[empty]
        top = np.argsort(-s)[:20]
        rows.append({"T": T, "ranking": name, "AUC": roc_auc_score(y, s), "p@20": y[top].mean(), "base": y.mean(),
                     "empty cells": int(empty.sum()), "filled": int(y.sum())})
    # examples of filled cells for the write-up: problem label, domain exemplar
    labels = dict(con.execute("SELECT cluster_id, label FROM clusters WHERE run_id = ?", [f"{RUN}@T{T}"]).fetchall())
    ex = [(P[i], d) for i, d in zip(*np.where(empty & (F >= 2)))]
    for p, d in ex[:5]:
        members = [x for x in dom_ids if dom_of[x] == d and year[x] <= T][:1]
        examples.append({"T": T, "problem community": labels.get(p, p), "new application (example statement)":
                         (dlabels.get(members[0]) or "")[:120] if members else "", "new papers": int(F[pix[p], d])})
    print(T, "empty", int(empty.sum()), "filled", int(y.sum()), flush=True)

R = pd.DataFrame(rows).groupby("ranking")[["AUC", "p@20", "base", "empty cells", "filled"]].mean().reset_index() \
    .sort_values("AUC", ascending=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
C.save(RUN, "exp19_applications", "rankings", R, title="Untried applications: which empty problem x domain cells fill",
       how_to_read="Problem communities x application domains (15 clusters of the domain statements up to T) at each "
       "cutoff (2019-2022). Cells empty at T; a hit is at least 2 papers in the cell in the next three years. **AUC** "
       "among empty cells and **p@20** (hit share of the top 20) averaged over cutoffs; **base** = hit rate, the p@20 of "
       "a random ranking. No ranking is fitted on outcomes.")
C.save(RUN, "exp19_applications", "examples", pd.DataFrame(examples), title="Untried applications: cells that filled",
       how_to_read="Examples of empty problem x application cells that received at least 2 papers in the next three years.")
