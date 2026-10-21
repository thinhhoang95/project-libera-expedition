"""Experiment 1: paper-level bridges. Which not-yet-cited innovation papers will problem cluster X adopt?

Candidates at cutoff T: (X, i) with i an innovation paper <= T outside X that no X paper has cited
by T. Hit: an X paper published in T+1..T+3 cites i. Writes data/projects/<run>/backtest/exp1_pairs.parquet. Usage: --run-id (default dro-pilot).
"""
import numpy as np
import pandas as pd
from scipy import sparse

from landscape import embed, experiments, store
from landscape.backtest import Assigner
from landscape.cli import _params

BASE, ALPHA = experiments.run_arg(), 0.5
con = experiments.connect_ro()
p = _params(con, BASE)
ver = p.embedding.version
papers = con.execute("""SELECT v.paper_id, p.year, coalesce(p.cited_by_count, 0) cbc FROM provenance v
                        JOIN papers p USING (paper_id) WHERE v.run_id = ?""", [BASE]).df()
year = papers.set_index("paper_id").year
refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v
                      ON v.paper_id = r.citing_id AND v.run_id = ?""", [BASE]).df()
pidx = {p_: k for k, p_ in enumerate(papers.paper_id)}
widx = {w: k for k, w in enumerate(refs.cited_id.unique())}
R = sparse.csr_matrix((np.ones(len(refs)), (refs.citing_id.map(pidx), refs.cited_id.map(widx))),
                      shape=(len(pidx), len(widx)))
R.data[:] = 1
refs["cy"] = refs.citing_id.map(year)
iid_all, iv_all = embed.load_vectors("innovation", version=ver)
lid_all, lv_all = embed.load_vectors("limitations", version=ver)
qid_all, qv_all = embed.load_vectors("problem", version=ver)
ipos = {x: k for k, x in enumerate(iid_all)}

rows = []
for T in (2020, 2021, 2022, 2023):
    run = f"{BASE}@T{T}"
    A = Assigner(con, run, ver)
    lab = A.assign("problem", ALPHA, papers.paper_id.tolist())          # every paper -> X
    past = lab[lab.index.map(year) <= T]
    fut = lab[(lab.index.map(year) > T) & (lab.index.map(year) <= T + 3)]
    # citations so far / future, from corpus papers
    cited_by_X_past = refs[refs.cy <= T].assign(X=lambda d: d.citing_id.map(lab)).dropna(subset=["X"]) \
        .groupby("X").cited_id.apply(set)
    fut_cites = refs[(refs.cy > T) & (refs.cy <= T + 3)].assign(X=lambda d: d.citing_id.map(lab)).dropna(subset=["X"]) \
        .groupby("X").cited_id.apply(set)
    pop = refs[refs.cy <= T].groupby("cited_id").size()
    # innovation candidates <= T
    cand = [x for x in past.index if x in ipos]
    cv = iv_all[[ipos[x] for x in cand]]
    cX = past[cand].values
    cyr = np.array([year[x] for x in cand])
    # X's limitation and problem vectors (<= T)
    lpos = {x: k for k, x in enumerate(lid_all)}
    qpos = {x: k for k, x in enumerate(qid_all)}
    # cluster x works matrix for bibliographic coupling
    clusters = sorted(past.unique())
    cl_ix = {c: k for k, c in enumerate(clusters)}
    M = sparse.csr_matrix((np.ones(len(past)), ([cl_ix[c] for c in past.values], [pidx[x] for x in past.index])),
                          shape=(len(clusters), len(pidx)))
    XR = M @ R  # cluster x works: how many X papers cite each work
    size = np.asarray(M.sum(axis=1)).ravel()
    RC = R[[pidx[x] for x in cand]]
    coupling = (XR @ RC.T).toarray() / size[:, None]          # clusters x candidates
    gc = lv_all.mean(axis=0); gc /= np.linalg.norm(gc)
    for X in clusters:
        members = past.index[past.values == X]
        L = [lpos[x] for x in members if x in lpos]
        Q = [qpos[x] for x in members if x in qpos]
        if len(Q) < 5:
            continue
        lv = lv_all[L] if L else np.zeros((0, 768), np.float32)
        if len(lv):
            lv = lv[(lv @ gc) <= np.quantile(lv_all @ gc, 0.8)]   # drop generic limitations (as in 6.4a)
        qc = qv_all[Q].mean(axis=0); qc /= np.linalg.norm(qc)
        keep = (cX != X)
        already = cited_by_X_past.get(X, set())
        keep &= np.array([c not in already for c in cand])
        idx = np.where(keep)[0]
        S_lim = (cv[idx] @ lv.T) if len(lv) else None
        lim_top = np.sort(S_lim, axis=1)[:, -3:].mean(axis=1) if S_lim is not None and S_lim.shape[1] >= 1 else np.full(len(idx), np.nan)
        lim_max = S_lim.max(axis=1) if S_lim is not None and S_lim.shape[1] >= 1 else np.full(len(idx), np.nan)
        hits = fut_cites.get(X, set())
        for j, k in enumerate(idx):
            i = cand[k]
            rows.append((T, X, i, len(lv), float(lim_top[j]), float(lim_max[j]), float(cv[k] @ qc),
                         float(coupling[cl_ix[X], k]), int(pop.get(i, 0)), int(cyr[k]), i in hits, int(size[cl_ix[X]])))
    print(T, "rows so far", len(rows))
df = pd.DataFrame(rows, columns=["T", "X", "i", "n_lim", "lim_sim3", "lim_max", "prob_sim", "coupling", "pop",
                                 "i_year", "hit", "X_size"])
df.to_parquet(experiments.work_dir(BASE) / "exp1_pairs.parquet")
print(df.groupby("T").agg(pairs=("hit", "size"), hits=("hit", "sum"), clusters=("X", "nunique")))
