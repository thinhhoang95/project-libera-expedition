"""Experiment 3: emerging micro-topics. Do dense pockets of recent papers keep attracting work?

For each paper x published in T-1..T: its k nearest neighbours among papers <= T (abstract vectors,
256-d). Burstiness = share of those neighbours published in T-1..T. Outcome: the local ball
(radius = similarity to the k-th neighbour) - share of T+1..T+3 papers inside it versus the share of
T-2..T papers inside it. Compared with the paper's cluster-level E', its citations, and random.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from landscape import embed, store
from landscape.cli import _params

BASE, K = "dro-pilot", 15
con = store.connect()
p = _params(con, BASE)
papers = con.execute("""SELECT v.paper_id, v.route, p.year FROM provenance v JOIN papers p USING (paper_id)
                        WHERE v.run_id = ?""", [BASE]).df()
year = papers.set_index("paper_id").year
ids, V = embed.load_vectors("abstract", ids=papers.paper_id.tolist(), version=p.embedding.version)
V = embed.truncate(V, 256)
yrs = np.array([year[i] for i in ids])
refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance a ON a.paper_id = r.citing_id AND a.run_id = ?
                      JOIN provenance b ON b.paper_id = r.cited_id AND b.run_id = ?""", [BASE, BASE]).df()
refs["cy"] = refs.citing_id.map(year)
rows = []
for T in (2020, 2021, 2022, 2023):
    run = f"{BASE}@T{T}"
    past = np.where(yrs <= T)[0]
    recent = np.where((yrs >= T - 1) & (yrs <= T))[0]
    window_past = np.where((yrs >= T - 2) & (yrs <= T))[0]
    future = np.where((yrs > T) & (yrs <= T + 3))[0]
    S = V[recent] @ V[past].T
    S[np.arange(len(recent)), np.searchsorted(past, recent)] = -1      # drop self
    nn = np.argsort(-S, axis=1)[:, :K]
    radius = np.take_along_axis(S, nn[:, -1:], axis=1)[:, 0]
    burst = (yrs[past][nn] >= T - 1).mean(axis=1)
    in_fut = (V[recent] @ V[future].T >= radius[:, None]).mean(axis=1)
    in_past = (V[recent] @ V[window_past].T >= radius[:, None]).mean(axis=1)
    growth = np.log((in_fut + 1e-3) / (in_past + 1e-3))
    pc = con.execute("""SELECT paper_id, cluster_id FROM paper_cluster WHERE run_id = ? AND facet = 'problem'
                        AND alpha = 0.3 AND resolution = 'medium'""", [run]).df().set_index("paper_id").cluster_id
    sc = con.execute("SELECT cluster_id, scores FROM viz_cluster_year WHERE run_id = ? AND year = ? AND scores IS NOT NULL",
                     [run, T]).df()
    E = {c: json.loads(s).get("E") for c, s in zip(sc.cluster_id, sc.scores)}
    cites = refs[refs.cy <= T].groupby("cited_id").size()
    for j, k in enumerate(recent):
        pid = ids[k]
        rows.append({"T": T, "paper_id": pid, "burst": burst[j], "growth": growth[j],
                     "cluster_E": E.get(pc.get(pid)), "cites": cites.get(pid, 0), "radius": radius[j]})
df = pd.DataFrame(rows)
df["cluster_E"] = df.cluster_E.astype(float)
df["hit"] = df.growth >= np.log(1.5)
df.to_parquet("data/backtest/exp3_pockets.parquet")
print(f"{len(df)} recent papers, hit rate (local share x1.5) {df.hit.mean():.3f}")
out = []
for name in ["burst", "cluster_E", "cites", "radius"]:
    d = df.dropna(subset=[name])
    rho = np.mean([spearmanr(g[name], g.growth).statistic for _, g in d.groupby("T")])
    pk = np.mean([g.nlargest(50, name).hit.mean() for _, g in d.groupby("T")])
    out.append({"ranking": name, "rho with local growth": rho, "AUC": roc_auc_score(d.hit, d[name]), "p@50": pk})
out.append({"ranking": "random", "rho with local growth": 0, "AUC": 0.5, "p@50": df.hit.mean()})
print(pd.DataFrame(out).round(3).to_string(index=False))

# ---- variant: local momentum (share of year-T papers in the ball vs year T-1), the local analogue of E'
rows2 = []
for T in (2020, 2021, 2022, 2023):
    past = np.where(yrs <= T)[0]
    recent = np.where((yrs >= T - 1) & (yrs <= T))[0]
    yT, yT1 = np.where(yrs == T)[0], np.where(yrs == T - 1)[0]
    S = V[recent] @ V[past].T
    S[np.arange(len(recent)), np.searchsorted(past, recent)] = -1
    radius = -np.partition(-S, K - 1, axis=1)[:, K - 1]
    mom = np.log(((V[recent] @ V[yT].T >= radius[:, None]).mean(axis=1) + 1e-3)
                 / ((V[recent] @ V[yT1].T >= radius[:, None]).mean(axis=1) + 1e-3))
    rows2.append(pd.DataFrame({"T": T, "paper_id": [ids[k] for k in recent], "local_momentum": mom}))
d2 = df.merge(pd.concat(rows2), on=["T", "paper_id"])
rho = np.mean([spearmanr(g.local_momentum, g.growth).statistic for _, g in d2.groupby("T")])
print(f"local momentum: rho {rho:.3f}, AUC {roc_auc_score(d2.hit, d2.local_momentum):.3f}, "
      f"p@50 {np.mean([g.nlargest(50, 'local_momentum').hit.mean() for _, g in d2.groupby('T')]):.3f}")
