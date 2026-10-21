import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from landscape import store
con = store.connect()
base = "dro-pilot"
papers = con.execute("SELECT v.paper_id, p.year, v.route FROM provenance v JOIN papers p USING (paper_id) WHERE v.run_id=?", [base]).df()
year = papers.set_index("paper_id").year
edges = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance a ON a.paper_id=r.citing_id AND a.run_id=?
                       JOIN provenance b ON b.paper_id=r.cited_id AND b.run_id=?""", [base, base]).df()
edges["cy"] = edges.citing_id.map(year)
def strat_auc(df, score, hit, strata):
    out = []
    for _, g in df.groupby(strata):
        if g[hit].nunique() == 2 and g[score].nunique() > 1:
            out.append(roc_auc_score(g[hit], g[score]))
    return round(float(np.mean(out)), 3) if out else None
rows = []
for T in (2020, 2021, 2022, 2023):
    run = f"{base}@T{T}"
    fut = edges[(edges.cy > T) & (edges.cy <= T + 3)].groupby("cited_id").size()
    past = edges[edges.cy <= T].groupby("cited_id").size()
    au = con.execute("SELECT s.paper_id, s.author_id FROM authorships s JOIN provenance v ON v.paper_id=s.paper_id AND v.run_id=?", [run]).df()
    pc = con.execute("""SELECT paper_id, cluster_id, sim_to_centroid FROM paper_cluster WHERE run_id=? AND facet='problem'
                        AND alpha=0.3 AND resolution='medium' AND cluster_id IS NOT NULL""", [run]).df()
    aa = au.merge(pc, on="paper_id")
    k = aa.groupby(["author_id", "cluster_id"]).size().unstack(fill_value=0)
    ka = k.sum(axis=1); k = k[ka >= 3]
    P = 1 - ((k.div(k.sum(axis=1), axis=0)) ** 2).sum(axis=1)
    A = pd.DataFrame({"P": P, "npap": k.sum(axis=1)})
    A["past"] = au.assign(n=au.paper_id.map(past).fillna(0)).groupby("author_id").n.sum().reindex(A.index).fillna(0)
    A["fut"] = au.assign(n=au.paper_id.map(fut).fillna(0)).groupby("author_id").n.sum().reindex(A.index).fillna(0)
    A["hit"] = A.fut > A.fut.quantile(0.9)
    A["pop_decile"] = pd.qcut(A.past.rank(method="first"), 10, labels=False)
    rows.append({"T": T, "what": "broker P (AUC overall)", "v": round(roc_auc_score(A.hit, A.P), 3)})
    rows.append({"T": T, "what": "broker P (AUC within popularity deciles)", "v": strat_auc(A, "P", "hit", "pop_decile")})
    rows.append({"T": T, "what": "popularity (AUC overall)", "v": round(roc_auc_score(A.hit, A.past), 3)})
    # novelty
    nv = con.execute("SELECT paper_id, facet, value FROM paper_scores WHERE run_id=? AND metric='novelty'", [run]).df()
    nvp = nv.pivot(index="paper_id", columns="facet", values="value")
    nvp = nvp[nvp.index.map(year).isin([T - 2, T - 1, T])]
    nvp["fut"] = nvp.index.map(fut).fillna(0); nvp["y"] = nvp.index.map(year)
    nvp["hit"] = nvp.fut > nvp.groupby("y").fut.transform(lambda s: s.quantile(0.75))
    nvp["sim"] = nvp.index.map(pc.set_index("paper_id").sim_to_centroid)
    nvp["route"] = nvp.index.map(papers.set_index("paper_id").route)
    for f in ("abstract", "innovation", "method", "problem"):
        rows.append({"T": T, "what": f"novelty {f} (AUC)", "v": round(roc_auc_score(nvp.hit, nvp[f].fillna(0)), 3)})
    core = nvp[nvp.sim >= nvp.sim.median()]
    rows.append({"T": T, "what": "novelty innovation, central half only (AUC)", "v": round(roc_auc_score(core.hit, core["innovation"].fillna(0)), 3)})
    rows.append({"T": T, "what": "centrality sim_to_centroid (AUC)", "v": round(roc_auc_score(nvp.dropna(subset=["sim"]).hit, nvp.dropna(subset=["sim"]).sim), 3)})
    rows.append({"T": T, "what": "share of top-5% novel (abstract) from semantic route", "v": round(float((nvp.sort_values("abstract", ascending=False).head(max(1, len(nvp)//20)).route == "semantic").mean()), 3)})
    rows.append({"T": T, "what": "share of all candidate papers from semantic route", "v": round(float((nvp.route == "semantic").mean()), 3)})
r = pd.DataFrame(rows).pivot(index="what", columns="T", values="v")
r["mean"] = r.mean(axis=1).round(3)
print(r.to_string())
