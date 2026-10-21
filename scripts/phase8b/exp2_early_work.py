"""Experiment 2: early-work detector. Which papers published in year T will be in the top quartile of
their year for corpus citations received in T+1..T+3? Only information available at T is used.

Robustness: the forward and semantic routes selected papers by their 2026 citation count, which
leaks the future into who is in the corpus. Results are also reported on author-route papers only
(taken newest-first, with no citation filter). Usage: --run-id (default dro-pilot); tables are saved to
data/backtest/<run>/experiments/ (explorer: Experiments view)."""
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from landscape import experiments, store
from landscape.cli import _params

BASE = experiments.run_arg()
con = experiments.connect_ro()
p = _params(con, BASE)
papers = con.execute("""SELECT v.paper_id, v.route, p.year FROM provenance v JOIN papers p USING (paper_id)
                        WHERE v.run_id = ?""", [BASE]).df()
year = papers.set_index("paper_id").year
route = papers.set_index("paper_id").route
refs_corpus = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r
                             JOIN provenance a ON a.paper_id = r.citing_id AND a.run_id = ?
                             JOIN provenance b ON b.paper_id = r.cited_id AND b.run_id = ?""", [BASE, BASE]).df()
refs_corpus["cy"] = refs_corpus.citing_id.map(year)
refs_all = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v
                          ON v.paper_id = r.citing_id AND v.run_id = ?""", [BASE]).df()
facets = con.execute("""SELECT paper_id, facet, text FROM facets WHERE facet IN
                        ('contribution_type', 'has_quantitative_result')""").df() \
    .pivot_table(index="paper_id", columns="facet", values="text", aggfunc="first")
au = con.execute("""SELECT s.paper_id, s.author_id FROM authorships s JOIN provenance v
                    ON v.paper_id = s.paper_id AND v.run_id = ?""", [BASE]).df()
au["year"] = au.paper_id.map(year)
pidx = {x: k for k, x in enumerate(papers.paper_id)}
widx = {w: k for k, w in enumerate(refs_all.cited_id.unique())}
R = sparse.csr_matrix((np.ones(len(refs_all)), (refs_all.citing_id.map(pidx), refs_all.cited_id.map(widx))),
                      shape=(len(pidx), len(widx)))
R.data[:] = 1
nrefs = np.asarray(R.sum(axis=1)).ravel()

rows = []
for T in (2020, 2021, 2022, 2023):
    run = f"{BASE}@T{T}"
    new = papers[papers.year == T].paper_id.tolist()
    pastp = papers[papers.year <= T].paper_id.tolist()
    rec_T = refs_corpus[refs_corpus.cy <= T].groupby("cited_id").size()
    fut = refs_corpus[(refs_corpus.cy > T) & (refs_corpus.cy <= T + 3)].groupby("cited_id").size()
    # bibliographic coupling with the corpus up to T (>= 2 shared references)
    Rn, Rp = R[[pidx[x] for x in new]], R[[pidx[x] for x in pastp]]
    C = (Rn @ Rp.T).tocsr()
    C.data = (C.data >= 2).astype(float)
    coup = np.maximum(np.asarray(C.sum(axis=1)).ravel() - 1, 0)  # minus the self-match
    # cluster context at T (problem, alpha 0.3, cutoff run): momentum-based score E' and centrality
    pc = con.execute("""SELECT paper_id, cluster_id, sim_to_centroid FROM paper_cluster WHERE run_id = ?
                        AND facet = 'problem' AND alpha = 0.3 AND resolution = 'medium'""", [run]).df().set_index("paper_id")
    sc = con.execute("""SELECT cluster_id, scores FROM viz_cluster_year WHERE run_id = ? AND year = ?
                        AND scores IS NOT NULL""", [run, T]).df()
    E = {c: json.loads(s).get("E") for c, s in zip(sc.cluster_id, sc.scores)}
    nov = con.execute("""SELECT paper_id, value FROM paper_scores WHERE run_id = ? AND metric = 'novelty'
                         AND facet = 'abstract'""", [run]).df().set_index("paper_id").value
    # author prominence before T: max corpus citations received by an author's earlier papers
    prev = au[au.year < T].assign(c=lambda d: d.paper_id.map(rec_T).fillna(0))
    a_prom = prev.groupby("author_id").c.sum()
    a_npap = prev.groupby("author_id").paper_id.nunique()
    for k, x in enumerate(new):
        auth = au[au.paper_id == x].author_id
        cl = pc.cluster_id.get(x) if x in pc.index else None
        ct = facets.contribution_type.get(x) if x in facets.index else None
        rows.append({"T": T, "paper_id": x, "route": route[x], "cites_so_far": rec_T.get(x, 0),
                     "coupling": coup[k], "n_refs": nrefs[pidx[x]], "cluster_E": E.get(cl) if cl else None,
                     "centrality": pc.sim_to_centroid.get(x) if x in pc.index else None,
                     "novelty": nov.get(x), "author_prom": a_prom.reindex(auth).max() if len(auth) else 0,
                     "author_papers": a_npap.reindex(auth).max() if len(auth) else 0, "n_authors": len(auth),
                     "survey": ct == "survey", "theory": ct == "theory",
                     "quant": (facets.has_quantitative_result.get(x) == "true") if x in facets.index else False,
                     "fut": fut.get(x, 0)})
df = pd.DataFrame(rows)
df["hit"] = df.fut > df.groupby("T").fut.transform(lambda s: s.quantile(0.75))
for c in ["cites_so_far", "coupling", "n_refs", "author_prom", "author_papers", "n_authors"]:
    df["l_" + c] = np.log1p(df[c].fillna(0).astype(float))
for c in ["cluster_E", "centrality", "novelty"]:
    df[c] = df[c].astype(float).fillna(df[c].astype(float).median())
df[["survey", "theory", "quant"]] = df[["survey", "theory", "quant"]].astype(float)
df.to_parquet(experiments.work_dir(BASE) / "exp2_papers.parquet")
FEATS = ["l_cites_so_far", "l_coupling", "l_n_refs", "cluster_E", "centrality", "novelty", "l_author_prom",
         "l_author_papers", "l_n_authors", "survey", "theory", "quant"]
NO_AUTH = [f for f in FEATS if not f.startswith("l_author")]
coefs = []
for name, feats in (("logit (all)", FEATS), ("logit (no author fame)", NO_AUTH)):
    df[name] = np.nan
    for T in df["T"].unique():
        tr, te = df["T"] != T, df["T"] == T
        s = StandardScaler().fit(df.loc[tr, feats])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(df.loc[tr, feats]), df.loc[tr, "hit"])
        df.loc[te, name] = m.decision_function(s.transform(df.loc[te, feats]))
    print(name, "coefficients (last fold):", dict(zip(feats, np.round(m.coef_[0], 2))))
    coefs += [{"model": name, "feature": f, "coefficient": float(c)} for f, c in zip(feats, m.coef_[0])]


def report(d, label):
    out = []
    for name in ["l_cites_so_far", "l_author_prom", "l_coupling", "l_n_refs", "cluster_E", "centrality", "novelty",
                 "logit (all)", "logit (no author fame)"]:
        pk = np.mean([g.nlargest(50, name).hit.mean() for _, g in d.groupby("T")])
        out.append({"ranking": name, "AUC": roc_auc_score(d.hit, d[name]), "p@50": pk})
    print(f"== {label}: {len(d)} papers, base rate {d.hit.mean():.3f}")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    return pd.DataFrame(out).assign(population=label, papers=len(d), base_rate=d.hit.mean())


tab = pd.concat([report(df, "all routes"),
                 report(df[df.route == "author"], "author route only (no citation-based selection)")])
experiments.save(BASE, "exp2_early_work", "rankings", tab, title="Early-work detector: rankings",
                 how_to_read="Papers published in year T; a hit is landing in the top quarter of that year for "
                 "corpus citations received in T+1..T+3. **AUC** is how well a ranking separates hits from the rest "
                 "(0.5 = chance); **p@50** the hit share among the top 50 per cutoff, to compare with **base_rate**. "
                 "Single features are shown next to the two leave-one-cutoff-out models. The author-route rows "
                 "are papers chosen without any citation filter: a check against selection bias, but small.")
experiments.save(BASE, "exp2_early_work", "coefficients", pd.DataFrame(coefs), title="Early-work detector: model weights",
                 how_to_read="Standardized logistic coefficients (last fold). Negative novelty means novel papers "
                 "draw fewer in-field citations in the first three years.")
