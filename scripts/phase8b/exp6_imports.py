"""Experiment 6: ideas borrowed from outside the field. Which external works does the field import
more and more, and does it pay to borrow from external trends?

Needs scripts/phase8b/external_mine.py (fetch + embed; facets optional) for the run.

A. Import map (descriptive): the subfields of the external works corpus papers cite, by citing year.
B. Lead indicator (predictive, leave-one-cutoff-out). Candidates at cutoff T are external works with
   at least 5 corpus citers by T, so their selection uses data up to T only (the fetched set is
   "cited >= 5 times ever", which would otherwise select on the outcome). Hit: the share of corpus
   papers citing the work in T+1..T+3, relative to T-2..T, is in the top quarter for T. The base
   model knows the in-field citation rate, its momentum, global citations by T and age; the
   tested families are the work's global trend at T, its semantic fit to what the corpus is asking
   now, and how foreign its subfield is to the corpus.
D. Live list for the explorer (Bridges › From outside the field): at the latest complete year, every
   fetched external work scored by the model fitted on all cutoffs, with its subfield, global trend,
   best-fitting problem cluster, closest corpus paper and the corpus papers citing it lately.
   Written to data/backtest/<run>/external/live_imports.parquet.
C. Borrowing style of corpus papers (features for exp5): for each paper, the global trend of the
   external works it cites (as of its year) and how far, semantically and by subfield, it
   borrows. Written to data/backtest/<run>/exp6_paper_borrowing.parquet for exp5_features.py.

Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a copy of the database.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from landscape import embed, experiments
from landscape.cli import _params

RUN = experiments.run_arg()
WORK = experiments.work_dir(RUN)
EXT = WORK / "external"
CUTOFFS = (2020, 2021, 2022, 2023)
con = experiments.connect_ro()
p = _params(con, RUN)
VER, DIM = p.embedding.version, p.embedding.graph_dim

papers = con.execute("""SELECT v.paper_id, p.year, json_extract_string(p.topics, '$[0].subfield') subfield
                        FROM provenance v JOIN papers p USING (paper_id) WHERE v.run_id = ? AND p.year IS NOT NULL""",
                     [RUN]).df()
year = papers.set_index("paper_id").year
n_year = papers.groupby("year").size()
works = pd.read_parquet(EXT / "works.parquet")
edges = pd.read_parquet(EXT / "edges.parquet").dropna(subset=["citing_year"])
edges["citing_year"] = edges.citing_year.astype(int)
meta = works.set_index("paper_id")

# global citations per year of each external work (OpenAlex counts_by_year)
gc = pd.DataFrame([{"paper_id": k, "year": c["year"], "cites": c.get("cited_by_count", 0)}
                   for k, s in zip(works.paper_id, works.counts_by_year) for c in json.loads(s)])
G = gc.pivot_table(index="paper_id", columns="year", values="cites", aggfunc="sum").fillna(0)


def gsum(ids, y0, y1):
    cols = [c for c in G.columns if y0 <= c <= y1]
    return G.reindex(ids)[cols].sum(axis=1).fillna(0).to_numpy() if cols else np.zeros(len(ids))


def load(path, ids=None):
    if not path.exists():
        return {}, np.zeros((0, DIM), np.float32)
    t = pq.read_table(path)
    pid = t.column("paper_id").to_pylist()
    v = np.asarray(t.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                   dtype=np.float32).reshape(len(pid), -1)
    v = embed.truncate(v, DIM)
    return {x: k for k, x in enumerate(pid)}, v


def corpus_vecs(facet):
    ids, v = embed.load_vectors(facet, ids=papers.paper_id.tolist(), version=VER)
    return {x: k for k, x in enumerate(ids)}, embed.truncate(v, DIM)


XA = load(EXT / "vectors_abstract.parquet")
XI = load(EXT / "vectors_innovation.parquet")
XM = load(EXT / "vectors_method.parquet")
CA, CP, CL = corpus_vecs("abstract"), corpus_vecs("problem"), corpus_vecs("limitations")
CI = corpus_vecs("innovation")


def rows_of(V, ids):
    pos, v = V
    ok = [x for x in ids if x in pos]
    return ok, (v[[pos[x] for x in ok]] if ok else np.zeros((0, DIM), np.float32))


def ext_vec(ids):
    """Innovation facet when extracted, else the method facet, else the abstract."""
    out = np.full((len(ids), DIM), np.nan, np.float32)
    for V in (XA, XM, XI):  # later sources overwrite earlier ones
        pos, v = V
        for k, x in enumerate(ids):
            if x in pos:
                out[k] = v[pos[x]]
    return out


# ---------------------------------------------------------------- A. import map
m = edges.merge(works[["paper_id", "subfield", "field"]], left_on="cited_id", right_on="paper_id", how="left")
m["subfield"] = m.subfield.fillna("(not fetched: cited < 5 times)")
imp = m[(m.citing_year >= 2016) & (m.citing_year <= 2025)].groupby(["citing_year", "subfield"]).size().rename("citations")
imp = imp.reset_index()
unfetched = imp[imp.subfield.str.startswith("(not fetched")].set_index("citing_year").citations / \
    imp.groupby("citing_year").citations.sum()
imp = imp[~imp.subfield.str.startswith("(not fetched")].copy()  # shares among works with metadata
imp["share"] = imp.citations / imp.groupby("citing_year").citations.transform("sum")
early_s = imp[imp.citing_year.between(2017, 2019)].groupby("subfield").share.mean()
late_s = imp[imp.citing_year.between(2023, 2025)].groupby("subfield").share.mean()
trend = pd.DataFrame({"share 2017-19": early_s, "share 2023-25": late_s}).fillna(0)
trend["change (points)"] = 100 * (trend["share 2023-25"] - trend["share 2017-19"])
trend = trend.sort_values("share 2023-25", ascending=False).reset_index().rename(columns={"index": "subfield"})
corpus_sf = papers.subfield.value_counts(normalize=True)
trend["share of corpus papers in this subfield"] = trend.subfield.map(corpus_sf).fillna(0)
print(trend.head(20).round(3).to_string(index=False))
experiments.save(RUN, "exp6_imports", "import_map", trend.head(40),
                 title="External imports: which subfields the field borrows from",
                 how_to_read="Citations from corpus papers to works outside the corpus, grouped by the cited work's "
                 "OpenAlex subfield, among the works cited at least 5 times (the ones fetched). The rest of the "
                 "external citations go to works cited fewer times: "
                 + ", ".join(f"{int(y)}: {v:.0%}" for y, v in unfetched.items() if 2017 <= y <= 2025)
                 + " of each year's external citations. "
                 "**share** columns are the subfield's share of all external citations in those citing years; "
                 "**change** is in percentage points. The last column shows how much of the corpus itself sits in "
                 "that subfield: a large import share with a small corpus share is a genuinely external source.")

# ---------------------------------------------------------------- B. lead indicator
cy = edges.groupby(["cited_id", "citing_year"]).citing_id.nunique().unstack(fill_value=0)
ext_ids = [x for x in works.paper_id if x in cy.index]
cy = cy.reindex(ext_ids)


def csum(y0, y1):
    cols = [c for c in cy.columns if y0 <= c <= y1]
    return cy[cols].sum(axis=1) if cols else pd.Series(0, index=cy.index)


def nsum(y0, y1):
    return float(n_year[(n_year.index >= y0) & (n_year.index <= y1)].sum())


def features(T, cand):
    before = csum(T - 2, T)[cand] / nsum(T - 2, T)
    F = pd.DataFrame({"paper_id": cand, "T": T})
    # base: in-field rate and momentum, global popularity, age
    F["l_rate_before"] = np.log(before.to_numpy() + 1e-4)
    F["infield_mom"] = np.log((csum(T, T)[cand].to_numpy() / max(nsum(T, T), 1) + 1e-4)
                              / (csum(T - 2, T - 1)[cand].to_numpy() / max(nsum(T - 2, T - 1), 1) + 1e-4))
    F["l_global_cum"] = np.log1p(gsum(cand, 0, T))
    F["age"] = T - meta.year.reindex(cand).fillna(T).to_numpy()
    # external trend at T (global citations, all of science)
    g_now, g_prev = gsum(cand, T, T), gsum(cand, T - 2, T - 1) / 2
    F["g_trend"] = np.log((g_now + 1) / (g_prev + 1))
    F["g_trend_3y"] = np.log((gsum(cand, T - 1, T) / 2 + 1) / (gsum(cand, T - 4, T - 2) / 3 + 1))
    # semantic fit with what the corpus asks now vs a few years ago
    X = ext_vec(cand)
    has = ~np.isnan(X).any(axis=1)
    Xs = np.nan_to_num(X)
    recent_ids = papers[papers.year.between(T - 1, T)].paper_id.tolist()
    older_ids = papers[papers.year.between(T - 4, T - 2)].paper_id.tolist()
    recent_ok, Pr = rows_of(CP, recent_ids)
    _, Po = rows_of(CP, older_ids)
    Srec = Xs @ Pr.T
    F["fit_now"] = np.sort(Srec, axis=1)[:, -20:].mean(axis=1)
    F["fit_shift"] = F.fit_now - np.sort(Xs @ Po.T, axis=1)[:, -20:].mean(axis=1)
    _, Lv = rows_of(CL, papers[papers.year <= T].paper_id.tolist())
    F["lim_match"] = np.sort(Xs @ Lv.T, axis=1)[:, -3:].mean(axis=1) if len(Lv) else np.nan
    for c in ("fit_now", "fit_shift", "lim_match"):
        F.loc[~has, c] = np.nan
    # externality: how much of the corpus (up to T) shares the work's subfield
    sf_T = papers[papers.year <= T].subfield.value_counts(normalize=True)
    F["externality"] = 1 - meta.subfield.reindex(cand).map(sf_T).fillna(0).to_numpy()
    F["_nearest_recent"] = [recent_ok[j] for j in Srec.argmax(axis=1)] if len(recent_ok) else None
    F.loc[~has, "_nearest_recent"] = None
    return F


rows = []
for T in CUTOFFS:
    cum = csum(0, T)
    cand = [x for x in ext_ids if cum[x] >= 5 and (meta.year.get(x) or 9999) <= T]
    F = features(T, cand)
    after = csum(T + 1, T + 3)[cand] / nsum(T + 1, T + 3)
    before = csum(T - 2, T)[cand] / nsum(T - 2, T)
    F["growth"] = np.log((after.to_numpy() + 1e-4) / (before.to_numpy() + 1e-4))
    F["after_n"] = csum(T + 1, T + 3)[cand].to_numpy()
    rows.append(F.drop(columns="_nearest_recent"))
    print(T, len(F), flush=True)

D = pd.concat(rows, ignore_index=True)
D["hit_ratio"] = D.growth > D.groupby("T").growth.transform(lambda s: s.quantile(0.75))
# strict: the import intensifies AND at least 3 corpus papers cite the work in T+1..T+3 (a ratio from
# a near-zero rate is mostly noise and mean reversion)
D["hit_strict"] = D.hit_ratio & (D.after_n >= 3)
for c in D.columns:
    if D[c].dtype.kind == "f" and D[c].isna().any():
        D[c] = D[c].fillna(D[c].median())
D.to_parquet(WORK / "exp6_candidates.parquet", index=False)

FAM = {"base": ["l_rate_before", "infield_mom", "l_global_cum", "age"],
       "external trend": ["g_trend", "g_trend_3y"],
       "semantic fit": ["fit_now", "fit_shift", "lim_match"],
       "externality": ["externality"]}
MODELS = {"base": FAM["base"], "base + external trend": FAM["base"] + FAM["external trend"],
          "base + semantic fit": FAM["base"] + FAM["semantic fit"], "base + externality": FAM["base"] + FAM["externality"],
          "all": sum(FAM.values(), []), "trend + fit only (no base)": FAM["external trend"] + FAM["semantic fit"]}


def loco(d, feats, hit):
    out = pd.Series(np.nan, index=d.index)
    for T in sorted(d["T"].unique()):
        tr, te = d["T"] != T, d["T"] == T
        s = StandardScaler().fit(d.loc[tr, feats])
        mdl = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(s.transform(d.loc[tr, feats]), d.loc[tr, hit])
        out[te] = mdl.decision_function(s.transform(d.loc[te, feats]))
    return out


def boot(d, y, a, b, n=300):
    """Bootstrap over works (a work appears at several cutoffs), not rows."""
    rng = np.random.default_rng(0)
    groups = d.groupby("paper_id").indices
    keys = list(groups)
    g = []
    for _ in range(n):
        i = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        if 0 < y[i].sum() < len(i):
            g.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(g, [5, 95])


tables, unis = [], []
for hit, label in (("hit_strict", "strict: intensifies and >= 3 citing papers after T"),
                   ("hit_ratio", "ratio only: share growth in the top quarter")):
    D["hit"] = D[hit]
    y = D.hit.to_numpy().astype(int)
    sc = {k: loco(D, f, "hit") for k, f in MODELS.items()}
    base = sc["base"].to_numpy()
    res = []
    for k, s_ in sc.items():
        lo, hi = boot(D, y, s_.to_numpy(), base) if k != "base" else (0.0, 0.0)
        D["_s"] = s_
        res.append({"model": k, "AUC": roc_auc_score(y, s_), "gain vs base": roc_auc_score(y, s_) - roc_auc_score(y, base),
                    "gain 90% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                    "p@50": np.mean([g.nlargest(50, "_s").hit.mean() for _, g in D.groupby("T")])})
    res = pd.DataFrame(res).assign(hit=label, candidates=len(D), base_rate=D.hit.mean())
    print(res.round(3).to_string(index=False))
    tables.append(res)
    unis.append(pd.DataFrame([{"hit": label, "family": fam, "feature": f, "AUC alone": roc_auc_score(y, D[f])}
                              for fam, fs in FAM.items() for f in fs]))
uni = pd.concat(unis)
uni["rho with growth"] = uni.feature.map({f: D[[f, "growth"]].corr(method="spearman").iloc[0, 1] for f in uni.feature.unique()})
print(uni.round(3).to_string(index=False))
how = ("Candidates at each cutoff T are external works already cited by at least 5 corpus papers by T. A **hit** "
       "is the work's import intensifying: the share of corpus papers citing it in T+1..T+3 versus T-2..T is in "
       "the top quarter for that T; the **strict** definition also needs at least 3 citing papers after T, because "
       "a ratio from a near-zero rate is mostly noise. The **base** model knows the in-field citation rate and its "
       "momentum, global citations by T and age. **external trend** is the work's global citation momentum at T "
       "(all of science); **semantic fit** is how close its innovation (or method / abstract) is to the problems "
       "the corpus posed in T-1..T, its shift towards them since T-4..T-2, and its match to stated limitations; "
       "**externality** is how little of the corpus shares its subfield. The 90% interval resamples whole works.")
experiments.save(RUN, "exp6_imports", "lead_indicator_models", pd.concat(tables),
                 title="External imports: what predicts more borrowing", how_to_read=how)
experiments.save(RUN, "exp6_imports", "lead_indicator_features", uni, title="External imports: single features",
                 how_to_read="Each feature alone: **AUC** for each hit definition (below 0.5 points the other way), "
                 "and its rank correlation with the import growth itself.")
D["hit"] = D.hit_strict

# ---------------------------------------------------------------- C. borrowing style of corpus papers
b = edges.merge(works[["paper_id", "subfield"]], left_on="cited_id", right_on="paper_id", how="inner")
b = b[b.citing_id.isin(set(papers.paper_id))]
out, sf_cache = [], {}
for (pid, yr), g in b.groupby(["citing_id", "citing_year"]):
    ids = g.cited_id.tolist()
    # only works already imported (>= 5 corpus citers) by the citing year, to keep the future out
    cum = cy.reindex(ids)[[c for c in cy.columns if c <= yr]].sum(axis=1)
    ids = [x for x in ids if cum.get(x, 0) >= 5]
    if not ids:
        continue
    gt = np.log((gsum(ids, yr, yr) + 1) / (gsum(ids, yr - 2, yr - 1) / 2 + 1))
    Xe = ext_vec(ids)
    ok = ~np.isnan(Xe).any(axis=1)
    dist = np.nan
    if ok.any() and pid in CA[0]:
        dist = float(1 - (Xe[ok] @ CA[1][CA[0][pid]]).mean())
    if yr not in sf_cache:
        sf_cache[yr] = papers[papers.year <= yr].subfield.value_counts(normalize=True)
    ext = 1 - meta.subfield.reindex(ids).map(sf_cache[yr]).fillna(0)
    out.append({"paper_id": pid, "b_trend_max": float(gt.max()), "b_trend_mean": float(gt.mean()),
                "b_n_trending": int((gt > np.log(1.5)).sum()), "b_distance": dist,
                "b_externality": float(ext.mean()), "b_n_subfields": int(meta.subfield.reindex(ids).nunique())})
pd.DataFrame(out).to_parquet(WORK / "exp6_paper_borrowing.parquet", index=False)
print(f"borrowing features for {len(out)} corpus papers")

# ---------------------------------------------------------------- D. live list
T_LIVE = int(papers.year.max()) - 1  # the latest complete publication year
feats = MODELS["all"]
fit_d = D.copy()
sc_ = StandardScaler().fit(fit_d[feats])
mdl = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc_.transform(fit_d[feats]), fit_d.hit)
recent_n = csum(T_LIVE - 2, T_LIVE)
L = features(T_LIVE, [x for x in ext_ids if (meta.year.get(x) or 9999) <= T_LIVE and recent_n.get(x, 0) >= 3])
for c in feats:
    L[c] = L[c].astype(float).fillna(D[c].median())
L["score"] = mdl.decision_function(sc_.transform(L[feats]))
pc = con.execute("""SELECT paper_id, cluster_id FROM paper_cluster WHERE run_id = ? AND facet = 'problem'
                    AND alpha = 0.5 AND resolution = 'medium' AND cluster_id IS NOT NULL""", [RUN]).df()
lab = dict(con.execute("SELECT cluster_id, label FROM clusters WHERE run_id = ?", [RUN]).fetchall())
pcl = pc.set_index("paper_id").cluster_id
# semantic distance from the corpus (abstracts): OpenAlex subfields mislabel many in-field works, so
# "outside the field" is the third of the list farthest from the corpus's own abstracts
xid, Xa = rows_of(XA, L.paper_id.tolist())
_, Ca = rows_of(CA, papers.paper_id.tolist())
dist = pd.Series(1 - np.sort(Xa @ Ca.T, axis=1)[:, -20:].mean(axis=1), index=xid)
# title-only records (old books, proceedings volumes) embed far from everything: abstracts only
with_abs = set(works.paper_id[works.abstract.fillna("").str.len() >= 100])
L["sem_distance"] = L.paper_id.map(dist).where(L.paper_id.isin(with_abs))
L["outside_field"] = L.sem_distance >= L.sem_distance.quantile(2 / 3)
L["cluster_id"] = L._nearest_recent.map(pcl)
L["cluster_label"] = L.cluster_id.map(lab)
L["closest_corpus_paper"] = L._nearest_recent
L["recent_citers"] = recent_n.reindex(L.paper_id).fillna(0).astype(int).to_numpy()  # T-2..T, as filtered
L["corpus_citers_total"] = csum(0, 9999).reindex(L.paper_id).fillna(0).astype(int).to_numpy()
L = L.merge(works[["paper_id", "title", "year", "subfield", "field", "venue", "cited_by_count"]], on="paper_id", how="left")
L = L.drop(columns="_nearest_recent").sort_values("score", ascending=False)
L.attrs = {}
L.assign(as_of=T_LIVE, model="exp6 all families, fitted on cutoffs " + ",".join(map(str, CUTOFFS))).to_parquet(
    EXT / "live_imports.parquet", index=False)
print(f"live list as of {T_LIVE}: {len(L)} external works; top 10:")
print(L.head(10)[["title", "subfield", "g_trend", "fit_now", "recent_citers", "cluster_label"]].to_string(index=False))
print("outside the field (semantic):")
print(L[L.outside_field & (L.year >= T_LIVE - 10)].head(12)[["title", "subfield", "g_trend", "recent_citers", "cluster_label"]].to_string(index=False))
