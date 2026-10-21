"""Experiment 8: can we see where a new community will form before it exists?

Every earlier test predicts uptake by communities that already exist at the cutoff. Here the
outcome is a *birth*, detected directly in embedding space because the Leiden communities are too
coarse (new work is absorbed into an existing community):

    frontier papers   papers from T+1..T+3 far from everything published up to T (top quarter by
                      1 - max cosine to papers <= T, abstract vectors)
    births            groups of >= 5 frontier papers close to each other (average-linkage, cosine
                      threshold = 90th percentile of frontier-frontier similarity)
    seeds (hits)      papers from T-2..T among the 3 nearest pre-T papers of >= 2 members of one birth

Each recent paper (T-2..T) is an *anchor*, scored with signals available at T. Because the seeds
of something new tend to sit at the edge of existing work themselves, the base model includes
isolation (distance to the nearest earlier papers); the question is what predicts a birth beyond
"being at the edge":

    local momentum, reference-pair novelty (pairs of references rarely cited together before),
    bridging (the paper's references spread across several cutoff communities' literatures),
    borrowing (global trend and distance of its external references, exp6), nearby stated
    limitations; citations so far and coupling as the bibliometric base.

Cutoffs T = 2019..2022 (outcomes complete through 2025); the cutoff-T communities come from the
cutoff runs. Also writes a live list (as of the latest complete year) of anchors the model ranks
highest: data/projects/<run>/backtest/exp8_regions_live.parquet, frozen by freeze_snapshot.py.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a copy of the database.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from landscape import embed, experiments
from landscape.cli import _params

RUN = experiments.run_arg()
WORK = experiments.work_dir(RUN)
CUTOFFS = (2019, 2020, 2021, 2022)
con = experiments.connect_ro()
P = _params(con, RUN)
VER, DIM = P.embedding.version, P.embedding.graph_dim

papers = con.execute("""SELECT v.paper_id, v.route, p.year FROM provenance v JOIN papers p USING (paper_id)
                        WHERE v.run_id = ? AND p.year IS NOT NULL""", [RUN]).df()
year = papers.set_index("paper_id").year
route = papers.set_index("paper_id").route
ids, V = embed.load_vectors("abstract", ids=papers.paper_id.tolist(), version=VER)
V = embed.truncate(V, DIM)
pos = {x: k for k, x in enumerate(ids)}
yrs = np.array([year[x] for x in ids])
lid, LV = embed.load_vectors("limitations", ids=papers.paper_id.tolist(), version=VER)
LV = embed.truncate(LV, DIM)
lyr = np.array([year[x] for x in lid])

refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v ON v.paper_id = r.citing_id
                      AND v.run_id = ?""", [RUN]).df()
refs = refs[refs.citing_id.isin(pos)]
corpus = set(papers.paper_id)
cites_in = refs[refs.cited_id.isin(corpus)].assign(cy=lambda d: d.citing_id.map(year))
pidx = {x: k for k, x in enumerate(ids)}
widx = {w: k for k, w in enumerate(refs.cited_id.unique())}
R = sparse.csr_matrix((np.ones(len(refs)), (refs.citing_id.map(pidx), refs.cited_id.map(widx))),
                      shape=(len(pidx), len(widx)))
R.data[:] = 1
borrow = pd.read_parquet(WORK / "exp6_paper_borrowing.parquet").set_index("paper_id") \
    if (WORK / "exp6_paper_borrowing.parquet").exists() else pd.DataFrame()


def cocitation_before(y):
    """Work x work co-citation counts from papers published before year y."""
    rows = np.where(yrs < y)[0]
    Ry = R[rows]
    return (Ry.T @ Ry).tocsr()


def pair_novelty(anchor_rows, y_of, cache={}):
    """Share of an anchor's reference pairs (both references cited at least twice before) never co-cited before
    the anchor's year, and the count of such pairs."""
    out_share, out_n = [], []
    for r in anchor_rows:
        y = y_of[r]
        if y not in cache:
            C = cocitation_before(y)
            cache[y] = (C, np.asarray(C.diagonal()).ravel())
        C, deg = cache[y]
        refs_r = R[r].indices
        refs_r = refs_r[deg[refs_r] >= 2]
        if len(refs_r) < 3:
            out_share.append(np.nan)
            out_n.append(0)
            continue
        sub = C[refs_r][:, refs_r].toarray()
        iu = np.triu_indices(len(refs_r), 1)
        z = (sub[iu] == 0)
        out_share.append(float(z.mean()))
        out_n.append(int(z.sum()))
    return np.array(out_share), np.array(out_n)


def anchors_features(T, community_run=None):
    """Signals for papers published in T-2..T, using data up to T only (communities from the cutoff-T run, or
    from community_run for the live list)."""
    pre = np.where(yrs <= T)[0]
    anc = np.where((yrs >= T - 2) & (yrs <= T))[0]
    F = pd.DataFrame({"paper_id": [ids[k] for k in anc], "T": T, "year": yrs[anc]})
    S = V[anc] @ V[pre].T
    S[np.arange(len(anc)), np.searchsorted(pre, anc)] = -2  # not itself
    top20 = np.sort(S, axis=1)[:, -20:]
    F["isolation"] = 1 - top20.mean(axis=1)          # distance from the nearest earlier work
    # novelty when published: 1 - max cos to papers before its own year
    F["novelty_at_pub"] = [1 - float((V[k] @ V[yrs < yrs[k]].T).max()) if (yrs < yrs[k]).any() else np.nan for k in anc]
    # local momentum at T (exp3): share of year-T papers in the 15-NN ball vs year T-1 papers
    radius = np.sort(S, axis=1)[:, -15]
    yT, yT1 = np.where(yrs == T)[0], np.where(yrs == T - 1)[0]
    F["local_momentum"] = np.log(((V[anc] @ V[yT].T >= radius[:, None]).mean(axis=1) + 1e-3)
                                 / ((V[anc] @ V[yT1].T >= radius[:, None]).mean(axis=1) + 1e-3))
    # reference-pair novelty
    share, n = pair_novelty(anc, yrs)
    F["pair_novelty"], F["l_novel_pairs"] = share, np.log1p(n)
    # bridging: entropy of the anchor's coupling across cutoff-T communities (problem, alpha 0.5, medium)
    pc = con.execute("""SELECT paper_id, cluster_id FROM paper_cluster WHERE run_id = ? AND facet = 'problem'
                        AND alpha = 0.5 AND resolution = 'medium' AND cluster_id IS NOT NULL""",
                     [community_run or f"{RUN}@T{T}"]).df()
    pc = pc[pc.paper_id.isin(pos)]
    cl = sorted(pc.cluster_id.unique())
    ci = {c: k for k, c in enumerate(cl)}
    M = sparse.csr_matrix((np.ones(len(pc)), (pc.cluster_id.map(ci), pc.paper_id.map(pidx))), shape=(len(cl), len(pidx)))
    CR = (M @ R)                                     # how often each community cites each work
    coup = (R[anc] @ CR.T).toarray()                 # anchors x communities: shared-reference mass
    tot = coup.sum(axis=1, keepdims=True)
    p = np.divide(coup, tot, out=np.zeros_like(coup), where=tot > 0)
    F["bridging_entropy"] = -(np.where(p > 0, p * np.log(p), 0)).sum(axis=1)
    F["top_community_share"] = p.max(axis=1)
    F["l_coupling"] = np.log1p(tot.ravel())
    F["l_cites_so_far"] = np.log1p(F.paper_id.map(cites_in[cites_in.cy <= T].groupby("cited_id").size()).fillna(0))
    # nearby stated limitations (<= T)
    Lm = LV[lyr <= T]
    F["limitations_near"] = np.sort(V[anc] @ Lm.T, axis=1)[:, -3:].mean(axis=1) if len(Lm) >= 3 else np.nan
    for c in ("b_trend_max", "b_distance"):
        F[c] = F.paper_id.map(borrow[c]) if c in borrow else np.nan
    F["route"] = F.paper_id.map(route)
    return F


def births(T):
    """Frontier groups in T+1..T+3 and their seeds among T-2..T papers."""
    pre = np.where(yrs <= T)[0]
    new = np.where((yrs > T) & (yrs <= T + 3))[0]
    Snp = V[new] @ V[pre].T
    nov = 1 - Snp.max(axis=1)
    fr = new[nov >= np.quantile(nov, 0.75)]
    Sf = V[fr] @ V[fr].T
    thr = np.quantile(Sf[np.triu_indices(len(fr), 1)], 0.90)
    Z = linkage(V[fr], method="average", metric="cosine")
    lab = fcluster(Z, t=1 - thr, criterion="distance")
    groups = [fr[lab == g] for g in np.unique(lab) if (lab == g).sum() >= 5]
    seeds, info = {}, []
    anc_set = set(np.where((yrs >= T - 2) & (yrs <= T))[0])
    for gi, g in enumerate(groups):
        S = V[g] @ V[pre].T
        near = pre[np.argsort(-S, axis=1)[:, :3]]
        u, c = np.unique(near, return_counts=True)
        sd = [k for k, n in zip(u, c) if n >= 2 and k in anc_set]
        for k in sd:
            seeds.setdefault(ids[k], []).append(gi)
        info.append({"T": T, "birth": gi, "size": len(g), "seeds": len(sd),
                     "routes": pd.Series([route[ids[k]] for k in g]).value_counts().to_dict(),
                     "example_titles": [ids[k] for k in g[:3]]})
    return seeds, info


rows, binfo = [], []
for T in CUTOFFS:
    F = anchors_features(T)
    seeds, info = births(T)
    F["hit"] = F.paper_id.isin(seeds)
    rows.append(F)
    binfo += info
    print(T, f"anchors {len(F)}, births {len(info)}, seed anchors {int(F.hit.sum())}", flush=True)
D = pd.concat(rows, ignore_index=True)
BI = pd.DataFrame(binfo)
titles = con.execute("SELECT paper_id, title FROM papers").df().set_index("paper_id").title
BI["example_titles"] = BI.example_titles.map(lambda l: " | ".join(str(titles.get(x, x))[:70] for x in l))
BI["routes"] = BI.routes.map(lambda d: ", ".join(f"{k} {v}" for k, v in d.items()))
for c in D.columns:
    if D[c].dtype.kind == "f":
        D[c] = D[c].fillna(D[c].median())
D.to_parquet(WORK / "exp8_anchors.parquet", index=False)

FAM = {"base (edge + bibliometric)": ["isolation", "novelty_at_pub", "l_cites_so_far", "l_coupling"],
       "local momentum": ["local_momentum"],
       "reference-pair novelty": ["pair_novelty", "l_novel_pairs"],
       "bridging communities": ["bridging_entropy", "top_community_share"],
       "borrowing": ["b_trend_max", "b_distance"],
       "nearby limitations": ["limitations_near"]}
base = FAM["base (edge + bibliometric)"]
MODELS = {"base": base, **{f"base + {k}": base + v for k, v in FAM.items() if not k.startswith("base")},
          "all": sum(FAM.values(), []), "isolation only": ["isolation"]}


def loco(d, feats):
    out = pd.Series(np.nan, index=d.index)
    for T in sorted(d["T"].unique()):
        tr, te = d["T"] != T, d["T"] == T
        s = StandardScaler().fit(d.loc[tr, feats])
        m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(s.transform(d.loc[tr, feats]),
                                                                                   d.loc[tr, "hit"])
        out[te] = m.decision_function(s.transform(d.loc[te, feats]))
    return out


def boot(y, a, b, n=300):
    rng = np.random.default_rng(0)
    g = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            g.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(g, [5, 95])


def evaluate(d, label):
    y = d.hit.to_numpy().astype(int)
    sc = {k: loco(d, f) for k, f in MODELS.items()}
    b = sc["base"].to_numpy()
    res = []
    for k, s in sc.items():
        lo, hi = boot(y, s.to_numpy(), b) if k != "base" else (0.0, 0.0)
        res.append({"population": label, "model": k, "AUC": roc_auc_score(y, s),
                    "gain vs base": roc_auc_score(y, s) - roc_auc_score(y, b), "gain 90% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                    "p@50": np.mean([g.assign(_s=s[g.index]).nlargest(50, "_s").hit.mean() for _, g in d.groupby("T")]),
                    "anchors": len(d), "seeds": int(y.sum()), "base_rate": y.mean()})
    uni = [{"population": label, "feature": f, "AUC alone": roc_auc_score(y, d[f])} for f in sum(FAM.values(), [])]
    return pd.DataFrame(res), pd.DataFrame(uni)


res, uni = zip(evaluate(D, "all anchors"), evaluate(D[D.route == "forward"], "forward route only"))
res, uni = pd.concat(res), pd.concat(uni)
pd.set_option("display.width", 220)
print(BI[["T", "birth", "size", "seeds", "routes"]].to_string(index=False))
print(res.round(3).to_string(index=False))
print(uni.round(3).to_string(index=False))
how = ("For each cutoff T (2019-2022), a **birth** is a group of at least 5 papers from T+1..T+3 that are far from "
       "everything published up to T and close to each other. Its **seeds** are the papers from T-2..T nearest to "
       "several of its members: where it grew from. Every paper from T-2..T is an anchor; the question is whether "
       "signals available at T rank the seeds first. The **base** model already knows how far a paper sits from "
       "earlier work (seeds of something new tend to be at the edge), its citations and its coupling. Gains with a "
       "90% interval above zero say a signal points to births beyond 'being at the edge'. Few births per cutoff: "
       "read the intervals.")
experiments.save(RUN, "exp8_birth", "models", res, title="Community birth: what points to where new work forms",
                 how_to_read=how)
experiments.save(RUN, "exp8_birth", "features", uni, title="Community birth: single features",
                 how_to_read="Each signal alone at ranking seed anchors (0.5 = chance; below 0.5 points the other way).")
experiments.save(RUN, "exp8_birth", "births", BI, title="Community birth: the detected births",
                 how_to_read="Each birth: its cutoff, size (papers in T+1..T+3), number of seed anchors, the expansion "
                 "routes its papers came from (a birth made only of semantic- or author-route papers may reflect how "
                 "the corpus was sampled rather than the field), and three example titles.")

# ---------------------------------------------------------------- live list
T_LIVE = int(yrs.max()) - 1
L = anchors_features(T_LIVE, community_run=RUN)  # no cutoff run for the live year: today's communities
for c in L.columns:
    if L[c].dtype.kind == "f":
        L[c] = L[c].fillna(D[c].median())
# forward route only (the population where the backtest found any signal; author-route and off-field
# outliers dominate otherwise), one record per title; low confidence: AUC about 0.6 in the backtest
feats = MODELS["all"]
Df = D[D.route == "forward"]
s = StandardScaler().fit(Df[feats])
m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(s.transform(Df[feats]), Df.hit)
L = L[L.route == "forward"].copy()
L["score"] = m.decision_function(s.transform(L[feats]))
L["title"] = L.paper_id.map(titles)
L = L.sort_values("score", ascending=False)
L = L[~L.title.fillna(L.paper_id).str.lower().str.replace(r"[^a-z0-9]", "", regex=True).duplicated()]
L["note"] = "low confidence: backtest AUC about 0.6 (forward route)"
L.head(200).to_parquet(WORK / "exp8_regions_live.parquet", index=False)
print(f"live (as of {T_LIVE}):")
print(L.head(15)[["title", "isolation", "local_momentum", "pair_novelty", "score"]].to_string(index=False))
