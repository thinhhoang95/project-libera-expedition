"""Experiment 5: does *what* a paper is about, the question it asks and who wrote it predict success
beyond rich-get-richer? Builds one row per paper published in year T (T = 2020..2023) with features
from data available at T only, in five families:

- base     rich-get-richer / bibliometric: corpus citations so far, coupling, references, team size,
           preprint, expansion route
- topic    the paper's semantic neighbourhood among earlier papers (abstract and problem vectors):
           how much attention similar earlier papers drew (heat), crowding, local momentum, and
           whether the paper sits ahead of the neighbourhood's drift
- question how many current problems (posed in T-2..T) the paper's innovation speaks to (reach),
           match to stated open limitations, typicality of the question, atypicality of the method
           chosen for it
- author   corpus-only credibility before T: track record (within-year citation percentile of earlier
           papers), topical expertise (similarity of this problem to the author's earlier problems)
- venue    shrunk mean track record of the venue's earlier papers

Outcomes: y_corpus = top quartile of corpus citations received in T+1..T+3 (as in exp2); y_global =
top quartile of OpenAlex cited_by_count (2026) within the publication year. Duplicate records
(same normalized title) are collapsed first. Writes data/backtest/exp5_papers.parquet.
"""
import re

import numpy as np
import pandas as pd
from scipy import sparse

from landscape import embed, store
from landscape.cli import _params

BASE = "dro-pilot"
CUTOFFS = (2020, 2021, 2022, 2023)
K = 20
con = store.connect(read_only=True)
p = _params(con, BASE)
VER, DIM = p.embedding.version, p.embedding.graph_dim

papers = con.execute("""SELECT v.paper_id, v.route, p.year, p.type, p.venue_id, p.title, p.doi,
                               coalesce(p.cited_by_count, 0) gcites
                        FROM provenance v JOIN papers p USING (paper_id)
                        WHERE v.run_id = ? AND p.year IS NOT NULL""", [BASE]).df()
# collapse duplicate records of one paper (preprint/article twins): keep the DOI'd, most-cited one
papers["tkey"] = papers.title.fillna("").str.lower().map(lambda t: re.sub(r"[^a-z0-9]", "", t))
papers = papers.assign(has_doi=papers.doi.notna()).sort_values(["has_doi", "gcites"], ascending=False)
generic = papers.tkey.str.len() < 12
dup = papers.duplicated("tkey") & ~generic
dups_dropped = int(dup.sum())
papers = papers[~dup].reset_index(drop=True)
year = papers.set_index("paper_id").year
keep = set(papers.paper_id)

refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v
                      ON v.paper_id = r.citing_id AND v.run_id = ?""", [BASE]).df()
refs = refs[refs.citing_id.isin(keep)]
rc = refs[refs.cited_id.isin(keep)].assign(cy=lambda d: d.citing_id.map(year))
au = con.execute("""SELECT s.paper_id, s.author_id FROM authorships s JOIN provenance v
                    ON v.paper_id = s.paper_id AND v.run_id = ?""", [BASE]).df()
au = au[au.paper_id.isin(keep)]
au["year"] = au.paper_id.map(year)
n_auth = au.groupby("paper_id").author_id.nunique()

# self-citations: citing and cited paper share an author (kept for features, removable from outcomes)
aset = au.groupby("paper_id").author_id.apply(frozenset)
rc["self"] = [bool(aset.get(a, frozenset()) & aset.get(b, frozenset())) for a, b in zip(rc.citing_id, rc.cited_id)]
pidx = {x: k for k, x in enumerate(papers.paper_id)}
widx = {w: k for k, w in enumerate(refs.cited_id.unique())}
R = sparse.csr_matrix((np.ones(len(refs)), (refs.citing_id.map(pidx), refs.cited_id.map(widx))),
                      shape=(len(pidx), len(widx)))
R.data[:] = 1
nrefs = np.asarray(R.sum(axis=1)).ravel()


def vecs(facet):
    ids, v = embed.load_vectors(facet, ids=papers.paper_id.tolist(), version=VER)
    v = embed.truncate(v, DIM)
    return {x: k for k, x in enumerate(ids)}, v


V = {f: vecs(f) for f in ("abstract", "problem", "method", "innovation", "limitations")}

# time-sliced global counts (scripts/phase8b/fetch_counts_by_year.py): citations a work received in
# each year, and citations / new works per author per year. "As of T" = years <= T only.
wc = pd.read_parquet("data/backtest/oa_work_counts.parquet").pivot_table(
    index="paper_id", columns="year", values="cites", aggfunc="sum").fillna(0)
ac = pd.read_parquet("data/backtest/oa_author_counts.parquet")
ac = ac[(ac.year >= 2010) & (ac.year <= 2026)]
acc = ac.pivot_table(index="author_id", columns="year", values="cites", aggfunc="sum").fillna(0)
acw = ac.pivot_table(index="author_id", columns="year", values="works", aggfunc="sum").fillna(0)


def wsum(m, ids, y0, y1):
    cols = [c for c in m.columns if y0 <= c <= y1]
    return m.reindex(ids)[cols].sum(axis=1).fillna(0) if cols else pd.Series(0.0, index=ids)


def mat(facet, ids):
    pos, v = V[facet]
    ok = [x for x in ids if x in pos]
    return ok, v[[pos[x] for x in ok]] if ok else np.zeros((0, DIM), np.float32)


def topk_mean(S, k):
    k = min(k, S.shape[1])
    return np.sort(S, axis=1)[:, -k:].mean(axis=1) if k else np.full(S.shape[0], np.nan)


rows = []
for T in CUTOFFS:
    pop = papers[(papers.year == T) & papers.type.isin(["article", "preprint", "conference-paper",
                                                         "proceedings-article"])].paper_id.tolist()
    pop = [x for x in pop if x in V["problem"][0] and x in V["abstract"][0]]
    cites_T = rc[rc.cy <= T].groupby("cited_id").size()
    fut = rc[(rc.cy > T) & (rc.cy <= T + 3)].groupby("cited_id").size()
    fut_ns = rc[(rc.cy > T) & (rc.cy <= T + 3) & ~rc.self].groupby("cited_id").size()
    # within-year citation percentile at T: the age-free "track record" of every earlier paper
    early = papers[papers.year < T][["paper_id", "year", "venue_id"]].copy()
    early["c"] = early.paper_id.map(cites_T).fillna(0)
    early["pct"] = early.groupby("year").c.rank(pct=True)
    pct = early.set_index("paper_id").pct
    # the same with global citations received by T, and received in year T alone (current attention)
    early["g"] = wsum(wc, early.paper_id.tolist(), 0, T).to_numpy()
    early["gnow"] = wsum(wc, early.paper_id.tolist(), T, T).to_numpy()
    early["gpct"] = early.groupby("year").g.rank(pct=True)
    early["gnow_pct"] = early.groupby("year").gnow.rank(pct=True)
    gpct, gnow_pct = early.set_index("paper_id").gpct, early.set_index("paper_id").gnow_pct
    F = pd.DataFrame({"paper_id": pop, "T": T})

    # ---- base
    pastp = papers[papers.year <= T].paper_id.tolist()
    C = (R[[pidx[x] for x in pop]] @ R[[pidx[x] for x in pastp]].T).tocsr()
    C.data = (C.data >= 2).astype(float)
    F["cites_so_far"] = [cites_T.get(x, 0) for x in pop]
    F["g_cites_so_far"] = wsum(wc, pop, 0, T).to_numpy()
    F["coupling"] = np.maximum(np.asarray(C.sum(axis=1)).ravel() - 1, 0)
    F["n_refs"] = [nrefs[pidx[x]] for x in pop]
    F["n_authors"] = [n_auth.get(x, 0) for x in pop]
    meta = papers.set_index("paper_id")
    F["preprint"] = [float(meta.type[x] == "preprint") for x in pop]
    F["route"] = [meta.route[x] for x in pop]

    # ---- topic: neighbourhood among earlier papers (T-5..T-1)
    nb_ids = papers[(papers.year < T) & (papers.year >= T - 5)].paper_id.tolist()
    for facet, tag in (("abstract", "t"), ("problem", "q")):
        _, X = mat(facet, pop)
        nid, N = mat(facet, nb_ids)
        S = X @ N.T
        top = np.argsort(-S, axis=1)[:, :K]
        npct = np.array([pct.get(x, np.nan) for x in nid])
        F[f"{tag}_heat"] = np.nanmean(npct[top], axis=1)
        F[f"{tag}_gheat"] = np.nanmean(np.array([gpct.get(x, np.nan) for x in nid])[top], axis=1)
        F[f"{tag}_gnow"] = np.nanmean(np.array([gnow_pct.get(x, np.nan) for x in nid])[top], axis=1)
        w = np.take_along_axis(S, top, axis=1)
        F[f"{tag}_heat_w"] = np.nansum(npct[top] * w, axis=1) / w.sum(axis=1)
        F[f"{tag}_typicality"] = w.mean(axis=1)
        if facet == "abstract":
            radius = w[:, -1]
            nyr = np.array([year[x] for x in nid])
            _, XT = mat("abstract", pop)
            same = (X @ XT.T) >= radius[:, None]
            F["t_crowding"] = (same.sum(axis=1) - 1) / max(len(pop), 1)
            prev = (S >= radius[:, None]) & (nyr == T - 1)[None, :]
            F["t_momentum"] = np.log((F.t_crowding + 1e-3) / (prev.sum(axis=1) / max((nyr == T - 1).sum(), 1) + 1e-3))
            # drift: is the paper ahead of where its 50-neighbourhood moved from T-5..T-3 to T-2..T-1?
            top50 = np.argsort(-S, axis=1)[:, :50]
            drift = []
            for i in range(len(pop)):
                nb = top50[i]
                rec, old = nb[nyr[nb] >= T - 2], nb[nyr[nb] <= T - 3]
                if len(rec) < 5 or len(old) < 5:
                    drift.append(np.nan)
                    continue
                d = N[rec].mean(0) - N[old].mean(0)
                off = X[i] - N[nb].mean(0)
                drift.append(float(off @ d / (np.linalg.norm(off) * np.linalg.norm(d) + 1e-9)))
            F["t_drift"] = drift

    # ---- question
    cur = papers[(papers.year >= T - 2) & (papers.year <= T)].paper_id.tolist()
    iid, I = mat("innovation", pop)
    cid, P = mat("problem", cur)
    Sip = I @ P.T
    Sip[np.array(iid)[:, None] == np.array(cid)[None, :]] = -1  # not the paper's own problem
    thr = np.quantile(Sip, 0.99)
    reach = pd.Series(topk_mean(Sip, 20), index=iid)
    breadth = pd.Series((Sip >= thr).sum(axis=1), index=iid)
    F["q_reach"] = F.paper_id.map(reach)
    F["q_breadth"] = F.paper_id.map(breadth)
    lid, L = mat("limitations", papers[papers.year < T].paper_id.tolist())
    if len(lid) >= 20:
        gc = L.mean(0)
        gc /= np.linalg.norm(gc)
        L = L[(L @ gc) <= np.quantile(L @ gc, 0.8)]
        F["q_open_match"] = F.paper_id.map(pd.Series(topk_mean(I @ L.T, 3), index=iid))
    # method atypicality for the question: distance of the paper's method from the methods its
    # problem-neighbours used
    _, Xp = mat("problem", pop)
    pid_nb, Pn = mat("problem", nb_ids)
    mid_nb_pos = V["method"][0]
    top = np.argsort(-(Xp @ Pn.T), axis=1)[:, :K]
    mx_id, Mx = mat("method", pop)
    mpos = {x: k for k, x in enumerate(mx_id)}
    atyp = []
    for i, x in enumerate(pop):
        nbm = [mid_nb_pos[pid_nb[j]] for j in top[i] if pid_nb[j] in mid_nb_pos]
        if x not in mpos or len(nbm) < 5:
            atyp.append(np.nan)
            continue
        m = V["method"][1][nbm].mean(0)
        atyp.append(1 - float(Mx[mpos[x]] @ m / np.linalg.norm(m)))
    F["q_method_atypical"] = atyp

    # ---- author credibility (corpus-only, before T)
    prev = au[au.year < T].copy()
    prev["pct"] = prev.paper_id.map(pct)
    a_track = prev.groupby("author_id").pct.mean()
    a_n = prev.groupby("author_id").paper_id.nunique()
    ppos, Pv = V["problem"]
    a_prob = {}
    for a, g in prev.groupby("author_id"):
        k = [ppos[x] for x in g.paper_id if x in ppos]
        if k:
            c = Pv[k].mean(0)
            a_prob[a] = c / np.linalg.norm(c)
    tr, ex, ce, npr, nhist = [], [], [], [], []
    for x in pop:
        A = au[au.paper_id == x].author_id.unique()
        hist = [a for a in A if a in a_track.index]
        nhist.append(len(hist))
        tr.append(max((a_track[a] for a in hist), default=np.nan))
        npr.append(max((a_n[a] for a in hist), default=0))
        e = [(float(Pv[ppos[x]] @ a_prob[a]), a_track[a]) for a in hist if a in a_prob and x in ppos]
        ex.append(max((s for s, _ in e), default=np.nan))
        ce.append(max((s * t for s, t in e), default=np.nan))
    F["a_track"], F["a_expert"], F["a_cred_expert"] = tr, ex, ce
    F["a_n_prior"], F["a_n_with_history"] = npr, nhist

    # ---- global author credibility as of T (OpenAlex yearly counts, all of the author's work)
    A = au[au.paper_id.isin(pop)][["paper_id", "author_id"]].drop_duplicates()
    aid = A.author_id.unique().tolist()
    c_cum, c_rec = wsum(acc, aid, 0, T), wsum(acc, aid, T - 2, T)
    w_cum = wsum(acw, aid, 0, T)
    A["c_cum"], A["c_rec"], A["w_cum"] = A.author_id.map(c_cum), A.author_id.map(c_rec), A.author_id.map(w_cum)
    A["rate"] = np.where(A.w_cum >= 3, A.c_cum / A.w_cum.clip(lower=1), np.nan)
    ga = A.groupby("paper_id").agg(c_cum=("c_cum", "max"), c_rec=("c_rec", "max"), w_cum=("w_cum", "max"),
                                   rate=("rate", "max"), team=("c_cum", lambda v: np.log1p(v).sum()))
    for c in ("c_cum", "c_rec", "w_cum", "rate", "team"):
        F["ga_" + c] = F.paper_id.map(ga[c]).to_numpy()

    # ---- venue credibility (shrunk to 0.5 with 5 pseudo-papers)
    vg = early.dropna(subset=["venue_id"]).groupby("venue_id").pct.agg(["sum", "count"])
    vp = (vg["sum"] + 2.5) / (vg["count"] + 5)
    F["v_prestige"] = [vp.get(meta.venue_id[x], 0.5) if isinstance(meta.venue_id[x], str) else 0.5 for x in pop]
    vgg = early.dropna(subset=["venue_id"]).groupby("venue_id").gpct.agg(["sum", "count"])
    vpg = (vgg["sum"] + 2.5) / (vgg["count"] + 5)
    F["v_gprestige"] = [vpg.get(meta.venue_id[x], 0.5) if isinstance(meta.venue_id[x], str) else 0.5 for x in pop]
    F["v_n_prior"] = [vg["count"].get(meta.venue_id[x], 0) if isinstance(meta.venue_id[x], str) else 0 for x in pop]

    # ---- imports from outside the corpus: external works (not in the corpus) that corpus papers cite.
    # A rising import is one whose citations from corpus papers in T-1..T outpace T-4..T-2.
    ext = refs[~refs.cited_id.isin(keep)].assign(cy=lambda d: d.citing_id.map(year))
    e_old = ext[(ext.cy >= T - 4) & (ext.cy <= T - 2)].groupby("cited_id").size()
    e_new = ext[(ext.cy >= T - 1) & (ext.cy <= T)].groupby("cited_id").size()
    e_all = ext[ext.cy <= T].groupby("cited_id").size()
    rise = (np.log((e_new.reindex(e_all.index).fillna(0) + 1) / 2) - np.log((e_old.reindex(e_all.index).fillna(0) + 1) / 3))
    rise = rise[e_all >= 3]
    mine = ext[ext.citing_id.isin(pop)]
    g = mine.assign(r=mine.cited_id.map(rise), n=mine.cited_id.map(e_all).fillna(1)).groupby("citing_id")
    n_in = rc[rc.citing_id.isin(pop)].groupby("citing_id").size()
    F["x_ext_share"] = [1 - n_in.get(x, 0) / max(nrefs[pidx[x]], 1) for x in pop]
    F["x_rising_max"] = F.paper_id.map(g.r.max()).to_numpy()
    F["x_n_rising"] = F.paper_id.map(g.r.apply(lambda v: (v > np.log(2)).sum())).fillna(0).to_numpy()
    # first adopters: external works cited at most once by corpus papers before this one (new imports)
    F["x_n_fresh"] = F.paper_id.map(g.n.apply(lambda v: (v <= 2).sum())).fillna(0).to_numpy()

    # ---- outcomes
    F["fut_corpus"] = [fut.get(x, 0) for x in pop]
    F["fut_corpus_ns"] = [fut_ns.get(x, 0) for x in pop]
    F["gcites"] = [meta.gcites[x] for x in pop]
    F["g3"] = wsum(wc, pop, T + 1, T + 3).to_numpy()  # global citations received in T+1..T+3
    rows.append(F)
    print(T, len(F), flush=True)

df = pd.concat(rows, ignore_index=True)
# field-normalized global outcomes: global counts mostly measure field citation norms (an ML paper
# out-cites an OR-theory paper of equal standing), so compare each paper with like papers only
sub = con.execute("""SELECT paper_id, json_extract_string(topics, '$[0].subfield') sf,
                            json_extract_string(topics, '$[0].field') fd FROM papers""").df().set_index("paper_id")
df["subfield"] = df.paper_id.map(sub.sf).fillna("?")
n_sf = df.groupby(["T", "subfield"]).paper_id.transform("size")
df["field_group"] = np.where(n_sf >= 15, df.subfield, df.paper_id.map(sub.fd).fillna("?"))
lg = np.log1p(df.gcites.astype(float))
df["g_field_rel"] = lg - lg.groupby([df["T"], df.field_group]).transform("median")
l3 = np.log1p(df.g3.astype(float))
df["g3_field_rel"] = l3 - l3.groupby([df["T"], df.field_group]).transform("median")
# peers: the 15 nearest same-year papers (abstract vectors, whole corpus, any route)
apos, Av = V["abstract"]
gl = np.log1p(papers.set_index("paper_id").gcites.astype(float))
rel, crel, crel_ns, rel3 = [], [], [], []
fut_all = {}
for T in CUTOFFS:
    for ns in (False, True):
        c = rc[(rc.cy > T) & (rc.cy <= T + 3) & (~rc.self if ns else True)].groupby("cited_id").size()
        fut_all[T, ns] = np.log1p(c.reindex(papers.paper_id).fillna(0).to_numpy())
fpos = {x: k for k, x in enumerate(papers.paper_id)}
for T, g in df.groupby("T"):
    same = [x for x in papers[papers.year == T].paper_id if x in apos]
    S = Av[[apos[x] for x in g.paper_id]] @ Av[[apos[x] for x in same]].T
    S[np.array(g.paper_id)[:, None] == np.array(same)[None, :]] = -2
    top = np.argsort(-S, axis=1)[:, :15]
    peer = gl.reindex(same).to_numpy()[top].mean(axis=1)
    rel.append(pd.Series(lg[g.index].to_numpy() - peer, index=g.index))
    g3s = np.log1p(wsum(wc, same, T + 1, T + 3).to_numpy())
    rel3.append(pd.Series(l3[g.index].to_numpy() - g3s[top].mean(axis=1), index=g.index))
    for ns, lst in ((False, crel), (True, crel_ns)):
        fa = fut_all[T, ns][[fpos[x] for x in same]]
        own = g.fut_corpus_ns if ns else g.fut_corpus
        lst.append(pd.Series(np.log1p(own.to_numpy().astype(float)) - fa[top].mean(axis=1), index=g.index))
df["g_peer_rel"] = pd.concat(rel)
df["c_peer_rel"] = pd.concat(crel)  # in-field citations relative to same-year semantic peers
df["c_peer_rel_ns"] = pd.concat(crel_ns)  # the same without self-citations
df["g3_peer_rel"] = pd.concat(rel3)
for c, y in (("g_field_rel", "y_field"), ("g_peer_rel", "y_peer"), ("c_peer_rel", "y_corpus_peer"),
             ("c_peer_rel_ns", "y_corpus_peer_ns"), ("g3", "y_g3"), ("g3_field_rel", "y_g3_field"),
             ("g3_peer_rel", "y_g3_peer")):
    df[y] = df[c] > df.groupby("T")[c].transform(lambda s: s.quantile(0.75))
df["y_corpus"] = df.fut_corpus > df.groupby("T").fut_corpus.transform(lambda s: s.quantile(0.75))
df["y_global"] = df.gcites > df.groupby("T").gcites.transform(lambda s: s.quantile(0.75))
df.attrs["dups_dropped"] = dups_dropped
df.to_parquet("data/backtest/exp5_papers.parquet")
print(f"{len(df)} papers; {dups_dropped} duplicate records collapsed")
print(df.describe().T.round(3).to_string())
