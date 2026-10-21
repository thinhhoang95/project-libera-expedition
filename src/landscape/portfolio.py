"""Product portfolio: the backtested results turned into live, per-run artefacts for the explorer's
Start here and My paper views (see docs/features/09-portfolio.md and findings-portfolio.md).

    landscape portfolio --run-id R      import map, live external imports, community warmth for every
                                        corpus paper              -> data/portfolio/<R>/
    landscape my-paper  --run-id R --doi D
                                        any paper by DOI: facets (one LLM call unless it is in the
                                        corpus), embeddings, warmth -> data/portfolio/<R>/papers/<id>.json

Models come from the experiments and are not refitted on the live run's outcomes:
- community warmth: src/landscape/models/community_warmth.json (experiment 10: within-paper AUC 0.81,
  hit@3 0.69 vs 0.25 random);
- external imports: the experiment 6 model (strict hit), fitted on the cutoffs at build time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import embed, external, store
from .backtest import Assigner
from .config import RunParams

log = logging.getLogger(__name__)
MODELS = Path(__file__).parent / "models"
ALPHA, RES = 0.5, "medium"          # the communities the bridges and warmth are defined on
DIM = 256


def out_dir(run_id: str) -> Path:
    d = store.DATA / "portfolio" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_score(name: str, X: pd.DataFrame) -> np.ndarray:
    spec = json.loads((MODELS / f"{name}.json").read_text())
    z = (X[spec["features"]].to_numpy(dtype=float) - np.array(spec["mean"])) / np.array(spec["scale"])
    return z @ np.array(spec["coef"]) + spec["intercept"]


class Field:
    """Corpus, references, vectors and live communities of a run, loaded once."""

    def __init__(self, con, run_id: str, p: RunParams):
        self.con, self.run, self.p = con, run_id, p
        ver = p.embedding.version
        self.papers = con.execute("""
            SELECT v.paper_id, v.route, p.year, p.title, p.type, p.doi,
                   json_extract_string(p.topics, '$[0].subfield') AS subfield
            FROM provenance v JOIN papers p USING (paper_id) WHERE v.run_id = ? AND p.year IS NOT NULL""",
                                  [run_id]).df()
        self.year = self.papers.set_index("paper_id").year
        ids = self.papers.paper_id.tolist()
        self.t_live = int(self.papers.year.max()) - 1   # the latest complete publication year
        refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v ON v.paper_id = r.citing_id
                              AND v.run_id = ?""", [run_id]).df()
        self.refs = refs
        self.pidx = {x: k for k, x in enumerate(ids)}
        self.widx = {w: k for k, w in enumerate(refs.cited_id.unique())}
        self.R = sparse.csr_matrix((np.ones(len(refs)), (refs.citing_id.map(self.pidx), refs.cited_id.map(self.widx))),
                                   shape=(len(ids), len(self.widx)))
        self.R.data[:] = 1
        self.vec768, self.vec256 = {}, {}
        for f in ("abstract", "problem", "method", "innovation", "limitations"):
            fid, v = embed.load_vectors(f, ids=ids, version=ver)
            self.vec768[f] = ({x: k for k, x in enumerate(fid)}, v)
            self.vec256[f] = ({x: k for k, x in enumerate(fid)}, embed.truncate(v, DIM))
        # live communities: every paper assigned to the nearest live centroid (as in the backtest)
        self.lab = Assigner(con, run_id, ver).assign("problem", ALPHA, ids)
        cl = con.execute("""SELECT cluster_id, label, kind, size FROM clusters WHERE run_id = ? AND facet = 'problem'
                            AND alpha = ? AND resolution = ?""", [run_id, ALPHA, RES]).df()
        self.clusters = cl.set_index("cluster_id")
        self.cids = [c for c in self.clusters.index if (self.lab == c).any()]

    def rows(self, facet: str, ids, dim=DIM):
        pos, v = (self.vec256 if dim == DIM else self.vec768)[facet]
        ok = [x for x in ids if x in pos]
        return ok, (v[[pos[x] for x in ok]] if ok else np.zeros((0, v.shape[1] if len(v) else dim), np.float32))


# ------------------------------------------------------------------ community warmth

class Warmth:
    """Score every live community for a paper with the experiment-10 model. Feature definitions follow the
    training data exactly (experiment 1 for coupling and the 768-d similarities, experiment 9 for recent
    closeness)."""

    def __init__(self, f: Field):
        self.f = f
        lab = f.lab
        members = {c: lab.index[lab.values == c].tolist() for c in f.cids}
        self.size = pd.Series({c: len(m) for c, m in members.items()})
        M = sparse.csr_matrix((np.ones(len(lab)), ([f.cids.index(c) if c in f.cids else 0 for c in lab.values],
                                                    [f.pidx[x] for x in lab.index])), shape=(len(f.cids), len(f.pidx)))
        self.XR = (M @ f.R).tocsr()                      # communities x works: how many members cite each work
        _, lv_all = f.vec768["limitations"]
        gc = lv_all.mean(axis=0)
        gc /= np.linalg.norm(gc)
        cut = np.quantile(lv_all @ gc, 0.8) if len(lv_all) else 1.0
        self.pc, self.lim, self.recent = [], [], []
        for c in f.cids:
            _, q = f.rows("problem", members[c], dim=768)
            q = q.mean(axis=0)
            self.pc.append(q / np.linalg.norm(q))
            _, lv = f.rows("limitations", members[c], dim=768)
            self.lim.append(lv[(lv @ gc) <= cut] if len(lv) else lv)
            rec = [x for x in members[c] if f.t_live - 2 <= f.year[x] <= f.t_live]
            _, a = f.rows("method", rec)
            _, b = f.rows("innovation", rec)
            self.recent.append(np.vstack([a, b]))
        self.pc = np.vstack(self.pc)

    def features(self, inn768: np.ndarray, ref_ids: list[str]) -> pd.DataFrame:
        """inn768: (n, 768) innovation vectors; ref_ids: one reference list per paper."""
        f = self.f
        n = len(inn768)
        r = sparse.lil_matrix((n, len(f.widx)))
        for k, refs in enumerate(ref_ids):
            for w in refs:
                if w in f.widx:
                    r[k, f.widx[w]] = 1
        coup = (r.tocsr() @ self.XR.T).toarray() / self.size[f.cids].to_numpy()[None, :]
        prob = inn768 @ self.pc.T
        lim = np.column_stack([np.sort(inn768 @ L.T, axis=1)[:, -3:].mean(axis=1) if len(L) >= 1 else prob[:, j]
                               for j, L in enumerate(self.lim)])
        i256 = embed.truncate(inn768, DIM)
        rec = np.column_stack([(i256 @ Rv.T).max(axis=1) if len(Rv) else np.full(n, -1.0) for Rv in self.recent])
        out = []
        for j, c in enumerate(f.cids):
            out.append(pd.DataFrame({"row": np.arange(n), "cluster_id": c, "l_size": np.log(self.size[c]),
                                     "lcoup": np.log1p(coup[:, j] * 100), "prob_sim": prob[:, j],
                                     "lim_sim3_f": lim[:, j], "sim_recent": rec[:, j].clip(-1)}))
        X = pd.concat(out, ignore_index=True)
        X["warmth"] = _model_score("community_warmth", X)
        X["content_fit"] = X.prob_sim
        return X


def _rank(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["rank"] = X.groupby("row").warmth.rank(ascending=False, method="first").astype(int)
    return X


def corpus_warmth(f: Field, w: Warmth) -> pd.DataFrame:
    """Warmth of every live community for every corpus paper with an innovation statement. The paper's
    own community is marked *home*; communities with a member already citing it, *already citing*."""
    ids, inn = f.rows("innovation", f.papers.paper_id.tolist(), dim=768)
    refs = f.refs.groupby("citing_id").cited_id.apply(list)
    X = w.features(inn, [refs.get(x, []) for x in ids])
    X["paper_id"] = [ids[r] for r in X.row]
    cites = f.refs[f.refs.cited_id.isin(set(ids))].assign(X=lambda d: d.citing_id.map(f.lab)).dropna(subset=["X"])
    citing = set(zip(cites.cited_id, cites.X))
    X["home"] = X.cluster_id.to_numpy() == X.paper_id.map(f.lab).to_numpy()
    X["already_citing"] = [(p, c) in citing for p, c in zip(X.paper_id, X.cluster_id)]
    cand = X[~X.home & ~X.already_citing]
    X = X.merge(_rank(cand)[["paper_id", "cluster_id", "rank"]], on=["paper_id", "cluster_id"], how="left")
    return X.drop(columns="row")


# ------------------------------------------------------------------ external imports (experiment 6)

def import_map(f: Field, works: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    m = edges.merge(works[["paper_id", "subfield"]], left_on="cited_id", right_on="paper_id", how="inner")
    y1 = f.t_live
    imp = m[m.citing_year.between(y1 - 8, y1)].groupby(["citing_year", "subfield"]).size().rename("citations").reset_index()
    imp["share"] = imp.citations / imp.groupby("citing_year").citations.transform("sum")
    early = imp[imp.citing_year.between(y1 - 8, y1 - 6)].groupby("subfield").share.mean()
    late = imp[imp.citing_year.between(y1 - 2, y1)].groupby("subfield").share.mean()
    t = pd.DataFrame({"share_early": early, "share_recent": late}).fillna(0)
    t["change_points"] = 100 * (t.share_recent - t.share_early)
    t["corpus_share"] = t.index.map(f.papers.subfield.value_counts(normalize=True)).fillna(0)
    t.attrs = {}
    return t.sort_values("share_recent", ascending=False).reset_index().rename(columns={"index": "subfield"}).assign(
        early_years=f"{y1 - 8}-{y1 - 6}", recent_years=f"{y1 - 2}-{y1}")


def live_imports(f: Field, works: pd.DataFrame, edges: pd.DataFrame, cutoffs=(2020, 2021, 2022, 2023)) -> pd.DataFrame:
    """Experiment 6 (strict hit) fitted on the cutoffs, applied at the latest complete year."""
    run = f.run
    n_year = f.papers.groupby("year").size()
    meta = works.set_index("paper_id")
    gc = pd.DataFrame([{"paper_id": k, "year": c["year"], "cites": c.get("cited_by_count", 0)}
                       for k, s in zip(works.paper_id, works.counts_by_year) for c in json.loads(s)])
    G = gc.pivot_table(index="paper_id", columns="year", values="cites", aggfunc="sum").fillna(0)
    XA, XM, XI = (external.vectors(run, x, DIM) for x in ("abstract", "method", "innovation"))
    cy = edges.groupby(["cited_id", "citing_year"]).citing_id.nunique().unstack(fill_value=0)
    ext_ids = [x for x in works.paper_id if x in cy.index]
    cy = cy.reindex(ext_ids)

    def gsum(ids, y0, y1):
        cols = [c for c in G.columns if y0 <= c <= y1]
        return G.reindex(ids)[cols].sum(axis=1).fillna(0).to_numpy() if cols else np.zeros(len(ids))

    def csum(y0, y1):
        cols = [c for c in cy.columns if y0 <= c <= y1]
        return cy[cols].sum(axis=1) if cols else pd.Series(0, index=cy.index)

    def nsum(y0, y1):
        return float(n_year[(n_year.index >= y0) & (n_year.index <= y1)].sum())

    def ext_vec(ids):
        out = np.full((len(ids), DIM), np.nan, np.float32)
        for pos, v in (XA, XM, XI):  # innovation, else method, else abstract
            for k, x in enumerate(ids):
                if x in pos:
                    out[k] = v[pos[x]]
        return out

    def features(T, cand):
        before = csum(T - 2, T)[cand] / nsum(T - 2, T)
        F = pd.DataFrame({"paper_id": cand, "T": T})
        F["l_rate_before"] = np.log(before.to_numpy() + 1e-4)
        F["infield_mom"] = np.log((csum(T, T)[cand].to_numpy() / max(nsum(T, T), 1) + 1e-4)
                                  / (csum(T - 2, T - 1)[cand].to_numpy() / max(nsum(T - 2, T - 1), 1) + 1e-4))
        F["l_global_cum"] = np.log1p(gsum(cand, 0, T))
        F["age"] = T - meta.year.reindex(cand).fillna(T).to_numpy()
        F["g_trend"] = np.log((gsum(cand, T, T) + 1) / (gsum(cand, T - 2, T - 1) / 2 + 1))
        F["g_trend_3y"] = np.log((gsum(cand, T - 1, T) / 2 + 1) / (gsum(cand, T - 4, T - 2) / 3 + 1))
        X = ext_vec(cand)
        has = ~np.isnan(X).any(axis=1)
        Xs = np.nan_to_num(X)
        rec_ok, Pr = f.rows("problem", f.papers[f.papers.year.between(T - 1, T)].paper_id)
        _, Po = f.rows("problem", f.papers[f.papers.year.between(T - 4, T - 2)].paper_id)
        Srec = Xs @ Pr.T
        F["fit_now"] = np.sort(Srec, axis=1)[:, -20:].mean(axis=1)
        F["fit_shift"] = F.fit_now - np.sort(Xs @ Po.T, axis=1)[:, -20:].mean(axis=1)
        _, Lv = f.rows("limitations", f.papers[f.papers.year <= T].paper_id)
        F["lim_match"] = np.sort(Xs @ Lv.T, axis=1)[:, -3:].mean(axis=1) if len(Lv) else np.nan
        for c in ("fit_now", "fit_shift", "lim_match"):
            F.loc[~has, c] = np.nan
        sf = f.papers[f.papers.year <= T].subfield.value_counts(normalize=True)
        F["externality"] = 1 - meta.subfield.reindex(cand).map(sf).fillna(0).to_numpy()
        F["nearest_corpus_paper"] = [rec_ok[j] for j in Srec.argmax(axis=1)] if len(rec_ok) else None
        F.loc[~has, "nearest_corpus_paper"] = None
        return F

    feats = ["l_rate_before", "infield_mom", "l_global_cum", "age", "g_trend", "g_trend_3y", "fit_now", "fit_shift",
             "lim_match", "externality"]
    hist = []
    for T in cutoffs:
        cum = csum(0, T)
        cand = [x for x in ext_ids if cum[x] >= 5 and (meta.year.get(x) or 9999) <= T]
        F = features(T, cand)
        after = csum(T + 1, T + 3)[cand] / nsum(T + 1, T + 3)
        before = csum(T - 2, T)[cand] / nsum(T - 2, T)
        F["growth"] = np.log((after.to_numpy() + 1e-4) / (before.to_numpy() + 1e-4))
        F["after_n"] = csum(T + 1, T + 3)[cand].to_numpy()
        hist.append(F)
    D = pd.concat(hist, ignore_index=True)
    D["hit"] = (D.growth > D.groupby("T").growth.transform(lambda s: s.quantile(0.75))) & (D.after_n >= 3)
    med = D[feats].median()
    D[feats] = D[feats].fillna(med)
    sc = StandardScaler().fit(D[feats])
    mdl = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(D[feats]), D.hit)
    T = f.t_live
    recent_n = csum(T - 2, T)
    L = features(T, [x for x in ext_ids if (meta.year.get(x) or 9999) <= T and recent_n.get(x, 0) >= 3])
    L[feats] = L[feats].fillna(med)
    L["score"] = mdl.decision_function(sc.transform(L[feats]))
    # outside the field = the third farthest from the corpus's abstracts (abstracts only: title-only records
    # embed far from everything; OpenAlex subfields mislabel many in-field works)
    xid = [x for x in L.paper_id if x in XA[0]]
    Xa = XA[1][[XA[0][x] for x in xid]]
    _, Ca = f.rows("abstract", f.papers.paper_id)
    dist = pd.Series(1 - np.sort(Xa @ Ca.T, axis=1)[:, -20:].mean(axis=1), index=xid)
    with_abs = set(works.paper_id[works.abstract.fillna("").str.len() >= 100])
    L["sem_distance"] = L.paper_id.map(dist).where(L.paper_id.isin(with_abs))
    L["outside_field"] = L.sem_distance >= L.sem_distance.quantile(2 / 3)
    L["closest_corpus_paper"] = L.nearest_corpus_paper
    L["cluster_id"] = L.nearest_corpus_paper.map(f.lab)
    L["cluster_label"] = L.cluster_id.map(f.clusters.label)
    L["recent_citers"] = recent_n.reindex(L.paper_id).fillna(0).astype(int).to_numpy()
    L["corpus_citers_total"] = csum(0, 9999).reindex(L.paper_id).fillna(0).astype(int).to_numpy()
    L = L.merge(works[["paper_id", "title", "year", "type", "subfield", "field", "venue", "cited_by_count"]],
                on="paper_id", how="left")
    return L.sort_values("score", ascending=False).assign(as_of=T)


# ------------------------------------------------------------------ second waves

def second_waves(f: Field, W: pd.DataFrame, L: pd.DataFrame | None, edges: pd.DataFrame | None) -> pd.DataFrame:
    """Established ideas reaching new audiences: a watchlist of diffusion already under way, with evidence.

    - paper:    a paper in the map (published at least 3 years ago) that communities other than its own began
                citing in the last three complete years;
    - external: a work outside the map the field cites more and more (experiment 6), with the communities
                that began citing it lately;
    - method:   a method community used for problem communities it had not been used for before.
    Each row: new audiences, citing (or using) papers in the last three years vs the three before, likely next
    audiences from the backtested models (community warmth, experiment 10; method transfer, experiment 4;
    closest communities for external works), and supporting papers. Ordered by new audiences and momentum,
    within each kind: a watchlist, not a forecast of importance.
    """
    T = f.t_live
    lab = f.lab
    label = f.clusters.label
    rows = []

    def audiences(cite: pd.DataFrame, home):
        """cite: citing_id, cited_id, citing_year, X (citing community)."""
        out = []
        for idea, g in cite.groupby("cited_id"):
            g = g[g.X != home.get(idea)] if home is not None else g
            first = g.groupby("X").citing_year.min()
            new = first[first >= T - 2].index.tolist()
            recent = g[g.citing_year.between(T - 2, T)]
            earlier = g[g.citing_year.between(T - 5, T - 3)]
            if not new or recent.citing_id.nunique() < 3:
                continue
            ev = recent[recent.X.isin(new)].sort_values("citing_year", ascending=False).citing_id.unique()[:6].tolist()
            out.append({"idea_id": idea, "new_audiences": new, "n_new": len(new),
                        "recent": int(recent.citing_id.nunique()), "earlier": int(earlier.citing_id.nunique()),
                        "evidence": ev})
        return pd.DataFrame(out)

    # papers in the map
    corpus = set(f.papers.paper_id)
    rc = f.refs[f.refs.cited_id.isin(corpus)].assign(citing_year=lambda d: d.citing_id.map(f.year), X=lambda d: d.citing_id.map(lab))
    rc = rc.dropna(subset=["X", "citing_year"])
    old = set(f.papers[f.papers.year <= T - 3].paper_id)
    A = audiences(rc[rc.cited_id.isin(old)], lab)
    if len(A):
        nxt = (W[W["rank"].notna() & (W["rank"] <= 3)].groupby("paper_id").cluster_id.apply(list)
               if W is not None else pd.Series(dtype=object))
        t = f.papers.set_index("paper_id")
        for r in A.itertuples():
            rows.append({"kind": "paper", "idea_id": r.idea_id, "idea": t.title.get(r.idea_id), "type": t["type"].get(r.idea_id),
                         "origin": label.get(lab.get(r.idea_id), ""), "year": t.year.get(r.idea_id),
                         "new_audiences": [label.get(x, x) for x in r.new_audiences], "n_new": r.n_new,
                         "recent": r.recent, "earlier": r.earlier,
                         "likely_next": [label.get(x, x) for x in nxt.get(r.idea_id, []) if x not in r.new_audiences][:3],
                         "evidence": r.evidence,
                         "backed_by": "citations by new communities (descriptive); likely next: community warmth, exp10"})
    # external works
    if L is not None and edges is not None:
        e = edges.assign(X=edges.citing_id.map(lab)).dropna(subset=["X"])
        B = audiences(e[e.cited_id.isin(set(L.paper_id))], None)
        Lx = L.set_index("paper_id")
        for r in B.itertuples():
            x = Lx.loc[r.idea_id]
            nxt = [x.cluster_id] if isinstance(x.cluster_id, str) and x.cluster_id not in r.new_audiences else []
            rows.append({"kind": "external", "idea_id": r.idea_id, "idea": x.title, "origin": x.subfield, "type": x.get("type"),
                         "year": x.year, "new_audiences": [label.get(c, c) for c in r.new_audiences], "n_new": r.n_new,
                         "recent": r.recent, "earlier": r.earlier, "likely_next": [label.get(c, c) for c in nxt],
                         "evidence": r.evidence, "outside_field": bool(x.outside_field), "import_score": float(x.score),
                         "backed_by": "exp6 import model (+0.025 AUC; no sharper top of list)"})
    # methods reaching new problems
    pm = f.con.execute("""SELECT problem_cluster, method_cluster, year, count, first_year FROM pm_matrix
                          WHERE run_id = ? AND alpha = 0.5""", [f.run]).df()
    if len(pm):
        mlab = dict(f.con.execute("SELECT cluster_id, label FROM clusters WHERE run_id = ?", [f.run]).fetchall())
        tr = f.con.execute("""SELECT subject FROM signals WHERE run_id = ? AND signal_type = 'bridge_method_transfer'
                              ORDER BY rank""", [f.run]).df()
        trs = pd.DataFrame([json.loads(s) for s in tr.subject]) if len(tr) else pd.DataFrame(
            columns=["method_cluster", "problem_cluster"])
        pc = f.con.execute("""SELECT a.paper_id, a.cluster_id AS p, b.cluster_id AS m, pp.year FROM paper_cluster a
                              JOIN paper_cluster b ON a.paper_id = b.paper_id AND b.run_id = a.run_id AND b.facet = 'method'
                                   AND b.alpha = 0.5 AND b.resolution = 'medium'
                              JOIN papers pp ON pp.paper_id = a.paper_id
                              WHERE a.run_id = ? AND a.facet = 'problem' AND a.alpha = 0.5 AND a.resolution = 'medium'""",
                           [f.run]).df()
        for m, g in pm.groupby("method_cluster"):
            cells = g.groupby("problem_cluster").agg(first=("first_year", "min"))
            recent_n = g[g.year.between(T - 2, T)].groupby("problem_cluster")["count"].sum()
            new = [p for p in cells.index if cells.loc[p, "first"] >= T - 2 and recent_n.get(p, 0) >= 2]
            if not new:
                continue
            ev = pc[(pc.m == m) & pc.p.isin(new) & pc.year.between(T - 2, T)].sort_values("year", ascending=False)
            nxt = trs[trs.method_cluster == m].problem_cluster.tolist()[:3] if len(trs) else []
            rows.append({"kind": "method", "idea_id": m, "idea": mlab.get(m, m), "origin": "",
                         "year": int(g.first_year.min()), "new_audiences": [label.get(p, mlab.get(p, p)) for p in new],
                         "n_new": len(new), "recent": int(g[g.year.between(T - 2, T)]["count"].sum()),
                         "earlier": int(g[g.year.between(T - 5, T - 3)]["count"].sum()),
                         "likely_next": [label.get(p, mlab.get(p, p)) for p in nxt],
                         "evidence": ev.paper_id.unique()[:6].tolist(),
                         "backed_by": "new problem x method cells (descriptive); likely next: method transfer, exp4"})
    S = pd.DataFrame(rows)
    if S.empty:
        return S
    # quiet uptake (experiment 9): communities whose recent methods / innovations moved towards the idea without
    # citing it: closest recent statement >= 0.894 (99th percentile of random innovation-method pairs, 256-d) and
    # closer than the community's statements from 3-5 years before
    THR = 0.894
    comm = {}
    for c in f.cids:
        mem = lab.index[lab.values == c]
        rec = [x for x in mem if T - 2 <= f.year[x] <= T]
        old_ = [x for x in mem if T - 5 <= f.year[x] <= T - 3]
        comm[c] = (np.vstack([f.rows("method", rec)[1], f.rows("innovation", rec)[1]]),
                   np.vstack([f.rows("method", old_)[1], f.rows("innovation", old_)[1]]))
    ext_inn = external.vectors(f.run, "innovation", DIM) if edges is not None else ({}, None)
    cited_by = {}
    if edges is not None:
        for k, g in edges.assign(X=edges.citing_id.map(lab)).dropna(subset=["X"]).groupby("cited_id"):
            cited_by[k] = set(g.X)
    for k, g in rc.groupby("cited_id"):
        cited_by.setdefault(k, set()).update(g.X)
    quiet = []
    for r in S.itertuples():
        if r.kind == "paper":
            v = f.rows("innovation", [r.idea_id])[1]
        elif r.kind == "external" and r.idea_id in ext_inn[0]:
            v = ext_inn[1][[ext_inn[0][r.idea_id]]]
        else:
            quiet.append([])
            continue
        if not len(v):
            quiet.append([])
            continue
        out = []
        for c, (Rv, Ev) in comm.items():
            if c in cited_by.get(r.idea_id, set()) or c == lab.get(r.idea_id) or not len(Rv):
                continue
            rmax = float((Rv @ v[0]).max())
            emax = float((Ev @ v[0]).max()) if len(Ev) else -1.0
            if rmax >= THR and rmax > emax:
                out.append(label.get(c, c))
        quiet.append(out)
    S["quiet_uptake"] = quiet
    S["n_quiet"] = S.quiet_uptake.map(len)
    S["momentum"] = np.log((S.recent + 1) / (S.earlier + 1))
    # a second wave is an idea that spreads *and* gains use: research works only (textbooks are cited by every
    # community), external works from the last 15 years, more citing papers lately than in the 3 years before
    research = S.get("type", pd.Series(index=S.index, dtype=object)).isin(["article", "preprint", "conference-paper", "proceedings-article"]) | (S.kind == "method")
    recent_enough = (S.kind != "external") | (S.year >= T - 15)
    growing = (S.kind == "method") | (S.recent > S.earlier)
    S = S[research & recent_enough & growing]
    S = S.sort_values(["kind", "n_new", "momentum"], ascending=[True, False, False])
    S["as_of"] = T
    return S


# ------------------------------------------------------------------ build

def build_portfolio(con, run_id: str, p: RunParams) -> dict:
    f = Field(con, run_id, p)
    out = out_dir(run_id)
    stats = {"as_of": f.t_live, "built_at": datetime.now().isoformat(timespec="seconds")}
    w = Warmth(f)
    W = corpus_warmth(f, w)
    W.to_parquet(out / "warmth.parquet", index=False)
    stats["warmth_rows"] = len(W)
    ed = external.ext_dir(run_id)
    L = edges = None
    if (ed / "works.parquet").exists() and (ed / "vectors_abstract.parquet").exists():
        works = pd.read_parquet(ed / "works.parquet")
        edges = pd.read_parquet(ed / "edges.parquet").dropna(subset=["citing_year"])
        edges["citing_year"] = edges.citing_year.astype(int)
        import_map(f, works, edges).to_parquet(out / "import_map.parquet", index=False)
        L = live_imports(f, works, edges)
        L.to_parquet(out / "live_imports.parquet", index=False)
        stats.update(external_works=len(works), live_imports=len(L), outside_field=int(L.outside_field.sum()))
        ext_share = len(edges) / max(len(edges) + len(f.refs[f.refs.cited_id.isin(set(f.papers.paper_id))]), 1)
        stats["external_reference_share"] = round(ext_share, 3)
    else:
        stats["external"] = "not built: run `landscape external` first"
    S = second_waves(f, W, L, edges)
    S.to_parquet(out / "second_waves.parquet", index=False)
    stats["second_waves"] = S.kind.value_counts().to_dict() if len(S) else {}
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


# ------------------------------------------------------------------ any paper by DOI

async def _facets_for(p: RunParams, title: str, abstract: str) -> dict:
    from .extract import FIELDS, PROMPT_V1, SCHEMA, validate
    from .llm import OpenRouter

    ep = p.extraction
    client = OpenRouter(ep.model, ep.quantizations, 1, 0.05)
    try:
        prompt = PROMPT_V1.replace("{title}", title or "").replace("{abstract}", abstract)
        obj, cost, err = await client.chat_json(prompt, SCHEMA, "facets", max_tokens=1200)
        if obj is None or not validate(obj, abstract)[0]:
            obj, c2, err = await client.chat_json(prompt, SCHEMA, "facets", max_tokens=1200)
            cost += c2
        if obj is None:
            raise RuntimeError(f"facet extraction failed: {err}")
        return {k: obj.get(k) for k in FIELDS} | {"_cost_usd": cost}
    finally:
        await client.close()


def analyse_doi(con, run_id: str, p: RunParams, doi: str) -> Path:
    """Fetch a paper by DOI, extract its facets (unless it is in the corpus), embed, and score community warmth."""
    from .clients import OA_SELECT, Budget, OpenAlex, short_id
    from .ingest import rebuild_abstract

    async def get():
        oa = OpenAlex(Budget(cap_usd=0.05))
        try:
            data = await oa.get("/works", {"filter": f"doi:{doi.lower().removeprefix('https://doi.org/')}",
                                           "per_page": 1, "select": OA_SELECT})
        finally:
            await oa.close()
        res = data.get("results") or []
        if not res:
            raise SystemExit(f"OpenAlex has no work with DOI {doi}")
        return res[0]

    w = asyncio.run(get())
    wid = short_id(w["id"]) or w["id"]
    f = Field(con, run_id, p)
    refs = [short_id(r) for r in (w.get("referenced_works") or [])]
    abstract = rebuild_abstract(w.get("abstract_inverted_index")) or ""
    in_corpus = wid in f.pidx
    if in_corpus:
        facets = dict(con.execute("SELECT facet, text FROM facets WHERE paper_id = ?", [wid]).fetchall())
        vec = {k: f.rows(k, [wid], dim=768)[1] for k in ("problem", "method", "innovation", "limitations", "abstract")}
    else:
        if len(abstract) < 100:
            raise SystemExit("This work has no abstract on OpenAlex; facets and warmth need one.")
        facets = asyncio.run(_facets_for(p, w.get("title") or "", abstract))
        E = embed.Embedder(p.embedding)
        texts = {k: facets.get(k) for k in ("problem", "method", "innovation", "limitations")
                 if isinstance(facets.get(k), str) and facets[k].strip()}
        texts["abstract"] = f"{w.get('title') or ''}. {abstract}"
        enc = E.encode(list(texts.values()), show_progress=False)
        vec = {k: enc[j:j + 1] for j, k in enumerate(texts)}
    if "innovation" not in vec or not len(vec["innovation"]):
        raise SystemExit("No innovation statement could be extracted for this paper.")
    W = Warmth(f)
    X = W.features(vec["innovation"], [refs])
    corpus_citers = set(f.refs[f.refs.cited_id == wid].citing_id)
    X["already_citing"] = X.cluster_id.isin({f.lab.get(x) for x in corpus_citers})
    # home: the paper's own community if it is in the corpus; else the community citing it most; else the one
    # closest to its problem (nearest centroid is unreliable for papers far from the map)
    pv = vec.get("problem")
    by_citers = pd.Series([f.lab.get(x) for x in corpus_citers]).dropna().value_counts()
    if in_corpus:
        X["home"] = X.cluster_id == f.lab.get(wid)
    elif len(by_citers):
        X["home"] = X.cluster_id == by_citers.index[0]
    elif pv is not None and len(pv):
        X["home"] = X.cluster_id == f.cids[int(np.argmax(W.pc @ (pv[0] / np.linalg.norm(pv[0]))))]
    else:
        X["home"] = False
    cand = X[~X.home & ~X.already_citing]
    X = X.merge(_rank(cand)[["row", "cluster_id", "rank"]], on=["row", "cluster_id"], how="left").drop(columns="row")
    rec = {"paper_id": wid, "doi": w.get("doi"), "title": w.get("title"), "year": w.get("publication_year"),
           "in_corpus": in_corpus, "abstract": abstract, "facets": {k: v for k, v in facets.items() if k != "_cost_usd"},
           "llm_cost_usd": facets.get("_cost_usd", 0.0), "n_refs": len(refs), "refs": refs,
           "vectors": {k: v[0].round(6).tolist() for k, v in vec.items() if len(v)},
           "warmth": X.to_dict("records"), "analysed_at": datetime.now().isoformat(timespec="seconds")}
    d = out_dir(run_id) / "papers"
    d.mkdir(exist_ok=True)
    path = d / f"{wid}.json"
    path.write_text(json.dumps(rec, default=float) + "\n")
    return path
