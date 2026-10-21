"""Phase 6: signals and scoring.

Every signal writes ranked rows to `signals` (score, features, evidence paper ids). All code
takes a cutoff year T and only uses papers and citations up to T, so the Phase 8 backtest can
reuse it. Cluster-level signals (6.1-6.3) are computed at every yearly cutoff (map statuses)
and ranked at the latest one on the whole corpus and without the semantic route. Bridges (6.4)
and influence (6.5) are computed at the latest year.

Hub clusters (Phase 5, kind = 'hub') are scored like any cluster but ranked in their own list
(`group_kind = 'hub'`), so the unstable core of a field is reported, not dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from . import embed, store
from .config import RunParams, SignalParams, model_file, run_vectors
from .identity import work_aliases
from .llm import LLMBudgetExceeded, OpenRouter

log = logging.getLogger(__name__)


def z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    out = np.zeros_like(x)
    if ok.sum() > 1 and np.nanstd(x) > 0:
        out[ok] = (x[ok] - np.nanmean(x)) / np.nanstd(x)
    return out


def slope(xs, ys) -> float:
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 2 or np.ptp(xs) == 0:
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


class Ctx:
    """Data shared by all signals of one run."""

    def __init__(self, con, run_id: str, p: RunParams):
        self.con, self.run_id, self.p, self.sp = con, run_id, p, p.signals
        self.papers = con.execute("""
            SELECT v.paper_id, p.year, v.route, p.title, p.doi FROM provenance v JOIN papers p USING (paper_id)
            WHERE v.run_id = ? AND p.year IS NOT NULL""", [run_id]).df()
        self.year = self.papers.set_index("paper_id").year
        self.route = self.papers.set_index("paper_id").route
        self.clusters = con.execute("""SELECT cluster_id, facet, alpha, resolution, label, kind, coherence,
                                              size, time_bucket_years FROM clusters WHERE run_id = ?""",
                                    [run_id]).df()
        self.kind = self.clusters.set_index("cluster_id").kind
        self.label = self.clusters.set_index("cluster_id").label
        self.coherence = self.clusters.set_index("cluster_id").coherence
        self.bucket = self.clusters.set_index("cluster_id").time_bucket_years
        pc = con.execute("""SELECT paper_id, facet, alpha, cluster_id, sim_to_centroid FROM paper_cluster
                            WHERE run_id = ? AND resolution = 'medium'""", [run_id]).df()
        self.pc = pc
        self.edges = con.execute("""
            SELECT r.citing_id, r.cited_id, r.is_influential, r.intents FROM refs r
            JOIN provenance a ON a.paper_id = r.citing_id AND a.run_id = ?
            JOIN provenance b ON b.paper_id = r.cited_id AND b.run_id = ?""", [run_id, run_id]).df()
        self.authorships = con.execute("""
            SELECT s.paper_id, s.author_id, s.institution_ids FROM authorships s
            JOIN provenance v ON v.paper_id = s.paper_id AND v.run_id = ?""", [run_id]).df()
        self.t_max = int(self.papers.year.max())
        per_year = self.papers.groupby("year").size()
        ok = [y for y in per_year.index if all(per_year.get(y - k, 0) >= self.sp.min_papers_per_year for k in range(3))]
        self.cutoffs = [int(y) for y in sorted(ok)]
        cents = pd.read_parquet(run_vectors(run_id) / f"cluster_centroids_{run_id}.parquet")
        self.centroid = {c: np.asarray(v, dtype=np.float32) for c, v in zip(cents.cluster_id, cents.centroid)}

    def assignment(self, facet: str, alpha: float) -> pd.DataFrame:
        a = self.pc[(self.pc.facet == facet) & (self.pc.alpha == alpha)].copy()
        a["year"] = a.paper_id.map(self.year)
        return a.dropna(subset=["year"])

    def in_variant(self, ids: pd.Series, variant: str) -> pd.Series:
        return ids.map(self.route) != "semantic" if variant == "no_semantic" else pd.Series(True, index=ids.index)


# =============================================================== paper novelty (6.2)

def paper_novelty(ctx: Ctx, facet: str) -> pd.Series:
    """nov_f(x) = 1 - max cos(e_f(x), e_f(y)) over papers y published before year(x). alpha = 0."""
    import faiss

    ids, v = embed.load_vectors(facet, ids=ctx.papers.paper_id.tolist(), version=ctx.p.embedding.version)
    if not ids:
        return pd.Series(dtype=float)
    yrs = np.array([ctx.year[i] for i in ids])
    nov = np.full(len(ids), np.nan)
    for y in np.unique(yrs):
        prior, cur = np.where(yrs < y)[0], np.where(yrs == y)[0]
        if len(prior) < 20:
            continue
        index = faiss.IndexFlatIP(v.shape[1])
        index.add(v[prior])
        s, _ = index.search(v[cur], 1)
        nov[cur] = 1 - s[:, 0]
    return pd.Series(nov, index=ids, name=facet)


# =============================================================== 6.1-6.3 cluster signals

def _growth(n: dict, N: dict, end: int, b: int) -> float:
    """Slope (per year) of log share over three buckets of b years ending at `end`."""
    xs, ys = [], []
    for k in range(3):
        yrs = range(end - (k + 1) * b + 1, end - k * b + 1)
        tot = sum(N.get(y, 0) for y in yrs)
        if tot == 0:
            return np.nan
        xs.append(np.mean(list(yrs)))
        ys.append(np.log((sum(n.get(y, 0) for y in yrs) + 0.5) / tot))
    return slope(xs, ys)


def cluster_signals_at(ctx: Ctx, facet: str, alpha: float, T: int, variant: str,
                       novelty: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster features at cutoff T; also returns the author migration rows (t-1 -> t)."""
    sp = ctx.sp
    a = ctx.assignment(facet, alpha)
    a = a[(a.year <= T) & ctx.in_variant(a.paper_id, variant).values]
    N = a.groupby("year").size().to_dict()
    ca = a.dropna(subset=["cluster_id"])
    if ca.empty:
        return pd.DataFrame(), pd.DataFrame()
    n_cy = ca.groupby(["cluster_id", "year"]).size()
    win = (ca.year >= T - 2) & (ca.year <= T)
    papers_win = ca[win]

    # author dominant cluster per year -> migration
    au = ctx.authorships.merge(ca[["paper_id", "cluster_id", "year"]], on="paper_id")
    dom = (au.groupby(["author_id", "year"]).cluster_id
           .agg(lambda s: s.value_counts().sort_index().idxmax()).rename("dom").reset_index())
    prev = dom.assign(year=dom.year + 1)
    mig = dom.merge(prev, on=["author_id", "year"], suffixes=("_to", "_from"))
    mig = mig[mig.dom_to != mig.dom_from]
    mig_win = mig[(mig.year >= T - 2) & (mig.year <= T)]
    inflow = mig_win.groupby("dom_to").size()
    outflow = mig_win.groupby("dom_from").size()

    # reference age of cluster papers' references (cited year known when the work was fetched)
    ref_age = _ref_age(ctx, ca[win])

    # new-author share: window authors with no earlier paper in the same cluster
    first = au.groupby(["author_id", "cluster_id"]).year.min().rename("first").reset_index()
    aw = au[(au.year >= T - 2)].merge(first, on=["author_id", "cluster_id"])
    new_share = aw.drop_duplicates(["author_id", "cluster_id"]).assign(new=lambda d: d["first"] >= T - 2) \
        .groupby("cluster_id").new.mean()

    # external inflow of received citations (citing paper <= T, inside the variant)
    cl = ca.set_index("paper_id").cluster_id
    e = ctx.edges[ctx.edges.citing_id.isin(a.paper_id)]
    e = e.assign(to=e.cited_id.map(cl), frm=e.citing_id.map(cl)).dropna(subset=["to"])
    ext = e.assign(external=e.frm != e.to).groupby("to").external.mean()
    received = e.groupby("to").size()

    nov_mean = (papers_win.assign(nov=papers_win.paper_id.map(novelty)).groupby("cluster_id").nov.mean()
                if novelty is not None and len(novelty) else pd.Series(dtype=float))

    rows = []
    for c, grp in ca.groupby("cluster_id"):
        n = n_cy[c].to_dict()
        b = int(ctx.bucket.get(c, 1) or 1)
        g = _growth(n, N, T, b)
        g_prev = _growth(n, N, T - b, b)
        rows.append({
            "cluster_id": c, "kind": ctx.kind.get(c), "label": ctx.label.get(c),
            "n_total": int(len(grp)), "activity": int(((grp.year >= T - 2) & (grp.year <= T)).sum()),
            "share_T": n.get(T, 0) / max(N.get(T, 0), 1), "bucket_years": b,
            "momentum": n.get(T, 0) / max(N.get(T, 0), 1) - n.get(T - 1, 0) / max(N.get(T - 1, 0), 1),
            "g": g, "dg": g - g_prev if not (np.isnan(g) or np.isnan(g_prev)) else np.nan,
            "mig_in": int(inflow.get(c, 0)), "mig_out": int(outflow.get(c, 0)),
            "mig_net": int(inflow.get(c, 0) - outflow.get(c, 0)),
            "ref_age": ref_age.get(c, np.nan), "new_author_share": new_share.get(c, np.nan),
            "novelty": nov_mean.get(c, np.nan), "coherence": ctx.coherence.get(c, np.nan),
            "ext_inflow": ext.get(c, np.nan), "received": int(received.get(c, 0)),
            "median_year": float(grp.year.median()),
        })
    df = pd.DataFrame(rows)
    elig = df.n_total >= sp.emergence_min_size
    # recipe score, kept for comparison; the Phase 8 backtest found it anti-predictive here
    df["E_recipe"] = np.nan
    df.loc[elig, "E_recipe"] = (z(df.loc[elig, "g"]) + z(df.loc[elig, "dg"]) + z(df.loc[elig, "new_author_share"])
                                - z(df.loc[elig, "ref_age"]) + z(df.loc[elig, "novelty"]))
    # backtested score (docs/features/08-backtest.md): only features whose sign held across
    # cutoffs and across outcome samples drawn from different expansion routes
    df["E_backtested"] = np.nan
    df.loc[elig, "E_backtested"] = sum(w * z(df.loc[elig, f]) for f, w in sp.emergence_weights.items())
    df["E"] = df["E_backtested"] if sp.emergence_score == "backtested" else df["E_recipe"]
    # direction uses the same score (growth = positive), for clusters big enough to trend
    df["D"] = np.where(df.n_total >= sp.direction_min_size, df.E, np.nan) if sp.emergence_score == "backtested" else df.g
    df["N"] = z(df.coherence) - z(df.ext_inflow.fillna(df.ext_inflow.median()))
    thr = df.loc[elig, "E"].quantile(sp.emerging_quantile) if elig.any() else np.inf
    rising = df.momentum > 0 if sp.emergence_score == "backtested" else df.g > 0
    df["status"] = None
    df.loc[(df.N >= sp.niche_z) & (df.activity < sp.a_min), "status"] = "dormant"
    df.loc[(df.N >= sp.niche_z) & (df.activity >= sp.a_min), "status"] = "niche"
    df.loc[elig & (df.E >= thr) & rising, "status"] = "emerging"
    df["facet"], df["alpha"], df["cutoff_year"], df["variant"] = facet, alpha, T, variant
    return df, mig


def _ref_age(ctx: Ctx, cp: pd.DataFrame) -> dict:
    if cp.empty:
        return {}
    ctx.con.register("_cp", cp[["paper_id", "cluster_id", "year"]])
    r = ctx.con.execute("""SELECT c.cluster_id, median(c.year - p.year) AS age FROM _cp c
                           JOIN refs r ON r.citing_id = c.paper_id JOIN papers p ON p.paper_id = r.cited_id
                           WHERE p.year IS NOT NULL AND p.year <= c.year GROUP BY 1""").df()
    ctx.con.unregister("_cp")
    return dict(zip(r.cluster_id, r.age))


def evidence_for(ctx: Ctx, facet: str, alpha: float, cluster_id: str, T: int, k: int = 10) -> list[str]:
    a = ctx.assignment(facet, alpha)
    a = a[(a.cluster_id == cluster_id) & (a.year <= T)]
    recent = a[a.year >= T - 2]
    pick = (recent if len(recent) >= 3 else a).sort_values("sim_to_centroid", ascending=False)
    return pick.paper_id.head(k).tolist()


def cluster_signal_rows(ctx: Ctx, df: pd.DataFrame) -> list[dict]:
    """Ranked signal rows at one cutoff/variant; stable and hub clusters ranked separately."""
    out = []
    feats = ["g", "dg", "momentum", "mig_net", "mig_in", "mig_out", "ref_age", "new_author_share", "novelty",
             "coherence", "ext_inflow", "activity", "n_total", "share_T", "E", "E_recipe", "E_backtested",
             "D", "N", "status", "bucket_years"]
    big = df.n_total >= ctx.sp.direction_min_size
    specs = [("direction_growing", df.D, big & df.D.notna() & (df.D > 0)),
             ("direction_shrinking", -df.D, big & df.D.notna() & (df.D < 0)),
             ("emerging_cluster", df.E, df.E.notna()),
             ("niche", df.N, df.status == "niche"),
             ("dormant", df.N, df.status == "dormant")]
    for sig, score, mask in specs:
        d = df[mask].assign(score=score[mask])
        for kind, grp in d.groupby(d.kind.fillna("stable")):
            grp = grp.sort_values("score", ascending=False).head(ctx.sp.top_n)
            for rank, r in enumerate(grp.itertuples(), 1):
                out.append(_row(ctx, sig, {"cluster_id": r.cluster_id, "facet": r.facet, "alpha": r.alpha,
                                           "label": r.label},
                                float(r.score), rank, {f: _clean(getattr(r, f)) for f in feats},
                                evidence_for(ctx, r.facet, r.alpha, r.cluster_id, r.cutoff_year),
                                r.cutoff_year, r.variant, kind))
    return out


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _row(ctx, sig, subject, score, rank, features, evidence, T, variant="all", kind="na"):
    return {"run_id": ctx.run_id, "signal_type": sig, "subject": json.dumps(subject, default=str),
            "score": float(score), "rank": int(rank), "features": json.dumps(features, default=_clean),
            "evidence_paper_ids": list(evidence), "judge_score": None, "judge_rationale": None,
            "cutoff_year": int(T), "variant": variant, "group_kind": kind}


# =============================================================== 6.4 bridges

def _flow_matrix(ctx: Ctx, facet: str, alpha: float, T: int) -> pd.Series:
    a = ctx.assignment(facet, alpha).dropna(subset=["cluster_id"])
    a = a[a.year <= T]
    cl = a.set_index("paper_id").cluster_id
    e = ctx.edges.assign(f=ctx.edges.citing_id.map(cl), t=ctx.edges.cited_id.map(cl)).dropna(subset=["f", "t"])
    return e.groupby(["f", "t"]).size()


def bridges(ctx: Ctx, T: int) -> list[dict]:
    sp, ver = ctx.sp, ctx.p.embedding.version
    alpha = sp.bridge_alpha
    prob = ctx.assignment("problem", alpha).dropna(subset=["cluster_id"])
    prob = prob[prob.year <= T]
    pcl = prob.set_index("paper_id").cluster_id
    F = _flow_matrix(ctx, "problem", alpha, T)
    rows: list[dict] = []

    # ---- (a) limitation -> innovation
    lid, lv = embed.load_vectors("limitations", ids=prob.paper_id.tolist(), version=ver)
    iid, iv = embed.load_vectors("innovation", ids=prob.paper_id.tolist(), version=ver)
    lim_inn_sims = None
    if lid and iid:
        gc = lv.mean(axis=0)
        gc /= np.linalg.norm(gc)
        generic = lv @ gc
        keep = generic <= np.quantile(generic, 1 - sp.generic_limitation_drop)
        lid = [x for x, k in zip(lid, keep) if k]
        lv = lv[keep]
        ly = np.array([ctx.year[x] for x in lid])
        iy = np.array([ctx.year[x] for x in iid])
        lc = np.array([pcl[x] for x in lid])
        ic = np.array([pcl[x] for x in iid])
        S = lv @ iv.T
        valid = (iy[None, :] >= ly[:, None]) & (ic[None, :] != lc[:, None])
        lim_inn_sims = S[valid]
        for X in np.unique(lc):
            rX = np.where(lc == X)[0]
            fx = {Y: F.get((X, Y), 0) for Y in np.unique(ic)}
            fmax = max(fx.values()) if fx else 0
            for Y in np.unique(ic):
                if Y == X:
                    continue
                cY = np.where(ic == Y)[0]
                sub = np.where(valid[np.ix_(rX, cY)], S[np.ix_(rX, cY)], -np.inf)
                flat = sub.ravel()
                k = min(sp.lim_topk, int(np.isfinite(flat).sum()))
                if k == 0:
                    continue
                top = np.argpartition(-flat, k - 1)[:k]
                top = top[np.argsort(-flat[top])]
                topm = float(flat[top].mean())
                # recipe uses power 1; the backtest found 0.5 keeps candidates near-missing
                # (median 1.5 existing citations) while ranking better (08-backtest.md)
                score = topm * (1 - fx.get(Y, 0) / (fmax + 1)) ** sp.lim_flow_penalty_power
                pairs = [(lid[rX[t // len(cY)]], iid[cY[t % len(cY)]], round(float(flat[t]), 4)) for t in top]
                rows.append(_row(ctx, "bridge_lim_inn", {"from_cluster": X, "to_cluster": Y,
                                                          "from_label": ctx.label.get(X), "to_label": ctx.label.get(Y)},
                                 score, 0, {"mean_topk_sim": round(topm, 4), "flow_xy": int(fx.get(Y, 0)),
                                            "flow_max": int(fmax),
                                            "pairs": [{"limitation": l, "innovation": i, "sim": s} for l, i, s in pairs],
                                            "kinds": [ctx.kind.get(X), ctx.kind.get(Y)]},
                                 [p for l, i, _ in pairs for p in (l, i)], T))

    # ---- (b) method transfer
    pm = ctx.con.execute("""SELECT problem_cluster, method_cluster, sum(count) n FROM pm_matrix
                            WHERE run_id = ? AND alpha = ? AND year <= ? GROUP BY ALL""",
                         [ctx.run_id, alpha, T]).df()
    if not pm.empty:
        M = pm.pivot(index="problem_cluster", columns="method_cluster", values="n").fillna(0)
        P = [p for p in M.index if p in ctx.centroid]
        C = np.stack([ctx.centroid[p] for p in P])
        sim = C @ C.T
        np.fill_diagonal(sim, -np.inf)
        meth = ctx.assignment("method", alpha).dropna(subset=["cluster_id"])
        meth = meth[meth.year <= T]
        cell_papers = (prob[["paper_id", "cluster_id"]].merge(meth[["paper_id", "cluster_id"]], on="paper_id",
                                                              suffixes=("_p", "_m")))
        # item-based collaborative filtering on the problem x method matrix: methods similar (by
        # which problems use them) to the ones P already uses. Backtest: AUC 0.70-0.80 vs the
        # recipe's centroid-neighbour score at about random (08b-signal-experiments.md).
        L = np.log1p(M.loc[P].to_numpy(dtype=float))
        Lc = L / np.maximum(np.linalg.norm(L, axis=0, keepdims=True), 1e-9)
        Sm = Lc.T @ Lc
        np.fill_diagonal(Sm, 0)
        CF = L @ Sm
        mcols = list(M.columns)
        for i, Pc in enumerate(P):
            nn = np.argsort(-sim[i])[:sp.transfer_neighbors]
            w = np.clip(sim[i, nn], 0, None)
            for m in M.columns:
                if M.loc[Pc, m] > 0:
                    continue
                has = np.array([M.loc[P[j], m] > 0 for j in nn])
                cf = float(CF[i, mcols.index(m)])
                if sp.transfer_method == "item_cf":
                    if cf <= 0:
                        continue
                elif not has.any():
                    continue
                score = float((w * has).sum() / w.sum()) if has.any() else 0.0
                # the indicator saturates at 1 for big method clusters; tie-break by how much
                # of the neighbours' output uses the method
                usage = float((w * np.array([M.loc[P[j], m] / max(M.loc[P[j]].sum(), 1) for j in nn])).sum() / w.sum())
                score += 1e-3 * usage
                recipe_score = score
                if sp.transfer_method == "item_cf":
                    score = cf
                similar_used = [mcols[j] for j in np.argsort(-Sm[mcols.index(m)])
                                if M.loc[Pc, mcols[j]] > 0][:3]
                donors = [P[j] for j in nn[has]][:3]
                ev = cell_papers[(cell_papers.cluster_id_m == m) & cell_papers.cluster_id_p.isin(donors)].paper_id.head(6)
                rows.append(_row(ctx, "bridge_method_transfer",
                                 {"problem_cluster": Pc, "method_cluster": m, "problem_label": ctx.label.get(Pc),
                                  "method_label": ctx.label.get(m)}, score, 0,
                                 {"item_cf": round(cf, 4), "recipe_transfer_score": round(recipe_score, 4),
                                  "similar_methods_already_used": [{"cluster": c, "label": ctx.label.get(c)}
                                                                   for c in similar_used],
                                  "neighbor_usage_share": round(usage, 4),
                                  "neighbors_using_method": [{"cluster": P[j], "label": ctx.label.get(P[j]),
                                                              "sim": round(float(sim[i, j]), 4)} for j in nn[has]],
                                  "kinds": [ctx.kind.get(Pc), ctx.kind.get(m)]},
                                 ev.tolist(), T))

    # ---- (c) unresolved limitations
    if lid and iid and lim_inn_sims is not None and len(lim_inn_sims):
        from sklearn.cluster import AgglomerativeClustering

        match_thr = float(np.quantile(lim_inn_sims, sp.unresolved_match_quantile)) \
            if sp.unresolved_match == "auto" else float(sp.unresolved_match)
        ll = lv @ lv.T
        dist_thr = 1 - float(np.quantile(ll[np.triu_indices(len(lid), 1)], sp.limitation_group_quantile))
        for X in np.unique(lc):
            rX = np.where(lc == X)[0]
            if len(rX) < sp.unresolved_min_papers:
                continue
            lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                          distance_threshold=dist_thr).fit_predict(lv[rX])
            for g in np.unique(lab):
                members = rX[lab == g]
                if len(members) < sp.unresolved_min_papers:
                    continue
                # like-for-like with the per-statement threshold: each limitation's best match
                # among innovations published after it; resolved if most members are matched
                y0 = int(np.median(ly[members]))
                best_each = np.array([float(S[m][iy > ly[m]].max()) if (iy > ly[m]).any() else 0.0
                                      for m in members])
                if np.mean(best_each >= match_thr) >= 0.5:
                    continue
                best = float(np.median(best_each))
                rows.append(_row(ctx, "bridge_unresolved_limitation",
                                 {"cluster_id": X, "label": ctx.label.get(X)},
                                 float(len(members) * (1 + match_thr - best)), 0,
                                 {"group_size": int(len(members)), "median_year": y0,
                                  "median_best_later_innovation_sim": round(best, 4),
                                  "share_members_matched": round(float(np.mean(best_each >= match_thr)), 3),
                                  "match_threshold": round(match_thr, 4),
                                  "kind": ctx.kind.get(X)},
                                 [lid[m] for m in members][:10], T))

    # ---- (d) semantic-but-not-cited
    Pids = [c for c in prob.cluster_id.unique() if c in ctx.centroid]
    if len(Pids) > 3:
        C = np.stack([ctx.centroid[c] for c in Pids])
        S = C @ C.T
        size = prob.cluster_id.value_counts()
        iu = np.triu_indices(len(Pids), 1)
        sims = S[iu]
        flows = np.array([(F.get((Pids[i], Pids[j]), 0) + F.get((Pids[j], Pids[i]), 0))
                          / (size[Pids[i]] * size[Pids[j]]) for i, j in zip(*iu)])
        s_thr = np.quantile(sims, 1 - sp.semantic_pair_top)
        f_thr = np.quantile(flows, sp.semantic_pair_flow_pct / 100)
        for k in np.where((sims >= s_thr) & (flows <= f_thr))[0]:
            i, j = iu[0][k], iu[1][k]
            ev = (evidence_for(ctx, "problem", alpha, Pids[i], T, 3) + evidence_for(ctx, "problem", alpha, Pids[j], T, 3))
            rows.append(_row(ctx, "bridge_semantic_not_cited",
                             {"cluster_a": Pids[i], "cluster_b": Pids[j], "label_a": ctx.label.get(Pids[i]),
                              "label_b": ctx.label.get(Pids[j])}, float(sims[k]), 0,
                             {"centroid_sim": round(float(sims[k]), 4),
                              "flow_ab": int(F.get((Pids[i], Pids[j]), 0)), "flow_ba": int(F.get((Pids[j], Pids[i]), 0)),
                              "kinds": [ctx.kind.get(Pids[i]), ctx.kind.get(Pids[j])]}, ev, T))

    # rank per type
    out = []
    for sig in {r["signal_type"] for r in rows}:
        grp = sorted([r for r in rows if r["signal_type"] == sig], key=lambda r: -r["score"])
        for rank, r in enumerate(grp[:sp.bridge_keep], 1):
            r["rank"] = rank
            out.append(r)
    return out


JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["plausibility", "rationale", "obvious_reason"],
    "properties": {
        "plausibility": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string", "description": "two sentences"},
        "obvious_reason": {"type": ["string", "null"],
                           "description": "an obvious reason it has not been done, or null"},
    },
}
JUDGE_PROMPT = """You assess candidate research opportunities found by an automatic analysis of a
research field. Judge only from the evidence below.

Candidate type: {kind}
{what}

Evidence papers (facets extracted from their abstracts):
{evidence}

Return:
- "plausibility": 1-5, how plausible and useful it is to pursue this opportunity (5 = clearly worth pursuing).
- "rationale": exactly two sentences explaining your rating.
- "obvious_reason": an obvious reason it has not been done yet (e.g. the combination makes no sense, it is
  already standard, data does not exist), or null if none."""

WHAT = {
    "bridge_lim_inn": "A limitation stated by papers in cluster A ({from_label}) may be addressed by an "
                      "innovation from cluster B ({to_label}), and A rarely cites B.",
    "bridge_method_transfer": "Method cluster ({method_label}) has never been applied to problem cluster "
                              "({problem_label}), but is used for similar problems.",
    "bridge_unresolved_limitation": "A group of papers in cluster ({label}) states a similar limitation that "
                                    "no later innovation in the corpus seems to address.",
    "bridge_semantic_not_cited": "Problem clusters ({label_a}) and ({label_b}) are semantically close but "
                                 "barely cite each other.",
}


def judge(ctx: Ctx, rows: list[dict]) -> float:
    """Send the top candidates per bridge type to a stronger LLM; reorder, never add."""
    sp = ctx.sp
    todo = [r for r in rows if r["signal_type"] in WHAT and r["rank"] <= sp.judge_top]
    if not todo:
        return 0.0
    ids = sorted({p for r in todo for p in r["evidence_paper_ids"]})
    ctx.con.register("_ids", pd.DataFrame({"paper_id": ids}))
    f = ctx.con.execute("""SELECT f.paper_id, f.facet, f.text, p.title FROM facets f JOIN papers p USING (paper_id)
                           WHERE f.paper_id IN (SELECT paper_id FROM _ids)
                             AND f.facet IN ('problem', 'method', 'innovation', 'limitations')""").df()
    ctx.con.unregister("_ids")
    facets = defaultdict(dict)
    titles = {}
    for r in f.itertuples():
        facets[r.paper_id][r.facet] = r.text
        titles[r.paper_id] = r.title
    client = OpenRouter(sp.judge_model, ["fp8", "bf16", "fp16", "fp32"], 16, sp.judge_spend_cap_usd)

    async def one(r):
        subj = json.loads(r["subject"])
        ev = []
        for pid in r["evidence_paper_ids"][:8]:
            fx = facets.get(pid, {})
            ev.append(f"- {titles.get(pid, pid)} ({int(ctx.year.get(pid, 0))})\n" + "\n".join(
                f"  {k}: {fx[k]}" for k in ("problem", "method", "innovation", "limitations") if fx.get(k)))
        prompt = JUDGE_PROMPT.format(kind=r["signal_type"].removeprefix("bridge_").replace("_", " "),
                                     what=WHAT[r["signal_type"]].format(**{k: v or "?" for k, v in subj.items()}),
                                     evidence="\n".join(ev))
        try:
            obj, _, err = await client.chat_json(prompt, JUDGE_SCHEMA, "bridge_judgment", max_tokens=400)
        except LLMBudgetExceeded:
            return
        if obj:
            r["judge_score"] = float(obj["plausibility"])
            r["judge_rationale"] = obj["rationale"]
            feats = json.loads(r["features"])
            feats["judge_obvious_reason"] = obj.get("obvious_reason")
            r["features"] = json.dumps(feats)

    async def run():
        try:
            await asyncio.gather(*(one(r) for r in todo))
        finally:
            await client.close()

    asyncio.run(run())
    # reorder within each type: judged candidates by (judge score, score); unjudged keep order after
    for sig in {r["signal_type"] for r in todo}:
        grp = [r for r in rows if r["signal_type"] == sig]
        judged = sorted([r for r in grp if r["judge_score"] is not None],
                        key=lambda r: (-r["judge_score"], -r["score"]))
        rest = sorted([r for r in grp if r["judge_score"] is None], key=lambda r: r["rank"])
        for rank, r in enumerate(judged + rest, 1):
            r["features"] = json.dumps({**json.loads(r["features"]), "pre_judge_rank": r["rank"]})
            r["rank"] = rank
    return client.spent


# =============================================================== 6.5 influence

def influence(ctx: Ctx, T: int, growth: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    sp = ctx.sp
    rows = []
    papers = ctx.papers[ctx.papers.year <= T]
    pidx = {p: i for i, p in enumerate(papers.paper_id)}
    # full reference lists of corpus papers (cited may be outside the corpus)
    ctx.con.register("_ids", papers[["paper_id"]])
    refs = ctx.con.execute("""SELECT citing_id, cited_id FROM refs WHERE citing_id IN (SELECT paper_id FROM _ids)""").df()
    ctx.con.unregister("_ids")
    cited_ids = {c: j for j, c in enumerate(refs.cited_id.unique())}
    Cm = sparse.csr_matrix((np.ones(len(refs)), (refs.citing_id.map(pidx), refs.cited_id.map(cited_ids))),
                           shape=(len(pidx), len(cited_ids)))
    Cc = Cm.tocsc()
    yrs = papers.year.to_numpy()

    # ---- disruption (CD) index; citers are corpus papers only
    cd = {}
    refs_of = refs.groupby("citing_id").cited_id.apply(list)
    for p, i in pidx.items():
        if p not in cited_ids:
            continue
        f = np.zeros(len(pidx), bool)
        f[Cc[:, cited_ids[p]].indices] = True
        own_refs = refs_of.get(p, [])
        if len(own_refs) < sp.cd_min_refs:
            continue  # without its reference list the CD index is trivially 1
        rp = [cited_ids[r] for r in own_refs if r in cited_ids]
        b = np.zeros(len(pidx), bool)
        if rp:
            b[np.unique(Cc[:, rp].indices)] = True
        later = (yrs >= yrs[i])
        later[i] = False
        f &= later
        b &= later
        n = (f | b).sum()
        if f.sum() < sp.cd_min_citers or n == 0:
            continue
        cd[p] = (float((f & ~b).sum() - (f & b).sum()) / n, int(f.sum()), int(n))
    cd_df = pd.DataFrame([(p, *v) for p, v in cd.items()], columns=["paper_id", "cd", "citers", "n"])
    # rank papers from the analysed period only: with corpus-only citers, textbooks and classics
    # trivially score CD ~ 1 (they are cited without their own, older references)
    cd_min_year = (ctx.cutoffs[0] if ctx.cutoffs else T) - sp.cd_years_before_first_cutoff
    ranked = cd_df[cd_df.paper_id.map(ctx.year) >= cd_min_year]
    for rank, r in enumerate(ranked.sort_values("cd", ascending=False).head(sp.top_n).itertuples(), 1):
        citers = [papers.paper_id.iloc[k] for k in Cc[:, cited_ids[r.paper_id]].indices[:5]]
        rows.append(_row(ctx, "disruptive_paper", {"paper_id": r.paper_id, "title": ctx.papers.set_index("paper_id").title.get(r.paper_id)},
                         r.cd, rank, {"cd": round(r.cd, 4), "corpus_citers": r.citers, "n": r.n},
                         [r.paper_id] + citers, T))

    # ---- seeders: authors of the earliest 10% of papers in clusters growing above the median
    a = ctx.assignment(sp.influence_facet, sp.influence_alpha).dropna(subset=["cluster_id"])
    a = a[a.year <= T]
    gr = growth.dropna(subset=["g"])
    growing = gr[gr.g > gr.g.median()].cluster_id
    early = (a[a.cluster_id.isin(growing)].sort_values("year").groupby("cluster_id")
             .apply(lambda d: d.head(max(1, int(np.ceil(len(d) * 0.1)))), include_groups=False).reset_index(level=0))
    au = ctx.authorships
    ea = au[au.paper_id.isin(early.paper_id)]
    seed_score = ea.groupby("author_id").paper_id.nunique().sort_values(ascending=False)
    names = ctx.con.execute("SELECT author_id, display_name FROM authors").df().set_index("author_id").display_name
    for rank, (aid, s) in enumerate(seed_score.head(sp.top_n).items(), 1):
        rows.append(_row(ctx, "seeder_author", {"author_id": aid, "name": names.get(aid)}, float(s), rank,
                         {"early_papers": int(s)}, ea[ea.author_id == aid].paper_id.tolist()[:10], T))

    # ---- brokers: participation coefficient over clusters (authors with >= 3 assigned papers)
    aa = au.merge(a[["paper_id", "cluster_id"]], on="paper_id")
    k = aa.groupby(["author_id", "cluster_id"]).size().unstack(fill_value=0)
    ka = k.sum(axis=1)
    k = k[ka >= sp.broker_min_papers]
    P = 1 - ((k.div(k.sum(axis=1), axis=0)) ** 2).sum(axis=1)
    br = pd.DataFrame({"P": P, "k": k.sum(axis=1), "clusters": (k > 0).sum(axis=1)}).sort_values(["P", "k"], ascending=False)
    for rank, r in enumerate(br.head(sp.top_n).itertuples(), 1):
        rows.append(_row(ctx, "broker_author", {"author_id": r.Index, "name": names.get(r.Index)}, float(r.P), rank,
                         {"participation": round(float(r.P), 4), "papers": int(r.k), "clusters": int(r.clusters)},
                         aa[aa.author_id == r.Index].paper_id.tolist()[:10], T))

    # ---- substantive citations received per author (S2 influential / methodology; sparse)
    e = ctx.edges
    has_method = e.intents.map(lambda x: isinstance(x, (list, np.ndarray)) and "methodology" in list(x))
    subst = e[e.is_influential.fillna(False).astype(bool) | has_method]
    subst_by_author = au[au.paper_id.isin(subst.cited_id)].merge(
        subst.groupby("cited_id").size().rename("n"), left_on="paper_id", right_index=True).groupby("author_id").n.sum()

    # recent papers: see early_work() (bibliographic coupling is one of its features)

    # ---- institutions: author scores through authorships, weighted 1 / #institutions
    inst = au.dropna(subset=["institution_ids"]).copy()
    inst = inst[inst.institution_ids.map(len) > 0]
    inst["w"] = 1 / inst.institution_ids.map(len)
    inst = inst.explode("institution_ids")
    inst_names = ctx.con.execute("SELECT institution_id, name, country FROM institutions").df().set_index("institution_id")
    for sig, score in (("institution_seeder", seed_score), ("institution_broker", br.P)):
        sc = inst[inst.author_id.isin(score.index)].assign(s=lambda d: d.author_id.map(score) * d.w)
        # one contribution per author-institution pair (an author's score is not multiplied by paper count)
        agg = sc.drop_duplicates(["author_id", "institution_ids"]).groupby("institution_ids").s.sum().sort_values(ascending=False)
        for rank, (iid, s) in enumerate(agg.head(sp.top_n_institutions).items(), 1):
            nm = inst_names.name.get(iid)
            authors = sc[sc.institution_ids == iid].author_id.unique().tolist()
            rows.append(_row(ctx, sig, {"institution_id": iid, "name": nm, "country": inst_names.country.get(iid)},
                             float(s), rank, {"authors": authors[:10]},
                             au[au.author_id.isin(authors)].paper_id.unique().tolist()[:10], T))

    paper_scores = cd_df.assign(metric="cd_index", facet="citation", value=cd_df.cd)[["paper_id", "metric", "facet", "value"]]
    extra = pd.DataFrame({"author_id": subst_by_author.index, "n": subst_by_author.values})
    log.info("influence: %d CD papers, %d seeders, %d brokers, %d authors with substantive citations",
             len(cd_df), len(seed_score), len(br), len(extra))
    return rows, paper_scores


# =============================================================== backtested paper-level signals

def model_score(name: str, X: pd.DataFrame, run_id: str) -> np.ndarray:
    """Logistic score from a model fitted on the Phase 8 cutoffs (scripts/phase8b/fit_models.py): the project's own
    if it has one, else the pilot's (config.model_file)."""
    spec = json.loads(model_file(run_id, name).read_text())
    Z = (X[spec["features"]].to_numpy(float) - np.array(spec["mean"])) / np.array(spec["scale"])
    return Z @ np.array(spec["coef"]) + spec["intercept"]


def _ref_matrix(ctx: Ctx, paper_ids: list[str]):
    ctx.con.register("_ids", pd.DataFrame({"paper_id": paper_ids}))
    refs = ctx.con.execute("SELECT citing_id, cited_id FROM refs WHERE citing_id IN (SELECT paper_id FROM _ids)").df()
    ctx.con.unregister("_ids")
    pidx = {p: k for k, p in enumerate(paper_ids)}
    widx = {w: k for k, w in enumerate(refs.cited_id.unique())}
    R = sparse.csr_matrix((np.ones(len(refs)), (refs.citing_id.map(pidx), refs.cited_id.map(widx))),
                          shape=(len(pidx), max(len(widx), 1)))
    R.data[:] = 1
    return R, pidx


def paper_bridges(ctx: Ctx, T: int) -> list[dict]:
    """Per problem cluster X: innovation papers no X paper cites yet, ranked by the backtested
    adoption model (popularity, bibliographic coupling with X, similarity to X's problems and to its
    stated limitations, age). 'Hidden gems' are candidates cited at most once in the corpus."""
    sp, ver = ctx.sp, ctx.p.embedding.version
    prob = ctx.assignment("problem", sp.bridge_alpha).dropna(subset=["cluster_id"])
    prob = prob[prob.year <= T]
    lab = prob.set_index("paper_id").cluster_id
    ids = ctx.papers[ctx.papers.year <= T].paper_id.tolist()
    aliases = work_aliases(ctx.papers[ctx.papers.year <= T], ctx.authorships)
    alias_records = defaultdict(list)
    for pid, work in aliases.items():
        alias_records[work].append(pid)
    R, pidx = _ref_matrix(ctx, ids)
    e = ctx.edges.assign(cy=ctx.edges.citing_id.map(ctx.year))
    e = e[e.cy <= T]
    distinct_citations = e.assign(
        source_work=e.citing_id.map(lambda i: aliases.get(i, i)),
        target_work=e.cited_id.map(lambda i: aliases.get(i, i))).drop_duplicates(["source_work", "target_work"])
    distinct_citations = distinct_citations[distinct_citations.source_work != distinct_citations.target_work]
    work_pop = distinct_citations.groupby("target_work").size()
    pop = pd.Series({pid: work_pop.get(aliases.get(pid, pid), 0) for pid in ids})
    cited_by = (e.assign(X=e.citing_id.map(lab), work=e.cited_id.map(lambda i: aliases.get(i, i)))
                .dropna(subset=["X"]).groupby("X").work.apply(set))
    iid, iv = embed.load_vectors("innovation", ids=list(lab.index), version=ver)
    lid, lv = embed.load_vectors("limitations", ids=list(lab.index), version=ver)
    qid, qv = embed.load_vectors("problem", ids=list(lab.index), version=ver)
    if not iid or not qid:
        return []
    if len(lid):
        gc = lv.mean(axis=0)
        gc /= np.linalg.norm(gc)
        generic_cut = np.quantile(lv @ gc, 1 - sp.generic_limitation_drop)
    lpos, qpos = {x: k for k, x in enumerate(lid)}, {x: k for k, x in enumerate(qid)}
    cX = lab.reindex(iid).to_numpy()
    clusters = sorted(lab.unique())
    cix = {c: k for k, c in enumerate(clusters)}
    Mx = sparse.csr_matrix((np.ones(len(lab)), ([cix[c] for c in lab.values], [pidx[x] for x in lab.index])),
                           shape=(len(clusters), len(pidx)))
    size = np.asarray(Mx.sum(axis=1)).ravel()
    coup = ((Mx @ R) @ R[[pidx[x] for x in iid]].T).toarray() / size[:, None]
    titles = ctx.papers.set_index("paper_id").title
    rows = []
    for X in clusters:
        members = lab.index[lab.values == X]
        Q = [qpos[x] for x in members if x in qpos]
        if len(Q) < 5:
            continue
        qc = qv[Q].mean(axis=0)
        qc /= np.linalg.norm(qc)
        L = [lpos[x] for x in members if x in lpos]
        Lv = lv[L] if L else np.zeros((0, iv.shape[1]), np.float32)
        Lids = [lid[k] for k in L]
        if len(Lv):
            keep = (Lv @ gc) <= generic_cut
            Lv, Lids = Lv[keep], [x for x, k in zip(Lids, keep) if k]
        already = cited_by.get(X, set())
        member_works = {aliases.get(i, i) for i in members}
        idx = np.array([k for k, x in enumerate(iid)
                        if aliases.get(x, x) not in member_works | already], dtype=int)
        if not len(idx):
            continue
        S = iv[idx] @ Lv.T if len(Lv) else None
        prob_sim = iv[idx] @ qc
        lim3 = np.sort(S, axis=1)[:, -3:].mean(axis=1) if S is not None and S.shape[1] else prob_sim
        feats = pd.DataFrame({
            "lpop": np.log1p([pop.get(iid[k], 0) for k in idx]),
            "lcoup": np.log1p(coup[cix[X], idx] * 100), "prob_sim": prob_sim, "lim_sim3_f": lim3,
            "age": [T - ctx.year[iid[k]] for k in idx]})
        sc = model_score("paper_bridges", feats, ctx.run_id)
        # two lists per cluster: most likely adoptions, and hidden gems (cited at most once so far),
        # ranked by the same model (long-tail AUC 0.84 in the backtest)
        order = np.argsort(-sc)
        seen_works = set()
        distinct = []
        for j in order:
            work = aliases.get(iid[idx[j]], iid[idx[j]])
            if work not in seen_works:
                seen_works.add(work)
                distinct.append(j)
        order = distinct
        gems = [j for j in order if feats.lpop.iloc[j] <= np.log1p(1)][:sp.paper_bridges_gems_per_cluster]
        picks = [(j, False) for j in order[:sp.paper_bridges_per_cluster]] + [(j, True) for j in gems]
        for rank, (j, gem_list) in enumerate(picks, 1):
            i = iid[idx[j]]
            best_lim = Lids[int(np.argmax(S[j]))] if S is not None and S.shape[1] else None
            rows.append(_row(ctx, "bridge_paper_gem" if gem_list else "bridge_paper", {"cluster_id": X, "label": ctx.label.get(X), "paper_id": i,
                                                   "title": titles.get(i), "from_cluster": cX[idx[j]]},
                             float(sc[j]), rank,
                             {"citations_so_far": int(pop.get(i, 0)), "hidden_gem": bool(pop.get(i, 0) <= 1),
                              "coupling": round(float(coup[cix[X], idx[j]]), 4),
                              "problem_sim": round(float(prob_sim[j]), 4), "limitation_sim": round(float(lim3[j]), 4),
                              "best_matching_limitation": best_lim, "kind": ctx.kind.get(X)},
                             [i] + ([best_lim] if best_lim else []), T, kind=str(ctx.kind.get(X) or "stable")))
            rows[-1]["features"] = json.dumps(json.loads(rows[-1]["features"]) | {
                "canonical_work": aliases.get(i, i), "alias_records": alias_records.get(aliases.get(i, i), [i])})
    out, seen = [], {}  # rank within each (cluster, list)
    for r in rows:
        key = (json.loads(r["subject"])["cluster_id"], r["signal_type"])
        seen[key] = seen.get(key, 0) + 1
        r["rank"] = seen[key]
        out.append(r)
    return out


def refresh_paper_bridges(con, run_id: str, p: RunParams) -> dict:
    """Refresh only the two paper-bridge lists, preserving all judged and other signal rows."""
    ctx = Ctx(con, run_id, p)
    rows = pd.DataFrame(paper_bridges(ctx, ctx.t_max))
    con.execute("BEGIN TRANSACTION")
    try:
        store.replace_run_rows(con, "signals", run_id, rows,
                               "AND signal_type IN ('bridge_paper', 'bridge_paper_gem')")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return {"paper_bridges": len(rows), "judge_spend_usd": 0.0}


def early_work(ctx: Ctx, T: int, growth: pd.DataFrame, novelty: pd.Series | None) -> list[dict]:
    """Recent papers (published T-1..T) likely to become top-cited for their year, from features
    available now (backtested model; author fame deliberately left out)."""
    sp = ctx.sp
    papers = ctx.papers[ctx.papers.year <= T]
    ids = papers.paper_id.tolist()
    R, pidx = _ref_matrix(ctx, ids)
    # new research only: OpenAlex dates new editions of old books as new works
    types = ctx.con.execute("SELECT paper_id, type FROM papers").df().set_index("paper_id")["type"]
    # allowlist: a blocklist missed a 2025 Cambridge book chapter typed "other"
    recent = [x for x in papers[papers.year >= T - 1].paper_id
              if types.get(x) in ("article", "preprint", "conference-paper", "proceedings-article")]
    if not recent:
        return []
    Rn = R[[pidx[x] for x in recent]]
    C = (Rn @ R.T).tocsr()
    C.data = (C.data >= 2).astype(float)
    coup = np.maximum(np.asarray(C.sum(axis=1)).ravel() - 1, 0)
    e = ctx.edges.assign(cy=ctx.edges.citing_id.map(ctx.year))
    cites = e[e.cy <= T].groupby("cited_id").size()
    a = ctx.assignment(sp.influence_facet, sp.influence_alpha).set_index("paper_id")
    E = growth.set_index("cluster_id").E if len(growth) else pd.Series(dtype=float)
    f = ctx.con.execute("""SELECT paper_id, facet, text FROM facets WHERE facet IN ('contribution_type',
                           'has_quantitative_result')""").df().pivot_table(index="paper_id", columns="facet",
                                                                          values="text", aggfunc="first")
    n_auth = ctx.authorships.groupby("paper_id").author_id.nunique()
    ct = f["contribution_type"] if "contribution_type" in f else pd.Series(dtype=object)
    hq = f["has_quantitative_result"] if "has_quantitative_result" in f else pd.Series(dtype=object)
    X = pd.DataFrame({"paper_id": recent})
    X["l_cites_so_far"] = np.log1p([cites.get(x, 0) for x in recent])
    X["l_coupling"] = np.log1p(coup)
    X["l_n_refs"] = np.log1p(np.asarray(Rn.sum(axis=1)).ravel())
    X["cluster_E"] = [E.get(a.cluster_id.get(x)) if x in a.index else np.nan for x in recent]
    X["centrality"] = [a.sim_to_centroid.get(x) if x in a.index else np.nan for x in recent]
    X["novelty"] = [novelty.get(x) if novelty is not None else np.nan for x in recent]
    X["l_n_authors"] = np.log1p([n_auth.get(x, 0) for x in recent])
    X["survey"] = [float(ct.get(x) == "survey") for x in recent]
    X["theory"] = [float(ct.get(x) == "theory") for x in recent]
    X["quant"] = [float(hq.get(x) == "true") for x in recent]
    for c in ("cluster_E", "centrality", "novelty"):
        X[c] = X[c].astype(float).fillna(X[c].astype(float).median())
    X["score"] = model_score("early_work", X.fillna(0), ctx.run_id)
    titles = ctx.papers.set_index("paper_id").title
    rows = []
    for rank, r in enumerate(X.sort_values("score", ascending=False).head(sp.top_n).itertuples(), 1):
        rows.append(_row(ctx, "early_work", {"paper_id": r.paper_id, "title": titles.get(r.paper_id)},
                         float(r.score), rank,
                         {"year": int(ctx.year[r.paper_id]), "citations_so_far": int(round(np.expm1(r.l_cites_so_far))),
                          "coupled_papers": int(round(np.expm1(r.l_coupling))), "cluster_E": round(float(r.cluster_E), 3),
                          "novelty": round(float(r.novelty), 4)}, [r.paper_id], T))
    return rows


def local_momentum(ctx: Ctx, T: int, k: int = 15) -> pd.DataFrame:
    """For papers from T-1..T: share of year-T papers inside the ball of its k nearest earlier
    neighbours versus the share of year T-1 papers (log ratio). Feeds the 'rising pockets' layer;
    backtest: top-50 hit rate about 2x random (modest)."""
    ids, v = embed.load_vectors("abstract", ids=ctx.papers.paper_id.tolist(), version=ctx.p.embedding.version)
    v = embed.truncate(v, ctx.p.embedding.graph_dim)
    yrs = np.array([ctx.year[i] for i in ids])
    order = np.argsort(yrs, kind="stable")
    ids, v, yrs = [ids[i] for i in order], v[order], yrs[order]
    past, recent = np.where(yrs <= T)[0], np.where((yrs >= T - 1) & (yrs <= T))[0]
    yT, yT1 = np.where(yrs == T)[0], np.where(yrs == T - 1)[0]
    if len(recent) == 0 or len(yT) == 0 or len(yT1) == 0:
        return pd.DataFrame()
    S = v[recent] @ v[past].T
    S[np.arange(len(recent)), np.searchsorted(past, recent)] = -1   # drop self
    radius = -np.partition(-S, k - 1, axis=1)[:, k - 1]
    mom = np.log(((v[recent] @ v[yT].T >= radius[:, None]).mean(axis=1) + 1e-3)
                 / ((v[recent] @ v[yT1].T >= radius[:, None]).mean(axis=1) + 1e-3))
    return pd.DataFrame({"paper_id": [ids[i] for i in recent], "metric": "local_momentum", "facet": "abstract",
                         "value": mom})


# =============================================================== viz tables

def viz_tables(ctx: Ctx, statuses: pd.DataFrame, migrations: list[pd.DataFrame]) -> dict:
    con, run = ctx.con, ctx.run_id
    # cluster status per playhead year (variant 'all')
    st_all = statuses[statuses.variant == "all"]
    cy = con.execute("SELECT cluster_id, year, n, share FROM cluster_year WHERE run_id = ?", [run]).df()
    sc = st_all.rename(columns={"cutoff_year": "year"})
    sc["scores"] = sc.apply(lambda r: json.dumps({k: _clean(r[k]) for k in ("g", "dg", "momentum", "E", "E_recipe",
                                                                         "D", "N", "mig_net", "new_author_share",
                                                                         "novelty", "ext_inflow", "ref_age")}
                                                 | {"kind": r["kind"]}), axis=1)
    v = cy.merge(sc[["cluster_id", "year", "status", "scores"]], on=["cluster_id", "year"], how="left")
    store.replace_run_rows(con, "viz_cluster_year", run, v.assign(run_id=run))

    # flows: citation (from citation_flows) and author migration
    cf = con.execute("""SELECT facet, alpha, year, from_cluster, to_cluster, count FROM citation_flows
                        WHERE run_id = ?""", [run]).df().assign(kind="citation")
    mg = pd.concat(migrations, ignore_index=True) if migrations else pd.DataFrame()
    if not mg.empty:
        mg = (mg.groupby(["facet", "alpha", "year", "dom_from", "dom_to"]).size().rename("count").reset_index()
              .rename(columns={"dom_from": "from_cluster", "dom_to": "to_cluster"}).assign(kind="migration"))
    flows = pd.concat([cf, mg], ignore_index=True)
    store.replace_run_rows(con, "viz_flow_year", run, flows.assign(run_id=run))

    # author paths on the layout of the influence facet/alpha
    lay = con.execute("""SELECT facet, alpha, paper_id, x, y FROM viz_layout WHERE run_id = ?""", [run]).df()
    au = ctx.authorships[["paper_id", "author_id"]]
    many = au.author_id.value_counts()
    au = au[au.author_id.isin(many[many >= 2].index)]
    ay = au.merge(lay, on="paper_id").assign(year=lambda d: d.paper_id.map(ctx.year))
    ay = ay.groupby(["facet", "alpha", "author_id", "year"])[["x", "y"]].mean().reset_index()
    store.replace_run_rows(con, "viz_author_year", run, ay.assign(run_id=run))
    return {"viz_cluster_year": len(v), "viz_flow_year": len(flows), "viz_author_year": len(ay)}


# =============================================================== orchestration

def build_signals(con, run_id: str, p: RunParams, skip_judge: bool = False) -> dict:
    ctx = Ctx(con, run_id, p)
    sp = ctx.sp
    T = ctx.t_max
    log.info("cutoffs %s, latest %d", ctx.cutoffs, T)

    # paper novelty per facet (alpha 0)
    nov = {f: paper_novelty(ctx, f) for f in sp.novelty_facets}
    paper_scores = [pd.DataFrame({"paper_id": s.index, "metric": "novelty", "facet": f, "value": s.values})
                    for f, s in nov.items()]

    rows, statuses, migrations = [], [], []
    for facet, alpha in sp.cluster_sets:
        for variant in ("all", "no_semantic"):
            for t in ctx.cutoffs:
                if variant == "no_semantic" and t != T:
                    continue  # the map shows the full corpus; the variant is ranked at the latest year
                df, mg = cluster_signals_at(ctx, facet, alpha, t, variant, nov.get(facet))
                if df.empty:
                    continue
                statuses.append(df)
                if variant == "all" and t == T:
                    migrations.append(mg.assign(facet=facet, alpha=alpha))
                if t == T:
                    rows += cluster_signal_rows(ctx, df)
        log.info("cluster signals done for %s a=%.1f", facet, alpha)
    statuses = pd.concat(statuses, ignore_index=True)

    # novel papers and new combinations (6.2)
    for f, s in nov.items():
        top = s.dropna()
        top = top[top.index.map(ctx.year) >= T - 2].sort_values(ascending=False).head(sp.top_n)
        for rank, (pid, v) in enumerate(top.items(), 1):
            rows.append(_row(ctx, "novel_paper", {"paper_id": pid, "facet": f,
                                                  "title": ctx.papers.set_index("paper_id").title.get(pid)},
                             float(v), rank, {"novelty": round(float(v), 4), "year": int(ctx.year[pid])}, [pid], T))
    pm = con.execute("""SELECT problem_cluster, method_cluster, sum(count) n, min(first_year) first_year
                        FROM pm_matrix WHERE run_id = ? AND alpha = ? AND year <= ? GROUP BY ALL""",
                     [run_id, sp.grid_alpha, T]).df()
    new = pm[pm.first_year >= T - 1].sort_values("n", ascending=False)
    pr = ctx.assignment("problem", sp.grid_alpha).set_index("paper_id").cluster_id
    me = ctx.assignment("method", sp.grid_alpha).set_index("paper_id").cluster_id
    both = pd.DataFrame({"p": pr, "m": me}).dropna()
    for rank, r in enumerate(new.head(sp.top_n).itertuples(), 1):
        ev = both[(both.p == r.problem_cluster) & (both.m == r.method_cluster)].index.tolist()[:10]
        rows.append(_row(ctx, "new_combination", {"problem_cluster": r.problem_cluster, "method_cluster": r.method_cluster,
                                                  "problem_label": ctx.label.get(r.problem_cluster),
                                                  "method_label": ctx.label.get(r.method_cluster)},
                         float(r.n), rank, {"first_year": int(r.first_year), "papers": int(r.n),
                                            "kinds": [ctx.kind.get(r.problem_cluster), ctx.kind.get(r.method_cluster)]},
                         ev, T))

    # bridges (6.4) + judge
    br = bridges(ctx, T)
    spend = 0.0 if skip_judge else judge(ctx, br)
    rows += br

    # influence (6.5) on the growth of the influence clustering at T
    g = statuses[(statuses.facet == sp.influence_facet) & (statuses.alpha == sp.influence_alpha)
                 & (statuses.cutoff_year == T) & (statuses.variant == "all")]
    inf_rows, cd_scores = influence(ctx, T, g)
    rows += inf_rows
    paper_scores.append(cd_scores)

    # backtested paper-level signals (Phase 8b)
    rows += paper_bridges(ctx, T)
    rows += early_work(ctx, T, g, nov.get("abstract"))
    paper_scores.append(local_momentum(ctx, T))

    sig = pd.DataFrame(rows)
    store.replace_run_rows(con, "signals", run_id, sig)
    ps = pd.concat(paper_scores, ignore_index=True).dropna(subset=["value"])
    store.replace_run_rows(con, "paper_scores", run_id, ps.assign(run_id=run_id))
    vz = viz_tables(ctx, statuses, migrations)
    counts = sig.groupby(["signal_type", "group_kind"]).size().rename("n").reset_index()
    return {"cutoffs": ctx.cutoffs, "signals": counts.to_dict("records"), "judge_spend_usd": spend,
            "status_counts_latest": statuses[(statuses.cutoff_year == T) & (statuses.variant == "all")]
            .status.value_counts().to_dict(), **vz}
