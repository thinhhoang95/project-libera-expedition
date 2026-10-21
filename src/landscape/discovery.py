"""Discovery artefacts for the product portfolio (Phase 10, experiments 17, 19, 20, 23; evidence in
docs/features/10-orientation-experiments.md). Built by orientation.build_orientation, so `landscape portfolio`
writes them next to the other portfolio files in data/projects/<run>/portfolio/:

    quiet_foundations.parquet   papers the field builds on without citing (exp17: being the uncited earliest source of
                                later work persists, AUC 0.83 like for like; citations 0.51). Also a third role in
                                reading_paths.parquet.
    tools_to_learn.parquet      per problem community, the method communities its recent papers use, with a settled /
                                in flux status (exp20: settled tools keep their share; newcomers using them stayed and
                                were taken up more)
    untried_applications.parquet
                                per problem community, application domains it has not used, ranked by semantic fit
                                (exp19: AUC 0.81, p@20 15% vs 5.6%; small sample: "use with care")
    domain_centroids.parquet    the application-domain centroids and examples, to rank domains for a paper analysed by
                                DOI (cosine with its problem vector)
    crowding.parquet            per paper: others posing the same question now, near duplicates, stated limitations
                                it answers (exp23: crowding marks attention, not risk; answering a limitation draws
                                the field's work towards you)
    crowding_thresholds.json    the thresholds, to compute the same gauge for a paper analysed by DOI

All "now" windows are the last three complete publication years. Thresholds and closeness as in the experiments:
0.894 for reuse (innovation vs method or innovation, 256-d); 99th percentile of random pairs for crowding.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import orientation as O

QUIET_EVIDENCE = ("Experiment 17: among papers of similar genericness, prior activity and citations, being the uncited "
                  "earliest source of later work persists (AUC 0.83; citations 0.51); 40% of quiet foundations are not "
                  "among their community's 50 most cited papers.")
TOOLS_EVIDENCE = ("Experiment 20: settled method communities (broad use, few competing variants, little innovation on "
                  "the tool itself) keep their share (rank correlation +0.22); newcomers who entered with them stayed "
                  "(+3.0 points) and were taken up (+6.2 points) more.")
APPS_EVIDENCE = ("Experiment 19: semantic fit between a problem community and an application domain found the empty "
                 "cells that filled (AUC 0.81; top 20: 15% vs 5.6% at random), better than co-usage. Only 24 cells "
                 "filled on the pilot: use with care.")
CROWD_EVIDENCE = ("Experiment 23: entries with recent near duplicates were taken up more (+9.5 points), entries into "
                  "busy questions too (+2.7): crowding marks attention, not risk, over three years. Answering a stated "
                  "limitation drew the community's later work towards the paper (+5.4 points), not more citations.")


# ------------------------------------------------------------------ quiet foundations (experiment 17)

def quiet_foundations(f: "O._Field", L: pd.DataFrame) -> pd.DataFrame:
    T = f.t_live
    st_ids = list(f.MP) + list(f.IP)
    ST = np.vstack([f.MV[[f.MP[x] for x in f.MP]], f.IV[[f.IP[x] for x in f.IP]]])
    st_year = np.array([f.year.get(x, 9999) for x in st_ids])
    inn_ids = list(f.IP)
    inn_year = np.array([f.year.get(x, 9999) for x in inn_ids])
    cites = set(zip(f.refs.citing_id, f.refs.cited_id))
    quiet, total = {}, {}
    rows = np.where((st_year >= 2018) & (st_year <= T))[0]
    for s in range(0, len(rows), 1000):
        blk = rows[s:s + 1000]
        S = ST[blk] @ f.IV.T
        for k, j in enumerate(blk):
            ok = np.where((inn_year < st_year[j]) & (S[k] >= O.THR))[0]
            if not len(ok):
                continue
            src = inn_ids[ok[np.argmin(inn_year[ok])]]
            p = st_ids[j]
            if p == src or f.team.get(p, frozenset()) & f.team.get(src, frozenset()):
                continue
            total.setdefault(src, set()).add(p)
            if (p, src) not in cites:
                quiet.setdefault(src, set()).add(p)
    cand = L[L.year <= T - 2].copy() if "year" in L else L.copy()
    cand = cand[cand.paper_id.isin(f.IP)]
    cand["attributed_quiet"] = cand.paper_id.map(lambda x: len(quiet.get(x, ())))
    cand["attributed"] = cand.paper_id.map(lambda x: len(total.get(x, ())))
    # prior density: close statements in the three years before the paper (busy areas stay busy)
    dens = {}
    for y0, g in cand.groupby("year"):
        m = (st_year >= y0 - 3) & (st_year < y0)
        S = f.IV[[f.IP[x] for x in g.paper_id]] @ ST[m].T if m.any() else np.zeros((len(g), 0))
        dens.update(dict(zip(g.paper_id, (S >= O.THR).sum(axis=1) if S.shape[1] else np.zeros(len(g), int))))
    cand["prior_density"] = cand.paper_id.map(dens).fillna(0)
    cand["dq"] = pd.qcut(cand.prior_density.rank(method="first"), 3, labels=False)
    cand["quiet_rank"] = cand.groupby(["genericness_quintile", "dq"]).attributed_quiet.rank(pct=True)
    cand["quiet_foundation"] = (cand.quiet_rank >= 0.9) & (cand.attributed_quiet >= 2)
    by_others = f.refs_in[f.refs_in.cy <= T]
    by_others = by_others[[not (f.team.get(a, frozenset()) & f.team.get(b, frozenset()))
                           for a, b in zip(by_others.citing_id, by_others.cited_id)]].groupby("cited_id").size()
    cand["cites_by_others"] = cand.paper_id.map(by_others).fillna(0).astype(int)
    cand["cluster_id"] = cand.paper_id.map(f.lab)
    top50 = {X: set(g.sort_values("cites_by_others", ascending=False).paper_id.head(50))
             for X, g in cand.dropna(subset=["cluster_id"]).groupby("cluster_id")}
    cand["outside_top50_cited"] = [x not in top50.get(X, set()) for x, X in zip(cand.paper_id, cand.cluster_id)]
    cand["title"] = cand.paper_id.map(f.title)
    cand["label"] = cand.cluster_id.map(f.labels)
    cand["builders_not_citing"] = cand.paper_id.map(
        lambda x: json.dumps(sorted(quiet.get(x, ()), key=lambda p: -f.year.get(p, 0))[:3]))
    cols = ["paper_id", "cluster_id", "label", "title", "year", "attributed_quiet", "attributed", "cites_by_others",
            "quiet_rank", "quiet_foundation", "outside_top50_cited", "builders_not_citing"]
    return cand[cols].sort_values(["quiet_foundation", "attributed_quiet"], ascending=False).assign(
        as_of=T, evidence=QUIET_EVIDENCE)


# ------------------------------------------------------------------ tools to learn (experiment 20)

def _method_labels(con, run):
    cl = con.execute("""SELECT cluster_id, label FROM clusters WHERE run_id = ? AND facet = 'method' AND alpha = 0.5
                        AND resolution = 'medium' AND kind IN ('stable', 'hub')""", [run]).df()
    return cl.set_index("cluster_id").label


def tools_to_learn(f: "O._Field", con) -> pd.DataFrame:
    T = f.t_live
    labels = _method_labels(con, f.run)
    cents = pd.read_parquet(O._project(f.run, "vectors") / f"cluster_centroids_{f.run}.parquet")
    c = cents[cents.cluster_id.isin(set(labels.index))]
    Cm = np.stack([np.asarray(x, dtype=np.float32) for x in c.centroid])
    fpos, fv = O._vectors("method", None)
    gpos, gv = O._vectors("method_gloss", None)
    g = np.array([gv[gpos[k]] if k in gpos else fv[i] for k, i in fpos.items()], dtype=np.float32)
    v = O._unit(0.5 * fv + 0.5 * g)
    mlab = pd.Series(np.array(c.cluster_id)[np.argmax(v @ Cm.T, axis=1)], index=list(fpos))
    mlab = mlab[mlab.index.isin(f.lab.index)]
    rec = [x for x in mlab.index if T - 2 <= f.year[x] <= T]
    ctype = dict(con.execute("SELECT paper_id, text FROM facets WHERE facet = 'contribution_type'").fetchall())
    traits = []
    for m in sorted(set(mlab)):
        mem = [x for x in rec if mlab[x] == m]
        if len(mem) < 10:
            continue
        if np.mean([ctype.get(x) == "survey" for x in mem]) >= 0.5:
            continue  # a community of literature reviews is not a tool to learn
        mv = f.MV[[f.MP[x] for x in mem if x in f.MP]]
        cen = O._unit(mv.mean(axis=0))
        inn = [float(f.IV[f.IP[x]] @ cen) for x in mem if x in f.IP]
        used = pd.Series([f.lab.get(x) for x in mem]).value_counts()
        traits.append({"method_cluster": m, "breadth": int((used >= 2).sum()), "dispersion": float(1 - (mv @ cen).mean()),
                       "inward": float(np.mean(np.array(inn) >= 0.85)) if inn else 0.0, "method_papers_recent": len(mem)})
    M = pd.DataFrame(traits)
    z = lambda s: (s - s.mean()) / (s.std() or 1)  # noqa: E731
    M["stable_score"] = z(M.breadth) - z(M.dispersion) - z(M.inward)
    hi, lo = M.stable_score.quantile([2 / 3, 1 / 3])
    M["status"] = np.where(M.stable_score >= hi, "settled", np.where(M.stable_score <= lo, "in flux", "in between"))
    M = M.set_index("method_cluster")
    rows = []
    for X in sorted(set(f.lab)):
        mem = [x for x in rec if f.lab.get(x) == X]
        used = pd.Series([mlab[x] for x in mem]).value_counts()
        for m, n in used.items():
            if n < 2 or m not in M.index:
                continue
            ex = [x for x in mem if mlab[x] == m][:2]
            rows.append({"cluster_id": X, "label": f.labels.get(X, X), "method_cluster": m,
                         "method_label": labels.get(m, m), "papers_here_recent": int(n), "status": M.loc[m, "status"],
                         "stable_score": float(M.loc[m, "stable_score"]), "breadth": int(M.loc[m, "breadth"]),
                         "dispersion": float(M.loc[m, "dispersion"]), "inward": float(M.loc[m, "inward"]),
                         "examples": json.dumps([{"paper_id": x, "title": f.title.get(x)} for x in ex])})
    return pd.DataFrame(rows).sort_values(["cluster_id", "papers_here_recent"], ascending=[True, False]).assign(
        as_of=T, recent_years=f"{T - 2}-{T}", evidence=TOOLS_EVIDENCE)


# ------------------------------------------------------------------ untried applications (experiment 19)

def untried_applications(f: "O._Field", con, k: int = 15, top: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    T = f.t_live
    DP, DV = O._vectors("domain_data")
    dom_ids = [x for x in DP if f.year.get(x, 9999) <= T and x in f.lab.index]
    D = DV[[DP[x] for x in dom_ids]]
    km = KMeans(k, n_init=10, random_state=0).fit(D)
    cent = O._unit(km.cluster_centers_)
    dom = np.argmax(D @ cent.T, axis=1)
    text = dict(con.execute("SELECT paper_id, text FROM facets WHERE facet = 'domain_data'").fetchall())
    ex = {}
    for d in range(k):
        idx = np.where(dom == d)[0]
        order = idx[np.argsort(-(D[idx] @ cent[d]))][:3]
        ex[d] = [text.get(dom_ids[i]) for i in order]
    C = pd.DataFrame({"domain_id": range(k), "papers": [int((dom == d).sum()) for d in range(k)],
                      "examples": [json.dumps(ex[d]) for d in range(k)], "centroid": list(cent.astype(float))})
    cell = pd.crosstab(pd.Series([f.lab[x] for x in dom_ids]), pd.Series(dom))
    rows = []
    for X in sorted(set(f.lab)):
        mem = [x for x in f.lab.index[f.lab.values == X] if x in f.PP and f.year[x] <= T]
        if len(mem) < 10:
            continue
        pc = O._unit(f.PV[[f.PP[x] for x in mem]].mean(axis=0))
        used = cell.loc[X] if X in cell.index else pd.Series(0, index=range(k))
        fit = cent @ pc
        cand = [d for d in np.argsort(-fit) if used.get(d, 0) == 0][:top]
        for r, d in enumerate(cand, 1):
            rows.append({"cluster_id": X, "label": f.labels.get(X, X), "rank": r, "domain_id": int(d),
                         "fit": float(fit[d]), "domain_examples": json.dumps(ex[d]), "domain_papers": int((dom == d).sum()),
                         "domains_used_here": int((used > 0).sum())})
    return pd.DataFrame(rows).assign(as_of=T, evidence=APPS_EVIDENCE), C.assign(as_of=T)


# ------------------------------------------------------------------ crowding and demand (experiment 23)

def crowding(f: "O._Field") -> tuple[pd.DataFrame, dict]:
    T = f.t_live
    rng = np.random.default_rng(0)
    LP, LV = O._vectors("limitations")

    def q99(A, B, n=4000):
        i, j = rng.integers(0, len(A), n), rng.integers(0, len(B), n)
        return float(np.quantile(np.sum(A[i] * B[j], axis=1), 0.99))

    th = {"same_question": q99(f.PV, f.PV), "same_method": q99(f.MV, f.MV), "answers_limitation": q99(f.IV, LV),
          "window": [T - 2, T], "evidence": CROWD_EVIDENCE}
    now = [x for x in f.PP if T - 2 <= f.year.get(x, 0) <= T]
    Pn = f.PV[[f.PP[x] for x in now]]
    lim = [x for x in LP if f.year.get(x, 9999) <= T]
    Lv = LV[[LP[x] for x in lim]]
    ids = [x for x in f.PP if x in f.year.index]
    rows = []
    for s in range(0, len(ids), 800):
        blk = ids[s:s + 800]
        S = f.PV[[f.PP[x] for x in blk]] @ Pn.T
        for k, x in enumerate(blk):
            tm = f.team.get(x, frozenset())
            same = [now[j] for j in np.where(S[k] >= th["same_question"])[0]
                    if now[j] != x and not (f.team.get(now[j], frozenset()) & tm)]
            dup = [p for p in same if p in f.MP and x in f.MP and float(f.MV[f.MP[p]] @ f.MV[f.MP[x]]) >= th["same_method"]]
            dem = []
            if x in f.IP and len(Lv):
                sl = Lv @ f.IV[f.IP[x]]
                dem = [lim[j] for j in np.argsort(-sl) if sl[j] >= th["answers_limitation"] and lim[j] != x][:5]
            rows.append({"paper_id": x, "same_question_now": len(same), "near_duplicates_now": len(dup),
                         "limitations_answered": len(dem),
                         "same_question_examples": json.dumps(sorted(same, key=lambda p: -float(S[k][now.index(p)]))[:3]),
                         "limitation_examples": json.dumps(dem[:3])})
    return pd.DataFrame(rows).assign(as_of=T), th


def build_discovery(f: "O._Field", con, L: pd.DataFrame, out) -> tuple[dict, pd.DataFrame]:
    Q = quiet_foundations(f, L)
    Q.to_parquet(out / "quiet_foundations.parquet", index=False)
    Tl = tools_to_learn(f, con)
    Tl.to_parquet(out / "tools_to_learn.parquet", index=False)
    A, Dc = untried_applications(f, con)
    A.to_parquet(out / "untried_applications.parquet", index=False)
    Dc.to_parquet(out / "domain_centroids.parquet", index=False)
    Cw, th = crowding(f)
    Cw.to_parquet(out / "crowding.parquet", index=False)
    (out / "crowding_thresholds.json").write_text(json.dumps(th, indent=2) + "\n")
    return {"quiet_foundations": int(Q.quiet_foundation.sum()),
            "quiet_outside_top50": int((Q.quiet_foundation & Q.outside_top50_cited).sum()),
            "tools_rows": len(Tl), "tools_settled": int(Tl.drop_duplicates("method_cluster").status.eq("settled").sum()),
            "untried_rows": len(A), "crowding_papers": len(Cw)}, Q
