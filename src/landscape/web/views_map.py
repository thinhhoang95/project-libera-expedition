"""Map & grid (Phase 6b explorer) as JSON: one map, one grid, one time axis, and the signal layers.

The map sends every paper once per (facet, α, resolution); the time window, layer filters and selection
are applied in the browser, as the Streamlit explorer did on each rerun.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .data import Router, artefact, columns, jl, memo, portfolio_file, q, records, titles

router = Router()
BRIDGE_TYPES = {"bridge_lim_inn": "Limitation → innovation", "bridge_method_transfer": "Method transfer",
                "bridge_unresolved_limitation": "Unresolved limitation",
                "bridge_semantic_not_cited": "Semantic but not cited"}


def _signals(p: str, types: tuple) -> pd.DataFrame:
    return q(f"""SELECT * FROM signals WHERE run_id = ? AND signal_type IN ({",".join("?" * len(types))})
                 ORDER BY signal_type, rank""", (p, *types))


@router.get("/p/{p}/map")
@memo(32)
def map_data(p: str, facet: str = "problem", alpha: float = 0.3, res: str = "medium") -> dict:
    years = q("""SELECT year, n FROM year_totals WHERE run_id = ? AND facet = 'corpus'
                 AND year IS NOT NULL ORDER BY year""", (p,))
    if years.empty:
        return {"built": False, "reason": "core"}
    y0 = int(max(years.year.min(), years.loc[years.n >= 20, "year"].min()))
    layout = q("""SELECT l.paper_id, l.x, l.y, p.year, left(p.title, 110) AS title, c.cluster_id FROM viz_layout l
                  JOIN papers p USING (paper_id)
                  LEFT JOIN paper_cluster c ON c.run_id = l.run_id AND c.paper_id = l.paper_id
                       AND c.facet = l.facet AND c.alpha = l.alpha AND c.resolution = ?
                  WHERE l.run_id = ? AND l.facet = ? AND l.alpha = ?""", (res, p, facet, alpha))
    if layout.empty:
        return {"built": False, "reason": "viz"}
    clusters = q("""SELECT c.cluster_id, c.label, c.description, c.size, c.stability, c.kind, c.centrality,
                           c.color, c.top_terms, c.parent_cluster_id, l.x, l.y
                    FROM clusters c LEFT JOIN viz_cluster_layout l USING (run_id, cluster_id)
                    WHERE c.run_id = ? AND c.facet = ? AND c.alpha = ? AND c.resolution = ?
                      AND c.kind IN ('stable', 'hub')""", (p, facet, alpha, res))
    cy = q("""SELECT cluster_id, year, n, share FROM cluster_year WHERE run_id = ? AND facet = ?
              AND alpha = ? AND resolution = ?""", (p, facet, alpha, res))
    ids = tuple(clusters.cluster_id)
    status = q(f"""SELECT cluster_id, year, status, scores FROM viz_cluster_year WHERE run_id = ?
                   AND scores IS NOT NULL AND cluster_id IN ({",".join("?" * len(ids)) or "''"})""", (p, *ids))
    layout["year"] = layout.year.fillna(0).astype(int)
    for c in ("x", "y"):
        layout[c] = layout[c].round(4)
    return {"built": True, "y0": y0, "y1": int(years.year.max()), "years": records(years),
            "have_signals": q("SELECT 1 FROM signals WHERE run_id = ? LIMIT 1", (p,)).shape[0] > 0,
            "points": columns(layout), "clusters": records(clusters), "cluster_year": columns(cy),
            "status": [{"cluster_id": c, "year": int(y), "status": s, "scores": jl(sc)}
                       for c, y, s, sc in zip(status.cluster_id, status.year, status.status, status.scores)]}


@router.get("/p/{p}/map/scores")
@memo(16)
def paper_scores(p: str, metric: str, facet: str | None = None) -> dict:
    if metric == "novelty":
        d = q("""SELECT paper_id, value FROM paper_scores WHERE run_id = ? AND metric = 'novelty' AND facet = ?""",
              (p, facet or "abstract"))
    else:
        d = q("""SELECT paper_id, value FROM paper_scores WHERE run_id = ? AND metric = 'local_momentum'""", (p,))
        if len(d):  # top 5% of the whole corpus, as in the explorer
            d = d[d.value >= d.value.quantile(0.95)]
    return columns(d)


@router.get("/p/{p}/map/flows")
@memo(32)
def flows(p: str, kind: str, facet: str, alpha: float) -> dict:
    f = q("""SELECT year, from_cluster, to_cluster, count FROM viz_flow_year WHERE run_id = ?
             AND kind = ? AND facet = ? AND alpha = ? AND from_cluster != to_cluster""", (p, kind, facet, alpha))
    return columns(f)


@router.get("/p/{p}/map/bridge-arcs")
@memo()
def bridge_arcs(p: str) -> list[dict]:
    b = _signals(p, ("bridge_lim_inn", "bridge_semantic_not_cited"))
    out = []
    for r in b.itertuples():
        s = jl(r.subject)
        out.append({"type": r.signal_type, "label": BRIDGE_TYPES[r.signal_type], "rank": int(r.rank),
                    "a": s.get("from_cluster", s.get("cluster_a")), "b": s.get("to_cluster", s.get("cluster_b")),
                    "judge": None if pd.isna(r.judge_score) else float(r.judge_score)})
    return out


@router.get("/p/{p}/map/authors")
@memo()
def authors(p: str) -> list[dict]:
    ppl = _signals(p, ("seeder_author", "broker_author"))
    seen = {}
    for r in ppl.itertuples():
        s = jl(r.subject)
        seen.setdefault(s["author_id"], {"author_id": s["author_id"],
                                         "label": f"{s.get('name')} ({r.signal_type.split('_')[0]} #{r.rank})"})
    return list(seen.values())


@router.get("/p/{p}/map/author-path")
def author_path(p: str, author: str, facet: str, alpha: float) -> list[dict]:
    return records(q("""SELECT year, x, y FROM viz_author_year WHERE run_id = ? AND author_id = ?
                        AND facet = ? AND alpha = ? ORDER BY year""", (p, author, facet, alpha)))


@router.get("/p/{p}/grid")
@memo(16)
def grid(p: str, alpha: float = 0.3) -> dict:
    pm = q("""SELECT problem_cluster, method_cluster, year, count, first_year FROM pm_matrix
              WHERE run_id = ? AND alpha = ?""", (p, alpha))
    if pm.empty:
        return {"built": False}
    cl = q("""SELECT cluster_id, label, size, kind FROM clusters WHERE run_id = ? AND alpha = ?
              AND resolution = 'medium' AND kind IN ('stable', 'hub')""", (p, alpha))
    rows = cl[cl.cluster_id.str.startswith("problem:")].sort_values("size", ascending=False)
    cols = cl[cl.cluster_id.str.startswith("method:")].sort_values("size", ascending=False)
    transfer = []
    if alpha == 0.5:
        tr = _signals(p, ("bridge_method_transfer",))
        tr = tr[tr["rank"] <= 25]
        transfer = [{"method": jl(s)["method_cluster"], "problem": jl(s)["problem_cluster"], "rank": int(r)}
                    for s, r in zip(tr.subject, tr["rank"])]
    return {"built": True, "rows": records(rows), "cols": records(cols), "cells": columns(pm), "transfer": transfer}


@router.get("/p/{p}/cluster/{cid}/papers")
def central_papers(p: str, cid: str, res: str = "medium", alpha: float = 0.3) -> list[dict]:
    return records(q("""SELECT p.paper_id, p.year, p.title, round(pc.sim_to_centroid, 3) AS sim
                        FROM paper_cluster pc JOIN papers p USING (paper_id)
                        WHERE pc.run_id = ? AND pc.cluster_id = ? AND pc.resolution = ? AND pc.alpha = ?
                        ORDER BY pc.sim_to_centroid DESC LIMIT 10""", (p, cid, res, alpha)))


@router.get("/p/{p}/ranking/no-semantic")
@memo()
def ranking_no_semantic(p: str) -> dict:
    sv = q("""SELECT subject, features FROM signals WHERE run_id = ? AND variant = 'no_semantic'
              AND signal_type IN ('emerging_cluster', 'direction_growing', 'direction_shrinking', 'niche', 'dormant')""", (p,))
    return {jl(s)["cluster_id"]: jl(f) for s, f in zip(sv.subject, sv.features)}


# ------------------------------------------------------------------ bridges

def _bridge_title(s: dict) -> str:
    if "from_label" in s:
        return f"{s['from_label']} → {s['to_label']}"
    if "problem_label" in s:
        return f"{s['method_label']} → {s['problem_label']}"
    if "label_a" in s:
        return f"{s['label_a']} ~ {s['label_b']}"
    return s.get("label", "?")


@router.get("/p/{p}/bridges")
@memo()
def bridges(p: str) -> dict:
    b = _signals(p, tuple(BRIDGE_TYPES))
    out = {t: [] for t in BRIDGE_TYPES}
    for r in b.itertuples():
        if r.rank > 50 and r.signal_type != "bridge_lim_inn":
            continue
        out[r.signal_type].append({"rank": int(r.rank), "candidate": _bridge_title(jl(r.subject)),
                                   "judge": None if pd.isna(r.judge_score) else float(r.judge_score),
                                   "score": round(float(r.score), 4)})
    return {"types": BRIDGE_TYPES, "lists": {k: v[:50] for k, v in out.items()}}


@router.get("/p/{p}/bridges/{typ}/{rank}")
def bridge_detail(p: str, typ: str, rank: int) -> dict:
    d = q("SELECT * FROM signals WHERE run_id = ? AND signal_type = ? AND rank = ?", (p, typ, rank))
    if d.empty:
        raise HTTPException(404)
    r = d.iloc[0]
    f, s = jl(r.features), jl(r.subject)
    ev = list(r.evidence_paper_ids) if r.evidence_paper_ids is not None else []
    tt = titles(ev)
    out = {"type": typ, "rank": rank, "title": _bridge_title(s), "subject": s,
           "judge": None if pd.isna(r.judge_score) else float(r.judge_score), "rationale": r.judge_rationale,
           "obvious_reason": f.get("judge_obvious_reason"), "evidence": [tt.get(x, {"paper_id": x}) for x in ev],
           "features": {k: v for k, v in f.items() if k not in ("judge_obvious_reason", "pairs")}}
    if typ == "bridge_lim_inn":
        pairs = f.get("pairs", [])
        ids = [x for pr in pairs for x in (pr["limitation"], pr["innovation"])]
        ft = q(f"""SELECT paper_id, facet, text FROM facets WHERE paper_id IN ({",".join("?" * len(ids)) or "''"})
                   AND facet IN ('limitations', 'innovation')""", ids).drop_duplicates(["paper_id", "facet"])
        txt = {(a, b): c for a, b, c in ft.itertuples(index=False)}
        pt = titles(ids)
        out["pairs"] = [{"limitation": {**pt.get(pr["limitation"], {}), "text": txt.get((pr["limitation"], "limitations"))},
                         "innovation": {**pt.get(pr["innovation"], {}), "text": txt.get((pr["innovation"], "innovation"))},
                         "sim": pr["sim"]} for pr in pairs]
    return out


@router.get("/p/{p}/paper-bridges")
@memo()
def paper_bridges(p: str) -> dict:
    b = _signals(p, ("bridge_paper", "bridge_paper_gem"))
    if b.empty:
        return {"clusters": [], "lists": {}}
    b["subj"] = b.subject.map(jl)
    b["cluster_id"] = b.subj.map(lambda s: s["cluster_id"])
    size = dict(q("SELECT cluster_id, size FROM clusters WHERE run_id = ?", (p,)).itertuples(index=False))
    all_labs = dict(q("SELECT cluster_id, label FROM clusters WHERE run_id = ?", (p,)).itertuples(index=False))
    labs = dict(zip(b.cluster_id, b.subj.map(lambda s: s.get("label") or s["cluster_id"])))
    kinds = dict(zip(b.cluster_id, b.group_kind))
    clusters = [{"cluster_id": c, "label": labs[c], "kind": kinds.get(c), "size": size.get(c)}
                for c in sorted(labs, key=lambda c: -(size.get(c) or 0))]
    lists: dict = {}
    for r in b.sort_values("rank").itertuples():
        f = jl(r.features)
        lists.setdefault(r.cluster_id, {"bridge_paper": [], "bridge_paper_gem": []})[r.signal_type].append({
            "rank": int(r.rank), "paper_id": r.subj["paper_id"], "title": r.subj.get("title") or r.subj["paper_id"],
            "from": all_labs.get(r.subj.get("from_cluster"), ""), "cites": f.get("citations_so_far"),
            "score": round(float(r.score), 2), "coupling": f.get("coupling"), "problem_sim": f.get("problem_sim"),
            "limitation_sim": f.get("limitation_sim"), "best_matching_limitation": f.get("best_matching_limitation"),
            "status": f.get("status"), "age": f.get("age"), "closeness": f.get("closeness")})
    return {"clusters": clusters, "lists": lists}


@router.get("/p/{p}/facet-text")
def facet_text(p: str, ids: str, facet: str) -> dict:
    idl = [x for x in ids.split(",") if x][:50]
    t = q(f"""SELECT paper_id, text FROM facets WHERE facet = ? AND paper_id IN ({",".join("?" * len(idl)) or "''"})""",
          (facet, *idl)).drop_duplicates("paper_id")
    tt = titles(idl)
    return {x: {**tt.get(x, {}), "text": dict(zip(t.paper_id, t.text)).get(x)} for x in idl}


# ------------------------------------------------------------------ outside the field

@router.get("/p/{p}/external-imports")
@memo()
def external_imports(p: str) -> dict:
    L = portfolio_file(p, "live_imports.parquet")
    if L is None:
        return {"built": False}
    d = L.assign(global_trend=np.exp(L.g_trend).round(2), score=L.score.round(2))
    return {"built": True, "as_of": int(L.as_of.iloc[0]),
            "rows": records(d[["paper_id", "title", "year", "subfield", "global_trend", "recent_citers", "cluster_id",
                               "cluster_label", "score", "outside_field"]])}


@router.get("/p/{p}/external-imports/{wid}")
def external_detail(p: str, wid: str) -> dict:
    L = portfolio_file(p, "live_imports.parquet")
    if L is None or wid not in set(L.paper_id):
        raise HTTPException(404)
    r = L[L.paper_id == wid].iloc[0]
    out = {"work": records(L[L.paper_id == wid][["paper_id", "title", "year", "venue", "cited_by_count",
                                                 "corpus_citers_total", "as_of", "closest_corpus_paper"]])[0]}
    f = artefact(p, "external", "facets.parquet")
    if f is not None:
        f = f[f.paper_id == wid].drop_duplicates("facet").set_index("facet").text
        out["facets"] = {k: f[k] for k in ("problem", "innovation", "limitations") if isinstance(f.get(k), str)}
    e = artefact(p, "external", "edges.parquet")
    if e is not None:
        riders = e[(e.cited_id == wid) & (e.citing_year >= int(r.as_of) - 2)].citing_id.unique().tolist()
        out["citing_recent"] = sorted(titles(riders).values(), key=lambda t: -(t["year"] or 0))
    return out


# ------------------------------------------------------------------ people

@router.get("/p/{p}/people")
@memo()
def people(p: str) -> dict:
    ppl = _signals(p, ("seeder_author", "broker_author", "institution_seeder", "institution_broker"))
    out = {}
    for typ in ("seeder_author", "broker_author", "institution_seeder", "institution_broker"):
        d = ppl[ppl.signal_type == typ].sort_values("rank").head(15)
        out[typ] = [{"rank": int(r), "name": jl(s).get("name"), "score": round(float(v), 3)}
                    for r, s, v in zip(d["rank"], d.subject, d.score)]
    ew = _signals(p, ("early_work",))
    ew = ew[ew.variant.fillna("all") == "all"].head(30) if "variant" in ew else ew.head(30)
    out["early_work"] = [{"rank": int(r), "paper_id": jl(s).get("paper_id"), "title": jl(s).get("title"),
                          "year": jl(f).get("year"), "cites": jl(f).get("citations_so_far"), "score": round(float(v), 2)}
                         for r, s, f, v in zip(ew["rank"], ew.subject, ew.features, ew.score)]
    return out
