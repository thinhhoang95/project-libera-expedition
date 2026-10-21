"""Facet categories: every facet's clusters as browsable categories, and one details sheet per category.

A category is a cluster of one facet (problem, method, innovation, result, limitations) at that facet's
reference setting: problem and method at α = 0.5 (where bridges and method transfer are computed), the others
at α = 0, all at medium resolution. The problem categories are the communities of Start here.

The sheet sends every paper of the category with its categories in the other facets and its venue (journal or
proceedings), so the browser can scope the trend, the distributions over the other facets and journals, and the
paper list to any time window without another request.
"""

from __future__ import annotations

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel

from . import timing as tm
from .data import Router, columns, jl, memo, q, records, titles
from .views_map import BRIDGE_TYPES, _bridge_title

router = Router()
RES = "medium"
FACETS = {"problem": 0.5, "method": 0.5, "innovation": 0.0, "result": 0.0, "limitations": 0.0}
SIGNAL_LABELS = {**BRIDGE_TYPES, "bridge_paper": "Paper to bring in", "bridge_paper_gem": "Hidden gem",
                 "new_combination": "New combination", "emerging_cluster": "Emerging (describe only)",
                 "direction_growing": "Growing (describe only)", "direction_shrinking": "Shrinking (describe only)",
                 "niche": "Niche (describe only)", "dormant": "Dormant (describe only)"}


def facet_of(cid: str) -> tuple[str, float]:
    """`problem:0.5:medium:003` → ("problem", 0.5)."""
    parts = cid.split(":")
    if len(parts) < 4 or parts[0] not in FACETS:
        raise HTTPException(404, f"Not a category id: {cid}")
    return parts[0], float(parts[1])


@memo(8)
def _clusters(p: str) -> pd.DataFrame:
    where = " OR ".join("(facet = ? AND alpha = ?)" for _ in FACETS)
    args = [x for f, a in FACETS.items() for x in (f, a)]
    return q(f"""SELECT c.facet, c.cluster_id, c.label, c.description, c.size, c.kind, c.color, c.top_terms,
                        c.stability, l.x IS NOT NULL AS on_map
                 FROM clusters c LEFT JOIN viz_cluster_layout l USING (run_id, cluster_id)
                 WHERE c.run_id = ? AND c.resolution = ? AND c.kind IN ('stable', 'hub') AND ({where})
                 ORDER BY c.facet, c.size DESC""", (p, RES, *args))


@memo(8)
def _membership(p: str) -> pd.DataFrame:
    """One row per corpus paper: year, citations, and its category in each facet (None = unassigned or no facet)."""
    where = " OR ".join("(pc.facet = ? AND pc.alpha = ?)" for _ in FACETS)
    args = [x for f, a in FACETS.items() for x in (f, a)]
    m = q(f"""SELECT pc.paper_id, pc.facet, pc.cluster_id FROM paper_cluster pc
              WHERE pc.run_id = ? AND pc.resolution = ? AND ({where})""", (p, RES, *args))
    ok = set(_clusters(p).cluster_id)
    m.loc[~m.cluster_id.isin(ok), "cluster_id"] = "-"  # has the facet, but no category ("-" keeps the denominator)
    wide = m.drop_duplicates(["paper_id", "facet"]).pivot(index="paper_id", columns="facet", values="cluster_id")
    papers = q("""SELECT p.paper_id, p.year, p.title, p.cited_by_count AS cites FROM papers p
                  JOIN provenance v USING (paper_id) WHERE v.run_id = ?""", (p,)).set_index("paper_id")
    out = papers.join(wide, how="left")
    for f in FACETS:
        if f not in out:
            out[f] = None
    return out


@memo(8)
def _cited_here(p: str) -> pd.Series:
    """Citations from corpus papers, by cited corpus paper."""
    d = q("""SELECT r.cited_id, count(*) AS n FROM refs r
             JOIN provenance v ON v.paper_id = r.citing_id AND v.run_id = ?
             JOIN provenance w ON w.paper_id = r.cited_id AND w.run_id = ?
             GROUP BY 1""", (p, p))
    return d.set_index("cited_id").n


def _as_of(p: str, y1: int) -> int:
    from .views_portfolio import _summary  # noqa: PLC0415 (views_portfolio imports a lot; only needed here)
    s = _summary(p)
    return int(s["as_of"]) if s else y1


@router.get("/p/{p}/catalogue")
@memo(8)
def catalogue(p: str) -> dict:
    """Every facet's categories with yearly paper counts, and each facet's yearly totals (the share denominators)."""
    cl = _clusters(p)
    if cl.empty:
        return {"built": False}
    M = _membership(p)
    M = M[M.year.notna()]
    years = q("""SELECT year, n FROM year_totals WHERE run_id = ? AND facet = 'corpus' AND year IS NOT NULL
                 ORDER BY year""", (p,))
    y1 = int(M.year.max())
    y0 = int(max(M.year.min(), years.loc[years.n >= 20, "year"].min())) if len(years) else int(M.year.min())
    span = range(y0, y1 + 1)
    facets = []
    for f, a in FACETS.items():
        sub = M[M[f].notna() & (M.year >= y0)]
        tot = sub.groupby("year").size().reindex(span, fill_value=0)
        cnt = sub[sub[f] != "-"].groupby([f, "year"]).size().unstack(fill_value=0).reindex(columns=span, fill_value=0)
        rows = cl[cl.facet == f].copy()
        rows["counts"] = [cnt.loc[c].astype(int).tolist() if c in cnt.index else [0] * len(span)
                          for c in rows.cluster_id]
        facets.append({"facet": f, "alpha": a, "papers": int(M[f].notna().sum()),
                       "assigned": int((M[f].notna() & (M[f] != "-")).sum()), "totals": tot.astype(int).tolist(),
                       "categories": records(rows.drop(columns=["facet"]))})
    return {"built": True, "y0": y0, "y1": y1, "as_of": _as_of(p, y1), "resolution": RES, "facets": facets}


@router.get("/p/{p}/category/{cid}")
@memo(32)
def category(p: str, cid: str) -> dict:
    """One category's papers, each with its categories in the other facets; in-facet citation flows; signals.

    Any cluster opens here (the map and cluster review use other α and resolutions); `is_category` says whether
    it is one of the facet's reference categories (the ones listed in the catalogue).
    """
    facet, _ = facet_of(cid)
    row = q("""SELECT c.facet, c.cluster_id, c.label, c.description, c.size, c.kind, c.color, c.top_terms, c.stability,
                      c.alpha, c.resolution, c.merged_into, l.x IS NOT NULL AS on_map
               FROM clusters c LEFT JOIN viz_cluster_layout l USING (run_id, cluster_id)
               WHERE c.run_id = ? AND c.cluster_id = ?""", (p, cid))
    if row.empty:
        raise HTTPException(404, f"No cluster {cid} in this project")
    meta = records(row)[0]
    alpha, res = float(meta["alpha"]), meta["resolution"]
    meta["is_category"] = alpha == FACETS[facet] and res == RES and meta["kind"] in ("stable", "hub")
    sims = q("""SELECT paper_id, round(sim_to_centroid, 3) AS sim FROM paper_cluster
                WHERE run_id = ? AND facet = ? AND resolution = ? AND alpha = ? AND coalesce(cluster_id, raw_cluster_id) = ?""",
             (p, facet, res, alpha, cid)).drop_duplicates("paper_id")
    M = _membership(p)
    ids = [x for x in sims.paper_id if x in M.index]
    ph = ",".join("?" * len(ids)) or "''"
    text = q(f"""SELECT paper_id, left(text, 420) AS text FROM facets WHERE facet = ? AND paper_id IN ({ph})""",
             (facet, *ids)).drop_duplicates("paper_id")
    lim = q(f"""SELECT DISTINCT paper_id FROM facets WHERE facet = 'limitations' AND text IS NOT NULL
                AND paper_id IN ({ph})""", ids).paper_id
    others = [f for f in FACETS if f != facet]
    P = M.loc[ids].reset_index()[["paper_id", "year", "title", "cites", *others]]
    P = P.merge(sims, on="paper_id", how="left").merge(text, on="paper_id", how="left")
    ven = q(f"""SELECT paper_id, venue_id, venue_name AS venue FROM papers WHERE paper_id IN ({ph})""", ids)
    P = P.merge(ven.drop_duplicates("paper_id"), on="paper_id", how="left")
    P["cites_here"] = P.paper_id.map(_cited_here(p)).fillna(0).astype(int)
    P["has_limitation"] = P.paper_id.isin(set(lim))
    P = P.sort_values("sim", ascending=False, na_position="last")
    for f in others:
        P[f] = P[f].where(P[f].notna() & (P[f] != "-"), None)
    flows = q("""SELECT year, from_cluster, to_cluster, count FROM viz_flow_year WHERE run_id = ? AND kind = 'citation'
                 AND facet = ? AND alpha = ? AND from_cluster != to_cluster AND (from_cluster = ? OR to_cluster = ?)""",
              (p, facet, alpha, cid, cid))
    labels = dict(q("""SELECT cluster_id, label FROM clusters WHERE run_id = ? AND facet = ? AND alpha = ? AND resolution = ?""",
                    (p, facet, alpha, res)).itertuples(index=False))
    return {"meta": meta, "papers": records(P), "flows": records(flows), "flow_labels": labels,
            "signals": _signals_for(p, cid)}


@router.get("/p/{p}/cooccurrence")
@memo(16)
def cooccurrence(p: str, a: str = "problem", b: str = "method") -> dict:
    """Papers per (category of facet a, category of facet b, year): the cross-facet matrix."""
    if a not in FACETS or b not in FACETS or a == b:
        raise HTTPException(422, "Pick two different facets")
    M = _membership(p)
    sub = M[M[a].notna() & M[b].notna() & (M[a] != "-") & (M[b] != "-") & M.year.notna()]
    g = sub.groupby([a, b, "year"]).size().reset_index(name="n").rename(columns={a: "a", b: "b"})
    g["year"] = g.year.astype(int)
    return columns(g)


@router.get("/p/{p}/venues")
@memo(8)
def venues(p: str) -> dict:
    """Corpus papers per (venue, year): the whole-map baseline for a category's journals."""
    d = q("""SELECT p.venue_id, p.year, count(*) AS n FROM papers p JOIN provenance v USING (paper_id)
             WHERE v.run_id = ? AND p.year IS NOT NULL GROUP BY 1, 2""", (p,))
    d["venue_id"] = d.venue_id.fillna("")
    d["year"] = d.year.astype(int)
    return columns(d)


def _signals_for(p: str, cid: str) -> list[dict]:
    """Opportunities that name this category: bridges, papers to bring in, method transfer, new combinations."""
    s = q("""SELECT signal_type, rank, subject, judge_score, score FROM signals WHERE run_id = ?
             AND signal_type IN ({}) AND CAST(subject AS VARCHAR) LIKE ? ORDER BY signal_type, rank""".format(
        ",".join("?" * len(SIGNAL_LABELS))), (p, *SIGNAL_LABELS, f'%"{cid}"%'))
    out = []
    T = tm.as_of(p)
    when = tm.paper_when(p, [jl(x).get("paper_id") for x in s.subject], T)
    for typ, g in s.groupby("signal_type", sort=False):
        for r in g.head(8).itertuples():
            subj = jl(r.subject)
            others = [v for k, v in subj.items() if isinstance(v, str) and v != cid and v.split(":")[0] in FACETS
                      and v.count(":") == 3]
            role = next((k for k, v in subj.items() if v == cid), None)
            paper = subj.get("paper_id")
            out.append({"type": typ, "label": SIGNAL_LABELS[typ], "rank": int(r.rank), "role": role,
                        "title": subj.get("title") if paper else _bridge_title(subj), "paper_id": paper,
                        "other": others[0] if others else None, "year": (when.get(paper) or {}).get("year"),
                        "timing": (when.get(paper) or {}).get("timing"),
                        "judge": None if pd.isna(r.judge_score) else float(r.judge_score)})
    return out


def categories_of(p: str, pid: str) -> list[dict]:
    """A paper's category in each facet (for the paper card)."""
    M = _membership(p)
    if pid not in M.index:
        return []
    cl = _clusters(p).set_index("cluster_id")
    out = []
    for f in FACETS:
        c = M.at[pid, f]
        if isinstance(c, str) and c in cl.index:
            out.append({"facet": f, "cluster_id": c, "label": cl.at[c, "label"], "color": cl.at[c, "color"],
                        "kind": cl.at[c, "kind"]})
    return out


def citations_of(p: str, pid: str, limit: int = 8) -> dict:
    """Corpus papers this paper cites, and corpus papers citing it (most cited here first)."""
    refs = q("""SELECT r.cited_id AS paper_id FROM refs r JOIN provenance v ON v.paper_id = r.cited_id AND v.run_id = ?
                WHERE r.citing_id = ?""", (p, pid)).paper_id.tolist()
    citers = q("""SELECT r.citing_id AS paper_id FROM refs r JOIN provenance v ON v.paper_id = r.citing_id
                  AND v.run_id = ? WHERE r.cited_id = ?""", (p, pid)).paper_id.tolist()
    ch = _cited_here(p)

    def top(ids):
        tt = titles(ids)
        rows = [{**tt[i], "cites_here": int(ch.get(i, 0))} for i in ids if i in tt]
        return sorted(rows, key=lambda r: (-r["cites_here"], -(r["year"] or 0)))[:limit]

    return {"cites_here": int(ch.get(pid, 0)), "refs_here": len(refs), "citers_here": len(citers),
            "refs": top(refs), "citers": top(citers)}


@router.get("/p/{p}/categories/search")
def search_categories(p: str, term: str, limit: int = 8) -> list[dict]:
    t = term.strip().lower()
    if len(t) < 2:
        return []
    cl = _clusters(p)
    hay = cl.label.str.lower() + " " + cl.top_terms.map(lambda x: " ".join(x) if x is not None else "").str.lower()
    hit = cl[hay.str.contains(t, regex=False)].copy()
    hit["starts"] = hit.label.str.lower().str.startswith(t)
    hit = hit.sort_values(["starts", "size"], ascending=False).head(limit)
    return records(hit[["facet", "cluster_id", "label", "size", "kind", "color"]])



class Ids(BaseModel):
    ids: list[str]


@router.post("/p/{p}/paper-meta")
def paper_meta(p: str, body: Ids) -> dict:
    """Year, citations overall (OpenAlex) and citations from this map, for any list of papers (sorting in lists)."""
    ids = [x for x in dict.fromkeys(body.ids) if isinstance(x, str)][:500]
    if not ids:
        return {}
    d = q(f"""SELECT paper_id, year, cited_by_count AS cites, venue_name AS venue FROM papers
              WHERE paper_id IN ({",".join("?" * len(ids))})""", ids)
    ch = _cited_here(p)
    return {r.paper_id: {"year": None if pd.isna(r.year) else int(r.year),
                         "cites": None if pd.isna(r.cites) else int(r.cites),
                         "cites_here": int(ch.get(r.paper_id, 0)), "venue": r.venue}
            for r in d.itertuples()}
