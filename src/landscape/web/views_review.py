"""Corpus view, facet review, cluster review, the paper card, search and saved judgments."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel

from .. import jobs
from . import views_facets
from .data import Router, jl, memo, q, records, run_params

router = Router()
FAITHFUL = ["problem", "domain_data", "method", "innovation", "result", "limitations"]
GLOSSES = ["problem_gloss", "method_gloss"]
FACET_ORDER = ["problem", "problem_class", "problem_gloss", "method", "method_family", "method_gloss",
               "innovation", "result", "limitations", "domain_data", "contribution_type"]


# ------------------------------------------------------------------ papers

@router.get("/p/{p}/paper/{pid}")
def paper_card(p: str, pid: str) -> dict:
    d = q("""SELECT p.paper_id, p.doi, p.title, p.year, p.venue_name, p.type, p.cited_by_count, p.abstract_raw,
                    p.abstract_source, v.route, v.hop, v.via_id FROM papers p
             LEFT JOIN provenance v ON v.paper_id = p.paper_id AND v.run_id = ?
             WHERE p.paper_id = ?""", (p, pid))
    if d.empty:
        raise HTTPException(404, f"{pid} not found")
    out = records(d)[0]
    out["authors"] = q("""SELECT a.display_name FROM authorships s JOIN authors a USING (author_id)
                          WHERE s.paper_id = ? ORDER BY s.position""", (pid,)).display_name.tolist()
    f = q("SELECT facet, text FROM facets WHERE paper_id = ?", (pid,)).drop_duplicates("facet").set_index("facet").text
    out["facets"] = [{"facet": k, "text": f[k]} for k in FACET_ORDER if k in f.index and isinstance(f[k], str)]
    out["link"] = f"https://doi.org/{out['doi']}" if out.get("doi") else f"https://openalex.org/{pid}"
    out["categories"] = views_facets.categories_of(p, pid)
    out["citations"] = views_facets.citations_of(p, pid) if out.get("route") else None
    return out


@router.get("/p/{p}/search")
def search(p: str, term: str, limit: int = 30) -> list[dict]:
    if len(term.strip()) < 2:
        return []
    return records(q("""SELECT p.paper_id, p.year, p.title, p.doi FROM papers p JOIN provenance v USING (paper_id)
                        WHERE v.run_id = ? AND (p.title ILIKE ? OR p.doi ILIKE ?) ORDER BY p.year DESC LIMIT ?""",
                     (p, f"%{term}%", f"%{term}%", limit)))


# ------------------------------------------------------------------ corpus

@router.get("/p/{p}/corpus")
@memo()
def corpus(p: str) -> dict:
    k = q("""SELECT count(*) n, avg(CASE WHEN abstract_raw IS NULL OR length(abstract_raw) < 100 THEN 1 ELSE 0 END) miss
             FROM provenance JOIN papers USING (paper_id) WHERE run_id = ?""", (p,)).iloc[0]
    spend_row = q("SELECT spend_usd FROM runs WHERE run_id = ?", (p,))
    spend = jl(spend_row.spend_usd.iloc[0]) if len(spend_row) else {}
    by = q("""SELECT p.year, v.route, count(*) n FROM provenance v JOIN papers p USING (paper_id)
              WHERE v.run_id = ? GROUP BY ALL ORDER BY 1""", (p,))
    by["year"] = np.where(by.year < 2010, "≤2009", by.year.astype("Int64").astype(str))
    by = by.groupby(["year", "route"], as_index=False).n.sum()
    routes = q("""SELECT route, hop, count(*) papers,
            round(avg(CASE WHEN abstract_raw IS NULL OR length(abstract_raw) < 100 THEN 1 ELSE 0 END), 3) missing_abstract,
            sum((abstract_source = 's2')::INT) s2_filled, sum((abstract_source = 'arxiv')::INT) arxiv_filled,
            min(year) first_year, max(year) last_year
        FROM provenance JOIN papers USING (paper_id) WHERE run_id = ? GROUP BY ALL ORDER BY ALL""", (p,))
    topics = q("""SELECT json_extract_string(p.topics, '$[0].name') topic,
                    sum((v.route != 'semantic')::INT) citation_routes, sum((v.route = 'semantic')::INT) semantic
             FROM provenance v JOIN papers p USING (paper_id) WHERE v.run_id = ?
             GROUP BY 1 ORDER BY citation_routes + semantic DESC LIMIT 15""", (p,))
    return {"papers": int(k.n), "missing_abstract": float(k.miss or 0), "spend": spend,
            "by_year": records(by), "routes": records(routes), "topics": records(topics)}


@router.get("/p/{p}/corpus/papers")
def corpus_papers(p: str, route: str) -> list[dict]:
    return records(q("""SELECT p.paper_id, p.year, p.title, v.hop, v.via_id, round(v.score, 3) score,
                             p.abstract_source FROM provenance v JOIN papers p USING (paper_id)
                      WHERE v.run_id = ? AND v.route = ? ORDER BY v.score DESC NULLS LAST, p.year DESC
                      LIMIT 500""", (p, route)))


# ------------------------------------------------------------------ facet review

def _extraction(p: str) -> tuple[str, str]:
    e = run_params(p).extraction
    return e.model, e.prompt_version


@memo(16)
def _facet_pool(p: str):
    model, pv = _extraction(p)
    ok = q("""SELECT s.paper_id, s.qa_flags FROM facet_status s JOIN provenance v USING (paper_id)
              WHERE v.run_id = ? AND s.model_id = ? AND s.prompt_version = ? AND s.status = 'ok'""", (p, model, pv))
    ok["h"] = [hashlib.md5(f"{p}|{x}".encode()).hexdigest() for x in ok.paper_id]
    gold = ok.sort_values("h").head(200).paper_id.tolist()
    flagged = ok[ok.qa_flags.fillna("{}").str.contains("too_long|tag_too_long")].sort_values("h").paper_id.tolist()
    lims = q("""SELECT f.paper_id FROM facets f JOIN provenance v USING (paper_id)
                WHERE v.run_id = ? AND f.facet = 'limitations' AND f.text IS NOT NULL
                  AND f.model_id = ? AND f.prompt_version = ? ORDER BY f.paper_id""", (p, model, pv)).paper_id.tolist()
    return {"gold": gold, "flagged": flagged, "limitations": lims}, dict(zip(ok.paper_id, ok.qa_flags))


@router.get("/p/{p}/facets/pool")
def facet_pool(p: str, mode: str = "gold", reviewer: str = "anonymous") -> dict:
    pools, _ = _facet_pool(p)
    pool = pools[mode]
    done = {jl(s.get("subject")).get("paper_id") for s in reviews_for(p, "facet_review", reviewer)}
    return {"pool": pool, "done": [x for x in pool if x in done], "gold_size": len(pools["gold"])}


@router.get("/p/{p}/facets/paper/{pid}")
def facet_paper(p: str, pid: str) -> dict:
    model, pv = _extraction(p)
    pools, flags = _facet_pool(p)
    m = q("SELECT title, year, abstract_raw, abstract_source FROM papers WHERE paper_id = ?", (pid,))
    if m.empty:
        raise HTTPException(404)
    f = q("SELECT facet, text FROM facets WHERE paper_id = ? AND model_id = ? AND prompt_version = ?",
          (pid, model, pv)).drop_duplicates("facet").set_index("facet").text
    get = lambda k: f.get(k) if isinstance(f.get(k), str) else None  # noqa: E731
    return {"paper_id": pid, **records(m)[0], "model": model, "prompt_version": pv,
            "faithful": [{"facet": k, "text": get(k)} for k in FAITHFUL],
            "glosses": [{"facet": k, "text": get(k)} for k in GLOSSES],
            "tags": {"problem_class": get("problem_class"), "method_family": get("method_family"),
                     "contribution_type": get("contribution_type")},
            "qa_flags": jl(flags.get(pid)) if flags.get(pid) else {}, "gold": pid in set(pools["gold"])}


@router.get("/p/{p}/facets/acceptance")
def facet_acceptance(p: str) -> dict | None:
    rows = [r for r in reviews_for(p, "facet_review") if jl(r["subject"]).get("gold_set")]
    if not rows:
        return None
    d = pd.DataFrame({"facet": [jl(r["subject"]).get("facet") for r in rows], "action": [r["action"] for r in rows]})
    faith = d[d.facet.isin(FAITHFUL)]
    lim = d[d.facet == "limitations"]
    gl = d[d.facet.isin(GLOSSES)]
    return {"supported": float((faith.action == "supported").sum() / max(len(faith), 1)),
            "limitations_unsupported": float((lim.action == "unsupported").sum() / max(len(lim), 1)),
            "gloss_ok": float((gl.action != "wrong variant").sum() / max(len(gl), 1)), "ratings": len(d)}


# ------------------------------------------------------------------ cluster review

@router.get("/p/{p}/clusters")
@memo(32)
def clusters(p: str, facet: str = "problem", alpha: float = 0.3, res: str = "medium") -> list[dict]:
    return records(q("""SELECT cluster_id, label, size, stability, kind, parent_cluster_id, label_model,
                               description, top_terms, color FROM clusters
                        WHERE run_id = ? AND facet = ? AND alpha = ? AND resolution = ? AND size >= 5
                        ORDER BY size DESC""", (p, facet, alpha, res)))


@router.get("/p/{p}/clusters/{cid}")
def cluster_detail(p: str, cid: str, facet: str = "problem", alpha: float = 0.3, res: str = "medium") -> dict:
    s = q("SELECT year, n, share FROM cluster_year WHERE run_id = ? AND cluster_id = ? ORDER BY year", (p, cid))
    central = q("""SELECT p.paper_id, p.year, p.title, f.text, round(pc.sim_to_centroid, 3) sim
                   FROM paper_cluster pc JOIN papers p USING (paper_id)
                   LEFT JOIN facets f ON f.paper_id = pc.paper_id AND f.facet = pc.facet
                   WHERE pc.run_id = ? AND pc.facet = ? AND pc.alpha = ? AND pc.resolution = ?
                     AND pc.raw_cluster_id = ?
                   ORDER BY pc.sim_to_centroid DESC LIMIT 12""", (p, facet, alpha, res, cid))
    prev = [r for r in reviews_for(p, "cluster_review") if jl(r["subject"]).get("cluster_id") == cid]
    return {"series": records(s), "central": records(central.drop_duplicates("paper_id")),
            "reviews": sorted(prev, key=lambda r: r["ts"] or "", reverse=True)}


# ------------------------------------------------------------------ judgments

class Review(BaseModel):
    view: str
    subject: dict
    action: str
    note: str = ""
    alpha: dict | None = None
    reviewer: str = "anonymous"


def reviews_for(p: str, view: str, reviewer: str | None = None) -> list[dict]:
    """Judgments already in the database (replica) plus the ones saved here and not yet copied there."""
    sql = "SELECT reviewer, view, subject, action, note, alpha, CAST(ts AS VARCHAR) ts FROM review_actions WHERE run_id = ? AND view = ?"
    args: list = [p, view]
    if reviewer:
        sql += " AND reviewer = ?"
        args.append(reviewer)
    old = records(q(sql, args))
    con = jobs.connect()
    sql = "SELECT reviewer, view, subject, action, note, alpha, ts FROM reviews WHERE run_id = ? AND view = ? AND synced = 0"
    new = [dict(r) for r in con.execute(sql + (" AND reviewer = ?" if reviewer else ""), args)]
    return old + new


@router.post("/p/{p}/reviews")
def save_review(p: str, r: Review) -> dict:
    con = jobs.connect()
    con.execute("""INSERT INTO reviews (run_id, reviewer, view, subject, action, note, alpha, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (p, r.reviewer or "anonymous", r.view, json.dumps(r.subject), r.action, r.note,
                 json.dumps(r.alpha) if r.alpha is not None else None, jobs.now()))
    return {"saved": True}


@router.get("/p/{p}/reviews")
def list_reviews(p: str, view: str, reviewer: str | None = None) -> list[dict]:
    return reviews_for(p, view, reviewer)


def sync_reviews() -> int:
    """Copy unsynced judgments into DuckDB review_actions (called by the pipeline worker, which may write)."""
    from datetime import datetime

    from .. import store

    con = jobs.connect()
    rows = con.execute("SELECT * FROM reviews WHERE synced = 0").fetchall()
    if not rows:
        return 0
    db = store.connect()
    try:
        for r in rows:
            db.execute("INSERT INTO review_actions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["run_id"], r["reviewer"], r["view"], r["subject"], r["action"], r["note"], r["alpha"],
                        datetime.fromisoformat(r["ts"])])
    finally:
        db.close()
    con.execute(f"UPDATE reviews SET synced = 1 WHERE id IN ({','.join('?' * len(rows))})", [r["id"] for r in rows])
    return len(rows)
