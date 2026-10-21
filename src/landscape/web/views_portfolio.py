"""Start here and My paper (the product portfolio, docs/features/09-portfolio.md) as JSON.

The computations follow the Streamlit views they replace (portfolio_view.py) line for line, so both
read the same numbers; the copy that explains each list lives in the frontend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .. import embed, portfolio
from . import discovery_views as dv
from ..research_lab import lab_dir
from .data import (Router, corpus_vectors, embedding_version, ext_vectors, jl, memo, portfolio_file, q, read_file,
                   records, titles)

router = Router()
WARMTH_SPEC = json.loads((portfolio.MODELS / "community_warmth.json").read_text())
REASONS = {"lcoup": "cites the same literature", "sim_recent": "its recent work is close to yours",
           "prob_sim": "asks related problems", "lim_sim3_f": "states limitations your work addresses",
           "l_size": "large community"}


def _lasting(p: str) -> dict:
    """paper_id -> lasting marker (experiment 14): True, False, or None when not assessed (before 2018)."""
    L = portfolio_file(p, "lasting.parquet")
    if L is None:
        return {}
    return {pid: (bool(la) if a else None) for pid, la, a in zip(L.paper_id, L.lasting, L.assessed)}


def _reading_paths(p: str) -> pd.DataFrame | None:
    return portfolio_file(p, "reading_paths.parquet")


def _entry_routes(p: str, clusters: list[str] | None = None) -> list[dict]:
    E = portfolio_file(p, "entry_routes.parquet")
    if E is None:
        return []
    if clusters is not None:
        E = E[E.cluster_id.isin(clusters)]
    out = records(E.drop(columns=["examples_taken_up"]))
    for r, ex in zip(out, E.examples_taken_up):
        r["examples_taken_up"] = jl(ex) if isinstance(ex, str) else list(ex or [])
    return out


def _survey_gaps(p: str, clusters: list[str] | None = None) -> list[dict] | None:
    """Per community: survey gap / thin / covered (experiment 15), gaps first."""
    G = portfolio_file(p, "survey_gaps.parquet")
    if G is None:
        return None
    if clusters is not None:
        G = G[G.cluster_id.isin(clusters)].set_index("cluster_id").reindex([c for c in clusters if c in set(G.cluster_id)]).reset_index()
    else:
        order = {"survey gap": 0, "thin": 1, "covered": 2, "not judged": 3}
        G = G.assign(_o=G.status.map(order)).sort_values(["_o", "ratio_to_field"]).drop(columns="_o")
    return records(G.drop(columns=["as_of"]))


def _entry_effects(p: str) -> dict | None:
    f = portfolio.out_dir(p) / "entry_effects.json"
    return json.loads(f.read_text()) if f.exists() else None


def _summary(p: str) -> dict | None:
    f = portfolio.out_dir(p) / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def _list(x) -> list:
    return [] if x is None or (isinstance(x, float) and np.isnan(x)) else list(x)


def _second_waves(S: pd.DataFrame | None) -> list[dict]:
    if S is None or S.empty:
        return []
    ev = titles([x for e in S.evidence for x in list(e)])
    out = []
    for r in S.to_dict("records"):
        out.append({"kind": r["kind"], "idea": r["idea"], "idea_id": r.get("idea_id"), "origin": r.get("origin"),
                    "year": None if pd.isna(r.get("year")) else int(r["year"]),
                    "n_new": int(r["n_new"]), "new_audiences": list(r["new_audiences"]),
                    "recent": int(r["recent"]), "earlier": int(r["earlier"]),
                    "n_quiet": int(r.get("n_quiet") or 0), "quiet_uptake": _list(r.get("quiet_uptake")),
                    "likely_next": list(r["likely_next"]), "backed_by": r.get("backed_by"),
                    "momentum": float(r.get("momentum") or 0),
                    "evidence": sorted((ev[x] for x in list(r["evidence"]) if x in ev),
                                       key=lambda t: -(t["year"] or 0))})
    return out


@router.get("/p/{p}/overview")
@memo()
def overview(p: str) -> dict:
    summ = _summary(p)
    if summ is None:
        return {"built": False}
    T = int(summ["as_of"])
    papers = q("SELECT p.year, v.route FROM papers p JOIN provenance v USING (paper_id) WHERE v.run_id = ?", (p,))
    cl = q("""SELECT cluster_id, label, kind, size FROM clusters WHERE run_id = ? AND facet = 'problem' AND alpha = 0.5
              AND resolution = 'medium' AND kind IN ('stable', 'hub')""", (p,))
    lab = q("""SELECT pc.cluster_id, p.year FROM paper_cluster pc JOIN papers p USING (paper_id)
               WHERE pc.run_id = ? AND pc.facet = 'problem' AND pc.alpha = 0.5 AND pc.resolution = 'medium'""", (p,))
    rec = lab[lab.year.between(T - 2, T)].groupby("cluster_id").size()
    prev = lab[lab.year.between(T - 5, T - 3)].groupby("cluster_id").size()
    n_rec, n_prev = max(lab.year.between(T - 2, T).sum(), 1), max(lab.year.between(T - 5, T - 3).sum(), 1)
    t = cl.assign(recent=cl.cluster_id.map(rec).fillna(0).astype(int),
                  share_now=cl.cluster_id.map(rec).fillna(0) / n_rec,
                  share_before=cl.cluster_id.map(prev).fillna(0) / n_prev)
    t["change"] = t.share_now - t.share_before
    t = t.sort_values("size", ascending=False)
    # yearly papers per community, for sparklines
    yrs = lab[lab.year.between(T - 9, T)].groupby(["cluster_id", "year"]).size().unstack(fill_value=0)
    yrs = yrs.reindex(columns=range(T - 9, T + 1), fill_value=0)
    t["series"] = [yrs.loc[c].tolist() if c in yrs.index else [0] * 10 for c in t.cluster_id]

    per_year = papers.groupby("year").size()
    out = {"built": True, "as_of": T, "built_at": summ["built_at"],
           "metrics": {"papers": len(papers), "year_lo": int(papers.year.quantile(0.05)),
                       "year_hi": int(papers.year.max()), "year_min": int(papers.year.min()),
                       "stable": int((cl.kind == "stable").sum()), "hubs": int((cl.kind == "hub").sum()),
                       "external_reference_share": summ.get("external_reference_share"),
                       "external_works": summ.get("external_works"),
                       "papers_by_year": [{"year": int(y), "n": int(n)} for y, n in per_year.items()
                                          if y >= T - 14]},
           "routes": {k: int(v) for k, v in papers.route.value_counts().items()},
           "communities": records(t[["cluster_id", "label", "kind", "size", "recent", "share_now", "change", "series"]])}

    im = portfolio_file(p, "import_map.parquet")
    if im is not None and len(im):
        top = im.head(10)
        out["import_map"] = {"early_years": str(top.early_years.iloc[0]), "recent_years": str(top.recent_years.iloc[0]),
                             "rows": records(top[["subfield", "share_early", "share_recent"]])}
    L = portfolio_file(p, "live_imports.parquet")
    if L is not None:
        o = L[L.outside_field & (L.year >= T - 10)].head(8)
        out["imported_now"] = records(o[["paper_id", "title", "subfield", "recent_citers", "cluster_label", "year"]])
        gap = L[~L.outside_field].sort_values("recent_citers", ascending=False).head(8)
        out["coverage_gaps"] = records(gap[["paper_id", "title", "year", "recent_citers", "cluster_label"]])
    ew = q("""SELECT subject, rank FROM signals WHERE run_id = ? AND signal_type = 'early_work' AND variant = 'all'
              ORDER BY rank LIMIT 8""", (p,))
    out["read_now"] = [{"paper_id": jl(s).get("paper_id"), "title": jl(s).get("title"), "rank": int(r)}
                       for s, r in zip(ew.subject, ew["rank"])]

    g = q("""SELECT subject, score FROM signals WHERE run_id = ? AND signal_type = 'bridge_paper_gem'
             ORDER BY score DESC LIMIT 10""", (p,))
    li = q("""SELECT subject, judge_score, judge_rationale FROM signals WHERE run_id = ? AND signal_type = 'bridge_lim_inn'
              AND judge_score IS NOT NULL ORDER BY judge_score DESC, rank LIMIT 6""", (p,))
    mt = q("""SELECT subject, judge_score FROM signals WHERE run_id = ? AND signal_type = 'bridge_method_transfer'
              ORDER BY rank LIMIT 6""", (p,))
    op = read_file(lab_dir(p) / "opportunities.parquet")
    out["opportunities"] = {
        "gems": [{"paper_id": jl(s).get("paper_id"), "title": jl(s).get("title"), "community": jl(s).get("label")}
                 for s in g.subject],
        "lim_inn": [{"from": jl(s).get("from_label"), "to": jl(s).get("to_label"), "judge": j, "why": w}
                    for s, j, w in zip(li.subject, li.judge_score, li.judge_rationale)],
        "method_transfer": [{"method": jl(s).get("method_label"), "problem": jl(s).get("problem_label"),
                             "judge": None if pd.isna(j) else float(j)} for s, j in zip(mt.subject, mt.judge_score)],
        "underconnected": None if op is None else records(
            op.sort_values("score", ascending=False).head(6)[["source_label", "target_label", "observed", "expected"]]),
    }
    out["second_waves"] = _second_waves(portfolio_file(p, "second_waves.parquet"))
    lasting = _lasting(p)
    for r in out["read_now"]:
        if r["paper_id"] in lasting:  # absent = no marker at all; None = not assessed
            r["lasting"] = lasting[r["paper_id"]]
    R = _reading_paths(p)
    out["reading_paths"] = None if R is None else records(R.drop(columns=["as_of"]))
    out["entry_routes"] = _entry_routes(p) if portfolio_file(p, "entry_routes.parquet") is not None else None
    out["entry_effects"] = _entry_effects(p)
    out["survey_gaps"] = _survey_gaps(p)
    # discovery (experiments 17-23): builders of quiet foundations in the reading paths, tools to learn
    if out["reading_paths"]:
        b = dv.builders_for(p)
        for row in out["reading_paths"]:
            if row["role"] == "quiet foundation":
                row["builders"] = b.get((row["cluster_id"], row["paper_id"]), [])
    out["tools"] = dv.tools(p)
    return out


# ================================================================== My paper

def _paper_record(p: str, pid: str) -> dict | None:
    """Corpus paper: warmth rows from the portfolio, vectors from the store (as portfolio_view._paper_record)."""
    W = portfolio_file(p, "warmth.parquet")
    if W is None:
        return None
    rows = W[W.paper_id == pid]
    if rows.empty:
        return None
    ver = embedding_version(p)
    vec = {}
    for f in ("problem", "method", "innovation", "limitations", "abstract"):
        pos, v = corpus_vectors(f, ver, 768)
        if pid in pos:
            vec[f] = v[pos[pid]]
    meta = q("SELECT title, year, doi, abstract_raw, venue_name, type, cited_by_count FROM papers WHERE paper_id = ?",
             (pid,)).iloc[0]
    facets = dict(q("SELECT facet, text FROM facets WHERE paper_id = ?", (pid,)).itertuples(index=False))
    refs = q("SELECT cited_id FROM refs WHERE citing_id = ?", (pid,)).cited_id.tolist()
    return {"paper_id": pid, "title": meta.title, "year": meta.year, "doi": meta.doi, "in_corpus": True,
            "abstract": meta.abstract_raw, "venue": meta.venue_name, "type": meta.type,
            "cited_by_count": None if pd.isna(meta.cited_by_count) else int(meta.cited_by_count),
            "facets": facets, "vectors": vec, "warmth": rows.to_dict("records"), "refs": refs}


def _analysed_path(p: str, pid: str) -> Path:
    return portfolio.out_dir(p) / "papers" / f"{pid}.json"


def _load_record(p: str, pid: str, source: str) -> dict:
    if source == "doi":
        path = _analysed_path(p, pid)
        if not path.exists():
            raise HTTPException(404, "This paper has not been analysed")
        rec = json.loads(path.read_text())
        rec["vectors"] = {k: np.array(v, dtype=np.float32) for k, v in rec["vectors"].items()}
        m = q("SELECT venue_name, type, cited_by_count, abstract_raw FROM papers WHERE paper_id = ?", (pid,))
        if len(m):
            rec.update(venue=m.venue_name.iloc[0], type=m.type.iloc[0], cited_by_count=int(m.cited_by_count.iloc[0] or 0))
            if not rec.get("abstract"):  # OpenAlex often has no abstract where S2 or arXiv filled one
                rec["abstract"] = m.abstract_raw.iloc[0]
        return rec
    rec = _paper_record(p, pid)
    if rec is None:
        raise HTTPException(422, "This paper has no extracted innovation statement, so warmth cannot be scored.")
    return rec


@router.get("/p/{p}/my-paper/analysed")
def analysed(p: str) -> list[dict]:
    d = portfolio.out_dir(p) / "papers"
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        if f.name.startswith("._"):
            continue
        r = json.loads(f.read_text())
        out.append({"paper_id": r["paper_id"], "title": r.get("title"), "year": r.get("year"), "doi": r.get("doi"),
                    "in_corpus": r.get("in_corpus"), "analysed_at": r.get("analysed_at")})
    return sorted(out, key=lambda r: r.get("analysed_at") or "", reverse=True)


def _why(row: dict) -> list[str]:
    z = {f: (row[f] - m) / s * c for f, m, s, c in zip(WARMTH_SPEC["features"], WARMTH_SPEC["mean"],
                                                         WARMTH_SPEC["scale"], WARMTH_SPEC["coef"])}
    return [REASONS[f] for f, v in sorted(z.items(), key=lambda kv: -kv[1]) if v > 0][:2]


@router.get("/p/{p}/my-paper/{pid}")
def my_paper(p: str, pid: str, source: str = "corpus") -> dict:
    return _my_paper(p, pid, source)


@memo(64)
def _my_paper(p: str, pid: str, source: str) -> dict:
    rec = _load_record(p, pid, source)
    W = pd.DataFrame(rec["warmth"])
    labels = q("""SELECT cluster_id, label, kind FROM clusters WHERE run_id = ? AND facet = 'problem' AND alpha = 0.5
                  AND resolution = 'medium'""", (p,)).set_index("cluster_id")
    W["community"] = W.cluster_id.map(labels.label)
    W["kind"] = W.cluster_id.map(labels.kind)
    home = W[W.home]
    cand = W[W["rank"].notna()].sort_values("rank").copy()
    lo, hi = cand.warmth.min(), cand.warmth.max()
    cand["warmth_rel"] = (cand.warmth - lo) / (hi - lo) if hi > lo else 1.0
    cand["why"] = [_why(r) for r in cand.to_dict("records")]
    ver = embedding_version(p)
    vecs = rec["vectors"]
    inn, prob = vecs.get("innovation"), vecs.get("problem")
    refs = set(rec.get("refs") or [])
    out = {"paper": {k: rec.get(k) for k in ("paper_id", "title", "year", "doi", "in_corpus", "abstract", "venue",
                                             "type", "cited_by_count", "analysed_at", "llm_cost_usd")},
           "source": source,
           "facets": {k: v for k, v in (rec.get("facets") or {}).items() if isinstance(v, str)},
           "home": None if home.empty else {"cluster_id": home.cluster_id.iloc[0], "label": home.community.iloc[0],
                                            "kind": home.kind.iloc[0]},
           "already_citing": W[W.already_citing & ~W.home].community.dropna().tolist(),
           "warm": [{"rank": int(r["rank"]), "cluster_id": r["cluster_id"], "community": r["community"],
                     "kind": r["kind"], "warmth": float(r["warmth_rel"]), "content_fit": round(float(r["content_fit"]), 3),
                     "why": r["why"]} for r in cand.to_dict("records")]}
    n_refs = rec.get("n_refs", len(refs))

    # citations within the map, by year (who already cites it)
    cit = q("""SELECT p.year, count(*) n FROM refs r JOIN papers p ON p.paper_id = r.citing_id
               JOIN provenance v ON v.paper_id = r.citing_id AND v.run_id = ? WHERE r.cited_id = ? GROUP BY 1 ORDER BY 1""",
            (p, rec["paper_id"]))
    out["stats"] = {"cited_by_count": rec.get("cited_by_count"), "map_citations": int(cit.n.sum()) if len(cit) else 0,
                    "map_citations_by_year": records(cit), "references": n_refs,
                    "refs_in_map": int(q(f"""SELECT count(*) n FROM provenance WHERE run_id = ? AND paper_id IN
                                            ({','.join('?' * len(refs))})""", (p, *refs)).n.iloc[0]) if refs else 0,
                    "warm_candidates": len(cand), "already_citing": len(out["already_citing"])}

    # whose stated problems it may answer
    lims = []
    if inn is not None:
        pos, Lv = corpus_vectors("limitations", ver, 768)
        ids = list(pos)
        s = Lv @ np.asarray(inn)
        order = np.argsort(-s)[:8]
        top = [(ids[k], float(s[k])) for k in order if ids[k] != rec["paper_id"]][:6]
        if top:
            tids = [t for t, _ in top]
            t = q(f"""SELECT p.paper_id, p.year, p.title, f.text FROM papers p JOIN facets f USING (paper_id)
                      WHERE f.facet = 'limitations' AND p.paper_id IN ({",".join("?" * len(tids))})""", tids)
            t = t.drop_duplicates("paper_id").set_index("paper_id")
            au = q(f"""SELECT s.paper_id, a.display_name FROM authorships s JOIN authors a USING (author_id)
                       WHERE s.paper_id IN ({",".join("?" * len(tids))}) AND s.position = 0""", tids)
            first = dict(zip(au.paper_id, au.display_name))
            lims = [{"paper_id": x, "limitation": t.text[x], "title": t.title[x],
                     "year": None if pd.isna(t.year[x]) else int(t.year[x]), "first_author": first.get(x),
                     "closeness": round(sim, 3)} for x, sim in top if x in t.index]
    out["limitations"] = lims

    # ideas from outside the field that fit
    L = portfolio_file(p, "live_imports.parquet")
    outside = []
    if L is not None and (prob is not None or inn is not None):
        q256 = embed.truncate(np.asarray(prob if prob is not None else inn)[None, :], portfolio.DIM)[0]
        best = {}
        for facet in ("innovation", "method", "abstract"):
            pos, v = ext_vectors(p, facet)
            for x in L.paper_id:
                if x in pos and x not in best:
                    best[x] = float(v[pos[x]] @ q256)
        o = L[L.outside_field].assign(fit=L.paper_id.map(best)).dropna(subset=["fit"])
        o = o[~o.paper_id.isin(refs | {rec["paper_id"]})].sort_values("fit", ascending=False).head(6)
        outside = records(o.assign(fit=o.fit.round(3))[["paper_id", "title", "subfield", "fit", "recent_citers", "year"]])
    out["outside"] = outside

    # papers to bring in (home community's list, closest to this paper first)
    hc = home.cluster_id.tolist()
    bring = []
    if hc:
        b = q("""SELECT signal_type, subject, score FROM signals WHERE run_id = ?
                 AND signal_type IN ('bridge_paper', 'bridge_paper_gem')""", (p,))
        b["s"] = b.subject.map(jl)
        b = b[b.s.map(lambda x: x["cluster_id"]) == hc[0]]
        b["paper_id"] = b.s.map(lambda x: x["paper_id"])
        b = b[~b.paper_id.isin(refs) & (b.paper_id != rec["paper_id"])]
        pos, v = corpus_vectors("innovation", ver, 768)
        b["fit"] = [float(v[pos[x]] @ np.asarray(prob)) if (x in pos and prob is not None) else np.nan for x in b.paper_id]
        b = b.drop_duplicates("paper_id").sort_values("fit" if prob is not None else "score", ascending=False).head(8)
        tt = titles(b.paper_id)
        bring = [{"paper_id": x, "title": s.get("title"), "year": tt.get(x, {}).get("year"),
                  "list": {"bridge_paper": "likely", "bridge_paper_gem": "hidden gem"}[t],
                  "fit": None if pd.isna(f) else round(float(f), 3)}
                 for x, s, t, f in zip(b.paper_id, b.s, b.signal_type, b.fit)]
    out["bring_in"] = bring

    methods = []
    if hc:
        mt = q("""SELECT subject, judge_score FROM signals WHERE run_id = ? AND signal_type = 'bridge_method_transfer'
                  ORDER BY rank""", (p,))
        methods = [{"method": jl(x).get("method_label"), "judge": None if pd.isna(j) else float(j)}
                   for x, j in zip(mt.subject, mt.judge_score) if jl(x).get("problem_cluster") == hc[0]][:6]
    out["methods"] = methods

    early = []
    ew = q("SELECT subject FROM signals WHERE run_id = ? AND signal_type = 'early_work' AND variant = 'all'", (p,))
    if len(ew) and vecs.get("abstract") is not None:
        s = [jl(x) for x in ew.subject]
        pos, v = corpus_vectors("abstract", ver, 768)
        a = np.asarray(vecs["abstract"])
        sc = [(x["paper_id"], x.get("title"), float(v[pos[x["paper_id"]]] @ a)) for x in s
              if x["paper_id"] in pos and x["paper_id"] != rec["paper_id"]]
        tt = titles([x for x, _, _ in sc])
        early = [{"paper_id": i, "title": t, "year": tt.get(i, {}).get("year"), "closeness": round(c, 3)}
                 for i, t, c in sorted(sc, key=lambda t: -t[2])[:6]]
    out["early_work"] = early

    # second waves near this paper
    SW = portfolio_file(p, "second_waves.parquet")
    near = set(home.community.dropna()) | set(cand.head(3).community.dropna())
    sw = []
    if SW is not None and len(SW) and near:
        hit = SW[[bool(near & (set(a) | set(b) | set(c))) for a, b, c in
                  zip(SW.new_audiences, SW.likely_next, SW.get("quiet_uptake", pd.Series([[]] * len(SW))))]]
        hit = hit[hit.idea_id != rec["paper_id"]]
        sw = _second_waves(hit.sort_values(["n_new", "momentum"], ascending=False).head(8))
    out["second_waves"] = sw

    # people active in the warmest communities (descriptive)
    top3 = cand.head(3).cluster_id.tolist()
    people = []
    if top3:
        y1 = int(q("SELECT max(p.year) y FROM papers p JOIN provenance v USING (paper_id) WHERE v.run_id = ?", (p,)).y.iloc[0])
        ppl = q(f"""SELECT a.display_name AS name, pc.cluster_id, count(DISTINCT s.paper_id) AS papers
                    FROM paper_cluster pc JOIN papers p USING (paper_id) JOIN authorships s ON s.paper_id = pc.paper_id
                    JOIN authors a ON a.author_id = s.author_id
                    WHERE pc.run_id = ? AND pc.facet = 'problem' AND pc.alpha = 0.5 AND pc.resolution = 'medium'
                      AND pc.cluster_id IN ({",".join("?" * len(top3))}) AND p.year >= ?
                    GROUP BY ALL ORDER BY papers DESC LIMIT 12""", (p, *top3, y1 - 3))
        ppl["community"] = ppl.cluster_id.map(labels.label)
        names = tuple(ppl.name)
        if names:
            hist = q(f"""SELECT a.display_name AS name, count(DISTINCT s.paper_id) AS earlier
                         FROM authors a JOIN authorships s USING (author_id) JOIN papers p ON p.paper_id = s.paper_id
                         JOIN provenance v ON v.paper_id = p.paper_id AND v.run_id = ?
                         WHERE a.display_name IN ({",".join("?" * len(names))}) AND p.year < ? GROUP BY 1""",
                     (p, *names, y1 - 3)).set_index("name").earlier
            ppl["earlier"] = ppl.name.map(hist).fillna(0).astype(int)
            ppl["track_record"] = np.where(ppl.earlier >= 2, "insider", "newer")
            people = records(ppl[["name", "community", "papers", "earlier", "track_record"]])
        out["people_since"] = y1 - 3
    out["people"] = people
    out["top3"] = top3

    # orientation (experiments 12-14): read this first in the home community, how people got in
    lasting = _lasting(p)
    for lst in (out["bring_in"], out["early_work"], out["limitations"]):
        for r in lst:
            if r["paper_id"] in lasting:
                r["lasting"] = lasting[r["paper_id"]]
    R = _reading_paths(p)
    if R is not None and hc:
        rp = R[(R.cluster_id == hc[0]) & ~R.paper_id.isin(refs) & (R.paper_id != rec["paper_id"])]
        out["reading_path"] = records(rp.drop(columns=["as_of"]))
    else:
        out["reading_path"] = None
    out["entry_routes"] = _entry_routes(p, hc + [c for c in top3 if c not in hc])
    out["entry_effects"] = _entry_effects(p)
    out["survey_gaps"] = _survey_gaps(p, hc + [c for c in top3 if c not in hc])
    # discovery (experiments 17-23)
    if out["reading_path"]:
        b = dv.builders_for(p)
        for row in out["reading_path"]:
            if row["role"] == "quiet foundation":
                row["builders"] = b.get((row["cluster_id"], row["paper_id"]), [])
    skip = refs | {rec["paper_id"]}
    out["quiet_foundations"] = dv.quiet_foundations(p, hc[0], skip) if hc else None
    out["tools"] = dv.tools(p, hc[:1]) if hc else None
    out["untried"] = dv.untried(p, hc[0], prob) if hc else None
    out["crowding"] = dv.crowding(p, rec, source)
    return out


@router.get("/p/{p}/my-paper/{pid}/community/{cid}")
def community_closest(p: str, pid: str, cid: str, source: str = "corpus") -> list[dict]:
    """A community's recent papers closest to this paper's innovation."""
    rec = _load_record(p, pid, source)
    inn = rec["vectors"].get("innovation")
    lab = q("""SELECT pc.paper_id, p.year, p.title FROM paper_cluster pc JOIN papers p USING (paper_id)
               WHERE pc.run_id = ? AND pc.cluster_id = ? ORDER BY p.year DESC""", (p, cid))
    pos, v = corpus_vectors("innovation", embedding_version(p), 768)
    lab = lab[lab.paper_id.isin(pos) & (lab.year >= lab.year.max() - 3)].drop_duplicates("paper_id").copy()
    if inn is None or lab.empty:
        return []
    lab["closeness"] = (v[[pos[x] for x in lab.paper_id]] @ np.asarray(inn)).astype(float).round(3)
    return records(lab.sort_values("closeness", ascending=False).head(6))
