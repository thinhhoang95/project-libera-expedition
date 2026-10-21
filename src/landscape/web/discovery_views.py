"""Discovery artefacts for Start here and My paper (src/landscape/discovery.py; evidence in
docs/features/10-orientation-experiments.md): quiet foundations, tools to learn, untried applications and the
crowding gauge. For a paper analysed by DOI, the untried-application fit and the crowding counts are computed
here the way discovery.py computes them for corpus papers (256-d unit vectors, the stored thresholds).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import embed, portfolio
from .data import corpus_vectors, embedding_version, jl, portfolio_file, q, records, titles

DIM = 256


def _evidence(df: pd.DataFrame | None) -> str | None:
    return str(df.evidence.iloc[0]) if df is not None and len(df) and "evidence" in df else None


def _json_list(x) -> list:
    v = jl(x) if isinstance(x, str) else x
    return list(v) if v is not None and not isinstance(v, dict) else []


def _unit256(v) -> np.ndarray | None:
    if v is None:
        return None
    return embed.truncate(np.asarray(v, dtype=np.float32).reshape(1, -1), DIM)[0]


# ------------------------------------------------------------------ quiet foundations

def builders_for(p: str) -> dict[tuple[str, str], list[dict]]:
    """(cluster_id, paper_id) -> papers that build on a quiet foundation without citing it."""
    Q = portfolio_file(p, "quiet_foundations.parquet")
    if Q is None:
        return {}
    Q = Q[Q.quiet_foundation]
    lists = {(c, x): _json_list(b) for c, x, b in zip(Q.cluster_id, Q.paper_id, Q.builders_not_citing)}
    tt = titles([y for v in lists.values() for y in v])
    return {k: [tt[y] for y in v if y in tt] for k, v in lists.items()}


def quiet_foundations(p: str, cluster: str, skip: set[str], n: int = 8) -> dict | None:
    Q = portfolio_file(p, "quiet_foundations.parquet")
    if Q is None:
        return None
    d = Q[(Q.cluster_id == cluster) & Q.quiet_foundation & ~Q.paper_id.isin(skip)].sort_values("quiet_rank", ascending=False)
    b = builders_for(p)
    rows = records(d.head(n)[["paper_id", "title", "year", "attributed_quiet", "cites_by_others", "outside_top50_cited"]])
    for r in rows:
        r["builders"] = b.get((cluster, r["paper_id"]), [])
    return {"rows": rows, "evidence": _evidence(Q)}


# ------------------------------------------------------------------ tools to learn

def tools(p: str, clusters: list[str] | None = None) -> dict | None:
    T = portfolio_file(p, "tools_to_learn.parquet")
    if T is None:
        return None
    if clusters is not None:
        T = T[T.cluster_id.isin(clusters)]
    order = {"settled": 0, "in between": 1, "in flux": 2}
    T = T.assign(_o=T.status.map(order).fillna(3)).sort_values(["cluster_id", "_o", "papers_here_recent"], ascending=[True, True, False])
    rows = records(T.drop(columns=["_o", "examples", "evidence", "as_of"], errors="ignore"))
    for r, ex in zip(rows, T.examples):
        r["examples"] = _json_list(ex)
    full = portfolio_file(p, "tools_to_learn.parquet")
    return {"rows": rows, "evidence": _evidence(full)}


# ------------------------------------------------------------------ untried applications

def untried(p: str, cluster: str, problem_vec) -> dict | None:
    U = portfolio_file(p, "untried_applications.parquet")
    if U is None:
        return None
    d = U[U.cluster_id == cluster].sort_values("rank")
    rows = records(d.drop(columns=["domain_examples", "evidence", "as_of"], errors="ignore"))
    for r, ex in zip(rows, d.domain_examples):
        r["domain_examples"] = _json_list(ex)
    C = portfolio_file(p, "domain_centroids.parquet")
    pv = _unit256(problem_vec)
    if C is not None and pv is not None and rows:
        cent = {int(i): np.asarray(c, dtype=np.float32) for i, c in zip(C.domain_id, C.centroid)}
        for r in rows:  # the paper's own fit with each domain its community has not used
            c = cent.get(int(r["domain_id"]))
            r["paper_fit"] = None if c is None else round(float(c @ pv), 3)
        rows.sort(key=lambda r: -(r.get("paper_fit") or -1))
    return {"rows": rows, "evidence": _evidence(U), "community": cluster}


# ------------------------------------------------------------------ crowding gauge

def _thresholds(p: str) -> dict | None:
    f = portfolio.out_dir(p) / "crowding_thresholds.json"
    return json.loads(f.read_text()) if f.exists() else None


def crowding(p: str, rec: dict, source: str) -> dict | None:
    th = _thresholds(p)
    if th is None:
        return None
    pid = rec["paper_id"]
    lo, hi = th["window"]
    C = portfolio_file(p, "crowding.parquet")
    row = C[C.paper_id == pid] if C is not None and source == "corpus" else None
    if row is not None and len(row):
        r = row.iloc[0]
        same, dups, lims = int(r.same_question_now), int(r.near_duplicates_now), int(r.limitations_answered)
        same_ex, lim_ex = _json_list(r.same_question_examples), _json_list(r.limitation_examples)
        computed = False
    else:  # a paper analysed by DOI: the same counts against the corpus (authors' own papers are not excluded)
        v = rec["vectors"]
        prob, meth, inn = _unit256(v.get("problem")), _unit256(v.get("method")), _unit256(v.get("innovation"))
        if prob is None:
            return {"available": False, "reason": "No problem statement to compare."}
        ver = embedding_version(p)
        yrs = q("SELECT p.paper_id, p.year FROM papers p JOIN provenance v USING (paper_id) WHERE v.run_id = ?", (p,))
        year = dict(zip(yrs.paper_id, yrs.year))
        PP, PV = corpus_vectors("problem", ver, DIM)
        now = [x for x in PP if x != pid and year.get(x) is not None and lo <= year[x] <= hi]
        s = PV[[PP[x] for x in now]] @ prob
        same_ids = [now[j] for j in np.argsort(-s) if s[j] >= th["same_question"]]
        dup = []
        if meth is not None:
            MP, MV = corpus_vectors("method", ver, DIM)
            dup = [x for x in same_ids if x in MP and float(MV[MP[x]] @ meth) >= th["same_method"]]
        lim_ids = []
        if inn is not None:
            LP, LV = corpus_vectors("limitations", ver, DIM)
            lid = [x for x in LP if x != pid and year.get(x, 9999) <= hi]
            sl = LV[[LP[x] for x in lid]] @ inn
            lim_ids = [lid[j] for j in np.argsort(-sl) if sl[j] >= th["answers_limitation"]][:5]
        same, dups, lims = len(same_ids), len(dup), len(lim_ids)
        same_ex, lim_ex, computed = same_ids[:3], lim_ids[:3], True
    tt = titles(same_ex + lim_ex)
    # where this paper sits among the corpus papers (for the gauge)
    pct = None
    if C is not None and len(C):
        pct = float((C.same_question_now < same).mean() + 0.5 * (C.same_question_now == same).mean())
    return {"available": True, "window": [lo, hi], "same_question": same, "near_duplicates": dups,
            "limitations_answered": lims, "percentile": pct, "computed_here": computed,
            "same_question_examples": [tt[x] for x in same_ex if x in tt],
            "limitation_examples": [tt[x] for x in lim_ex if x in tt], "evidence": th.get("evidence")}
