"""When an opportunity sits in time: is it fresh, part of a line of work that is active now, or a revival of
one that has gone quiet?

Everything here describes the map as sampled at the portfolio's as-of year T; none of it was backtested as a
predictor. Three ingredients:
- a **line**: one cluster's yearly papers and its share of the facet, recent (T-2..T) against before (T-5..T-3);
- a paper's **when**: its year, its latest citation from the field, and the line its idea belongs to (its
  innovation category, else its problem category);
- **flow change**: citations from one cluster to another, recent against before (is a gap already closing?).
"""

from __future__ import annotations

import pandas as pd

from .data import memo, q

SPAN = 10          # years of series sent for sparklines
RECENT = 3         # T-2..T against T-5..T-3


def label(ratio: float | None, recent: int) -> str:
    if ratio is None or recent < 3:
        return "quiet"
    if ratio >= 1.15:
        return "growing"
    if ratio >= 0.85:
        return "steady"
    if ratio >= 0.5:
        return "shrinking"
    return "quiet"


@memo(8)
def _cluster_years(p: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Papers and facet totals per (cluster, year), for every cluster of every facet, α and resolution."""
    cy = q("SELECT cluster_id, year, n, n_total FROM cluster_year WHERE run_id = ? AND year IS NOT NULL", (p,))
    n = cy.pivot_table(index="cluster_id", columns="year", values="n", aggfunc="sum", fill_value=0)
    tot = cy.pivot_table(index="cluster_id", columns="year", values="n_total", aggfunc="max", fill_value=0)
    return n, tot


@memo(8)
def _dense_from(p: str) -> int:
    """First year with at least 20 corpus papers (earlier years are sparse references, not a trend)."""
    y = q("SELECT min(year) FROM year_totals WHERE run_id = ? AND facet = 'corpus' AND n >= 20", (p,)).iloc[0, 0]
    return 0 if y is None or pd.isna(y) else int(y)


def line(p: str, cid: str | None, T: int) -> dict | None:
    if not cid:
        return None
    n, tot = _cluster_years(p)
    if cid not in n.index:
        return None
    get = lambda df, y: int(df.at[cid, y]) if y in df.columns else 0  # noqa: E731
    yrs = range(max(T - SPAN + 1, _dense_from(p)), T + 1)
    rec = sum(get(n, y) for y in range(T - RECENT + 1, T + 1))
    bef = sum(get(n, y) for y in range(T - 2 * RECENT + 1, T - RECENT + 1))
    rt = sum(get(tot, y) for y in range(T - RECENT + 1, T + 1))
    bt = sum(get(tot, y) for y in range(T - 2 * RECENT + 1, T - RECENT + 1))
    sr, sb = (rec / rt if rt else 0.0), (bef / bt if bt else None)
    ratio = sr / sb if sb else (None if rec == 0 else 9.99)
    return {"cluster_id": cid, "series": [get(n, y) for y in yrs], "from": yrs[0], "recent": rec, "before": bef,
            "share_recent": round(sr, 4), "share_before": None if sb is None else round(sb, 4),
            "ratio": None if ratio is None else round(ratio, 2), "trend": label(ratio, rec)}


@memo(8)
def _citations(p: str, T: int) -> pd.DataFrame:
    """Per cited corpus paper: latest citing year from the field, and citations in T-2..T."""
    d = q("""SELECT r.cited_id, max(c.year) AS last_cited, sum((c.year BETWEEN ? AND ?)::INT) AS cites_recent
             FROM refs r JOIN provenance v ON v.paper_id = r.citing_id AND v.run_id = ?
             JOIN papers c ON c.paper_id = r.citing_id
             JOIN provenance w ON w.paper_id = r.cited_id AND w.run_id = ?
             GROUP BY 1""", (T - RECENT + 1, T, p, p))
    return d.set_index("cited_id")


@memo(8)
def _idea_line(p: str) -> dict:
    """paper -> the category its idea belongs to: innovation (α 0, medium), else problem (α 0.5, medium)."""
    from .views_facets import _membership  # noqa: PLC0415 (views_facets imports views_map)
    M = _membership(p)
    inn = M.innovation.where(M.innovation.notna() & (M.innovation != "-"))
    prob = M.problem.where(M.problem.notna() & (M.problem != "-"))
    return inn.fillna(prob).dropna().to_dict()


def paper_when(p: str, pids: list[str], T: int) -> dict[str, dict]:
    """Timing for papers: fresh (T-2..T), else the trend of the line its idea belongs to."""
    ids = [x for x in dict.fromkeys(pids) if isinstance(x, str)]
    if not ids:
        return {}
    yrs = dict(q(f"SELECT paper_id, year FROM papers WHERE paper_id IN ({','.join('?' * len(ids))})", ids)
               .itertuples(index=False))
    C, L = _citations(p, T), _idea_line(p)
    out = {}
    for pid in ids:
        y = yrs.get(pid)
        y = None if y is None or pd.isna(y) else int(y)
        ln = line(p, L.get(pid), T)
        if y is not None and y >= T - RECENT + 1:
            timing = "fresh"
        elif ln is None:
            timing = "older"
        else:
            timing = {"growing": "active line", "steady": "active line", "shrinking": "fading line",
                      "quiet": "quiet line"}[ln["trend"]]
        last = C.at[pid, "last_cited"] if pid in C.index else None
        out[pid] = {"year": y, "age": None if y is None else T - y, "timing": timing,
                    "last_cited": None if last is None or pd.isna(last) else int(last),
                    "cites_recent": int(C.at[pid, "cites_recent"]) if pid in C.index else 0, "line": ln}
    return out


@memo(8)
def _flows(p: str) -> pd.DataFrame:
    return q("""SELECT facet, alpha, from_cluster, to_cluster, year, count FROM viz_flow_year
                WHERE run_id = ? AND kind = 'citation'""", (p,))


def flow_change(p: str, a: str, b: str, T: int) -> dict:
    """Citations from cluster a to cluster b, recent against before."""
    f = _flows(p)
    f = f[(f.from_cluster == a) & (f.to_cluster == b)]
    rec = int(f[f.year.between(T - RECENT + 1, T)]["count"].sum())
    bef = int(f[f.year.between(T - 2 * RECENT + 1, T - RECENT)]["count"].sum())
    return {"recent": rec, "before": bef,
            "state": "closing" if rec > max(bef, 0) * 1.5 and rec >= 3 else "open" if rec <= 1 else "steady"}


def year_map(ids) -> dict[str, int]:
    ids = [x for x in dict.fromkeys(ids) if isinstance(x, str)]
    if not ids:
        return {}
    d = q(f"SELECT paper_id, year FROM papers WHERE paper_id IN ({','.join('?' * len(ids))}) AND year IS NOT NULL", ids)
    return {a: int(b) for a, b in d.itertuples(index=False)}


def pair_when(pairs: list[dict], T: int, years: dict[str, int] | None = None) -> dict:
    """Limitation → innovation evidence pairs: when the limitations were stated and the innovations published."""
    years = years if years is not None else year_map([x for pr in pairs for x in (pr["limitation"], pr["innovation"])])
    ly = sorted(years[x["limitation"]] for x in pairs if x["limitation"] in years)
    iy = sorted(years[x["innovation"]] for x in pairs if x["innovation"] in years)
    lim_last, inn_last = (ly[-1] if ly else None), (iy[-1] if iy else None)
    if inn_last is not None and inn_last >= T - RECENT + 1:
        timing = "fresh answer"
    elif lim_last is not None and lim_last >= T - RECENT + 1:
        timing = "still stated"
    else:
        timing = "older pairing"
    return {"limitation_years": ly, "innovation_years": iy, "limitation_last": lim_last, "innovation_last": inn_last,
            "timing": timing}


def as_of(p: str) -> int:
    from .views_portfolio import _summary  # noqa: PLC0415
    s = _summary(p)
    if s:
        return int(s["as_of"])
    y = q("SELECT max(year) FROM year_totals WHERE run_id = ? AND facet = 'corpus'", (p,)).iloc[0, 0]
    return int(y) - 1
