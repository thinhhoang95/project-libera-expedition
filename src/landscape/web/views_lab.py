"""Research lab (Phase 8c) and Experiments (Phase 8/8b) as JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from .. import experiments
from ..research_lab import lab_dir
from .data import Router, memo, q, read_file, records, titles

router = Router()
PORTFOLIO_DOC = Path(__file__).resolve().parents[3] / "docs" / "features" / "findings-portfolio.md"


def _lab(p: str, name: str) -> pd.DataFrame:
    df = read_file(lab_dir(p) / name)
    if df is None:
        raise HTTPException(404, f"{name} not built")
    return df


def _evidence(ids) -> list[dict]:
    ids = list(ids) if ids is not None else []
    t = titles(ids)
    return [t[x] for x in ids if x in t]


@router.get("/p/{p}/lab")
@memo()
def lab(p: str) -> dict:
    f = lab_dir(p) / "summary.json"
    if not f.exists():
        return {"built": False}
    audit = json.loads(f.read_text())
    year = audit["complete_through"]
    routes = _lab(p, "routes.parquet")
    recent = routes[routes.year.between(year - 7, year + 1)].copy()
    recent["share"] = recent.n / recent.groupby("year").n.transform("sum")
    trends = _lab(p, "trends.parquet").sort_values("composition_change", key=abs, ascending=False)
    opp = _lab(p, "opportunities.parquet")
    lb = _lab(p, "live_bridges.parquet")
    labels = q("SELECT cluster_id, label, kind FROM clusters WHERE run_id = ?", (p,)).set_index("cluster_id")
    bridge_clusters = [{"cluster_id": c, "label": labels.label.get(c, c), "kind": labels.kind.get(c)}
                       for c in sorted(lb.X.unique())]
    people = _lab(p, "people.parquet")
    topics = people.drop_duplicates("cluster_id")[["cluster_id", "label"]]
    metrics = _lab(p, "bridge_metrics.csv")
    return {"built": True, "audit": {k: v for k, v in audit.items() if k not in ("models", "alias_fix")},
            "routes": records(recent),
            "trends": records(trends.drop(columns=["evidence", "routes"], errors="ignore")),
            "opportunities": records(opp.drop(columns=["evidence"])),
            "bridge_clusters": bridge_clusters, "topics": records(topics),
            "metrics": records(metrics), "geometry": records(_lab(p, "geometry.parquet").round(3))}


@router.get("/p/{p}/lab/trend/{cid}")
def lab_trend(p: str, cid: str) -> dict:
    series = _lab(p, "timeseries.parquet")
    trends = _lab(p, "trends.parquet")
    row = trends[trends.cluster_id == cid]
    return {"series": records(series[series.cluster_id == cid]),
            "evidence": _evidence(row.evidence.iloc[0]) if len(row) else []}


@router.get("/p/{p}/lab/bridges")
def lab_bridges(p: str, cluster: str, mode: str = "broader") -> list[dict]:
    b = _lab(p, "live_bridges.parquet")
    g = b[b.X == cluster]
    chosen = g[g.diverse_pick] if mode == "broader" else (g[g.baseline_pick] if mode == "likely"
                                                          else g.nlargest(10, "contrastive_model"))
    chosen = chosen.sort_values("contrastive_model" if mode == "contrastive" else "baseline_model", ascending=False)
    labels = q("SELECT cluster_id, label FROM clusters WHERE run_id = ?", (p,)).set_index("cluster_id").label
    out = chosen[["i", "title", "pop", "age", "method", "n_lim", "baseline_model", "contrastive_model",
                  "best_limitation", "limitation_text", "innovation_text"]].copy()
    out["method_label"] = out.method.map(labels).fillna("Unassigned")
    return records(out.round(4))


@router.get("/p/{p}/lab/opportunity")
def lab_opportunity(p: str, source: str, target: str) -> list[dict]:
    o = _lab(p, "opportunities.parquet")
    row = o[(o.source == source) & (o.target == target)]
    return _evidence(row.evidence.iloc[0]) if len(row) else []


@router.get("/p/{p}/lab/people")
def lab_people(p: str, topic: str, role: str = "external_adoption") -> list[dict]:
    people = _lab(p, "people.parquet")
    g = people[people.cluster_id == topic].sort_values([role, "recent_fractional_works"], ascending=False).head(20)
    out = records(g.drop(columns=["evidence"]))
    for r, ev in zip(out, g.evidence):
        r["evidence"] = list(ev)
    return out


@router.post("/p/{p}/titles")
def paper_titles(p: str, ids: list[str]) -> list[dict]:
    return _evidence(ids[:100])


# ------------------------------------------------------------------ experiments

def _doc_sections(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out, head = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            head = line[3:].strip()
            out[head] = []
        elif head:
            out[head].append(line)
    return {h: "\n".join(v).strip() for h, v in out.items()}


@router.get("/p/{p}/experiments")
def experiments_index(p: str) -> dict:
    from ..experiments import ORDER, PILOT_DECISIONS, SCRIPTS

    sec = _doc_sections(PORTFOLIO_DOC)
    idx = experiments.load_index(p)
    by_exp: dict[str, list] = {}
    for name, meta in idx.items():
        by_exp.setdefault(meta["experiment"], []).append({"name": name, **meta})
    exps = [e for e in ORDER if e in by_exp] + sorted(e for e in by_exp if e not in ORDER)
    groups = [{"experiment": e, "title": by_exp[e][0]["title"].split(":")[0], "decision": PILOT_DECISIONS.get(e),
               "scripts": SCRIPTS.get(e, []), "last_saved": max(t["saved_at"] for t in by_exp[e]),
               "tables": sorted(by_exp[e], key=lambda t: t["name"])} for e in exps]
    bt = experiments.work_dir(p) / "summary.csv"
    return {"portfolio": {k: sec.get(k, "") for k in (
                "The portfolio", "Verdicts", "Protocols, for reading the numbers",
                "What the evidence says about how attention works (pilot)", "Implications for the product",
                "Borrowed ideas (exp6)")},
            "backtest": records(pd.read_csv(bt)) if bt.exists() else None, "groups": groups}


@router.get("/p/{p}/experiments/table/{name}")
def experiment_table(p: str, name: str) -> dict:
    idx = experiments.load_index(p)
    if name not in idx:
        raise HTTPException(404)
    return {"meta": idx[name], "rows": records(experiments.load(p, name))}
