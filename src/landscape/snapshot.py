"""Score a frozen prospective snapshot (scripts/phase8b/freeze_snapshot.py) against papers published after it.

    uv run landscape score-snapshot --run-id dro-pilot --date 2026-10-21 [--months 12]

1. Integrity: every frozen file is checked against the SHA-256 in docs/snapshots/<run>-<date>.json (in git).
2. New papers: corpus papers not in the frozen corpus_ids, published after the snapshot date and within the
   horizon. They enter the corpus through the run's usual routes (`landscape expand`, `extract`, `embed`), so
   collect them first; the explorer must be closed.
3. Each list is scored with its pre-registered hit definition, next to its frozen baseline. New papers are
   assigned to the frozen communities (nearest frozen centroid, problem / method at alpha 0.5, medium).

Writes data/snapshots/<run>/<date>/score_<today>_<months>m.json. Lists without an automatic scorer yet are
reported as `manual`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import embed, store
from .config import RunParams


def _assign(ids: list[str], cent: pd.DataFrame, facet: str, version: str, allowed: set[str]) -> pd.Series:
    """Nearest frozen centroid among the communities the lists use (stable and hub ones, as the Assigner)."""
    c = cent[cent.cluster_id.str.startswith(f"{facet}:0.5:medium:") & cent.cluster_id.isin(allowed)]
    if c.empty or not ids:
        return pd.Series(dtype=object)
    C = np.stack([np.asarray(v, dtype=np.float32) for v in c.centroid])
    pid, v = embed.blended(facet, 0.5, ids, dim=768, version=version)
    return pd.Series(np.array(c.cluster_id)[np.argmax(v @ C.T, axis=1)], index=pid)


def score_snapshot(con, run_id: str, p: RunParams, snap_date: str, months: int = 12) -> dict:
    d = store.DATA / "snapshots" / run_id / snap_date
    man_path = Path("docs/snapshots") / f"{run_id}-{snap_date}.json"
    if not man_path.exists():
        raise SystemExit(f"no manifest {man_path}")
    man = json.loads(man_path.read_text())
    bad = [f for f, h in man["sha256"].items()
           if not (d / f).exists() or hashlib.sha256((d / f).read_bytes()).hexdigest() != h]
    report = {"run": run_id, "snapshot_date": snap_date, "scored_at": datetime.now().isoformat(timespec="seconds"),
              "horizon_months": months, "integrity": "ok" if not bad else f"MISMATCH: {bad}", "lists": {}}
    L = {f.stem: pd.read_parquet(d / f) for f in d.glob("*.parquet") if not f.name.startswith("._")}
    t0 = date.fromisoformat(snap_date)
    t1 = date(t0.year + (t0.month - 1 + months) // 12, (t0.month - 1 + months) % 12 + 1, min(t0.day, 28))
    frozen = set(L["corpus_ids"].paper_id)
    new = con.execute("""SELECT p.paper_id, p.pub_date, p.year FROM papers p JOIN provenance v USING (paper_id)
                         WHERE v.run_id = ?""", [run_id]).df()
    new = new[~new.paper_id.isin(frozen)]
    new["d"] = pd.to_datetime(new.pub_date).dt.date
    new = new[(new.d > t0) & (new.d <= t1)]
    report["new_papers"] = len(new)
    report["window"] = f"{t0} .. {t1}"
    if new.empty:
        report["status"] = ("no papers published after the snapshot are in the corpus yet: collect them with the run's "
                            "routes (landscape expand / extract / embed), then score again")
        return _write(d, report, months)
    ids = new.paper_id.tolist()
    cent = L.get("cluster_centroids")
    if cent is None:
        cent = pd.read_parquet(d / "cluster_centroids.parquet")
    ver = p.embedding.version
    used_p, used_m = set(), set()
    for k, df in L.items():
        for col in ("cluster_id", "problem_cluster", "from_cluster", "to_cluster"):
            if col in df:
                used_p |= set(df[col].dropna())
        if "method_cluster" in df:
            used_m |= set(df.method_cluster.dropna())
    prob = _assign(ids, cent, "problem", ver, used_p)
    meth = _assign(ids, cent, "method", ver, used_m)
    refs = con.execute("SELECT citing_id, cited_id FROM refs").df()
    refs = refs[refs.citing_id.isin(set(ids))].assign(X=lambda r: r.citing_id.map(prob))
    cited_by = refs.groupby("cited_id").citing_id.nunique()
    cited_by_X = set(zip(refs.cited_id, refs.X))

    def per_community_precision(lst: pd.DataFrame) -> float:
        hits = [np.mean([(pid, X) in cited_by_X for pid in g.paper_id]) for X, g in lst.groupby("cluster_id")]
        return float(np.mean(hits)) if hits else float("nan")

    def put(name, value, baseline, note):
        report["lists"][name] = {"score": value, "baseline": baseline, "hit": note}

    if "bridge_paper" in L:
        put("bridge_paper", per_community_precision(L["bridge_paper"]),
            per_community_precision(L["bridge_paper_baseline_popularity"]), "p@10 per community: a new paper in it cites the work")
    if "bridge_paper_gem" in L:
        put("bridge_paper_gem", per_community_precision(L["bridge_paper_gem"]),
            per_community_precision(L["bridge_paper_gem_baseline_random"]), "p@5 per community (random baseline)")
    if "community_warmth" in L:
        def warm_rate(lst):
            got = lst[lst.paper_id.isin(cited_by.index)]
            if got.empty:
                return float("nan")
            return float(got.groupby("paper_id").apply(lambda g: any((g.name, X) in cited_by_X for X in g.cluster_id)).mean())
        put("community_warmth", warm_rate(L["community_warmth"]), warm_rate(L["community_warmth_baseline_size"]),
            "share of papers cited by any new paper whose 3 listed communities include a citing one")
    if "early_work" in L:
        coh = L["early_work_cohort"].assign(c=lambda x: x.paper_id.map(cited_by).fillna(0))
        cut = coh.groupby("year").c.transform(lambda s: s.quantile(0.75))
        top = set(coh[(coh.c > cut) & (coh.c > 0)].paper_id)
        put("early_work", float(L["early_work"].paper_id.isin(top).mean()),
            float(L["early_work_baseline_citations"].paper_id.isin(top).mean()),
            "share of the list in the top quarter of its cohort for citations from new papers")
    if "external_imports" in L:
        per_year = months / 12

        def imp_rate(lst, recent):
            n = lst.paper_id.map(cited_by).fillna(0)
            return float(((n >= 3) & (n / per_year >= 1.5 * recent / 3)).mean())
        e = L["external_imports"]
        b = L["external_imports_baseline_rate"]
        put("external_imports", imp_rate(e, e.recent_citers), imp_rate(b, b.recent_citers),
            "new citing papers per year >= 1.5x the frozen 3-year rate, and at least 3")
    if "method_transfer" in L:
        cells = pd.DataFrame({"p": prob, "m": meth}).dropna().value_counts()

        def cell_rate(lst):
            return float(np.mean([cells.get((r.problem_cluster, r.method_cluster), 0) >= 2 for r in lst.itertuples()]))
        put("method_transfer", cell_rate(L["method_transfer"]), cell_rate(L["method_transfer_baseline_popularity"]),
            "at least 2 new papers in the cell")
    for name in ("bridge_lim_inn", "rising_pockets", "emerging_cluster", "birth_regions"):
        if name in L:
            report["lists"][name] = {"score": "manual", "baseline": "manual",
                                     "hit": next((x["hit"] for x in man["lists"] if x["list"] == name), "")}
    report["status"] = "scored"
    return _write(d, report, months)


def _write(d: Path, report: dict, months: int) -> dict:
    (d / f"score_{date.today().isoformat()}_{months}m.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report
