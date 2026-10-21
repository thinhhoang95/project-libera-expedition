"""Shared helpers for the Phase 10 experiments (orientation: entry strategies, reading paths, durable ideas).

Self-contained on purpose: it reads the migrated data layout directly (data/shared/ for the database and the
vectors, data/projects/<run>/ for per-run files) and saves result tables where the Experiments view reads them
(data/projects/<run>/backtest/experiments/, same index.json format as landscape.experiments).

Environment: LANDSCAPE_DB may point at a read-only copy of the database (e.g. while an app holds the lock).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "data" / "shared"
RESEARCH = ("article", "preprint", "conference-paper", "proceedings-article")
THR_REUSE = 0.894   # 99th percentile of random innovation-method cosine, 256-d (experiment 9)


def run_arg(default: str = "dro-pilot") -> str:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-id", default=default)
    return ap.parse_known_args()[0].run_id


def project(run: str, *parts: str) -> Path:
    p = ROOT / "data" / "projects" / run
    for x in parts:
        p = p / x
    p.mkdir(parents=True, exist_ok=True)
    return p


def connect():
    return duckdb.connect(os.environ.get("LANDSCAPE_DB", str(SHARED / "landscape.duckdb")), read_only=True)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def vectors(facet: str, ids=None, dim: int | None = 256) -> tuple[dict, np.ndarray]:
    """{paper_id: row} and vectors of one facet (truncated to `dim` and renormalized, or full if None)."""
    t = pq.read_table(SHARED / "vectors" / f"{facet}.parquet", columns=["paper_id", "vector"])
    pid = t.column("paper_id").to_pylist()
    v = np.asarray(t.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                   dtype=np.float32).reshape(len(pid), -1)
    if ids is not None:
        keep = set(ids)
        sel = [k for k, x in enumerate(pid) if x in keep]
        pid, v = [pid[k] for k in sel], v[sel]
    v = unit(v[:, :dim]) if dim else v
    return {x: k for k, x in enumerate(pid)}, v


def blended(facet: str, gloss: str, alpha: float, ids) -> tuple[dict, np.ndarray]:
    """768-d (1-a) e_facet + a e_gloss, renormalized (papers without a gloss keep e_facet), as in embed.blended."""
    fpos, fv = vectors(facet, ids, dim=None)
    gpos, gv = vectors(gloss, list(fpos), dim=None)
    g = np.array([gv[gpos[k]] if k in gpos else fv[i] for k, i in fpos.items()], dtype=np.float32)
    return fpos, unit((1 - alpha) * fv + alpha * g)


def assign(con, run: str, cutoff_run: str, ids) -> pd.Series:
    """Every paper -> nearest stable/hub medium problem community (alpha 0.5) of `cutoff_run` (as the Assigner)."""
    cents = pd.read_parquet(project(run, "vectors") / f"cluster_centroids_{cutoff_run}.parquet")
    keep = set(con.execute("""SELECT cluster_id FROM clusters WHERE run_id = ? AND facet = 'problem' AND alpha = 0.5
                              AND resolution = 'medium' AND kind IN ('stable', 'hub')""", [cutoff_run]).df().cluster_id)
    c = cents[cents.cluster_id.isin(keep)]
    C = np.stack([np.asarray(x, dtype=np.float32) for x in c.centroid])
    pos, v = blended("problem", "problem_gloss", 0.5, ids)
    lab = np.array(c.cluster_id)[np.argmax(v @ C.T, axis=1)]
    return pd.Series(lab, index=list(pos))


def base_tables(con, run: str):
    papers = con.execute("""SELECT v.paper_id, v.route, p.year, p.type, p.title FROM provenance v
                            JOIN papers p USING (paper_id) WHERE v.run_id = ? AND p.year IS NOT NULL""", [run]).df()
    au = con.execute("""SELECT s.paper_id, s.author_id FROM authorships s JOIN provenance v
                        ON v.paper_id = s.paper_id AND v.run_id = ?""", [run]).df()
    refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v ON v.paper_id = r.citing_id
                          AND v.run_id = ?""", [run]).df()
    year = papers.set_index("paper_id").year
    au["year"] = au.paper_id.map(year)
    refs["cy"] = refs.citing_id.map(year)
    team = au.groupby("paper_id").author_id.apply(frozenset)
    return papers, au, refs, year, team


def save(run: str, experiment: str, table: str, df: pd.DataFrame, *, title: str, how_to_read: str = "") -> None:
    d = project(run, "backtest", "experiments")
    name = f"{experiment}__{table}"
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:  # lists -> readable strings for the viewer
        if df[c].map(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
            df[c] = df[c].map(lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple, np.ndarray)) else x)
    df.to_parquet(d / f"{name}.parquet", index=False)
    idx_path = d / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[name] = {"experiment": experiment, "table": table, "title": title, "how_to_read": how_to_read, "params": {},
                 "rows": int(len(df)), "saved_at": datetime.now().isoformat(timespec="seconds")}
    idx_path.write_text(json.dumps(idx, indent=2) + "\n")


def boot_diff(groups: list[np.ndarray], values: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, n: int = 500,
              seed: int = 0) -> tuple[float, float]:
    """90% interval of mean(values[a]) - mean(values[b]), resampling whole groups (e.g. authors or communities)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        a, b = values[i][mask_a[i]], values[i][mask_b[i]]
        if len(a) and len(b):
            out.append(a.mean() - b.mean())
    return tuple(np.percentile(out, [5, 95])) if out else (np.nan, np.nan)
