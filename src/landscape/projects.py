"""Projects: one seed (or a few) and everything built from it, under data/projects/<project>/.

A project is a base run (`runs.run_id` without `@T`); its cutoff runs <project>@T<year> live in the same
folder. Papers, facets and paper-level vectors stay shared (data/shared/) so a paper is fetched,
extracted and embedded once whichever project reaches it.

    projects/<project>/
        project.json        name, description, seeds, created_at (written by new-run and the web app)
        inputs/             files given at creation (deep-research report)
        vectors/            cluster and author-year centroids of the run and its cutoff runs
        external/  portfolio/  backtest/  research_lab/  snapshots/
        logs/               one log per pipeline step run by the job queue

`migrate_layout` moves a data folder from the flat layout (data/landscape.duckdb, data/raw,
data/portfolio/<run>, ...) into this one. Every move is a rename on the same volume and is recorded
in data/migration.json, so it is fast and can be undone by hand.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from .config import APP_DIR, DATA, PROJECTS, SHARED, project_dir

FACETS = ("problem", "method", "innovation", "result", "limitations")


def meta_path(project: str) -> Path:
    return project_dir(project, mkdir=False) / "project.json"


def read_meta(project: str) -> dict:
    p = meta_path(project)
    meta = json.loads(p.read_text()) if p.exists() else {}
    return {"id": project, "name": project, **meta}


def write_meta(project: str, **fields) -> dict:
    meta = read_meta(project)
    meta.update({k: v for k, v in fields.items() if v is not None})
    meta.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    project_dir(project)
    meta_path(project).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def list_projects() -> list[str]:
    if not PROJECTS.exists():
        return []
    return sorted(unquote(d.name) for d in PROJECTS.iterdir()
                  if d.is_dir() and not d.name.startswith(("._", ".")))


# ------------------------------------------------------------------ migration from the flat layout

def _plan(data: Path, runs: list[str]) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    shared = data / "shared"
    for name in ("landscape.duckdb", "landscape.duckdb.wal"):
        moves.append((data / name, shared / name))
    moves.append((data / "raw", shared / "raw"))

    def proj(run: str, *parts: str) -> Path:
        return project_dir(run, *parts, mkdir=False)

    vec = data / "vectors"
    if vec.is_dir():
        for f in sorted(vec.iterdir()):
            if f.name.startswith("._"):
                continue
            m = re.fullmatch(r"cluster_centroids_(.+)\.parquet", f.name) or \
                re.fullmatch(rf"author_year_(?:{'|'.join(FACETS)})_(.+)\.parquet", f.name)
            moves.append((f, proj(m.group(1), "vectors", f.name) if m else shared / "vectors" / f.name))
    for kind in ("portfolio", "external", "snapshots", "research_lab"):
        d = data / kind
        if d.is_dir():
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and not sub.name.startswith("._"):
                    moves.append((sub, proj(unquote(sub.name), kind)))
    runs_dir = data / "runs"
    if runs_dir.is_dir():
        for sub in sorted(runs_dir.iterdir()):
            if sub.is_dir() and not sub.name.startswith("._"):
                moves.append((sub, proj(sub.name, "inputs")))
    bt = data / "backtest"
    if bt.is_dir():
        # directories first: the files below land inside the moved backtest/<run> folder
        for f in sorted(bt.iterdir(), key=lambda f: (not f.is_dir(), f.name)):
            if f.name.startswith("._"):
                continue
            if f.is_dir():
                moves.append((f, proj(f.name, "backtest")))
            elif f.name.endswith("_summary.csv"):
                moves.append((f, proj(f.name[: -len("_summary.csv")], "backtest", "summary.csv")))
            elif f.suffix == ".parquet" and f.stem in runs:
                moves.append((f, proj(f.stem, "backtest", "results.parquet")))
            elif len(runs) == 1:  # pilot-only analysis files (scripts/phase8)
                moves.append((f, proj(runs[0], "backtest", f.name)))
    # step logs written next to the database: they belong to the only project, if there is one
    for f in sorted(data.glob("*.log")):
        if not f.name.startswith("._"):
            moves.append((f, proj(runs[0], "logs", f.name) if len(runs) == 1 else data / "legacy" / f.name))
    return [(s, t) for s, t in moves if s.exists()]


def needs_migration(data: Path = DATA) -> bool:
    return (data / "landscape.duckdb").exists() and not (SHARED / "landscape.duckdb").exists()


def migrate_layout(dry_run: bool = False, data: Path = DATA) -> list[tuple[str, str]]:
    """Move a flat data folder into shared/ + projects/<project>/. Close every process using the database first."""
    import duckdb

    runs: list[str] = []
    notes: dict[str, dict] = {}
    db = data / "landscape.duckdb"
    if db.exists():
        con = duckdb.connect(str(db), read_only=True)
        try:
            rows = con.execute("SELECT run_id, notes FROM runs WHERE run_id NOT LIKE '%@T%'").fetchall()
        finally:
            con.close()
        runs = [r for r, _ in rows]
        notes = {r: json.loads(n) if n else {} for r, n in rows}
    moves = _plan(data, runs)
    for s, t in moves:
        if t.exists():
            raise SystemExit(f"refusing to overwrite {t} (moving {s}); nothing was moved")
    done = []
    for s, t in moves:
        done.append((str(s.relative_to(data)), str(t.relative_to(data))))
        if not dry_run:
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(t))
    if not dry_run:
        for run in runs:
            if not meta_path(run).exists():
                write_meta(run, seed_dois=notes[run].get("seed_dois") or [], seed_ids=notes[run].get("seed_ids") or [])
        for kind in ("vectors", "portfolio", "external", "snapshots", "research_lab", "runs", "backtest"):
            d = data / kind
            if d.is_dir():
                for junk in d.glob("._*"):  # macOS AppleDouble files on exFAT
                    junk.unlink(missing_ok=True)
                if not any(d.iterdir()):
                    d.rmdir()
        APP_DIR.mkdir(parents=True, exist_ok=True)
        log = data / "migration.json"
        prev = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(prev + [{"at": datetime.now().isoformat(timespec="seconds"), "moves": done}],
                                  indent=1) + "\n")
    return done
