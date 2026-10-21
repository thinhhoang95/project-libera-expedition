"""Read replica of the database for the web app.

DuckDB lets one process write a file, and no other process may open it, even read-only, while it does.
Pipeline steps write for hours, so the web app never opens the main file: it reads a copy that is
published after every pipeline step (and on start-up when the main file is newer and free). Copies are
versioned (replica/landscape-<stamp>.duckdb, named by replica/current), so a reader keeps its file
until it switches, on any file system.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import duckdb

from .config import APP_DIR, DB_PATH

log = logging.getLogger(__name__)
REPLICA_DIR = APP_DIR / "replica"
POINTER = REPLICA_DIR / "current"


def current() -> Path | None:
    try:
        name = POINTER.read_text().strip()
    except FileNotFoundError:
        return None
    p = REPLICA_DIR / name
    return p if name and p.exists() else None


def stale() -> bool:
    cur = current()
    if not DB_PATH.exists():
        return False
    if cur is None:
        return True
    src_m = max(DB_PATH.stat().st_mtime, (DB_PATH.with_name(DB_PATH.name + ".wal").stat().st_mtime
                                          if DB_PATH.with_name(DB_PATH.name + ".wal").exists() else 0))
    return src_m > cur.stat().st_mtime + 1


def publish() -> Path | None:
    """Copy the main database to a new replica. Returns None when a writer holds the file (try again later)."""
    if not DB_PATH.exists():
        return None
    try:
        guard = duckdb.connect(str(DB_PATH), read_only=True)  # shared lock: no writer while we copy
    except duckdb.Error as e:
        log.info("replica not published (database busy): %s", e)
        return None
    REPLICA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    dst = REPLICA_DIR / f"landscape-{stamp}.duckdb"
    try:
        tmp = dst.with_suffix(".tmp")
        shutil.copyfile(DB_PATH, tmp)
        wal = DB_PATH.with_name(DB_PATH.name + ".wal")
        if wal.exists():
            shutil.copyfile(wal, dst.with_name(dst.name + ".wal"))
        tmp.replace(dst)
    finally:
        guard.close()
    ptr_tmp = POINTER.with_suffix(".tmp")
    ptr_tmp.write_text(dst.name)
    ptr_tmp.replace(POINTER)
    _cleanup(keep=dst)
    log.info("published replica %s", dst.name)
    return dst


def _cleanup(keep: Path, grace_s: float = 900) -> None:
    """Remove replicas older than the grace period (readers switch within seconds)."""
    now = time.time()
    for p in REPLICA_DIR.glob("landscape-*.duckdb*"):
        if p.name.startswith(keep.name) or p.name.startswith("._"):
            continue
        try:
            if now - p.stat().st_mtime > grace_s:
                p.unlink()
        except OSError:
            pass


def ensure_fresh() -> Path | None:
    """Publish when the main file is newer than the replica and nobody is writing it."""
    if stale():
        publish()
    return current()
