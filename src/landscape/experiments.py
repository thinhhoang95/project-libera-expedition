"""Saved results of the signal experiments (Phase 8b, scripts/phase8b/), one folder per run.

Every experiment is run on one corpus, so its results describe that corpus: another field may
give different answers. The scripts therefore save every result table, adopted or not, under
data/backtest/<run>/experiments/, and the explorer's Experiments view shows them all.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import store

BACKTEST = store.DATA / "backtest"


def run_arg(default: str = "dro-pilot") -> str:
    """--run-id for the experiment scripts (other arguments are left to the script)."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-id", default=default)
    return ap.parse_known_args()[0].run_id


def connect_ro():
    """Read-only database connection for the experiment scripts. LANDSCAPE_DB may point at a copy
    of the database file, e.g. while the explorer holds the lock on the original."""
    import os

    import duckdb

    return duckdb.connect(os.environ.get("LANDSCAPE_DB", str(store.DB_PATH)), read_only=True)


def work_dir(run: str) -> Path:
    """Intermediate files of the experiments (per-pair and per-paper tables)."""
    d = BACKTEST / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(run: str) -> Path:
    d = work_dir(run) / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(run: str, experiment: str, table: str, df: pd.DataFrame, *, title: str, how_to_read: str = "",
         params: dict | None = None) -> None:
    """Write one result table and register it in the run's index (replacing an older version)."""
    d = results_dir(run)
    name = f"{experiment}__{table}"
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(d / f"{name}.parquet", index=False)
    idx_path = d / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[name] = {"experiment": experiment, "table": table, "title": title, "how_to_read": how_to_read,
                 "params": params or {}, "rows": int(len(df)), "saved_at": datetime.now().isoformat(timespec="seconds")}
    idx_path.write_text(json.dumps(idx, indent=2) + "\n")


def load_index(run: str) -> dict:
    p = BACKTEST / run / "experiments" / "index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load(run: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(BACKTEST / run / "experiments" / f"{name}.parquet")
