"""Saved results of the signal experiments (Phase 8b, scripts/phase8b/), one folder per run.

Every experiment is run on one corpus, so its results describe that corpus: another field may
give different answers. The scripts therefore save every result table, adopted or not, under
data/projects/<run>/backtest/experiments/, and the explorer's Experiments view shows them all.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import store
from .config import project_dir


# what was decided on the pilot; another run should be judged on its own tables
PILOT_DECISIONS = {
    "exp1_paper_bridges": "adopted: signals bridge_paper / bridge_paper_gem (Bridges tab, Papers to bring in)",
    "exp2_early_work": "adopted: signal early_work (People tab)",
    "exp3_micro_topics": "burstiness dropped; local momentum adopted lightly (map layer Rising pockets)",
    "exp4_method_transfer": "adopted: item CF for bridge_method_transfer (grid diamonds)",
    "exp5_beyond_rich_get_richer": "findings only; not used by any signal yet",
    "exp6_imports": "adopted as a review list (Bridges tab, From outside the field)",
    "exp7_coupling": "findings: coupling needs no received citations but defines the reachable set",
    "exp8_birth": "not demonstrated; live list frozen in the prospective snapshot (low confidence)",
    "exp9_realized": "findings: 95% of realized uptake is uncited; no special foresight from semantics",
    "exp10_warmth": "adopted: community warmth (My paper view; models/community_warmth.json)",
    "exp11_leaders": "not supported: no follow-up wave after big authors or labs step in",
}
ORDER = list(PILOT_DECISIONS)
SCRIPTS = {
    "exp1_paper_bridges": ["exp1_paper_bridges.py", "exp1_evaluate.py"],
    "exp2_early_work": ["exp2_early_work.py"],
    "exp3_micro_topics": ["exp3_micro_topics.py"],
    "exp4_method_transfer": ["exp4_method_transfer.py"],
    "exp5_beyond_rich_get_richer": ["fetch_counts_by_year.py", "exp6_imports.py", "exp5_features.py",
                                    "exp5_evaluate.py [logit] [hgb]"],
    "exp6_imports": ["exp6_imports.py (after `landscape external`)"],
    "exp7_coupling": ["exp7_coupling.py"],
    "exp8_birth": ["exp8_birth.py"],
    "exp9_realized": ["exp9_realized.py", "exp9_within_candidate.py"],
    "exp10_warmth": ["exp10_warmth.py"],
    "exp11_leaders": ["exp11_leaders.py"],
}


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
    return project_dir(run, "backtest")


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
    p = project_dir(run, "backtest", "experiments", mkdir=False) / "index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load(run: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(project_dir(run, "backtest", "experiments", mkdir=False) / f"{name}.parquet")
