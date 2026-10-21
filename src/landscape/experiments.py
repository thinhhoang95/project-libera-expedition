"""Saved results of the experiments (scripts/phase8/, phase8b/, phase10/, and the research lab's semantic ablation),
one folder per run.

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
    "phase8_route_robustness": "findings: several cluster features flip sign with the outcome's expansion route",
    "phase8_direction": "adopted: E' = momentum + acceleration - new-author share replaces the recipe's E",
    "phase8_flow_penalty": "adopted: flow-penalty power 0.5 for limitation -> innovation (SignalParams)",
    "phase8_people_novelty": "descriptive only: brokers lose to popularity; novelty anti-predicts 3-year citations",
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
    "exp12_entry": "describe: How people got in (entry_routes.parquet); anchored entries with an insider do better",
    "exp13_reading": "adopted: Read this first (reading_paths.parquet), classics interleaved with method sources",
    "exp14_durable": "describe: Lasting marker (lasting.parquet); breadth lasts, novelty fades",
    "exp15_lifecycle": "survey gap adopted (survey_gaps.parquet); lifecycle mix describe only; theory wanted dropped",
    "exp16_open_questions": "not demonstrated: recurring-question list kept for inspection only",
    "exp17_quiet": "adopted: Quiet foundations (quiet_foundations.parquet; third role in Read this first)",
    "exp18_twins": "not supported: shared problem structure adds nothing beyond topic",
    "exp19_applications": "use with care: Untried applications (untried_applications.parquet); 24 filled cells",
    "exp20_tools": "adopted: Tools to learn (tools_to_learn.parquet), settled vs in flux",
    "exp21_gateway": "not product-grade: like for like the effect is small (AUC 0.535)",
    "exp22_23_entry_context": "exp22 not supported; exp23 describe: crowding gauge on My paper (crowding.parquet)",
    "exp24_saturation": "open: too few abstracts state a gain over a named baseline; needs full texts",
    "exp25_insider_reads": "not adopted: supports Read this first's classics",
    "exp26_tempo": "not adopted: tempo is not a stable trait (hint only)",
    "exp27_testbeds": "not demonstrated: abstracts rarely name shared testbeds",
    "exp28_disagreements": "not demonstrated: 2 conflicts in 3,360 same-question pairs; needs full texts",
    "lab_semantic_ablation": "findings: semantics adds most for uncited papers (paper bridges, chronological test)",
}
ORDER = list(PILOT_DECISIONS)

# Every experiment script in run order: (pipeline step, script from the repo root, fixed arguments, experiments whose
# tables it feeds). Each takes --run-id; scripts/run_experiments.py runs one step's scripts in this order. The
# Phase 8 analyses need the backtest's cutoff runs; Phase 8b also needs `external` (exp6); Phase 10 also needs the
# 2019 cutoff (prepare-cutoffs) and makes the only paid calls (exp24, exp28: about $0.55 once, cached).
RUNS: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
    ("experiments-8", "scripts/phase8/route_robustness.py", (), ("phase8_route_robustness",)),
    ("experiments-8", "scripts/phase8/loco_feature_selection.py", (), ("phase8_direction",)),
    ("experiments-8", "scripts/phase8/loco_smoothed_hits.py", (), ("phase8_direction",)),
    ("experiments-8", "scripts/phase8/lim_flow_penalty.py", (), ("phase8_flow_penalty",)),
    ("experiments-8", "scripts/phase8/people_and_novelty.py", (), ("phase8_people_novelty",)),
    ("experiments-8b", "scripts/phase8b/exp1_paper_bridges.py", (), ("exp1_paper_bridges",)),
    ("experiments-8b", "scripts/phase8b/exp1_evaluate.py", (), ("exp1_paper_bridges",)),
    ("experiments-8b", "scripts/phase8b/exp2_early_work.py", (), ("exp2_early_work",)),
    ("experiments-8b", "scripts/phase8b/exp3_micro_topics.py", (), ("exp3_micro_topics",)),
    ("experiments-8b", "scripts/phase8b/exp4_method_transfer.py", (), ("exp4_method_transfer",)),
    ("experiments-8b", "scripts/phase8b/fetch_counts_by_year.py", (), ("exp5_beyond_rich_get_richer", "exp11_leaders")),
    ("experiments-8b", "scripts/phase8b/exp6_imports.py", (), ("exp6_imports", "exp5_beyond_rich_get_richer")),
    ("experiments-8b", "scripts/phase8b/exp5_features.py", (), ("exp5_beyond_rich_get_richer",)),
    ("experiments-8b", "scripts/phase8b/exp5_evaluate.py", (), ("exp5_beyond_rich_get_richer",)),
    ("experiments-8b", "scripts/phase8b/exp7_coupling.py", (), ("exp7_coupling",)),
    ("experiments-8b", "scripts/phase8b/exp8_birth.py", (), ("exp8_birth",)),
    ("experiments-8b", "scripts/phase8b/exp9_realized.py", (), ("exp9_realized",)),
    ("experiments-8b", "scripts/phase8b/exp9_within_candidate.py", (), ("exp9_realized",)),
    ("experiments-8b", "scripts/phase8b/exp10_warmth.py", (), ("exp10_warmth",)),
    ("experiments-8b", "scripts/phase8b/exp11_leaders.py", (), ("exp11_leaders",)),
    ("experiments-10", "scripts/phase10/exp12_entry.py", (), ("exp12_entry",)),
    ("experiments-10", "scripts/phase10/exp13_reading.py", (), ("exp13_reading",)),
    ("experiments-10", "scripts/phase10/exp14_durable.py", (), ("exp14_durable",)),
    ("experiments-10", "scripts/phase10/exp15_lifecycle.py", (), ("exp15_lifecycle",)),
    ("experiments-10", "scripts/phase10/exp16_open_questions.py", (), ("exp16_open_questions",)),
    ("experiments-10", "scripts/phase10/exp17_quiet.py", (), ("exp17_quiet",)),
    ("experiments-10", "scripts/phase10/exp18_twins.py", (), ("exp18_twins",)),
    ("experiments-10", "scripts/phase10/exp19_applications.py", (), ("exp19_applications",)),
    ("experiments-10", "scripts/phase10/exp20_tools.py", (), ("exp20_tools",)),
    ("experiments-10", "scripts/phase10/exp21_gateway.py", (), ("exp21_gateway",)),
    ("experiments-10", "scripts/phase10/exp22_23_entry_context.py", (), ("exp22_23_entry_context",)),
    ("experiments-10", "scripts/phase10/exp24_saturation.py", (), ("exp24_saturation",)),
    ("experiments-10", "scripts/phase10/exp25_insider_reads.py", (), ("exp25_insider_reads",)),
    ("experiments-10", "scripts/phase10/exp26_tempo.py", (), ("exp26_tempo",)),
    ("experiments-10", "scripts/phase10/exp27_testbeds.py", (), ("exp27_testbeds",)),
    ("experiments-10", "scripts/phase10/exp28_disagreements.py", (), ("exp28_disagreements",)),
]
STEP_SCRIPTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
SCRIPTS: dict[str, list[str]] = {"lab_semantic_ablation": ["scripts/phase8c/semantic_ablation.py"]}
STEP_OF: dict[str, str] = {"lab_semantic_ablation": "research-lab"}
for _step, _path, _args, _exps in RUNS:
    STEP_SCRIPTS.setdefault(_step, []).append((_path, _args))
    for _e in _exps:
        SCRIPTS.setdefault(_e, []).append(" ".join((_path, *_args)))
        STEP_OF.setdefault(_e, _step)


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
