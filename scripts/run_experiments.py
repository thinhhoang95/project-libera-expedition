"""Run one pipeline step's experiment scripts in order (the list is landscape.experiments.RUNS).

    uv run python scripts/run_experiments.py --step experiments-8b --run-id my-field   [--list]

Steps: experiments-8 (Phase 8 backtest analyses), experiments-8b (signal experiments 1-11), experiments-10
(orientation experiments 12-28), experiments-11 (time experiments 29-30). The job queue runs them as optional pipeline steps (src/landscape/jobs.py).

- Each script runs in its own process from the repository root with --run-id, so it opens the database itself
  (read-only, except fetch_counts_by_year.py, which logs its OpenAlex calls).
- Resuming: the scripts done so far are recorded in data/projects/<run>/backtest/experiments/.progress-<step>.json.
  After an interruption or a failure the step starts again at the first unfinished script. The file is removed once
  every script has finished, so the next run redoes them all (their inputs may have changed).
- Spend: a script line "@@spend <key>=<usd>" (the paid LLM reads of experiments 24 and 28) is added to the run's
  spend in the database once the script has exited.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime

from landscape import experiments
from landscape.config import ROOT

SPEND = re.compile(r"^@@spend (\w+)=([0-9.]+)\s*$")


def record_spend(run: str, spend: dict[str, float]) -> None:
    from landscape import store
    from landscape.cli import _add_spend

    try:
        con = store.connect()
    except Exception as ex:  # e.g. the Streamlit explorer holds the database: report, do not fail the experiments
        print(f"could not record spend {spend} for {run}: {ex}", flush=True)
        return
    try:
        for key, usd in spend.items():
            _add_spend(con, run, key, usd)
    finally:
        con.close()


def run_script(path: str, args: tuple[str, ...], run: str) -> tuple[int, dict[str, float]]:
    """Run one script, passing its output through; returns the exit code and the spend it reported."""
    spend: dict[str, float] = {}
    proc = subprocess.Popen([sys.executable, str(ROOT / path), *args, "--run-id", run], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            m = SPEND.match(line)
            if m:
                spend[m[1]] = spend.get(m[1], 0.0) + float(m[2])
            print(line, end="", flush=True)
        return proc.wait(), spend
    except KeyboardInterrupt:  # the worker interrupts the whole process group: let the script unwind first
        proc.wait()
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=sorted(experiments.STEP_SCRIPTS))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--list", action="store_true", help="list the scripts and what is done, run nothing")
    a = ap.parse_args()
    scripts = experiments.STEP_SCRIPTS[a.step]
    state_path = experiments.results_dir(a.run_id) / f".progress-{a.step}.json"
    done = set(json.loads(state_path.read_text())["done"]) if state_path.exists() else set()
    n = len(scripts)
    if a.list:
        for k, (path, args) in enumerate(scripts, 1):
            print(f"{k:>2}. {'done ' if path in done else '     '}{' '.join((path, *args))}")
        return 0
    if done:
        print(f"resuming {a.step}: {len(done)} of {n} scripts already done", flush=True)
    for k, (path, args) in enumerate(scripts, 1):
        if path in done:
            continue
        print(f"\n=== experiments {k}/{n}: {path} · {datetime.now():%H:%M:%S}", flush=True)
        code, spend = run_script(path, args, a.run_id)
        if spend:
            record_spend(a.run_id, spend)
        if code != 0:
            print(f"{path} failed (exit {code}); the next run of {a.step} starts here", flush=True)
            return code
        done.add(path)
        state_path.write_text(json.dumps({"step": a.step, "done": sorted(done)}, indent=2) + "\n")
    state_path.unlink(missing_ok=True)
    print(f"\n{a.step}: {n} scripts done for {a.run_id}; results in the Experiments view", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
