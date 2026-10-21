"""Superseded by the pipeline step `landscape external` (src/landscape/external.py), which writes to
data/projects/<run>/external/. Kept so the commands in older notes still work.

Usage: uv run python scripts/phase8b/external_mine.py [fetch] [extract] [embed] [--run-id R] [--min-cites 5]
"""
import argparse
import sys

from landscape.cli import main

ap = argparse.ArgumentParser()
ap.add_argument("stages", nargs="*", default=["fetch", "extract", "embed"])
ap.add_argument("--run-id", default="dro-pilot")
ap.add_argument("--min-cites", type=int, default=5)
ap.add_argument("--llm-cap", type=float, default=2.0)
a = ap.parse_args()
sys.exit(main(["external", "--run-id", a.run_id, "--stages", *a.stages, "--min-cites", str(a.min_cites),
               "--llm-cap", str(a.llm_cap)]))
