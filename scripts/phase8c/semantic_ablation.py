"""Separate semantic information from received citations and bibliographic coupling.

Reads frozen research-lab candidates only (`landscape research-lab` runs the same ablation at the end). Purged
chronological split: train on the cutoffs whose outcomes were complete before the test cutoff (2019 -> 2022 on the
pilot). No DB access, downloads or paid calls. All populations use models fitted on the same full training
population. Uncited means no received corpus citations at the test cutoff. Saves research_lab/semantic_ablation.csv
and the Experiments view's table.
"""

import argparse
import json

import pandas as pd

from landscape.research_lab import lab_dir, save_semantic_ablation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="dro-pilot")
    args = ap.parse_args()
    path = lab_dir(args.run_id)
    data = pd.read_parquet(path / "bridge_candidates.parquet")
    complete_through = json.loads((path / "summary.json").read_text())["complete_through"]
    print(save_semantic_ablation(args.run_id, data, complete_through).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
