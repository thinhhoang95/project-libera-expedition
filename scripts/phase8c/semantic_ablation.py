"""Separate semantic information from received citations and bibliographic coupling.

Reads frozen research-lab candidates only. Train 2019, test 2022, outcomes 2023-2025.
No DB access, downloads or paid calls. All populations use models fitted on the same full
training population. Uncited means no received corpus citations at the test cutoff.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from landscape.research_lab import fitted_scores, lab_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="dro-pilot")
    args = ap.parse_args()
    path = lab_dir(args.run_id)
    data = pd.read_parquet(path / "bridge_candidates.parquet")
    train, test = data[data["T"] == 2019], data[data["T"] == 2022].copy()
    models = {"citations + age": ["lpop", "age"],
              "bibliometric base": ["lpop", "lcoup", "age"],
              "semantic only": ["prob_sim", "lim_sim3_f"],
              "base + semantics": ["lpop", "lcoup", "age", "prob_sim", "lim_sim3_f"]}
    rows = []
    for name, features in models.items():
        test["score"] = fitted_scores(train, test, features)
        for label, mask in (("all", np.ones(len(test), bool)),
                            ("cited <=1", test["pop"] <= 1), ("uncited", test["pop"] == 0)):
            group = test[mask]
            precision = np.mean([g.nlargest(10, "score").hit.mean() for _, g in group.groupby("X")])
            rows.append({"train_cutoff": 2019, "test_cutoff": 2022, "outcome_years": "2023-2025",
                         "population": label, "model": name, "n": len(group), "hits": int(group.hit.sum()),
                         "auc": roc_auc_score(group.hit, group.score), "p@10": precision})
    results = pd.DataFrame(rows)
    results.to_csv(path / "semantic_ablation.csv", index=False)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
