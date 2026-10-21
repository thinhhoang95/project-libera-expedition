"""Experiment 9 control: rank communities *for each candidate idea*, so an idea that is generic (close to
everything in embedding space) cannot score well just by being generic. For every candidate i with at least
one positive and one negative community at a cutoff, the AUC of each raw signal at picking which communities
realize (or cite) i; averaged over candidates. Reads exp9_pairs.parquet. Usage: --run-id (default dro-pilot).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from landscape import experiments

RUN = experiments.run_arg()
d = pd.read_parquet(experiments.work_dir(RUN) / "exp9_pairs.parquet")
SIGNALS = {"problem similarity": "prob_sim", "limitation similarity": "lim_sim3_f", "coupling": "lcoup",
           "community size": "X_size", "recent closeness (sim_recent)": "sim_recent"}
rows = []
for y in ("hit", "realized", "realized_uncited"):
    for name, f in SIGNALS.items():
        aucs = [roc_auc_score(g[y], g[f]) for _, g in d.groupby(["T", "i"]) if 0 < g[y].sum() < len(g)]
        rows.append({"outcome": y, "signal": name, "mean AUC across communities": np.mean(aucs),
                     "share of candidates above 0.5": np.mean(np.array(aucs) > 0.5), "candidates": len(aucs)})
R = pd.DataFrame(rows)
print(R.round(3).to_string(index=False))
experiments.save(RUN, "exp9_realized", "within_candidate", R, title="Realized opportunities: which community, for a given idea",
                 how_to_read="Control for generic ideas: for each candidate idea, communities are ranked by each raw "
                 "signal, and the AUC says how well it picks the communities that later realize (or cite) that idea. "
                 "An idea that is close to everything cannot help here. 0.5 = chance. *recent closeness* is how close "
                 "the community's recent papers already were to the idea (a persistence baseline).")
