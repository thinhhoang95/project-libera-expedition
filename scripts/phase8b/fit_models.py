"""Fit the production models on all backtest cutoffs of a run and put them in production for it.

Inputs: data/projects/<run>/backtest/exp1_pairs.parquet, exp2_papers.parquet and community_warmth_model.json (run the
Phase 8b experiments first: pipeline step experiments-8b; --run-id, default dro-pilot).
Output: data/projects/<run>/models/{paper_bridges,hidden_gems,early_work,community_warmth}.json (hidden_gems needs
experiment 30's exp30_paper_bridges.parquet: step experiments-11), which signals.py and
portfolio.py then use for the run and its cutoff runs instead of the committed pilot models (config.model_file).
Run `signals` and `portfolio` afterwards for the lists to use them (template "own-models" does all three).
With --as-default the models replace the committed defaults in src/landscape/models/ (the pilot's, for every
project without its own).
The honest accuracy estimates are the leave-one-cutoff-out numbers from the experiment scripts.
"""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from landscape import experiments
from landscape.config import DEFAULT_MODELS, project_dir

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", default="dro-pilot")
ap.add_argument("--as-default", action="store_true", help="replace the committed models every project falls back to")
a = ap.parse_args()
RUN = a.run_id
WORK = experiments.work_dir(RUN)
warmth = WORK / "community_warmth_model.json"
missing = [p.name for p in (WORK / "exp1_pairs.parquet", WORK / "exp2_papers.parquet", warmth) if not p.exists()]
if missing:
    raise SystemExit(f"missing {', '.join(missing)} in {WORK}: run the Phase 8b experiments first (step experiments-8b)")
OUT = DEFAULT_MODELS if a.as_default else project_dir(RUN, "models")


def fit(df, feats, target, name, note):
    s = StandardScaler().fit(df[feats])
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(df[feats]), df[target])
    spec = {"name": name, "features": feats, "mean": s.mean_.round(6).tolist(), "scale": s.scale_.round(6).tolist(),
            "coef": m.coef_[0].round(6).tolist(), "intercept": float(m.intercept_[0]),
            "trained_on": sorted(int(t) for t in df["T"].unique()), "n_rows": int(len(df)),
            "n_positive": int(df[target].sum()), "run": RUN, "note": note}
    (OUT / f"{name}.json").write_text(json.dumps(spec, indent=2))
    print(name, dict(zip(feats, spec["coef"])))


d1 = pd.read_parquet(WORK / "exp1_pairs.parquet")
d1["lpop"] = np.log1p(d1["pop"])
d1["age"] = d1["T"] - d1["i_year"]
d1["lcoup"] = np.log1p(d1["coupling"] * 100)
d1["lim_sim3_f"] = d1.lim_sim3.fillna(d1.prob_sim)
fit(d1, ["lpop", "lcoup", "prob_sim", "lim_sim3_f", "age"], "hit", "paper_bridges",
    "P(problem cluster X cites innovation paper i within 3 years | not cited yet). "
    "Leave-one-cutoff-out estimates for this run: Experiments view, experiment 1 (pilot: p@10 per cluster 0.234, "
    "popularity 0.186, AUC 0.84).")

status = WORK / "exp30_paper_bridges.parquet"
if status.exists():
    s = pd.read_parquet(status, columns=["T", "X", "i", "r", "status"])
    g = d1.merge(s, on=["T", "X", "i"])
    g = g[g["pop"] <= 1].assign(being_taken=lambda x: (x.status == "being taken").astype(float),
                                closeness=lambda x: x.r)
    fit(g, ["lpop", "lcoup", "prob_sim", "lim_sim3_f", "age", "being_taken", "closeness"], "hit", "hidden_gems",
        "P(X cites a paper cited at most once so far within 3 years | not cited yet), with whether X's recent "
        "statements already reach the paper's innovation. Estimates for this run: Experiments view, experiment 30 "
        "(pilot: p@500 0.138 vs 0.102 without the status, leave-one-cutoff-out; 0.072 vs 0.046 chronological).")
else:
    print(f"hidden_gems: no {status.name} (run experiments-11); gems keep the paper_bridges model")

d2 = pd.read_parquet(WORK / "exp2_papers.parquet")
feats2 = ["l_cites_so_far", "l_coupling", "l_n_refs", "cluster_E", "centrality", "novelty", "l_n_authors",
          "survey", "theory", "quant"]
fit(d2, feats2, "hit", "early_work",
    "P(a paper from year T is top-quartile for its year in corpus citations over T+1..T+3), without author fame. "
    "Leave-one-cutoff-out estimates for this run: Experiments view, experiment 2 (pilot: AUC 0.815, p@50 0.67).")

shutil.copyfile(warmth, OUT / "community_warmth.json")
print("community_warmth", json.loads(warmth.read_text())["coef"])
print(f"models for {RUN} -> {OUT}; run `signals` and `portfolio` for the lists to use them")
