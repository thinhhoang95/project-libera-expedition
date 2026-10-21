"""Fit the production models for the paper-level signals on all backtest cutoffs and save them.

Inputs: data/backtest/exp1_pairs.parquet and exp2_papers.parquet (run exp1 / exp2 first).
Output: src/landscape/models/{paper_bridges,early_work}.json (committed; read by signals.py).
The honest accuracy estimates are the leave-one-cutoff-out numbers from the experiment scripts.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = Path("src/landscape/models")
OUT.mkdir(parents=True, exist_ok=True)


def fit(df, feats, target, name, note):
    s = StandardScaler().fit(df[feats])
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(df[feats]), df[target])
    spec = {"name": name, "features": feats, "mean": s.mean_.round(6).tolist(), "scale": s.scale_.round(6).tolist(),
            "coef": m.coef_[0].round(6).tolist(), "intercept": float(m.intercept_[0]),
            "trained_on": sorted(int(t) for t in df["T"].unique()), "n_rows": int(len(df)),
            "n_positive": int(df[target].sum()), "note": note}
    (OUT / f"{name}.json").write_text(json.dumps(spec, indent=2))
    print(name, dict(zip(feats, spec["coef"])))


d1 = pd.read_parquet("data/backtest/exp1_pairs.parquet")
d1["lpop"] = np.log1p(d1["pop"])
d1["age"] = d1["T"] - d1["i_year"]
d1["lcoup"] = np.log1p(d1["coupling"] * 100)
d1["lim_sim3_f"] = d1.lim_sim3.fillna(d1.prob_sim)
fit(d1, ["lpop", "lcoup", "prob_sim", "lim_sim3_f", "age"], "hit", "paper_bridges",
    "P(problem cluster X cites innovation paper i within 3 years | not cited yet). "
    "Leave-one-cutoff-out: p@10 per cluster 0.234 (popularity 0.186, random 0.017), AUC 0.84, long-tail AUC 0.84.")

d2 = pd.read_parquet("data/backtest/exp2_papers.parquet")
feats2 = ["l_cites_so_far", "l_coupling", "l_n_refs", "cluster_E", "centrality", "novelty", "l_n_authors",
          "survey", "theory", "quant"]
fit(d2, feats2, "hit", "early_work",
    "P(a paper from year T is top-quartile for its year in corpus citations over T+1..T+3), without author fame. "
    "Leave-one-cutoff-out: AUC 0.815, p@50 0.67 (citations so far 0.62, coupling 0.50, base rate 0.19).")
