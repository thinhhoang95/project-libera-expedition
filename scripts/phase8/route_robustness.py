import numpy as np, pandas as pd
from scipy.stats import spearmanr
from landscape import store
from landscape.backtest import cluster_feature_table
from landscape.cli import _params
con = store.connect(); p = _params(con, "dro-pilot")
feats = ["E", "novelty", "new_author_share", "g", "dg", "momentum", "ref_age", "ext_inflow", "coherence", "share_T", "mig_net", "cit_growth"]
res = {}
for name, routes in [("all routes", None), ("forward only", {"forward"}), ("non-forward", {"seed", "back", "author", "semantic"}), ("author+semantic", {"author", "semantic"})]:
    d = pd.concat([cluster_feature_table(con, "dro-pilot", T, p, routes) for T in (2020, 2021, 2022, 2023)])
    d = d[d.n_total >= 10]
    col = {}
    for f in feats:
        cs = [spearmanr(g[f], g.y_logratio).statistic for _, g in d.groupby(["cutoff", "scope"])
              if g[[f, "y_logratio"]].dropna().shape[0] >= 6 and g[f].nunique() > 1]
        col[f] = round(float(np.nanmean(cs)), 3)
    res[name] = col
print(pd.DataFrame(res).to_string())
