import numpy as np, pandas as pd
from scipy.stats import spearmanr
from landscape import experiments
from landscape.backtest import cluster_feature_table
from landscape.cli import _params
RUN = experiments.run_arg()
con = experiments.connect_ro(); p = _params(con, RUN)
feats = ["E", "novelty", "new_author_share", "g", "dg", "momentum", "ref_age", "ext_inflow", "coherence", "share_T", "mig_net", "cit_growth"]
res = {}
for name, routes in [("all routes", None), ("forward only", {"forward"}), ("non-forward", {"seed", "back", "author", "semantic"}), ("author+semantic", {"author", "semantic"})]:
    d = pd.concat([cluster_feature_table(con, RUN, T, p, routes) for T in (2020, 2021, 2022, 2023)])
    d = d[d.n_total >= 10]
    col = {}
    for f in feats:
        cs = [spearmanr(g[f], g.y_logratio).statistic for _, g in d.groupby(["cutoff", "scope"])
              if g[[f, "y_logratio"]].dropna().shape[0] >= 6 and g[f].nunique() > 1]
        col[f] = round(float(np.nanmean(cs)), 3)
    res[name] = col
R = pd.DataFrame(res)
print(R.to_string())
experiments.save(RUN, "phase8_route_robustness", "feature_signs", R.rename_axis("feature").reset_index(),
                 title="Route robustness: do cluster features keep their sign across expansion routes",
                 how_to_read="Mean Spearman correlation between each cluster feature at cutoff T and the cluster's future "
                 "share change (log ratio), over cutoffs 2020-2023 and clusterings, with the outcome measured on papers "
                 "from different expansion routes. A feature whose sign flips between columns reflects how the corpus was "
                 "sampled more than where the field is going.")
