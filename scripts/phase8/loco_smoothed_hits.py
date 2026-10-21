import numpy as np, pandas as pd
from scipy.stats import spearmanr
from landscape import experiments
from landscape.config import project_dir
RUN = experiments.run_arg()
tabs = {k: pd.read_parquet(project_dir(RUN, "backtest") / f"cluster_features_{k}.parquet") for k in ("all", "forward", "nonforward")}
CUT = (2020, 2021, 2022, 2023)
FEATS = ["novelty", "new_author_share", "g", "dg", "momentum", "ref_age", "ext_inflow", "coherence", "share_T", "mig_net", "cit_growth", "N"]
for d in tabs.values():
    d["hs_emerging"] = d.share_fut3 >= 1.5 * d.share_past3
    d["hs_growing"] = d.share_fut3 > d.share_past3
    d["hs_shrinking"] = (d.share_fut3 < d.share_past3) & (d.s2 < d.s1)
def corr(d, f):
    cs = [spearmanr(g[f], g.y_logratio).statistic for _, g in d.groupby(["cutoff", "scope"]) if g[[f, "y_logratio"]].dropna().shape[0] >= 6 and g[f].nunique() > 1]
    return float(np.nanmean(cs)) if cs else 0.0
zs = lambda g, f: ((g[f].astype(float) - g[f].mean()) / (g[f].std() or 1)).fillna(0)
pk = lambda g, s, h, k: g.assign(s=s).sort_values("s", ascending=False).head(k)[h].mean()
rows = []
for T in CUT:
    w = {}
    for f in FEATS:
        c = [corr(v[v.cutoff != T], f) for v in tabs.values()]
        if all(x > 0.05 for x in c) or all(x < -0.05 for x in c): w[f] = float(np.sign(c[0]))
    for scope, g in tabs["all"][tabs["all"].cutoff == T].groupby("scope"):
        new = sum(wt * zs(g, f) for f, wt in w.items())
        for name, sc in [("E (recipe)", zs(g, "E")), ("robust score", new), ("size", zs(g, "share_T")), ("random", pd.Series(np.random.default_rng(T).random(len(g)), index=g.index))]:
            rows.append({"ranking": name, "emerging p@3": pk(g, sc, "hs_emerging", 3), "growing p@5": pk(g, sc, "hs_growing", 5),
                         "shrinking p@5": pk(g, -sc, "hs_shrinking", 5)})
    for k in ("hs_emerging", "hs_growing", "hs_shrinking"): pass
r = pd.DataFrame(rows)
print("SMOOTHED hits (3-year means before vs after T), held-out cutoffs")
R = r.groupby("ranking").mean()
print(R.round(3).to_string())
a = tabs["all"]; base = a[["hs_emerging", "hs_growing", "hs_shrinking"]].mean().round(3).to_dict(); print("base rates:", base)
experiments.save(RUN, "phase8_direction", "loco_smoothed_hits", R.reset_index(),
                 title="Cluster direction: sign-robust features against the recipe's E (smoothed hits)",
                 how_to_read="As the recipe-hit table, scored with smoothed hits: the mean share over the three years "
                 "after T against the three years before (emerging = at least 1.5x). Compare each column with the "
                 "random row and the base rates in the parameters.",
                 params={"base rates": base})
