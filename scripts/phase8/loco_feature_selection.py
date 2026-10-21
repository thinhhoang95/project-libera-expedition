"""Leave-one-cutoff-out: pick sign-robust features on 3 cutoffs, test on the 4th."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from landscape import experiments
from landscape.config import project_dir
from landscape.backtest import cluster_feature_table
from landscape.cli import _params
RUN = experiments.run_arg()
con = experiments.connect_ro(); p = _params(con, RUN)
CUT = (2020, 2021, 2022, 2023)
SAMPLES = {"all": None, "forward": {"forward"}, "nonforward": {"seed", "back", "author", "semantic"}}
FEATS = ["novelty", "new_author_share", "g", "dg", "momentum", "ref_age", "ext_inflow", "coherence",
         "share_T", "mig_net", "cit_growth", "N"]
tabs = {k: pd.concat([cluster_feature_table(con, RUN, T, p, r) for T in CUT]) for k, r in SAMPLES.items()}
for k in tabs: tabs[k] = tabs[k][tabs[k].n_total >= 10].reset_index(drop=True)
def corr(d, f):
    cs = [spearmanr(g[f], g.y_logratio).statistic for _, g in d.groupby(["cutoff", "scope"])
          if g[[f, "y_logratio"]].dropna().shape[0] >= 6 and g[f].nunique() > 1]
    return float(np.nanmean(cs)) if cs else 0.0
def zs(g, f):
    x = g[f].astype(float); return ((x - x.mean()) / (x.std() or 1)).fillna(0)
def pk(g, score, hit, k):
    top = g.assign(s=score).sort_values("s", ascending=False).head(k); return top[hit].mean()
rows, chosen = [], {}
for T in CUT:
    train = {k: v[v.cutoff != T] for k, v in tabs.items()}
    w = {}
    for f in FEATS:
        c = [corr(train[k], f) for k in SAMPLES]
        if all(x > 0.05 for x in c) or all(x < -0.05 for x in c):
            w[f] = float(np.sign(c[0]))
    chosen[T] = w
    test = tabs["all"][tabs["all"].cutoff == T]
    for scope, g in test.groupby("scope"):
        new = sum(wt * zs(g, f) for f, wt in w.items()) if w else pd.Series(0, index=g.index)
        for name, sc in [("E (recipe)", zs(g, "E")), ("robust score", new), ("g (recipe growth)", zs(g, "g")),
                         ("size", zs(g, "share_T")), ("momentum only", zs(g, "momentum"))]:
            rows.append({"cutoff": T, "scope": scope, "ranking": name,
                         "rho_future": spearmanr(sc, g.y_logratio).statistic,
                         "emerging p@3": pk(g, sc, "hit_emerging", 3), "growing p@5": pk(g, sc, "hit_growing", 5),
                         "shrinking p@5 (low score)": pk(g, -sc, "hit_shrinking", 5),
                         "base_emerging": g.hit_emerging.mean(), "base_growing": g.hit_growing.mean(),
                         "base_shrinking": g.hit_shrinking.mean()})
r = pd.DataFrame(rows)
print("features chosen per held-out cutoff:", chosen)
R = r.groupby("ranking")[[c for c in r.columns if c not in ("cutoff", "scope", "ranking")]].mean()
print(R.round(3).to_string())
for k, v in tabs.items(): v.to_parquet(project_dir(RUN, "backtest") / f"cluster_features_{k}.parquet")
experiments.save(RUN, "phase8_direction", "loco_recipe_hits", R.reset_index(),
                 title="Cluster direction: sign-robust features against the recipe's E (recipe hits)",
                 how_to_read="Leave-one-cutoff-out: features whose sign agrees under every outcome sample are picked on "
                 "three cutoffs and scored on the fourth (the robust score, E'), against the recipe's E, recent growth, "
                 "size and momentum. rho_future is the correlation with the future share change; p@k uses the recipe's "
                 "single-year hits; base_* are the base rates.",
                 params={"features chosen per held-out cutoff": {str(k): v for k, v in chosen.items()}})
