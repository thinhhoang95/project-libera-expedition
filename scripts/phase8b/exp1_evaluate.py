"""Evaluate rankings for experiment 1 (paper-level bridges), leave-one-cutoff-out for learned models.
Saves its tables to data/projects/<run>/backtest/experiments/ (explorer: Experiments view)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from landscape import experiments

RUN = experiments.run_arg()
df = pd.read_parquet(experiments.work_dir(RUN) / "exp1_pairs.parquet")
df["lpop"] = np.log1p(df["pop"])
df["age"] = df["T"] - df["i_year"]
df["lcoup"] = np.log1p(df["coupling"] * 100)
df["has_lim"] = df.n_lim > 0
df["lim_sim3_f"] = df.lim_sim3.fillna(df.prob_sim)        # clusters without limitations fall back to problems
FEATS_ALL = ["lpop", "lcoup", "prob_sim", "lim_sim3_f", "age"]
FEATS_NOPOP = ["lcoup", "prob_sim", "lim_sim3_f", "age"]


def per_cluster_p_at(d, score, k=10):
    out = []
    for _, g in d.groupby(["T", "X"]):
        if g.hit.sum() == 0:
            continue
        out.append(g.nlargest(k, score).hit.mean())
    return np.mean(out)


def evaluate(d, label):
    coefs = []
    res = {}
    for name in ["random", "lpop", "lcoup", "prob_sim", "lim_sim3_f", "-age",
                 "logit (all features)", "logit (no popularity)"]:
        res[name] = {}
    # learned scores, leave one cutoff out
    for name, feats in (("logit (all features)", FEATS_ALL), ("logit (no popularity)", FEATS_NOPOP)):
        d[name] = np.nan
        for T in sorted(d["T"].unique()):
            tr, te = d["T"] != T, d["T"] == T
            sc = StandardScaler().fit(d.loc[tr, feats])
            m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(d.loc[tr, feats]), d.loc[tr, "hit"])
            d.loc[te, name] = m.decision_function(sc.transform(d.loc[te, feats]))
            if T == sorted(d["T"].unique())[-1]:
                print(f"  {name} coefficients (fit without {T}):", dict(zip(feats, np.round(m.coef_[0], 2))))
                coefs.extend({"model": name, "feature": f, "coefficient": float(c), "fold left out": int(T)}
                             for f, c in zip(feats, m.coef_[0]))
    d["random"] = np.random.default_rng(0).random(len(d))
    d["-age"] = -d["age"]
    rows = []
    for name in res:
        rows.append({"ranking": name, "p@10 per cluster": per_cluster_p_at(d, name),
                     "AUC": roc_auc_score(d.hit, d[name]),
                     "AUC long tail (pop<=1)": roc_auc_score(d[d["pop"] <= 1].hit, d[d["pop"] <= 1][name])})
    print(f"== {label}: {len(d)} pairs, {int(d.hit.sum())} hits, base rate {d.hit.mean():.4f}, "
          f"long-tail base rate {d[d['pop'] <= 1].hit.mean():.4f}")
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    return out.assign(population=label, pairs=len(d), hits=int(d.hit.sum()), base_rate=d.hit.mean(),
                      long_tail_base_rate=d[d["pop"] <= 1].hit.mean()), pd.DataFrame(coefs).assign(population=label)


tabs, coefs = zip(*(evaluate(df.copy(), "all clusters"),
                    evaluate(df[df.has_lim].copy(), "clusters with stated limitations only")))
experiments.save(RUN, "exp1_paper_bridges", "rankings", pd.concat(tabs), title="Paper-level bridges: rankings",
                 how_to_read="Candidates are (problem cluster X, innovation paper i) pairs where no X paper cites i "
                 "yet at cutoff T; a hit is an X paper citing i in T+1..T+3. **p@10 per cluster** is the share of "
                 "hits among each cluster's top 10, averaged over clusters with at least one hit. **AUC long tail** "
                 "restricts to papers cited at most once, where popularity cannot help. Logistic models are scored "
                 "leave-one-cutoff-out. Compare each ranking with **random** and **lpop** (popularity).")
experiments.save(RUN, "exp1_paper_bridges", "coefficients", pd.concat(coefs), title="Paper-level bridges: model weights",
                 how_to_read="Standardized logistic coefficients from the fold that leaves out the last cutoff. "
                 "Larger means the feature pushes a pair up the ranking more; features are correlated, so read "
                 "them together.")
