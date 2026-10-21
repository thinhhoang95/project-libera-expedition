"""Shared helpers for the Phase 11 experiments (time: the age of an opportunity's evidence, and whether it is still
open). Band tables with observed / expected against the score, and model comparisons under both protocols."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)


def mom(a, b):
    return np.log((a + 1) / (b + 1))


def cosine_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return (M / np.where(n == 0, 1, n)) @ (M / np.where(n == 0, 1, n)).T


def pct_rank(s: pd.Series, by: pd.Series) -> pd.Series:
    return s.groupby(by).rank(pct=True)


def by_band(d, band, hit, score, top_k=None, group=None):
    """Hit rate and observed/expected per band; expected = hit rate of the candidate's score quintile at its cutoff.
    90% interval of O/E by resampling `group` (communities)."""
    d = d.copy()
    d["q"] = d.groupby("T")[score].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    d["exp"] = d.groupby(["T", "q"])[hit].transform("mean")
    if top_k:
        d["top"] = d.groupby("T")[score].rank(ascending=False, method="first") <= top_k
    groups = d[group].astype(str).values if group else np.arange(len(d)).astype(str)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    rows = []
    for b, g in d.groupby(band, observed=True):
        o, e = g[hit].mean(), g["exp"].mean()
        boot = []
        bmask = (d[band] == b).values
        h, ex = d[hit].values.astype(float), d["exp"].values
        for _ in range(300):
            ix = np.concatenate([gi[x] for x in rng.choice(ug, len(ug))])
            m = bmask[ix]
            if m.sum() and ex[ix][m].sum() > 0:
                boot.append(h[ix][m].sum() / ex[ix][m].sum())
        lo, hi = np.percentile(boot, [5, 95]) if boot else (np.nan, np.nan)
        r = {"band": b, "candidates": len(g), "hits": int(g[hit].sum()), "hit rate": o,
             "expected from score": e, "observed / expected": o / e if e else np.nan, "O/E 90% low": lo,
             "O/E 90% high": hi}
        if top_k:
            t = g[g.top]
            r.update({f"in top {top_k}": len(t), f"hit rate in top {top_k}": t[hit].mean() if len(t) else np.nan})
        rows.append(r)
    return pd.DataFrame(rows)


def compare(d, hit, base_feats, time_feats, k, label, score_name):
    """Production score vs logistic models with and without the time features, LOCO and chronological."""
    rows = []
    Ts = sorted(d["T"].unique())
    for protocol in ("LOCO", "chronological"):
        for T in Ts:
            tr_T = [t for t in Ts if t != T] if protocol == "LOCO" else [t for t in Ts if t + 3 <= T]
            if not tr_T:
                continue
            tr, te = d[d["T"].isin(tr_T)], d[d["T"] == T]
            if te[hit].sum() == 0 or tr[hit].sum() == 0:
                continue
            sc = {"production score": te[score_name].values}
            for name, feats in (("model: score features", base_feats), ("model: + time features", base_feats + time_feats)):
                s = StandardScaler().fit(tr[feats])
                m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(tr[feats]), tr[hit])
                sc[name] = m.decision_function(s.transform(te[feats]))
                if name.endswith("time features") and T == Ts[-1]:
                    for f, c in zip(feats, m.coef_[0]):
                        rows.append({"list": label, "protocol": protocol, "T": T, "ranking": "coefficient: " + f,
                                     "AUC": float(c)})
            y = te[hit].values
            for name, s in sc.items():
                top = np.argsort(-s)[:k]
                rows.append({"list": label, "protocol": protocol, "T": T, "ranking": name, "AUC": roc_auc_score(y, s),
                             f"p@{k}": y[top].mean(), "base": y.mean(), "hits": int(y.sum()), "candidates": len(y)})
    r = pd.DataFrame(rows)
    coef = r[r.ranking.str.startswith("coefficient")][["list", "protocol", "T", "ranking", "AUC"]].rename(
        columns={"AUC": "value"})
    r = r[~r.ranking.str.startswith("coefficient")]
    summ = r.groupby(["list", "protocol", "ranking"])[["AUC", f"p@{k}", "base", "hits", "candidates"]].mean()
    summ["cutoffs"] = r.groupby(["list", "protocol", "ranking"]).T.apply(lambda s: ", ".join(map(str, sorted(s))))
    return summ.reset_index(), coef
