"""Evaluate experiment 5: incremental value of topic, question, author and venue features over the
bibliometric base, leave-one-cutoff-out, for two outcomes and three populations.

Reports, per (outcome, population): AUC and p@50 of each model, the AUC gain over the base model with a
paired bootstrap 90% interval, and each feature's AUC *within deciles of the base model's score* (how
much it separates hits from misses among papers the base model scores alike).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("data/backtest/exp5_papers.parquet")
for c in ("cites_so_far", "coupling", "n_refs", "n_authors", "q_breadth", "a_n_prior", "v_n_prior", "x_n_rising", "x_n_fresh"):
    df["l_" + c] = np.log1p(df[c].astype(float))
for r in ("semantic", "author", "back"):
    df["route_" + r] = (df.route == r).astype(float)
df["a_has_history"] = (df.a_n_with_history > 0).astype(float)

FAM = {
    "base": ["l_cites_so_far", "l_coupling", "l_n_refs", "l_n_authors", "preprint",
             "route_semantic", "route_author", "route_back"],
    "topic": ["t_heat", "t_typicality", "t_crowding", "t_momentum", "t_drift", "q_heat"],
    "question": ["q_reach", "q_breadth", "l_q_breadth", "q_open_match", "q_typicality", "q_method_atypical"],
    "author": ["a_track", "a_expert", "a_cred_expert", "l_a_n_prior", "a_has_history"],
    "venue": ["v_prestige", "l_v_n_prior"],
    "imports": ["x_ext_share", "x_rising_max", "l_x_n_rising", "l_x_n_fresh"],
}
for f in sum(FAM.values(), []):
    if f not in ("a_track", "a_expert", "a_cred_expert"):
        df[f] = df[f].astype(float).fillna(df[f].astype(float).median())
for f in ("a_track", "a_expert", "a_cred_expert"):  # no history: neutral value + the indicator
    df[f] = df[f].astype(float).fillna(df[f].astype(float).median())

MODELS = {
    "base": FAM["base"],
    "base + topic": FAM["base"] + FAM["topic"],
    "base + question": FAM["base"] + FAM["question"],
    "base + author": FAM["base"] + FAM["author"],
    "base + venue": FAM["base"] + FAM["venue"],
    "base + imports": FAM["base"] + FAM["imports"],
    "base + topic + question": FAM["base"] + FAM["topic"] + FAM["question"],
    "all": sum(FAM.values(), []),
    "semantic only (topic + question)": FAM["topic"] + FAM["question"],
    "credibility only (author + venue)": FAM["author"] + FAM["venue"],
}


def loco(d, feats, y, kind):
    out = pd.Series(np.nan, index=d.index)
    for T in sorted(d["T"].unique()):
        tr, te = d["T"] != T, d["T"] == T
        if kind == "logit":
            s = StandardScaler().fit(d.loc[tr, feats])
            m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(s.transform(d.loc[tr, feats]), d.loc[tr, y])
            out[te] = m.decision_function(s.transform(d.loc[te, feats]))
        else:
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30,
                                               l2_regularization=1.0, random_state=0)
            m.fit(d.loc[tr, feats], d.loc[tr, y])
            out[te] = m.predict_proba(d.loc[te, feats])[:, 1]
    return out


def p_at(d, score, y, k=50):
    return np.mean([g.nlargest(min(k, len(g) // 4), score)[y].mean() for _, g in d.groupby("T")])


def boot_gain(yv, a, b, n=300, seed=0):
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(n):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].min() == yv[i].max():
            continue
        g.append(roc_auc_score(yv[i], a[i]) - roc_auc_score(yv[i], b[i]))
    return np.percentile(g, [5, 95])


def within_decile_auc(d, f, y, base_score):
    q = pd.qcut(base_score.rank(method="first"), 10, labels=False)
    a, w = [], []
    for _, g in d.groupby(q):
        if g[y].nunique() == 2:
            a.append(roc_auc_score(g[y], g[f]))
            w.append(len(g))
    return np.average(a, weights=w)


def run(d, y, label, kind="logit"):
    d = d.copy()
    print(f"\n=== {label} · outcome {y} · {kind} · {len(d)} papers, base rate {d[y].mean():.3f}")
    sc = {m: loco(d, f, y, kind) for m, f in MODELS.items()}
    yv = d[y].to_numpy().astype(int)
    base = sc["base"].to_numpy()
    res = []
    for m, s in sc.items():
        auc = roc_auc_score(yv, s)
        lo, hi = boot_gain(yv, s.to_numpy(), base) if m != "base" else (0, 0)
        d["_s"] = s
        res.append({"model": m, "AUC": auc, "gain vs base": auc - roc_auc_score(yv, base),
                    "gain 90% CI": f"[{lo:+.3f}, {hi:+.3f}]", "p@50": p_at(d, "_s", y)})
    print(pd.DataFrame(res).round(3).to_string(index=False))
    if kind == "logit":
        uni = []
        for fam, fs in FAM.items():
            for f in fs:
                if d[f].nunique() < 2:
                    continue
                uni.append({"family": fam, "feature": f, "AUC alone": roc_auc_score(yv, d[f]),
                            "AUC within base deciles": within_decile_auc(d, f, y, sc["base"])})
        u = pd.DataFrame(uni)
        u["|edge|"] = (u["AUC within base deciles"] - 0.5).abs()
        print(u.sort_values("|edge|", ascending=False).drop(columns="|edge|").round(3).to_string(index=False))


kinds = [a for a in sys.argv[1:] if a in ("logit", "hgb")] or ["logit"]
OUTCOMES = [a for a in sys.argv[1:] if a.startswith("y_")] or ["y_global", "y_field", "y_peer", "y_corpus"]
for kind in kinds:
    for y in OUTCOMES:
        run(df, y, "all routes", kind)
        run(df[df.route == "forward"], y, "forward route only", kind)
        run(df[df.cites_so_far == 0], y, "cold start (no corpus citations at T)", kind)
