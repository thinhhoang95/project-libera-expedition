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

from landscape import experiments

RUN = experiments.run_arg()
df = pd.read_parquet(experiments.work_dir(RUN) / "exp5_papers.parquet")
for c in ("cites_so_far", "coupling", "n_refs", "n_authors", "q_breadth", "a_n_prior", "v_n_prior", "x_n_rising", "x_n_fresh", "b_n_trending",
          "g_cites_so_far", "ga_c_cum", "ga_c_rec", "ga_w_cum", "ga_rate"):
    if c in df.columns:
        df["l_" + c] = np.log1p(df[c].astype(float))
for r in ("semantic", "author", "back"):
    df["route_" + r] = (df.route == r).astype(float)
df["a_has_history"] = (df.a_n_with_history > 0).astype(float)

FAM = {
    "base": ["l_cites_so_far", "l_g_cites_so_far", "l_coupling", "l_n_refs", "l_n_authors", "preprint",
             "route_semantic", "route_author", "route_back"],
    "topic": ["t_heat", "t_typicality", "t_crowding", "t_momentum", "t_drift", "q_heat",
              "t_gheat", "t_gnow", "q_gheat", "q_gnow"],
    "question": ["q_reach", "q_breadth", "l_q_breadth", "q_open_match", "q_typicality", "q_method_atypical"],
    "author": ["a_track", "a_expert", "a_cred_expert", "l_a_n_prior", "a_has_history"],
    "venue": ["v_prestige", "v_gprestige", "l_v_n_prior"],
    "author_global": ["l_ga_c_cum", "l_ga_c_rec", "l_ga_w_cum", "l_ga_rate", "ga_team"],
    "imports": ["x_ext_share", "x_rising_max", "l_x_n_rising", "l_x_n_fresh"],
    "borrowing": ["b_trend_max", "b_trend_mean", "l_b_n_trending", "b_distance", "b_externality", "b_n_subfields"],
}
FAM = {k: [f for f in v if f in df.columns or f.removeprefix("l_") in df.columns] for k, v in FAM.items()}
FAM = {k: v for k, v in FAM.items() if v}
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
    "base + author_global": FAM["base"] + FAM["author_global"],
    **({"base + borrowing": FAM["base"] + FAM["borrowing"]} if "borrowing" in FAM else {}),
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
    meta = dict(outcome=y, population=label, model_kind=kind, papers=len(d), base_rate=d[y].mean())
    models.append(pd.DataFrame(res).assign(**meta))
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
        features.append(u.drop(columns="|edge|").assign(**meta))


kinds = [a for a in sys.argv[1:] if a in ("logit", "hgb")] or ["logit"]
OUTCOMES = [a for a in sys.argv[1:] if a.startswith("y_")] or [
    "y_corpus", "y_corpus_peer", "y_corpus_peer_ns", "y_global", "y_field", "y_peer", "y_g3", "y_g3_field", "y_g3_peer"]
models, features = [], []
for kind in kinds:
    for y in OUTCOMES:
        run(df, y, "all routes", kind)
        run(df[df.route == "forward"], y, "forward route only", kind)
        run(df[df.cites_so_far == 0], y, "cold start (no corpus citations at T)", kind)
        run(df[df.g_cites_so_far == 0], y, "global cold start (no citations anywhere at T)", kind)

OUTCOME_NOTES = (
    "Outcomes (all: top quarter of papers from the same year T): **y_corpus** citations from corpus papers in "
    "T+1..T+3; **y_corpus_peer** the same relative to the 15 most similar same-year papers; **y_corpus_peer_ns** "
    "that without self-citations (the strictest in-field test); **y_global** OpenAlex citations as of 2026; "
    "**y_field** that relative to the paper's subfield; **y_peer** relative to its 15 most similar same-year "
    "papers; **y_g3**, **y_g3_field**, **y_g3_peer** the same three with global citations received in T+1..T+3 only.")
for kind in kinds:
    experiments.save(RUN, "exp5_beyond_rich_get_richer", f"models_{kind}", pd.concat(models).query("model_kind == @kind"),
                     title=f"Topic, question, credibility and imports: models ({kind})",
                     how_to_read="Each row adds one feature family to the bibliometric **base** model (citations so "
                     "far, coupling, references, team size, preprint, route) and scores it leave-one-cutoff-out. "
                     "**gain vs base** is the AUC difference; the **90% CI** comes from a paired bootstrap, so a gain "
                     "whose interval stays above zero is a real improvement on this corpus. Pick an outcome and a "
                     "population: *forward route only* removes most route effects, *cold start* keeps papers with "
                     "no corpus citations at T, *global cold start* those uncited anywhere. " + OUTCOME_NOTES)
experiments.save(RUN, "exp5_beyond_rich_get_richer", "features", pd.concat(features),
                 title="Topic, question, credibility and imports: single features",
                 how_to_read="**AUC alone** ranks papers by the feature by itself (below 0.5 means the feature points "
                 "the other way). **AUC within base deciles** compares papers the base model scores alike: values "
                 "away from 0.5 are what the feature adds beyond rich-get-richer. " + OUTCOME_NOTES)
