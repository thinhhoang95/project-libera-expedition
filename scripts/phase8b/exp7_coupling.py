"""Experiment 7: how much of the paper-bridge signal needs the citation network, and what is left
for semantics where the network is blind?

Bibliographic coupling uses the candidate's own reference list, so it exists the day a paper
appears. It still only sees candidates that already cite what the community cites. This script
scores the exp1 candidate pairs (problem community X, innovation paper i, not yet cited by X) in
strata where the network gives less and less:

    fresh            i published in the cutoff year T (no time to be cited)
    uncited          i has no corpus citations by T
    zero coupling    no X paper shares a single reference with i
    combinations of the above

Models are trained on all pairs (leave-one-cutoff-out, and a chronological split: train 2020,
test 2023) and evaluated inside each stratum. Hit: an X paper published in T+1..T+3 cites i.
Needs exp1_paper_bridges.py output. Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at
a copy of the database.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from landscape import experiments

RUN = experiments.run_arg()
WORK = experiments.work_dir(RUN)
con = experiments.connect_ro()
d = pd.read_parquet(WORK / "exp1_pairs.parquet")
nref = con.execute("""SELECT r.citing_id AS i, count(*) AS n_refs FROM refs r JOIN provenance v
                      ON v.paper_id = r.citing_id AND v.run_id = ? GROUP BY 1""", [RUN]).df().set_index("i").n_refs
d["n_refs_i"] = d.i.map(nref).fillna(0)
d["lpop"] = np.log1p(d["pop"])
d["age"] = d["T"] - d.i_year
d["lcoup"] = np.log1p(d.coupling * 100)
d["lcoup_norm"] = np.log1p(100 * d.coupling / d.n_refs_i.clip(lower=1))  # per reference of i
d["lim_sim3_f"] = d.lim_sim3.fillna(d.prob_sim)
d["hit"] = d.hit.astype(int)

MODELS = {
    "citations + age": ["lpop", "age"],
    "+ coupling": ["lpop", "age", "lcoup"],
    "+ coupling per reference": ["lpop", "age", "lcoup_norm"],
    "semantic only": ["prob_sim", "lim_sim3_f"],
    "citations + age + semantic": ["lpop", "age", "prob_sim", "lim_sim3_f"],
    "all (citations, coupling, semantic)": ["lpop", "age", "lcoup", "prob_sim", "lim_sim3_f"],
}
STRATA = {
    "all candidates": lambda x: np.ones(len(x), bool),
    "fresh (published in T)": lambda x: x.i_year == x["T"],
    "uncited at T": lambda x: x["pop"] == 0,
    "zero coupling": lambda x: x.coupling == 0,
    "uncited, zero coupling": lambda x: (x["pop"] == 0) & (x.coupling == 0),
    "fresh, zero coupling": lambda x: (x.i_year == x["T"]) & (x.coupling == 0),
}


def fit_score(train, test, feats):
    s = StandardScaler().fit(train[feats])
    m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s.transform(train[feats]), train.hit)
    return m.decision_function(s.transform(test[feats]))


def protocols():
    """(name, scores DataFrame aligned to the evaluated rows)."""
    loco = pd.DataFrame(index=d.index, columns=list(MODELS), dtype=float)
    for T in sorted(d["T"].unique()):
        tr, te = d["T"] != T, d["T"] == T
        for k, f in MODELS.items():
            loco.loc[te, k] = fit_score(d[tr], d[te], f)
    yield "leave-one-cutoff-out (2020-2023)", d, loco
    tr, te = d[d["T"] == 2020], d[d["T"] == 2023]
    chrono = pd.DataFrame({k: fit_score(tr, te, f) for k, f in MODELS.items()}, index=te.index)
    yield "chronological (train 2020, test 2023; 2026 partial)", te, chrono


def p_at(x, s, k=10):
    """Top-k precision per community, among communities with at least one hit in the stratum."""
    v = []
    for _, g in x.assign(_s=s).groupby(["T", "X"]):
        if g.hit.sum():
            v.append(g.nlargest(k, "_s").hit.mean())
    return np.mean(v) if v else np.nan


def boot_gain(x, a, b, n=200):
    """Bootstrap over communities x cutoffs (the unit a list is shown for)."""
    rng = np.random.default_rng(0)
    groups = list(x.groupby(["T", "X"]).indices.values())
    y, out = x.hit.to_numpy(), []
    for _ in range(n):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(out, [5, 95]) if out else (np.nan, np.nan)


rows, gains = [], []
for proto, data, sc in protocols():
    for sname, f in STRATA.items():
        m = np.asarray(f(data), bool)
        x, s = data[m], sc[m]
        if x.hit.sum() < 5 or x.hit.nunique() < 2:
            continue
        for k in MODELS:
            rows.append({"protocol": proto, "stratum": sname, "model": k, "pairs": len(x), "hits": int(x.hit.sum()),
                         "base_rate": x.hit.mean(), "AUC": roc_auc_score(x.hit, s[k]), "p@10": p_at(x, s[k].to_numpy())})
        rows.append({"protocol": proto, "stratum": sname, "model": "random", "pairs": len(x), "hits": int(x.hit.sum()),
                     "base_rate": x.hit.mean(), "AUC": 0.5,
                     "p@10": p_at(x, np.random.default_rng(0).random(len(x)))})
        for a, b in (("citations + age + semantic", "citations + age"),
                     ("all (citations, coupling, semantic)", "+ coupling"),
                     ("semantic only", "citations + age")):
            lo, hi = boot_gain(x.reset_index(drop=True), s[a].to_numpy(), s[b].to_numpy())
            gains.append({"protocol": proto, "stratum": sname, "comparison": f"{a}  vs  {b}",
                          "AUC gain": roc_auc_score(x.hit, s[a]) - roc_auc_score(x.hit, s[b]),
                          "90% CI": f"[{lo:+.3f}, {hi:+.3f}]"})
        print(proto, "|", sname, len(x), int(x.hit.sum()), flush=True)

R = pd.DataFrame(rows)
G = pd.DataFrame(gains)
pd.set_option("display.width", 220)
print(R.pivot_table(index=["protocol", "stratum"], columns="model", values="AUC").round(3).to_string())
print(R.pivot_table(index=["protocol", "stratum"], columns="model", values="p@10").round(3).to_string())
print(G.round(3).to_string(index=False))
how = ("Candidate pairs from experiment 1: a problem community X at cutoff T and an innovation paper i that no X paper "
       "cites yet; a **hit** is an X paper citing i in T+1..T+3. Models are trained on all pairs and scored inside each "
       "**stratum**, where the citation network gives less and less: *fresh* papers were published in T, *uncited* "
       "ones have no corpus citations at T, *zero coupling* means no X paper shares a single reference with i. "
       "**p@10** is the hit share in each community's top 10 when only that stratum is listed (communities with at "
       "least one hit). *Chronological* trains on 2020 only and tests 2023, so no future outcome is used in training.")
experiments.save(RUN, "exp7_coupling", "models_by_stratum", R, title="Coupling vs semantics: models by stratum", how_to_read=how)
experiments.save(RUN, "exp7_coupling", "semantic_gains", G, title="Coupling vs semantics: what semantics adds",
                 how_to_read="AUC gain of adding the two semantic similarities (problem, limitation) to the "
                 "bibliometric models, per stratum; the 90% interval resamples communities x cutoffs. *semantic only vs "
                 "citations + age* asks whether meaning alone beats the citation baseline.")
