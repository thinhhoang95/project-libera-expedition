"""Experiment 18: structural twins -- same mathematical structure, different application.

Pairs of problem communities (cutoff runs 2019-2022, alpha 0.5, medium, >= 15 papers up to T). At T:
    topic       cosine of the communities' mean problem statements
    structure   cosine of their mean problem *glosses* (the textbook form of the problem)
    application cosine of their mean application-domain statements (domain_data facet)
    citations   citations between them up to T, per pair of papers
    convergence_before   share of their papers from T-2..T whose method or innovation is within 0.894 of the other
                         community's innovations from before
Outcome: convergence in T+1..T+3 (the same share for their new papers, against the other's innovations up to T),
top quarter = hit. Twins = structure in the top quarter and application in the bottom half.
Tests: partial rank correlation of structure with later convergence, holding topic, application, citations, prior
convergence and sizes fixed; logistic gain of structure (leave-one-cutoff-out); twins vs pairs with the same topic
similarity decile. Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
PP, PV = C.vectors("problem", ids)
GP, GV = C.vectors("problem_gloss", ids)
DP, DV = C.vectors("domain_data", ids)
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)


def mean_vec(pos, V, pids):
    k = [pos[x] for x in pids if x in pos]
    return C.unit(V[k].mean(axis=0)) if len(k) >= 3 else None


def stack(pids):
    a = [MV[MP[x]] for x in pids if x in MP] + [IV[IP[x]] for x in pids if x in IP]
    return a


def conv_share(new_ids, other_inn):
    """Share of papers in new_ids with a method or innovation within THR of any of other_inn."""
    if not len(new_ids) or not len(other_inn):
        return 0.0
    hits = 0
    for x in new_ids:
        vs = [MV[MP[x]]] if x in MP else []
        vs += [IV[IP[x]]] if x in IP else []
        if vs and (np.vstack(vs) @ other_inn.T).max() >= C.THR_REUSE:
            hits += 1
    return hits / len(new_ids)


rows = []
for T in range(2019, 2023):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    comm = {}
    for X in sorted(set(lab)):
        mem = lab.index[lab.values == X]
        pre = [x for x in mem if year[x] <= T]
        if len(pre) < 15:
            continue
        comm[X] = {"pre": pre, "recent": [x for x in mem if T - 2 <= year[x] <= T],
                   "new": [x for x in mem if T + 1 <= year[x] <= T + 3],
                   "topic": mean_vec(PP, PV, pre), "struct": mean_vec(GP, GV, pre), "app": mean_vec(DP, DV, pre),
                   "inn_pre": IV[[IP[x] for x in pre if x in IP]],
                   "inn_older": IV[[IP[x] for x in pre if x in IP and year[x] <= T - 3]]}
    rT = refs[refs.cy <= T]
    for X, Y in itertools.combinations(sorted(comm), 2):
        a, b = comm[X], comm[Y]
        if a["app"] is None or b["app"] is None:
            continue
        sx, sy = set(a["pre"]), set(b["pre"])
        cxy = int((rT.citing_id.isin(sx) & rT.cited_id.isin(sy)).sum() + (rT.citing_id.isin(sy) & rT.cited_id.isin(sx)).sum())
        before = (conv_share(a["recent"], b["inn_older"]) + conv_share(b["recent"], a["inn_older"])) / 2
        after = (conv_share(a["new"], b["inn_pre"]) + conv_share(b["new"], a["inn_pre"])) / 2
        rows.append({"T": T, "X": X, "Y": Y, "topic": float(a["topic"] @ b["topic"]),
                     "structure": float(a["struct"] @ b["struct"]), "application": float(a["app"] @ b["app"]),
                     "citations": cxy / (len(sx) * len(sy)), "convergence_before": before, "convergence_after": after,
                     "size": np.log(len(sx) * len(sy))})
    print(T, "pairs", sum(1 for r in rows if r["T"] == T), flush=True)

D = pd.DataFrame(rows)
D["hit"] = D.convergence_after >= D.groupby("T").convergence_after.transform(lambda s: s.quantile(0.75))
D["twin"] = (D.structure >= D.groupby("T").structure.transform(lambda s: s.quantile(0.75))) & \
            (D.application <= D.groupby("T").application.transform("median"))
D["l_citations"] = np.log1p(D.citations * 1e4)
D.to_parquet(C.project(RUN, "backtest") / "exp18_twins.parquet", index=False)

CTRL = ["topic", "application", "l_citations", "convergence_before", "size"]


def resid(v, Z):
    Z = np.column_stack([np.ones(len(Z)), Z])
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


res = []
Z = D[CTRL].to_numpy()
for f in ("structure", "topic", "application"):
    others = [c for c in CTRL if c != f]
    Zf = D[others].to_numpy()
    rho = spearmanr(resid(D[f].to_numpy(), Zf), resid(D.convergence_after.to_numpy(), Zf)).statistic
    rs = np.random.default_rng(0)
    bs = []
    for _ in range(500):
        i = rs.integers(0, len(D), len(D))
        bs.append(spearmanr(resid(D[f].to_numpy()[i], Zf[i]), resid(D.convergence_after.to_numpy()[i], Zf[i])).statistic)
    res.append({"measure": f"partial rho of {f} with later convergence", "value": rho,
                "90% CI": f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]", "pairs x cutoffs": len(D)})


def loco(feats):
    out = pd.Series(np.nan, index=D.index)
    for T in sorted(D["T"].unique()):
        tr, te = D["T"] != T, D["T"] == T
        s = StandardScaler().fit(D.loc[tr, feats])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s.transform(D.loc[tr, feats]), D.loc[tr, "hit"])
        out[te] = m.decision_function(s.transform(D.loc[te, feats]))
    return out


b = loco(CTRL)
s = loco(CTRL + ["structure"])
y = D.hit.astype(int).to_numpy()
rs = np.random.default_rng(0)
g = []
for _ in range(500):
    i = rs.integers(0, len(D), len(D))
    if 0 < y[i].sum() < len(i):
        g.append(roc_auc_score(y[i], s.to_numpy()[i]) - roc_auc_score(y[i], b.to_numpy()[i]))
res.append({"measure": "AUC gain of structure over topic, application, citations, prior convergence, size (LOCO)",
            "value": roc_auc_score(y, s) - roc_auc_score(y, b), "90% CI": f"[{np.percentile(g, 5):+.3f}, {np.percentile(g, 95):+.3f}]",
            "pairs x cutoffs": len(D)})
D["topic_decile"] = D.groupby("T").topic.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
m = D.groupby(["T", "topic_decile"]).apply(lambda g: pd.Series({
    "twin": g[g.twin].hit.mean() if g.twin.any() else np.nan, "other": g[~g.twin].hit.mean() if (~g.twin).any() else np.nan,
    "n_twin": int(g.twin.sum())})).dropna()
res.append({"measure": "hit rate, twins vs other pairs with the same topic similarity (decile-matched)",
            "value": float(np.average(m.twin - m.other, weights=m.n_twin)), "90% CI": "",
            "pairs x cutoffs": int(D.twin.sum())})
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
C.save(RUN, "exp18_twins", "results", R, title="Structural twins: does shared structure across applications predict convergence?",
       how_to_read="Pairs of problem communities at each cutoff (2019-2022). **structure** = similarity of their problem "
       "glosses (the textbook form of the problem), **topic** = of their problem statements, **application** = of their "
       "application domains. Outcome: in the next three years, how many of each community's new papers have a method or "
       "innovation within 0.894 of the other community's innovations (convergence), top quarter = hit. The partial "
       "correlations hold the other similarities, existing citations, prior convergence and sizes fixed; the AUC gain "
       "adds structure to all of those. **Twins** = top quarter of structure with a below-median application similarity.")
