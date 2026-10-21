"""Experiment 14: durable ideas vs flashes -- which ideas are still *used* years later?

Papers published in 2018-2019 (research types, with an innovation statement). Use of an idea = papers by other
authors whose method or innovation statement is within 0.894 of the paper's innovation (experiment 9: most uptake is
uncited), per paper published in the window:
    early   the two years after publication
    late    2023-2025
Among ideas used early (>= 2 such papers), **durable** = late rate >= early rate (still used, or more); the rest
**faded**. Separately, **late bloomers** = not used early, used in 2023-2025 (>= 2 papers).

Traits visible at publication (data up to the publication year): breadth (how many communities' earlier papers are
already within 0.894 of the idea), generality of the problem (mean similarity to the 50 closest earlier problems),
novelty, contribution type, stated limitations, quantitative result, genericness (mean similarity to random
statements: ideas close to everything look "used" everywhere), early non-self citations. Each trait's AUC for
durable vs faded and for late bloomers, per publication year (sign stability), and a two-fold model (fit on one year,
score the other). Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
ptype = papers.set_index("paper_id").type
PP, PV = C.vectors("problem", ids)
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
facets = con.execute("""SELECT paper_id, facet, text FROM facets WHERE facet IN ('contribution_type',
                        'has_quantitative_result', 'limitations')""").df()
fac = facets.pivot_table(index="paper_id", columns="facet", values="text", aggfunc="first")
lab = C.assign(con, RUN, f"{RUN}@T2019", ids)

# all method + innovation statements with their paper and year
st_ids = [x for x in MP] + [x for x in IP]
ST = np.vstack([MV[[MP[x] for x in MP]], IV[[IP[x] for x in IP]]])
st_year = np.array([year[x] for x in st_ids])
rng = np.random.default_rng(0)
rand = ST[rng.choice(len(ST), 1000, replace=False)]
n_by_year = papers.groupby("year").size()

rows = []
for y0 in (2018, 2019):
    cand = [x for x in papers[(papers.year == y0) & papers.type.isin(C.RESEARCH)].paper_id if x in IP and x in PP]
    early_w, late_w = (y0 + 1, y0 + 2), (2023, 2025)
    n_early = n_by_year.loc[early_w[0]:early_w[1]].sum()
    n_late = n_by_year.loc[late_w[0]:late_w[1]].sum()
    prev = np.where(st_year <= y0)[0]
    probs_prev = [x for x in PP if year[x] <= y0]
    Pprev = PV[[PP[x] for x in probs_prev]]
    innov_prev = [x for x in IP if year[x] < y0]
    Iprev = IV[[IP[x] for x in innov_prev]]
    for x in cand:
        v = IV[IP[x]]
        tm = team.get(x, frozenset())
        s = ST @ v
        close = np.where(s >= C.THR_REUSE)[0]
        users = {}
        for j in close:
            p = st_ids[j]
            if p != x and not (team.get(p, frozenset()) & tm):
                users[p] = st_year[j]
        uy = np.array(list(users.values())) if users else np.array([])
        e = ((uy >= early_w[0]) & (uy <= early_w[1])).sum()
        l_ = ((uy >= late_w[0]) & (uy <= late_w[1])).sum()
        prev_close = [st_ids[j] for j in prev[s[prev] >= C.THR_REUSE] if st_ids[j] != x]
        breadth = len({lab.get(p) for p in prev_close if isinstance(lab.get(p), str)})
        cit = refs[(refs.cited_id == x) & refs.cy.between(y0, y0 + 1)]
        early_cites = sum(1 for c in cit.citing_id if not (team.get(c, frozenset()) & tm))
        f = fac.loc[x] if x in fac.index else pd.Series(dtype=object)
        rows.append({"paper_id": x, "year": y0, "early_users": int(e), "late_users": int(l_),
                     "early_rate": e / n_early, "late_rate": l_ / n_late,
                     "breadth": breadth, "generality": float(np.sort(Pprev @ PV[PP[x]])[-50:].mean()),
                     "novelty": 1 - float((Iprev @ v).max()) if len(Iprev) else np.nan,
                     "genericness": float((rand @ v).mean()), "early_cites": early_cites,
                     "theory": f.get("contribution_type") == "theory", "method": f.get("contribution_type") == "method",
                     "application": f.get("contribution_type") == "application",
                     "states_limitation": isinstance(f.get("limitations"), str) and len(f.get("limitations") or "") > 0,
                     "quantitative": f.get("has_quantitative_result") == "true"})
    print(y0, len(cand), flush=True)

D = pd.DataFrame(rows)
D["used_early"] = D.early_users >= 2
D["durable"] = D.used_early & (D.late_rate >= D.early_rate)
D["late_bloomer"] = (D.early_users == 0) & (D.late_users >= 2)
D.to_parquet(C.project(RUN, "backtest") / "exp14_ideas.parquet", index=False)
TRAITS = ["breadth", "generality", "novelty", "genericness", "early_cites", "theory", "method", "application",
          "states_limitation", "quantitative"]
for t in TRAITS:
    D[t] = D[t].astype(float)
    D[t] = D[t].fillna(D[t].median())

res = []
for outcome, pop in (("durable", D[D.used_early]), ("late_bloomer", D[~D.used_early])):
    for t in TRAITS:
        per_year = [roc_auc_score(g[outcome], g[t]) for _, g in pop.groupby("year") if g[outcome].nunique() == 2]
        res.append({"outcome": outcome, "trait": t, "AUC": roc_auc_score(pop[outcome], pop[t]),
                    "AUC 2018": per_year[0] if per_year else np.nan, "AUC 2019": per_year[1] if len(per_year) > 1 else np.nan,
                    "papers": len(pop), "positive rate": pop[outcome].mean()})
    # two-fold model: fit on one publication year, score the other
    sc = pd.Series(np.nan, index=pop.index)
    base = pd.Series(np.nan, index=pop.index)
    for y0 in (2018, 2019):
        tr, te = pop.year != y0, pop.year == y0
        for feats, out in ((TRAITS, sc), (["early_cites", "genericness"], base)):
            s = StandardScaler().fit(pop.loc[tr, feats])
            m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s.transform(pop.loc[tr, feats]), pop.loc[tr, outcome])
            out[te] = m.decision_function(s.transform(pop.loc[te, feats]))
    res.append({"outcome": outcome, "trait": "model: all traits (fit on the other year)", "AUC": roc_auc_score(pop[outcome], sc),
                "papers": len(pop), "positive rate": pop[outcome].mean()})
    res.append({"outcome": outcome, "trait": "model: early citations + genericness only", "AUC": roc_auc_score(pop[outcome], base),
                "papers": len(pop), "positive rate": pop[outcome].mean()})
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
C.save(RUN, "exp14_durable", "traits", R, title="Durable ideas vs flashes: what is visible at publication",
       how_to_read="Ideas (innovation statements) of papers from 2018-2019. **Use** = papers by other authors whose "
       "method or innovation is within 0.894 of the idea, cited or not (experiment 9), per paper published in the "
       "window. Among ideas used in the two years after publication, **durable** ones are used at least as much in "
       "2023-2025; **late bloomers** were not used early but are in 2023-2025. Each trait is known at publication: "
       "**breadth** (how many communities' earlier work is already close to the idea), **generality** of the problem, "
       "novelty, contribution type, a stated limitation, a quantitative result, **genericness** (closeness to "
       "everything, a control) and early citations by others. AUC per publication year shows whether a trait's "
       "direction is stable; the models are fitted on one year and scored on the other.")

# ---- the same traits among equally generic ideas (genericness quintiles): what survives the artefact
def strat_auc(df, t, y):
    a, w = [], []
    for _, g in df.groupby("gq"):
        if g[y].nunique() == 2 and g[t].nunique() > 1:
            a.append(roc_auc_score(g[y], g[t]))
            w.append(len(g))
    return np.average(a, weights=w) if a else np.nan


res2 = []
for outcome, pop in (("durable", D[D.used_early]), ("late_bloomer", D[~D.used_early])):
    pop = pop.copy()
    pop["gq"] = pd.qcut(pop.genericness.rank(method="first"), 5, labels=False)
    rs = np.random.default_rng(0)
    for t in [x for x in TRAITS if x != "genericness"]:
        bs = [strat_auc(pop.sample(len(pop), replace=True, random_state=int(rs.integers(1_000_000_000))), t, outcome)
              for _ in range(300)]
        res2.append({"outcome": outcome, "trait": t, "AUC within genericness quintiles": strat_auc(pop, t, outcome),
                     "90% CI": f"[{np.nanpercentile(bs, 5):.3f}, {np.nanpercentile(bs, 95):.3f}]",
                     "papers": len(pop), "positives": int(pop[outcome].sum())})
R2 = pd.DataFrame(res2)
print(R2.round(3).to_string(index=False))
C.save(RUN, "exp14_durable", "traits_controlled", R2, title="Durable ideas: traits among equally generic ideas",
       how_to_read="The same traits, compared only among ideas of similar genericness (quintiles of closeness to "
       "everything), because generic ideas look 'used' everywhere. An interval that stays above 0.5 points to "
       "durability, below 0.5 to fading. Late bloomers are too few on the pilot to read.")
