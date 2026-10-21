"""Experiment 16: long-standing open questions. Which questions keep being asked without an answer that ends them?

For each cutoff T (2019..2022), using statements published up to T only:

    question groups   problem statements (problem facet, 256-d) from 2010..T, one record per normalized title, grouped by
                      average linkage at cosine 0.93 (just above the 99.9th percentile of random pairs, 0.923). A group
                      needs >= 3 statements, >= 2 distinct years and >= 2 papers with no author in common; the 20% most
                      generic groups (closest to the mean problem statement) are dropped.
    active            asked at least once in T-2..T.
    teams             unrelated teams asking it: papers counted greedily, skipping any that shares an author with one
                      already counted.
    standing          active, first asked at least 4 years before T, by at least 3 unrelated teams.

"Answered" cannot be read from abstracts: for the median question, the best innovation statement from outside the
group is closer to the question (cosine 0.927) than the askers' own innovations (0.902), so closeness of wording does
not certify an answer. The diagnostics table reports this; the live list shows the closest proposals as attempts,
not as answers.

Outcome: still asked in T+1..T+3, i.e. a later problem statement whose mean cosine to the group's members is >= 0.93.
Validation: does the breadth of askers (teams), and the standing label, predict being still asked beyond the group's
age, size and recent recurrence? Compared like for like (size x recent strata) and with a logistic model, intervals
resampling question groups.
Also writes the recurring questions of the latest complete year (asked by >= 3 unrelated teams) to
data/projects/<run>/backtest/exp16_recurring_questions.parquet: descriptive only, not used by the product, since
neither an answer nor persistence beyond recent activity could be validated.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C

RUN = C.run_arg()
GROUP_COS = 0.93
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
papers["tkey"] = papers.title.fillna("").str.lower().map(lambda t: re.sub(r"[^a-z0-9]", "", t))
dedup = papers.sort_values("year").drop_duplicates("tkey")
dedup = dedup[(dedup.tkey.str.len() >= 12) | (dedup.tkey == "")]
ids = dedup.paper_id.tolist()
PP, PV = C.vectors("problem", ids)
IP, IV = C.vectors("innovation", ids)
gmean = C.unit(PV.mean(axis=0))
rng = np.random.default_rng(0)
ia, pb = rng.integers(0, len(IV), 20000), rng.integers(0, len(PV), 20000)
ANSWER_COS = float(np.quantile(np.sum(IV[ia] * PV[pb], axis=1), 0.9999))
print(f"attempt threshold (99.99th pct of random innovation-problem cosine): {ANSWER_COS:.3f}")
text = con.execute("SELECT paper_id, text FROM facets WHERE facet = 'problem'").df().set_index("paper_id").text
title = papers.set_index("paper_id").title


def groups_at(T):
    pid = [x for x in ids if x in PP and 2010 <= year[x] <= T]
    V = PV[[PP[x] for x in pid]]
    lab = fcluster(linkage(V, method="average", metric="cosine"), t=1 - GROUP_COS, criterion="distance")
    out = []
    for g in np.unique(lab):
        k = np.where(lab == g)[0]
        mem = [pid[i] for i in k]
        yrs = [year[x] for x in mem]
        if len(mem) < 3 or len(set(yrs)) < 2:
            continue
        teams = [team.get(x, frozenset()) for x in mem]
        if not any(not (teams[a] & teams[b]) for a in range(len(mem)) for b in range(a + 1, len(mem))):
            continue
        cent = C.unit(V[k].mean(axis=0))
        out.append({"members": mem, "vecs": V[k], "cent": cent, "first": min(yrs), "last": max(yrs), "years": yrs})
    gen = np.array([float(g["cent"] @ gmean) for g in out])
    cut = np.quantile(gen, 0.8) if len(gen) else 1.0
    return [g for g, s in zip(out, gen) if s <= cut]


def n_teams(mem):
    seen, n = set(), 0
    for x in sorted(mem, key=lambda x: year[x]):
        t = team.get(x, frozenset())
        if not (t & seen):
            n += 1
        seen |= t
    return n


def describe(g, T):
    yrs = pd.Series(g["years"])
    recent = int((yrs >= T - 2).sum())
    mem = set(g["members"])
    askers = set().union(*[team.get(x, frozenset()) for x in mem])
    inn = [x for x in ids if x in IP and g["first"] <= year[x] <= T and x not in mem
           and not (team.get(x, frozenset()) & askers)]
    s = IV[[IP[x] for x in inn]] @ g["cent"] if inn else np.array([])
    own = [x for x in mem if x in IP]
    own_med = float(np.median(IV[[IP[x] for x in own]] @ g["cent"])) if own else np.nan
    hits = [(year[inn[i]], inn[i], float(s[i])) for i in np.where(s >= ANSWER_COS)[0]]
    answered = bool(hits)
    first_ans = min(h[0] for h in hits) if hits else None
    closed = False
    if answered:
        before = ((yrs >= g["first"]) & (yrs <= first_ans)).sum() / max(first_ans - g["first"] + 1, 1)
        after = (yrs > first_ans).sum() / max(T - first_ans, 1)
        closed = (T > first_ans) and after < 0.5 * before
    best = sorted(hits, key=lambda h: -h[2])[:3]
    teams = n_teams(g["members"])
    return {"size": len(g["members"]), "age": T - g["first"], "recent": recent, "active": recent > 0,
            "teams": teams, "standing": recent > 0 and T - g["first"] >= 4 and teams >= 3,
            "answered": answered, "first_answer": first_ans, "closed": closed, "attempts": [h[1] for h in best],
            "best_outside": float(s.max()) if len(s) else np.nan, "own_median": own_med}


rows = []
for T in range(2019, 2023):
    G = groups_at(T)
    later = [x for x in ids if x in PP and T < year[x] <= T + 3]
    Lv = PV[[PP[x] for x in later]] if later else np.zeros((0, PV.shape[1]))
    for gi, g in enumerate(G):
        d = describe(g, T)
        if not d["active"]:
            continue
        still = bool(len(Lv)) and bool(((Lv @ g["vecs"].T).mean(axis=1) >= GROUP_COS).any())
        rows.append({"T": T, "group": f"{T}:{gi}", **{k: v for k, v in d.items() if k != "attempts"},
                     "still_asked": still})
    print(T, len(G), "groups,", sum(1 for r in rows if r["T"] == T), "active", flush=True)
D = pd.DataFrame(rows)
D["l_size"], D["l_recent"], D["l_teams"] = np.log(D["size"]), np.log1p(D.recent), np.log(D.teams)
DIAG = pd.DataFrame([{"active question groups": len(D), "with an outside attempt above threshold": float(D.answered.mean()),
                      "median best outside innovation cosine": float(D.best_outside.median()),
                      "median askers' own innovation cosine": float(D.own_median.median()),
                      "attempt threshold": ANSWER_COS}])

# like for like: size x recent strata (age is part of the standing label)
D["size_bin"] = pd.cut(D["size"], [2, 4, 8, 1000], labels=["3-4", "5-8", "9+"])
D["recent_bin"] = pd.cut(D.recent, [0, 1, 3, 1000], labels=["1", "2-3", "4+"])
D["many_teams"] = D.teams >= D.groupby("T").teams.transform("median")
tab = []
for flag, label in (("standing", "standing (4+ years, 3+ unrelated teams, asked recently)"),
                    ("many_teams", "asked by more unrelated teams than the median")):
    diffs, w, bs = [], [], []
    for _, g in D.groupby(["T", "size_bin", "recent_bin"], observed=True):
        a, b = g[g[flag]], g[~g[flag]]
        if len(a) and len(b):
            diffs.append(a.still_asked.mean() - b.still_asked.mean())
            w.append(len(a))
    for _ in range(500):
        smp = D.iloc[rng.integers(0, len(D), len(D))]
        dd, ww = [], []
        for _, g in smp.groupby(["T", "size_bin", "recent_bin"], observed=True):
            a, b = g[g[flag]], g[~g[flag]]
            if len(a) and len(b):
                dd.append(a.still_asked.mean() - b.still_asked.mean())
                ww.append(len(a))
        if ww:
            bs.append(np.average(dd, weights=ww))
    tab.append({"label": label, "active groups": len(D), "labelled": int(D[flag].sum()),
                "still asked if labelled (precision)": float(D[D[flag]].still_asked.mean()),
                "still asked otherwise": float(D[~D[flag]].still_asked.mean()), "base rate": float(D.still_asked.mean()),
                "matched difference (same size, recent activity)": float(np.average(diffs, weights=w)) if w else np.nan,
                "90% CI": f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]" if bs else ""})
TA = pd.DataFrame(tab)


def loco(feats):
    out = pd.Series(np.nan, index=D.index)
    for T in sorted(D["T"].unique()):
        tr, te = D["T"] != T, D["T"] == T
        if D.loc[tr, "still_asked"].nunique() < 2:
            continue
        sc_ = StandardScaler().fit(D.loc[tr, feats])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc_.transform(D.loc[tr, feats]),
                                                                           D.loc[tr, "still_asked"])
        out[te] = m.decision_function(sc_.transform(D.loc[te, feats]))
    return out


D["standing_f"] = D.standing.astype(float)
base = ["age", "l_size", "l_recent"]
sc = {"age + size + recent recurrence": loco(base), "+ unrelated teams asking": loco(base + ["l_teams"]),
      "+ standing label": loco(base + ["standing_f"])}
ok = sc["age + size + recent recurrence"].notna()
y = D.still_asked[ok].astype(int).to_numpy()
b = sc["age + size + recent recurrence"][ok].to_numpy()
mrows = []
for k, sv in sc.items():
    sv = sv[ok].to_numpy()
    g = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            g.append(roc_auc_score(y[i], sv[i]) - roc_auc_score(y[i], b[i]))
    mrows.append({"model": k, "AUC": roc_auc_score(y, sv), "gain vs base": roc_auc_score(y, sv) - roc_auc_score(y, b),
                  "90% CI": f"[{np.percentile(g, 5):+.3f}, {np.percentile(g, 95):+.3f}]", "groups": int(ok.sum()),
                  "base rate": float(y.mean())})
TM = pd.DataFrame(mrows)
pd.set_option("display.width", 250)
print(DIAG.round(3).to_string(index=False))
print(TA.round(3).to_string(index=False))
print(TM.round(3).to_string(index=False))
C.save(RUN, "exp16_open_questions", "answer_detection", DIAG, title="Open questions: can an answer be read from abstracts?",
       how_to_read="For each active question group, the closest innovation statement from outside the group (other teams) "
       "against the askers' own innovations. When outside proposals are routinely closer to the question than the "
       "askers' own work, closeness of wording cannot certify that a question was answered.")
C.save(RUN, "exp16_open_questions", "standing_vs_others", TA, title="Standing questions: still asked three years later?",
       how_to_read="Question groups (problem statements grouped at cosine 0.93, at least 3 statements over 2+ years from "
       "unrelated teams; the most generic fifth dropped), active at cutoff T. **Standing** = first asked 4+ years before, "
       "by 3+ unrelated teams. **Still asked** = a statement in T+1..T+3 joins the group. The matched difference "
       "compares groups of the same size and recent activity.")
C.save(RUN, "exp16_open_questions", "models", TM, title="Standing questions: does breadth of askers add to recurrence?",
       how_to_read="Leave-one-cutoff-out logistic models of 'still asked in T+1..T+3'. The base knows the question's age, "
       "size and how often it was asked in T-2..T. A gain with an interval above zero means how many unrelated teams ask "
       "a question carries information beyond its recent activity.")

# ---------------------------------------------------------------- live list
T = int(papers.year.max()) - 1
G = groups_at(T)
live = []
for g in G:
    d = describe(g, T)
    if not d["active"] or d["teams"] < 3:
        continue
    mem = sorted(g["members"], key=lambda x: year[x])
    k = int(np.argmax(g["vecs"] @ g["cent"]))
    live.append({"question": text.get(g["members"][k]), "first_asked": g["first"], "last_asked": g["last"],
                 "open_years": T - g["first"], "times_asked": d["size"], "asked_recently": d["recent"],
                 "unrelated_teams": d["teams"], "standing": d["standing"], "oldest_paper": mem[0], "newest_paper": mem[-1],
                 "papers": mem, "attempts": d["attempts"], "as_of": T,
                 "community": None})
L = pd.DataFrame(live)
if len(L):
    lab = C.assign(con, RUN, RUN, sorted({x for m in L.papers for x in m}))
    L["community"] = [pd.Series([lab.get(x) for x in m]).mode().iloc[0] if any(lab.get(x) for x in m) else None
                      for m in L.papers]
    L = L.sort_values(["standing", "unrelated_teams", "open_years"], ascending=False)
    L.to_parquet(C.project(RUN, "backtest") / "exp16_recurring_questions.parquet", index=False)
    print(f"live open questions as of {T}: {len(L)}")
    print(L.head(10)[["question", "first_asked", "times_asked", "unrelated_teams"]].to_string(index=False))
