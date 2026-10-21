"""Experiment 21: gateway reads -- papers whose readers go on to join their community.

Communities: problem communities of the 2021 cutoff run (alpha 0.5, medium). For a paper p in community Y, an
*outsider citer* is an author citing p (in a paper of theirs) with no earlier paper in Y; they *entered* if they publish
in Y afterwards (in a paper that is not the citing one).
    train   outsider citers whose first citation of p is in 2018-2020; entered by the end of 2021
    test    outsider citers whose first citation of p is in 2021-2022; entered by the end of 2025
Gateway score = (entered + 1) / (outsider citers + 2) in the train window, known in 2021. Does it predict the test
entry rate of new citers, beyond popularity (citations to p up to 2021) and community size? Papers with >= 3 test
citers; rank correlations with 90% intervals; and at the citer level, the AUC of each paper score for whether a new
citer entered.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
lab = C.assign(con, RUN, f"{RUN}@T2021", ids)
a2 = au.assign(Y=au.paper_id.map(lab)).dropna(subset=["Y"])
first_in = a2.groupby(["author_id", "Y"]).year.min()
papers_in = a2.groupby(["author_id", "Y"]).apply(lambda g: sorted(zip(g.year, g.paper_id))).to_dict()
cit = refs[refs.cited_id.isin(set(lab.index))].merge(au[["paper_id", "author_id"]], left_on="citing_id",
                                                     right_on="paper_id").drop(columns="paper_id")
cit["Y"] = cit.cited_id.map(lab)
cit = cit.dropna(subset=["Y", "cy"])
cit = cit[[a not in team.get(p, frozenset()) for a, p in zip(cit.author_id, cit.cited_id)]]  # not the paper's authors
first_cite = cit.sort_values("cy").drop_duplicates(["author_id", "cited_id"])


def entered(a, Y, after_year, by_year, citing):
    for y, p in papers_in.get((a, Y), []):
        if after_year <= y <= by_year and p != citing:
            return True
    return False


rows = []
for r in first_cite.itertuples():
    f = first_in.get((r.author_id, r.Y))
    if f is not None and f < r.cy:
        continue  # already in the community: not an outsider
    if 2018 <= r.cy <= 2020:
        rows.append({"p": r.cited_id, "Y": r.Y, "window": "train", "entered": entered(r.author_id, r.Y, r.cy, 2021, r.citing_id)})
    elif 2021 <= r.cy <= 2022:
        rows.append({"p": r.cited_id, "Y": r.Y, "window": "test", "entered": entered(r.author_id, r.Y, r.cy, 2025, r.citing_id)})
Z = pd.DataFrame(rows)
tr = Z[Z.window == "train"].groupby("p").entered.agg(["sum", "size"])
te = Z[Z.window == "test"].groupby("p").entered.agg(["mean", "size"])
pop = refs[refs.cy <= 2021].groupby("cited_id").size()
size = lab.value_counts()
P = te[te["size"] >= 3].join(tr, how="inner", rsuffix="_train")
P["gateway"] = (P["sum"] + 1) / (P["size_train"] + 2)
P["popularity"] = P.index.map(pop).fillna(0)
P["community_size"] = P.index.map(lab).map(size)
P = P.rename(columns={"mean": "test_entry_rate", "size": "test_citers"})
res = []
for s in ("gateway", "popularity", "community_size"):
    rs = np.random.default_rng(0)
    d = P[[s, "test_entry_rate"]].astype(float)
    bs = [spearmanr(*d.sample(len(d), replace=True, random_state=int(rs.integers(1_000_000_000))).to_numpy().T).statistic
          for _ in range(500)]
    res.append({"level": "paper", "score": s, "value": spearmanr(d[s], d.test_entry_rate).statistic,
                "measure": "rank correlation with new citers' entry rate", "90% CI": f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]",
                "papers / citers": len(P)})
T2 = Z[(Z.window == "test") & Z.p.isin(P.index)].copy()
for s in ("gateway", "popularity", "community_size"):
    T2[s] = T2.p.map(P[s])
    res.append({"level": "citer", "score": s, "value": roc_auc_score(T2.entered, T2[s]), "measure": "AUC: did a new citer enter",
                "90% CI": "", "papers / citers": len(T2)})
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(f"train outsider citers {int((Z.window == 'train').sum())}, test {int((Z.window == 'test').sum())}, "
      f"test entry rate {Z[Z.window == 'test'].entered.mean():.3f}")
print(R.round(3).to_string(index=False))
P.reset_index().rename(columns={"index": "paper_id", "p": "paper_id"}).to_parquet(C.project(RUN, "backtest") / "exp21_gateway.parquet", index=False)
C.save(RUN, "exp21_gateway", "results", R, title="Gateway reads: papers whose readers join their community",
       how_to_read="Communities of the 2021 cutoff. An outsider citer cites a paper of a community they have not published "
       "in; they *entered* if they publish there later. The **gateway** score of a paper is its outsider citers' entry rate "
       "in 2018-2020 (known in 2021); does it predict the entry rate of its new outsider citers in 2021-2022 (entries "
       "through 2025)? Compared with the paper's popularity and its community's size; papers with at least 3 new citers.")

# ---- per reader, within the same community and popularity tercile: is it more than "less popular papers"?
T2["Y"] = T2.p.map(lab)
T2["pop_t"] = pd.qcut(T2.popularity.rank(method="first"), 3, labels=False)


def strat(df):
    a, w = [], []
    for _, g in df.groupby(["Y", "pop_t"]):
        if g.entered.nunique() == 2 and g.gateway.nunique() > 1:
            a.append(roc_auc_score(g.entered, g.gateway))
            w.append(len(g))
    return np.average(a, weights=w)


rs = np.random.default_rng(0)
groups = list(T2.reset_index(drop=True).groupby("p").indices.values())
T2r = T2.reset_index(drop=True)
bs = [strat(T2r.iloc[np.concatenate([groups[j] for j in rs.integers(0, len(groups), len(groups))])]) for _ in range(300)]
extra = pd.DataFrame([{"level": "citer", "score": "gateway, within community x popularity tercile", "value": strat(T2),
                       "measure": "AUC: did a new citer enter", "90% CI": f"[{np.percentile(bs, 5):.3f}, {np.percentile(bs, 95):.3f}]",
                       "papers / citers": len(T2)}])
R = pd.concat([R, extra], ignore_index=True)
print(extra.round(3).to_string(index=False))
C.save(RUN, "exp21_gateway", "results", R, title="Gateway reads: papers whose readers join their community",
       how_to_read="Communities of the 2021 cutoff. An outsider citer cites a paper of a community they have not published "
       "in; they *entered* if they publish there later. The **gateway** score of a paper is its outsider citers' entry rate "
       "in 2018-2020 (known in 2021); does it predict the entry rate of its new outsider citers in 2021-2022 (entries "
       "through 2025)? Paper level: rank correlation over papers with at least 3 new citers. Citer level: AUC for whether "
       "a new citer entered, also within the same community and popularity tercile (resampling papers).")
