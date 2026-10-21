"""Experiment 29: does an opportunity's age change its chance of paying off?

Every opportunity list rests on dated evidence: a limitation stated in some year, a method a neighbouring community
started using in some year, a paper published in some year. The backtests so far score a list at cutoff T against
T+1..T+3, but never ask whether old evidence converts as well as fresh evidence. Three lists, with their own hit
definitions (3-year horizon):

(a) Method transfer (exp4, cluster level, alpha 0.5 medium): empty problem x method cells at T, ranked by item CF.
    Hit = >= 2 papers in the cell in T+1..T+3 (recipe), also >= 1. Time features: *exposure* (years since one of the
    problem's 10 most similar communities by method profile first used the method; "none" when none has), method
    age (T - median year of the method's papers up to T), method and problem momentum (log of papers in T-2..T over
    T-5..T-3, plus one), problem activity (papers in T-2..T).
(b) Limitation -> innovation (the cutoff runs' bridge_lim_inn lists, 200 per cutoff 2020-2023, scored as production:
    mean top-k similarity x flow penalty^0.5). Time features: age of the limitations and of the innovations (T -
    median year). Hit = strict: a paper of X in T+1..T+3 cites a listed innovation; recipe (Phase 8): or the X -> Y
    citation flow triples.
(c) Paper bridges (exp1 candidate pairs, cutoffs 2020-2023). Time feature: the paper's age. Hit = X cites it in
    T+1..T+3.

Analyses, per list:
- hit rate by age band, over all candidates and over the top of the list;
- observed / expected by band, where expected is the hit rate of the candidate's score quintile at its cutoff: an
  age effect that is not the score's;
- does a model with the time features beat the production score? Leave-one-cutoff-out (LOCO) and chronological
  (train only on cutoffs whose outcome window closed by the test cutoff, T' + 3 <= T);
- horizon (method transfer, cutoffs 2019 and 2020, whose 5-year outcomes are complete): hits within 1, 3, 5 years.

Writes the candidate tables to backtest/exp29_<list>.parquet (experiment 30 reads them).
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import by_band, compare, cosine_rows, mom, pct_rank
from landscape import experiments
from landscape.backtest import Assigner
from landscape.cli import _params

RUN = experiments.run_arg()
WORK = experiments.work_dir(RUN)
EXP = "exp29_opportunity_age"
con = experiments.connect_ro()
VER = _params(con, RUN).embedding.version
papers = con.execute("""SELECT v.paper_id, p.year FROM provenance v JOIN papers p USING (paper_id)
                        WHERE v.run_id = ? AND p.year IS NOT NULL""", [RUN]).df()
year = papers.set_index("paper_id").year
ids = papers.paper_id.tolist()
edges = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance a ON a.paper_id = r.citing_id
                       AND a.run_id = ?""", [RUN]).df()
edges["cy"] = edges.citing_id.map(year)
pd.set_option("display.width", 220)


# ---------------------------------------------------------------- (a) method transfer
mt_rows, horizon_rows = [], []
for T in (2019, 2020, 2021, 2022, 2023):
    A = Assigner(con, f"{RUN}@T{T}", VER)
    both = pd.DataFrame({"P": A.assign("problem", 0.5, ids), "m": A.assign("method", 0.5, ids)}).dropna()
    both["year"] = both.index.map(year)
    Pc, mc = sorted(both.P.unique()), sorted(both.m.unique())
    past = both[both.year <= T]
    M = pd.crosstab(past.P, past.m).reindex(index=Pc, columns=mc, fill_value=0).values.astype(float)

    def future(h):
        f = both[(both.year > T) & (both.year <= T + h)]
        return pd.crosstab(f.P, f.m).reindex(index=Pc, columns=mc, fill_value=0).values
    F3, F1, F5 = future(3), future(1), future(5)
    L = np.log1p(M)
    Sm = cosine_rows(L.T)
    np.fill_diagonal(Sm, 0)
    CF = L @ Sm
    POP = np.outer(M.sum(1), M.sum(0))
    Sp = cosine_rows(L)
    np.fill_diagonal(Sp, -np.inf)
    nn = np.argsort(-Sp, axis=1)[:, :10]
    first = past.groupby(["P", "m"]).year.min().unstack().reindex(index=Pc, columns=mc).values  # first use, NaN = never
    expo_first = np.nanmin(np.where(np.isnan(first[nn]), np.inf, first[nn]), axis=1)          # problems x methods
    share = (M[nn] > 0).mean(1)
    m_med = past.groupby("m").year.median().reindex(mc).values
    cnt = lambda col, keys, lo, hi: both[(both.year >= lo) & (both.year <= hi)][col].value_counts().reindex(keys, fill_value=0).values
    m_mom = mom(cnt("m", mc, T - 2, T), cnt("m", mc, T - 5, T - 3))
    p_rec, p_old = cnt("P", Pc, T - 2, T), cnt("P", Pc, T - 5, T - 3)
    for i, j in zip(*np.where((M == 0) & (CF > 0))):     # production lists only cells with item CF > 0
        ef = expo_first[i, j]
        mt_rows.append({"T": T, "P": Pc[i], "m": mc[j], "cf": CF[i, j], "pop": POP[i, j], "neighbour_share": share[i, j],
                        "exposure_age": T - ef if np.isfinite(ef) else np.nan, "method_age": T - m_med[j],
                        "method_momentum": m_mom[j], "problem_momentum": mom(p_rec[i], p_old[i]),
                        "problem_recent": p_rec[i], "hit2": F3[i, j] >= 2, "hit1": F3[i, j] >= 1,
                        "hit2_1y": F1[i, j] >= 2, "hit2_5y": F5[i, j] >= 2 if T <= 2020 else np.nan})
    print(f"method transfer T={T}: {len(Pc)} problems x {len(mc)} methods")
mt = pd.DataFrame(mt_rows)
mt["no_exposure"] = mt.exposure_age.isna().astype(float)
mt["exposure_age_f"] = mt.exposure_age.fillna(0)
mt["cf_rank"], mt["pop_rank"] = pct_rank(mt.cf, mt["T"]), pct_rank(mt["pop"], mt["T"])
mt["exposure band"] = pd.cut(mt.exposure_age, [-1, 2, 5, 8, 100], labels=["0-2 y", "3-5 y", "6-8 y", "9+ y"]
                             ).cat.add_categories("no neighbour uses it").fillna("no neighbour uses it")
mt["method age band"] = pd.cut(mt.method_age, [-100, 2, 4, 6, 100], labels=["0-2 y", "3-4 y", "5-6 y", "7+ y"])
mt["method momentum band"] = pd.cut(mt.method_momentum, [-10, -0.3, 0.3, 10], labels=["fading", "flat", "rising"])
mt["problem momentum band"] = pd.cut(mt.problem_momentum, [-10, -0.3, 0.3, 10], labels=["fading", "flat", "rising"])
mt.to_parquet(WORK / "exp29_method_transfer.parquet")

bands = []
for band in ("exposure band", "method age band", "method momentum band", "problem momentum band"):
    b = by_band(mt, band, "hit2", "cf", top_k=50, group="P")
    bands.append(b.assign(list="method transfer", feature=band, hit=">= 2 papers in T+1..T+3"))
mt_cmp, mt_coef = compare(mt, "hit2", ["cf_rank", "pop_rank"],
                          ["exposure_age_f", "no_exposure", "method_age", "method_momentum", "problem_momentum",
                           "problem_recent"], 50, "method transfer", "cf")

for T in (2019, 2020):
    g = mt[mt["T"] == T]
    for b, gg in g.groupby("exposure band", observed=True):
        horizon_rows.append({"T": T, "exposure band": b, "candidates": len(gg), "hit within 1 y": gg.hit2_1y.mean(),
                             "hit within 3 y": gg.hit2.mean(), "hit within 5 y": gg.hit2_5y.astype(float).mean(),
                             "share of 5-y hits arriving after year 3": 1 - gg.hit2.sum() / max(gg.hit2_5y.astype(float).sum(), 1)})
    for b, gg in mt[mt["T"] <= 2020].groupby("exposure band", observed=True) if T == 2020 else ():
        h5 = gg.hit2_5y.astype(float)
        horizon_rows.append({"T": "2019 + 2020", "exposure band": b, "candidates": len(gg), "hit within 1 y": gg.hit2_1y.mean(),
                             "hit within 3 y": gg.hit2.mean(), "hit within 5 y": h5.mean(),
                             "share of 5-y hits arriving after year 3": 1 - gg.hit2.sum() / max(h5.sum(), 1)})
horizon = pd.DataFrame(horizon_rows).astype({"T": str})

# ---------------------------------------------------------------- (b) limitation -> innovation
li_rows = []
for T in (2020, 2021, 2022, 2023):
    run = f"{RUN}@T{T}"
    sig = con.execute("SELECT subject, features FROM signals WHERE run_id = ? AND signal_type = 'bridge_lim_inn'",
                      [run]).df()
    lab = Assigner(con, run, VER).assign("problem", 0.5, ids)
    e = edges.assign(f=edges.citing_id.map(lab), t=edges.cited_id.map(lab))
    before = e[(e.cy > T - 3) & (e.cy <= T)].groupby(["f", "t"]).size() / 3
    after = e[(e.cy > T) & (e.cy <= T + 3)].groupby(["f", "t"]).size() / 3
    fut = e[(e.cy > T) & (e.cy <= T + 3)]
    for s_, f_ in zip(sig.subject, sig.features):
        s, f = json.loads(s_), json.loads(f_)
        X, Y = s["from_cluster"], s["to_cluster"]
        lims = sorted({p["limitation"] for p in f["pairs"]})
        inns = sorted({p["innovation"] for p in f["pairs"]})
        ly, iy = [year[x] for x in lims if x in year.index], [year[x] for x in inns if x in year.index]
        strict = bool(fut[(fut.f == X) & fut.cited_id.isin(inns)].shape[0])
        recipe = strict or after.get((X, Y), 0) >= 3 * max(before.get((X, Y), 0), 1 / 3)
        li_rows.append({"T": T, "X": X, "Y": Y, "limitations": lims, "innovations": inns,
                        "sim": f["mean_topk_sim"], "flow_xy": f["flow_xy"], "flow_max": f["flow_max"],
                        "score": f["mean_topk_sim"] * (1 - f["flow_xy"] / (f["flow_max"] + 1)) ** 0.5,
                        "limitation_age": T - np.median(ly), "innovation_age": T - np.median(iy),
                        "newest_evidence_age": T - max(ly + iy), "hit_strict": strict, "hit_recipe": recipe})
li = pd.DataFrame(li_rows)
li["limitation age band"] = pd.cut(li.limitation_age, [-1, 1, 3, 5, 100], labels=["0-1 y", "2-3 y", "4-5 y", "6+ y"])
li["innovation age band"] = pd.cut(li.innovation_age, [-1, 1, 3, 5, 100], labels=["0-1 y", "2-3 y", "4-5 y", "6+ y"])
li["sim_rank"], li["flow_rank"] = pct_rank(li.sim, li["T"]), pct_rank(li.flow_xy / (li.flow_max + 1), li["T"])
li.to_parquet(WORK / "exp29_lim_inn.parquet")
for band in ("limitation age band", "innovation age band"):
    for hit in ("hit_strict", "hit_recipe"):
        b = by_band(li, band, hit, "score", top_k=20, group="X")
        bands.append(b.assign(list="limitation -> innovation", feature=band, hit=hit))
li_cmp, li_coef = compare(li, "hit_strict", ["sim_rank", "flow_rank"], ["limitation_age", "innovation_age"], 20,
                          "limitation -> innovation (strict hit)", "score")

# ---------------------------------------------------------------- (c) paper bridges
pb = pd.read_parquet(WORK / "exp1_pairs.parquet")
pb["age"] = pb["T"] - pb.i_year
pb["lpop"], pb["lcoup"] = np.log1p(pb["pop"]), np.log1p(pb.coupling * 100)
pb["lim_sim3_f"] = pb.lim_sim3.fillna(pb.prob_sim)
pb["age band"] = pd.cut(pb.age, [-1, 2, 5, 10, 200], labels=["0-2 y", "3-5 y", "6-10 y", "11+ y"])
# production model without its age term (LOCO), so the bands show what age adds beyond the other features
feats = ["lpop", "lcoup", "prob_sim", "lim_sim3_f"]
pb["score_no_age"] = np.nan
for T in sorted(pb["T"].unique()):
    tr, te = pb["T"] != T, pb["T"] == T
    s = StandardScaler().fit(pb.loc[tr, feats])
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(s.transform(pb.loc[tr, feats]), pb.loc[tr, "hit"])
    pb.loc[te, "score_no_age"] = m.decision_function(s.transform(pb.loc[te, feats]))
pb["age (years)"] = pb.age.clip(upper=11).map(lambda a: f"{a} y" if a < 6 else ("6-10 y" if a <= 10 else "11+ y"))
pb["age (years)"] = pd.Categorical(pb["age (years)"], ["0 y", "1 y", "2 y", "3 y", "4 y", "5 y", "6-10 y", "11+ y"])
per_cutoff = []
for pop_name, sub in (("all candidates", pb), ("hidden gems (cited <= 1x)", pb[pb["pop"] <= 1])):
    for band in ("age band", "age (years)"):
        b = by_band(sub, band, "hit", "score_no_age", top_k=500, group="X")
        bands.append(b.assign(list=f"paper bridges: {pop_name}", feature=band, hit="X cites it in T+1..T+3"))
    for T, g in sub.groupby("T"):   # is the age pattern the same at every cutoff?
        b = by_band(g, "age band", "hit", "score_no_age", group="X")
        per_cutoff.append(b[["band", "candidates", "hits", "observed / expected"]].assign(list=pop_name, T=T))
per_cutoff = pd.DataFrame(pd.concat(per_cutoff)).pivot_table(index=["list", "band"], columns="T",
                                                              values="observed / expected", observed=True).reset_index()
for b in ("3-5 y", "6-10 y", "11+ y"):   # age bands as dummies: the dip is not linear in age
    pb["band " + b] = (pb["age band"] == b).astype(float)
dummies = ["band 3-5 y", "band 6-10 y", "band 11+ y"]
pb_cmp, pb_coef = compare(pb, "hit", feats, ["age"], 500, "paper bridges", "score_no_age")
gems = pb[pb["pop"] <= 1]
for name, d_, tf in (("paper bridges (age bands)", pb, dummies), ("hidden gems", gems, ["age"]),
                     ("hidden gems (age bands)", gems, dummies)):
    c, k = compare(d_, "hit", feats, tf, 500, name, "score_no_age")
    pb_cmp, pb_coef = pd.concat([pb_cmp, c]), pd.concat([pb_coef, k])

# ---------------------------------------------------------------- save
B = pd.concat(bands, ignore_index=True)
cols = ["list", "feature", "hit", "band"] + [c for c in B.columns if c not in ("list", "feature", "hit", "band")]
B = B[cols]
print(B.round(3).to_string(index=False))
CMP = pd.concat([mt_cmp, li_cmp, pb_cmp], ignore_index=True)
print(CMP.round(3).to_string(index=False))
COEF = pd.concat([mt_coef, li_coef, pb_coef], ignore_index=True)
print(COEF.round(3).to_string(index=False))
print(horizon.round(3).to_string(index=False))
print(per_cutoff.round(3).to_string(index=False))
experiments.save(RUN, EXP, "paper_bridges_by_cutoff", per_cutoff,
                 title="Opportunity age: paper bridges, observed / expected by age band at each cutoff",
                 how_to_read="The age-band table for paper bridges, split by cutoff: a pattern that holds at every "
                 "cutoff is not one cohort's accident.")

experiments.save(RUN, EXP, "age_bands", B, title="Opportunity age: hit rate by the age of the evidence",
                 how_to_read="One row per list, time feature and band. **hit rate** is the share of candidates that "
                 "paid off by the list's own hit definition (3-year horizon). **expected from score** is the hit rate "
                 "of the candidate's score quintile at its cutoff, so **observed / expected** below 1 means the band "
                 "converts worse than its score promises: an age effect the production score does not see (90% "
                 "interval by resampling communities). Method transfer: all empty cells with item CF > 0, cutoffs "
                 "2019-2023; *exposure* = years since one of the 10 communities with the most similar method profile "
                 "first used the method. Limitation -> innovation: the cutoff runs' lists (200 per cutoff, 2020-2023). "
                 "Paper bridges: experiment 1's candidates; the score is the production model without its age term.")
experiments.save(RUN, EXP, "time_features", CMP, title="Opportunity age: do time features beat the production score?",
                 how_to_read="Production score vs logistic models on the score's features with and without the time "
                 "features. **LOCO** fits on the other cutoffs; **chronological** only on cutoffs whose outcomes were "
                 "complete by the test cutoff (T' + 3 <= T), so it lists fewer cutoffs. p@k: 50 for method transfer, "
                 "20 for limitation -> innovation, 500 for paper bridges (about 10 per community).")
experiments.save(RUN, EXP, "coefficients", COEF, title="Opportunity age: standardized coefficients of the time model",
                 how_to_read="Coefficients of the model with time features, fitted for the last test cutoff. Negative "
                 "on an age means older evidence converts less, other features equal.")
experiments.save(RUN, EXP, "horizon", horizon, title="Opportunity age: method transfer hits within 1, 3 and 5 years",
                 how_to_read="Cutoffs 2019 and 2020, whose 5-year outcomes are complete. If hits keep arriving after "
                 "year 3 in a band, a 3-year backtest undercounts it.")
