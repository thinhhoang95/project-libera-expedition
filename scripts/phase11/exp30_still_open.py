"""Experiment 30: is the opportunity still open when the list shows it?

Age alone does not say whether a gap is still there. An item can be stale because the community already took it
before the list was made, or it can be in the middle of being taken. Experiment 9's content measure makes the second
case visible without citations: a statement of community X (method or innovation, 256-d) at least THR close to the
idea's innovation statement (THR = 0.894, the 99th percentile of random innovation-method pairs).

Status at cutoff T, from evidence up to T only (the idea's own paper is left out of X's papers):
- **closed**: X already cites a listed work (limitation -> innovation; paper bridges exclude cited works by
  construction);
- **being taken**: not closed, and X's papers of T-2..T reach the idea (closest statement >= THR) and are closer than
  its papers of T-5..T-3;
- **open**: neither.
Method transfer cells are empty by construction. There the status is the proximity of the problem community's recent
methods to the method community's centroid (mean of the 3 closest papers of T-2..T, alpha 0.5 blended, 768-d), by
tertile at the cutoff, and its direction against T-5..T-3 (approaching / receding).

Outcomes after T (3 years): the list's own hit (citation, or for method transfer >= 2 papers in the cell); content
uptake (experiment 9: X reaches the idea in T+1..T+3, closer than it was in T-2..T); reached (X reaches it in
T+1..T+3 at all).

Tables: the share of each list by status (all candidates and the top of the list); hit rates and observed / expected
(against the candidate's score quintile) by status; list precision counting only open items; whether status adds to
the score (LOCO and chronological).
Needs experiment 29's candidate tables (backtest/exp29_*.parquet). Usage: --run-id; LANDSCAPE_DB may point at a copy.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import by_band, compare
from landscape import embed, experiments
from landscape.backtest import Assigner
from landscape.cli import _params

RUN = experiments.run_arg()
WORK = experiments.work_dir(RUN)
EXP = "exp30_still_open"
THR = 0.894
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


def load(f):
    i, v = embed.load_vectors(f, ids=ids, version=VER)
    return {x: k for k, x in enumerate(i)}, embed.truncate(v, 256)


IN, ME = load("innovation"), load("method")


def stack(pids):
    """Method and innovation statements of a set of papers, stacked (256-d)."""
    out = [V[[P_[x] for x in pids if x in P_]] for P_, V in ((ME[0], ME[1]), (IN[0], IN[1]))]
    return np.vstack(out) if sum(len(o) for o in out) else np.zeros((0, 256), np.float32)


def closest(S, V):
    """For each row of V, the highest cosine to any statement in S (-1 when S is empty)."""
    return (V @ S.T).max(1) if len(S) and len(V) else np.full(len(V), -1.0)


def periods(lab, T):
    yr = lab.index.map(year)
    return {k: lab[(yr >= lo) & (yr <= hi)] for k, (lo, hi) in
            {"old": (T - 5, T - 3), "rec": (T - 2, T), "fut": (T + 1, T + 3)}.items()}


def status_of(r, o, closed=False):
    return "closed" if closed else ("being taken" if (r >= THR and r > o) else "open")


share_rows, bands, cmps, prec_rows = [], [], [], []

# ---------------------------------------------------------------- limitation -> innovation
li = pd.read_parquet(WORK / "exp29_lim_inn.parquet")
out = []
for T, g in li.groupby("T"):
    lab = Assigner(con, f"{RUN}@T{T}", VER).assign("problem", 0.5, ids)
    per = periods(lab, T)
    cites_by = edges[edges.cy <= T].assign(f=lambda e: e.citing_id.map(lab))
    for X, gx in g.groupby("X"):
        st = {k: v[v == X].index for k, v in per.items()}
        for idx, row in gx.iterrows():
            inns = [x for x in row.innovations if x in IN[0]]
            V = IN[1][[IN[0][x] for x in inns]]
            r, o, f = (closest(stack([p for p in st[k] if p not in set(row.innovations)]), V) for k in ("rec", "old", "fut"))
            closed = bool(cites_by[(cites_by.f == X) & cites_by.cited_id.isin(row.innovations)].shape[0])
            taken = any(ri >= THR and ri > oi for ri, oi in zip(r, o))
            out.append({"idx": idx, "status": "closed" if closed else ("being taken" if taken else "open"),
                        "content_uptake": bool(np.any((f >= THR) & (f > r))), "reached": bool(np.any(f >= THR))})
li = li.join(pd.DataFrame(out).set_index("idx"))
li["closed"], li["being_taken"] = (li.status == "closed").astype(float), (li.status == "being taken").astype(float)
li["top20"] = li.groupby("T").score.rank(ascending=False, method="first") <= 20
for hit in ("hit_strict", "hit_recipe", "content_uptake"):
    bands.append(by_band(li, "status", hit, "score", top_k=20, group="X").assign(list="limitation -> innovation", hit=hit))
for scope, d in (("all 200", li), ("top 20", li[li.top20])):
    share_rows.append(d.status.value_counts(normalize=True).rename("share").reset_index().assign(
        list="limitation -> innovation", scope=scope))
for hit in ("hit_strict", "hit_recipe"):
    for scope, d in (("top 20 as listed", li[li.top20]), ("top 20 among items open or being taken",
                     li[li.status != "closed"].assign(r=lambda x: x.groupby("T").score.rank(ascending=False, method="first"))
                     .query("r <= 20"))):
        prec_rows.append({"list": "limitation -> innovation", "hit": hit, "scope": scope,
                          "precision": d.groupby("T")[hit].mean().mean(), "items": len(d)})
c, _ = compare(li, "hit_strict", ["sim_rank", "flow_rank"], ["closed", "being_taken"], 20,
               "limitation -> innovation (strict hit)", "score")
cmps.append(c)
li.drop(columns=["limitations", "innovations"]).to_parquet(WORK / "exp30_lim_inn.parquet")
print("limitation -> innovation done")

# ---------------------------------------------------------------- paper bridges
pb = pd.read_parquet(WORK / "exp1_pairs.parquet")
pb["age"] = pb["T"] - pb.i_year
pb["lpop"], pb["lcoup"] = np.log1p(pb["pop"]), np.log1p(pb.coupling * 100)
pb["lim_sim3_f"] = pb.lim_sim3.fillna(pb.prob_sim)
feats = ["lpop", "lcoup", "prob_sim", "lim_sim3_f", "age"]
parts = []
for T, g in pb.groupby("T"):
    lab = Assigner(con, f"{RUN}@T{T}", VER).assign("problem", 0.5, ids)
    per = periods(lab, T)
    for X, gx in g.groupby("X"):
        st = {k: stack(v[v == X].index) for k, v in per.items()}
        has = gx.i.isin(IN[0]).values
        V = IN[1][[IN[0][x] for x in gx.i[has]]]
        r, o, f = (np.full(len(gx), np.nan) for _ in range(3))
        for arr, k in ((r, "rec"), (o, "old"), (f, "fut")):
            arr[has] = closest(st[k], V)
        parts.append(pd.DataFrame({"r": r, "o": o, "f": f}, index=gx.index))
    print(f"paper bridges T={T}")
pb = pb.join(pd.concat(parts))
pb = pb[pb.r.notna()].copy()      # candidates without an innovation statement have no status
pb["status"] = np.where((pb.r >= THR) & (pb.r > pb.o), "being taken", "open")
pb["being_taken"], pb["closeness"] = (pb.status == "being taken").astype(float), pb.r
pb["content_uptake"] = (pb.f >= THR) & (pb.f > pb.r)
# the production model's score (all features, LOCO), as the expected-rate reference
pb["score"] = np.nan
for T in sorted(pb["T"].unique()):
    tr, te = pb["T"] != T, pb["T"] == T
    s = StandardScaler().fit(pb.loc[tr, feats])
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(s.transform(pb.loc[tr, feats]), pb.loc[tr, "hit"])
    pb.loc[te, "score"] = m.decision_function(s.transform(pb.loc[te, feats]))
pb["top10"] = pb.groupby(["T", "X"]).score.rank(ascending=False, method="first") <= 10
for pop_name, sub in (("paper bridges", pb), ("hidden gems (cited <= 1x)", pb[pb["pop"] <= 1])):
    for hit in ("hit", "content_uptake"):
        bands.append(by_band(sub, "status", hit, "score", group="X").assign(list=pop_name, hit=hit))
    for scope, d in (("all candidates", sub), ("top 10 per community", sub[sub.top10])):
        share_rows.append(d.status.value_counts(normalize=True).rename("share").reset_index().assign(
            list=pop_name, scope=scope))
    c, _ = compare(sub, "hit", feats, ["being_taken", "closeness"], 500, pop_name, "score")
    cmps.append(c)
pb[["T", "X", "i", "pop", "age", "hit", "score", "r", "o", "f", "status", "content_uptake"]].to_parquet(
    WORK / "exp30_paper_bridges.parquet")

# ---------------------------------------------------------------- method transfer
mt = pd.read_parquet(WORK / "exp29_method_transfer.parquet")
mpid, mvec = embed.blended("method", 0.5, None, dim=768, version=VER)
mpos = {x: k for k, x in enumerate(mpid)}
add = []
for T, g in mt.groupby("T"):
    A = Assigner(con, f"{RUN}@T{T}", VER)
    lab = A.assign("problem", 0.5, ids)
    per = periods(lab, T)
    cent = {m: np.asarray(A.cent[m], dtype=np.float32) for m in g.m.unique()}
    for P, gp in g.groupby("P"):
        sims = {}
        for k in ("rec", "old"):
            pp = [mpos[x] for x in per[k][per[k] == P].index if x in mpos]
            sims[k] = mvec[pp] if pp else None
        for idx, row in gp.iterrows():
            top3 = lambda S: float(np.sort(S @ cent[row.m])[-3:].mean()) if S is not None else np.nan
            add.append({"idx": idx, "proximity": top3(sims["rec"]), "proximity_before": top3(sims["old"])})
mt = mt.join(pd.DataFrame(add).set_index("idx"))
mt["approach"] = (mt.proximity - mt.proximity_before).fillna(0)
mt["proximity tertile"] = mt.groupby("T").proximity.transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                                                               labels=["far", "middle", "near"]))
mt["status"] = mt["proximity tertile"].astype(str) + ", " + np.where(mt.approach > 0, "approaching", "receding")
mt["top50"] = mt.groupby("T").cf.rank(ascending=False, method="first") <= 50
for band in ("status", "proximity tertile"):
    bands.append(by_band(mt, band, "hit2", "cf", top_k=50, group="P").assign(list="method transfer", hit="hit2 (" + band + ")"))
for scope, d in (("all candidates", mt), ("top 50", mt[mt.top50])):
    share_rows.append(d.status.value_counts(normalize=True).rename("share").reset_index().assign(
        list="method transfer", scope=scope))
mt["proximity_f"] = mt.proximity.fillna(mt.proximity.mean())
c, _ = compare(mt, "hit2", ["cf_rank", "pop_rank"], ["proximity_f", "approach"], 50, "method transfer", "cf")
cmps.append(c)

# ---------------------------------------------------------------- save
S = pd.concat(share_rows, ignore_index=True)[["list", "scope", "status", "share"]]
B = pd.concat(bands, ignore_index=True)
B = B[["list", "hit", "band"] + [c for c in B.columns if c not in ("list", "hit", "band")]]
CMP = pd.concat(cmps, ignore_index=True)
PREC = pd.DataFrame(prec_rows)
for t in (S, B, PREC, CMP):
    print(t.round(3).to_string(index=False))
experiments.save(RUN, EXP, "list_status", S, title="Still open: share of each list by status at its cutoff",
                 how_to_read="**closed**: the community already cites a listed work; **being taken**: its recent papers "
                 "already reach the idea and are closer than three to five years before (experiment 9's content "
                 "measure, no citation needed); **open**: neither. Method transfer cells are empty by construction; "
                 "their status is how close the problem community's recent methods are to the method (tertile at the "
                 "cutoff) and whether they moved towards it.")
experiments.save(RUN, EXP, "status_outcomes", B, title="Still open: what happens next, by status",
                 how_to_read="Hit rate after the cutoff by status, and observed / expected against the candidate's "
                 "score quintile (90% interval by resampling communities). hit_strict / hit: the list's citation hit; "
                 "hit_recipe: Phase 8's (citations or flow triples); content_uptake: experiment 9's realized uptake.")
experiments.save(RUN, EXP, "precision_open_only", PREC,
                 title="Still open: limitation -> innovation precision with closed items removed",
                 how_to_read="Precision of the top 20 per cutoff as the list stands, and after removing items the "
                 "community already cites (the next items move up). A drop means closed items were carrying the "
                 "backtest.")
experiments.save(RUN, EXP, "status_features", CMP, title="Still open: does the status add to the score?",
                 how_to_read="As experiment 29's model table, with the status (closed, being taken, closeness; for "
                 "method transfer proximity and approach) as the added features.")
