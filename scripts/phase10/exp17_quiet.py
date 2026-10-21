"""Experiment 17: quiet foundations -- papers people build on but rarely cite.

Experiment 9: most real uptake is uncited. For each cutoff T (2019-2022) and paper p published up to T-2 (with an
innovation statement):
    reuse      papers by other authors, published after p and up to T, whose method or innovation is within 0.894 of
               p's innovation
    quiet      the part of that reuse whose papers do not cite p
    cites      citations to p from other authors up to T
    quietness  log(1 + quiet) - log(1 + cites); quiet foundations = top tenth of quiet reuse among papers of
               similar genericness (quintiles; generic ideas look "used" everywhere) with at least 3 quiet reuses

Tests:
 A. Still built on: does quiet reuse at T find papers reused by others in T+1..T+3 (>= 3 papers), beyond citations?
    AUC of each ranking within genericness quintiles; and among papers with the same citation count bucket.
 B. Missed by citation lists: share of quiet foundations outside the 50 most cited papers of their community.
Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
st_ids = list(MP) + list(IP)
ST = np.vstack([MV[[MP[x] for x in MP]], IV[[IP[x] for x in IP]]])
st_year = np.array([year.get(x, 9999) for x in st_ids])
st_paper = np.array(st_ids)
rand = ST[np.random.default_rng(0).choice(len(ST), 1000, replace=False)]
cite_pairs = set(zip(refs.citing_id, refs.cited_id))

rows = []
for T in range(2019, 2023):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    cand = [x for x in IP if year.get(x, 9999) <= T - 2]
    V = IV[[IP[x] for x in cand]]
    cites_T = refs[refs.cy <= T]
    for s in range(0, len(cand), 400):
        S = V[s:s + 400] @ ST.T
        for k, x in enumerate(cand[s:s + 400]):
            tm = team.get(x, frozenset())
            close = np.where(S[k] >= C.THR_REUSE)[0]
            users = {}
            for j in close:
                p = st_paper[j]
                if p != x and st_year[j] > year[x] and not (team.get(p, frozenset()) & tm):
                    users[p] = st_year[j]
            before = [p for p, y in users.items() if y <= T]
            after = [p for p, y in users.items() if T + 1 <= y <= T + 3]
            quiet = [p for p in before if (p, x) not in cite_pairs]
            rows.append({"T": T, "paper_id": x, "year": int(year[x]), "X": lab.get(x),
                         "reuse": len(before), "quiet": len(quiet), "after": len(after),
                         "genericness": float((rand @ V[s + k]).mean())})
    c = cites_T.groupby("cited_id").apply(lambda g: sum(1 for a in g.citing_id if not (team.get(a, frozenset())
                                                                                        & team.get(g.name, frozenset()))))
    for r in rows:
        if r["T"] == T:
            r["cites"] = int(c.get(r["paper_id"], 0))
    print(T, len(cand), flush=True)

D = pd.DataFrame(rows)
D["quietness"] = np.log1p(D.quiet) - np.log1p(D.cites)
D["gq"] = D.groupby("T").genericness.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
D["quiet_rank"] = D.groupby(["T", "gq"]).quiet.rank(pct=True)
D["foundation"] = (D.quiet_rank >= 0.9) & (D.quiet >= 3)
D["still_built_on"] = D.after >= 3
D["cite_bucket"] = pd.cut(D.cites, [-1, 0, 2, 5, 15, 1e9], labels=False)
D.to_parquet(C.project(RUN, "backtest") / "exp17_quiet.parquet", index=False)


def strat_auc(df, score, y, by):
    a, w = [], []
    for _, g in df.groupby(by):
        if g[y].nunique() == 2 and g[score].nunique() > 1:
            a.append(roc_auc_score(g[y], g[score]))
            w.append(len(g))
    return np.average(a, weights=w) if a else np.nan


def ci(df, score, y, by, n=300):
    rs = np.random.default_rng(0)
    b = [strat_auc(df.sample(len(df), replace=True, random_state=int(rs.integers(1_000_000_000))), score, y, by)
         for _ in range(n)]
    return f"[{np.nanpercentile(b, 5):.3f}, {np.nanpercentile(b, 95):.3f}]"


res = []
for score in ("quiet", "cites", "reuse", "quietness"):
    for by, name in ((["T", "gq"], "within genericness quintiles"), (["T", "gq", "cite_bucket"],
                                                                     "within genericness x citation buckets")):
        res.append({"ranking": score, "comparison": name, "AUC (still built on in T+1..T+3)": strat_auc(D, score, "still_built_on", by),
                    "90% CI": ci(D, score, "still_built_on", by), "papers x cutoffs": len(D),
                    "base rate": D.still_built_on.mean()})
R = pd.DataFrame(res)
# B: how many quiet foundations a citation list would miss
miss = []
for (T, X), g in D[D.X.notna()].groupby(["T", "X"]):
    top50 = set(g.sort_values("cites", ascending=False).paper_id.head(50))
    f = g[g.foundation]
    if len(f):
        miss.append({"T": T, "X": X, "foundations": len(f), "outside top 50 cited": int((~f.paper_id.isin(top50)).sum())})
M = pd.DataFrame(miss)
summary = pd.DataFrame([{"quiet foundations": int(D.foundation.sum()),
                         "share outside their community's 50 most cited": M["outside top 50 cited"].sum() / max(M.foundations.sum(), 1),
                         "still built on, foundations": D[D.foundation].still_built_on.mean(),
                         "still built on, same genericness and citation bucket, others":
                             D[~D.foundation & D.set_index(["T", "gq", "cite_bucket"]).index.isin(
                                 D[D.foundation].set_index(["T", "gq", "cite_bucket"]).index)].still_built_on.mean()}])
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
print(summary.round(3).to_string(index=False))
C.save(RUN, "exp17_quiet", "rankings", R, title="Quiet foundations: still built on, beyond citations",
       how_to_read="For each cutoff T (2019-2022), papers published up to T-2. **quiet** = papers by other authors that "
       "reuse the idea (method or innovation within 0.894) without citing it, up to T; **cites** = citations by other "
       "authors up to T. Outcome: the idea is reused by at least 3 other papers in T+1..T+3. Rankings are compared "
       "among papers of similar genericness (generic ideas look used everywhere), and also within citation buckets: an "
       "AUC above 0.5 there means quiet reuse finds still-used papers that citation counts do not separate.")
C.save(RUN, "exp17_quiet", "summary", summary, title="Quiet foundations: how many, and how many citation lists miss",
       how_to_read="Quiet foundations = top tenth of quiet reuse among equally generic papers (and >= 3 quiet reuses). "
       "How many a list of each community's 50 most cited papers would miss, and how often foundations are still built "
       "on compared with papers of the same genericness and citation bucket.")

# ---- stricter: credit only the earliest close source, and hold the neighbourhood's prior density fixed
inn_ids = list(IP)
inn_year = np.array([year.get(x, 9999) for x in inn_ids])
rows2 = []
for T in range(2019, 2023):
    sub = D[D["T"] == T].set_index("paper_id")
    pool = np.where(inn_year <= T)[0]
    src_before, src_after = {}, {}
    for j in np.where((st_year >= 2018) & (st_year <= T + 3))[0]:
        p, yp = st_paper[j], st_year[j]
        ok = pool[inn_year[pool] < yp]
        if not len(ok):
            continue
        s = IV[ok] @ ST[j]
        hit = ok[s >= C.THR_REUSE]
        if not len(hit):
            continue
        src = inn_ids[hit[np.argmin(inn_year[hit])]]
        if src not in sub.index or team.get(p, frozenset()) & team.get(src, frozenset()):
            continue
        d = src_before if yp <= T else src_after
        d.setdefault(src, set()).add(p)
    for x in sub.index:
        b = src_before.get(x, set())
        rows2.append({"T": T, "paper_id": x, "attributed": len(b),
                      "attributed_quiet": sum(1 for p in b if (p, x) not in cite_pairs),
                      "attributed_after": len(src_after.get(x, set()))})
    print("attribution", T, flush=True)
A = D.merge(pd.DataFrame(rows2), on=["T", "paper_id"])
# prior density: statements close to the idea published in the 3 years before it (how busy its area already was)
dens = {}
for y0, g in A.drop_duplicates("paper_id").groupby("year"):
    m = (st_year >= y0 - 3) & (st_year < y0)
    if not m.any():
        dens.update({x: 0 for x in g.paper_id})
        continue
    S = IV[[IP[x] for x in g.paper_id]] @ ST[m].T
    dens.update(dict(zip(g.paper_id, (S >= C.THR_REUSE).sum(axis=1).astype(int))))
A["prior_density"] = A.paper_id.map(dens)
A["dq"] = pd.qcut(A.prior_density.rank(method="first"), 3, labels=False)
A["still_attributed"] = A.attributed_after >= 2
res3 = []
for y, yname in (("still_built_on", "reused by >= 3 in T+1..T+3"), ("still_attributed", "earliest source of >= 2 in T+1..T+3")):
    for score in ("attributed_quiet", "attributed", "quiet", "cites"):
        by = ["T", "gq", "dq", "cite_bucket"]
        res3.append({"outcome": yname, "ranking": score, "AUC within genericness x prior density x citation buckets":
                     strat_auc(A, score, y, by), "90% CI": ci(A, score, y, by), "base rate": A[y].mean()})
R3 = pd.DataFrame(res3)
print(R3.round(3).to_string(index=False))
A.to_parquet(C.project(RUN, "backtest") / "exp17_quiet_attributed.parquet", index=False)
C.save(RUN, "exp17_quiet", "rankings_strict", R3, title="Quiet foundations: attributed sources, density held fixed",
       how_to_read="Stricter versions. **attributed** credits a paper only when it is the earliest close source of a later "
       "method or innovation (as the reading paths' method lineage); **attributed_quiet** counts those that do not cite "
       "it. Compared among papers of similar genericness, similar prior density (how many close statements appeared in "
       "the 3 years before it: busy areas stay busy) and similar citations. Outcomes: reused by at least 3 papers in "
       "T+1..T+3, or the earliest source of at least 2.")
