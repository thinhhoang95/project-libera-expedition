"""Experiment 25: what insiders read that newcomers miss.

For each community X (problem communities of the cutoff-T run, alpha 0.5, medium) and cutoff T (2019-2022), from
X's papers of T-3..T-1 only:

    insider papers    a paper with an author who already had 2+ papers in X before it
    newcomer papers   an author's first paper in X, with no insider co-author
    classics          the 10 works X's papers cited most
    insider-only      among works cited by at least 3 insider papers and not classics, the 10 with the highest ratio of
                      citation rate in insider papers to citation rate in newcomer papers

Entries (experiment 12's table: an author's first paper in X published in T, with controls and outcomes) are compared
on whether the entry paper cites any insider-only work, holding fixed the controls of experiment 12 (strategy,
insider co-author, newcomer to the map, team and community size, route, references inside the map, year) and whether
it cites any classic. Outcomes: the author stayed (another paper in X in T+1..T+3), the paper was taken up by others
(non-self citations, top quarter of year-T papers), X's later work moved towards it (experiment 9). Adjusted
differences with 90% intervals resampling communities.

Check that insider-only works carry knowledge, not habit: for those in the map, the share that are a close source
(cosine >= 0.894) of X's methods and innovations from T-2..T, against classics and other works X cites.
Also writes the live lists (latest complete year) to data/projects/<run>/backtest/exp25_insider_reads_live.parquet.
Needs scripts/phase10/exp12_entry.py output. Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a copy.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
title = papers.set_index("paper_id").title
E = pd.read_parquet(C.project(RUN, "backtest") / "exp12_entries.parquet")
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)
refs_by = refs.groupby("citing_id").cited_id.apply(set)


def stack(pids):
    a = [MV[MP[x]] for x in pids if x in MP] + [IV[IP[x]] for x in pids if x in IP]
    return np.vstack(a) if a else np.zeros((0, MV.shape[1]), np.float32)


def lists_at(lab: pd.Series, T: int, k: int = 10):
    """Insider-only and classic lists per community, from X's papers of T-3..T-1; plus a source-of-methods check."""
    a2 = au.assign(X=au.paper_id.map(lab)).dropna(subset=["X"]).sort_values("year")
    out, check = {}, []
    for X, g in a2.groupby("X"):
        win = g[(g.year >= T - 3) & (g.year <= T - 1)].paper_id.unique()
        prior = g[g.year < T]
        n_before = {}
        ins_p, new_p = [], []
        for p in win:
            yp = year[p]
            authors = g[g.paper_id == p].author_id
            counts = [prior[(prior.author_id == a) & (prior.year < yp)].paper_id.nunique() for a in authors]
            if any(c >= 2 for c in counts):
                ins_p.append(p)
            elif all(c == 0 for c in counts):
                new_p.append(p)
        if len(ins_p) < 5 or len(new_p) < 5:
            continue
        cited_all = pd.Series([w for p in win for w in refs_by.get(p, ())]).value_counts()
        classics = cited_all.index[:k].tolist()
        ins = pd.Series([w for p in ins_p for w in refs_by.get(p, ())]).value_counts()
        new = pd.Series([w for p in new_p for w in refs_by.get(p, ())]).value_counts()
        cand = ins[(ins >= 3) & ~ins.index.isin(classics)]
        if cand.empty:
            continue
        score = ((cand + 0.5) / (len(ins_p) + 1)) / ((new.reindex(cand.index).fillna(0) + 0.5) / (len(new_p) + 1))
        insider_only = score.sort_values(ascending=False).index[:k].tolist()
        out[X] = {"insider_only": insider_only, "classics": classics, "n_insider_papers": len(ins_p),
                  "n_newcomer_papers": len(new_p), "ins_counts": ins, "new_counts": new}
        # knowledge check: is the work a close source of X's current methods / innovations?
        cur = stack([p for p in lab.index[lab.values == X] if T - 2 <= year.get(p, 0) <= T])
        others = [w for w in cited_all.index if w not in set(classics) | set(insider_only)][:50]
        for kind, lst in (("insider-only", insider_only), ("classic", classics), ("other cited", others)):
            for w in lst:
                if w in IP and len(cur):
                    check.append({"T": T, "X": X, "list": kind, "source_of_current": bool((cur @ IV[IP[w]] >= C.THR_REUSE).any())})
    return out, pd.DataFrame(check)


rows, checks = [], []
for T in sorted(E["T"].unique()):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    L, chk = lists_at(lab, T)
    checks.append(chk)
    e = E[(E["T"] == T) & E.X.isin(L)]
    for r in e.itertuples():
        refs_x = refs_by.get(r.paper_id, set())
        rows.append({**r._asdict(), "cites_insider_only": bool(refs_x & set(L[r.X]["insider_only"])),
                     "cites_classic": bool(refs_x & set(L[r.X]["classics"]))})
    print(T, "communities with lists", len(L), "entries", len(e), flush=True)
D = pd.DataFrame(rows).drop(columns=["Index"])
CHK = pd.concat(checks)
STRATS = ["method import", "new problem for X", "bridge"]


def design(d):
    X = pd.DataFrame({s: (d.strategy == s).astype(float) for s in STRATS})
    X["cites insider-only"] = d.cites_insider_only.astype(float)
    X["cites a classic"] = d.cites_classic.astype(float)
    X["insider co-author"] = d.insider.astype(float)
    X["newcomer to the map"] = d.newcomer.astype(float)
    X["log team"] = np.log1p(d.team)
    X["log community size"] = np.log(d["size"])
    X["forward route"] = d.forward.astype(float)
    X["log in-map references"] = np.log1p(d.n_inmap_refs)
    for T in sorted(d["T"].unique())[1:]:
        X[f"T{T}"] = (d["T"] == T).astype(float)
    return X


TRAITS = ["cites insider-only", "cites a classic"]


def adjusted(d, y):
    X, yy = design(d), d[y].astype(int)
    m = LogisticRegression(max_iter=3000, C=1.0).fit(X, yy)
    out = {}
    for s in TRAITS:
        Z1, Z0 = X.copy(), X.copy()
        Z1[s], Z0[s] = 1, 0
        out[s] = m.predict_proba(Z1)[:, 1].mean() - m.predict_proba(Z0)[:, 1].mean()
    return out


res = []
for y, unit, d in (("stayed", "author", D), ("taken_up", "paper", D.drop_duplicates("paper_id")),
                   ("moved_towards", "paper", D.drop_duplicates("paper_id"))):
    d = d.dropna(subset=[y]).reset_index(drop=True)
    est = adjusted(d, y)
    groups = list(d.groupby(["T", "X"]).indices.values())
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        try:
            boots.append(adjusted(d.iloc[i].reset_index(drop=True), y))
        except ValueError:
            pass
    for s in TRAITS:
        col = "cites_insider_only" if s == "cites insider-only" else "cites_classic"
        b = np.array([x[s] for x in boots])
        res.append({"outcome": y, "unit": unit, "trait": s, "entries with it": int(d[col].sum()), "entries": len(d),
                    "raw rate with": float(d[d[col]][y].mean()), "raw rate without": float(d[~d[col]][y].mean()),
                    "adjusted difference": est[s], "90% CI": f"[{np.percentile(b, 5):+.3f}, {np.percentile(b, 95):+.3f}]"})
R = pd.DataFrame(res)
K = CHK.groupby("list").source_of_current.agg(["mean", "size"]).reset_index().rename(
    columns={"mean": "share that are sources of current methods", "size": "works in the map"})
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
print(K.round(3).to_string(index=False))
how = ("Entries (experiment 12) whose paper cites at least one of the community's **insider-only** works: works its "
       "established authors (2+ earlier papers there) cited far more than its newcomers did in the three years before, "
       "excluding the community's 10 most cited works (**classics**). Adjusted differences hold fixed the entry "
       "strategy, an insider co-author, newcomer to the map, team and community size, route, references inside the map, "
       "year, and the other reading trait; intervals resample communities. Association, not cause.")
C.save(RUN, "exp25_insider_reads", "effects", R, title="Insider reads: do newcomers who read them do better?",
       how_to_read=how)
C.save(RUN, "exp25_insider_reads", "knowledge_check", K, title="Insider reads: knowledge or habit?",
       how_to_read="For works in the map: the share that are a close source (cosine >= 0.894) of the community's methods "
       "and innovations of the last three years, for insider-only works, classics and other works the community cites.")

# ---------------------------------------------------------------- live lists
T = int(papers.year.max())  # lists use T-3..T-1, i.e. through the latest complete year
lab = C.assign(con, RUN, RUN, ids)
L, _ = lists_at(lab, T)
labels = dict(con.execute("SELECT cluster_id, label FROM clusters WHERE run_id = ?", [RUN]).fetchall())
live = []
for X, v in L.items():
    for rank, w in enumerate(v["insider_only"], 1):
        live.append({"cluster_id": X, "label": labels.get(X, X), "rank": rank, "paper_id": w, "title": title.get(w),
                     "year": year.get(w), "insider_papers_citing": int(v["ins_counts"].get(w, 0)),
                     "newcomer_papers_citing": int(v["new_counts"].get(w, 0)), "n_insider_papers": v["n_insider_papers"],
                     "n_newcomer_papers": v["n_newcomer_papers"], "window": f"{T - 3}-{T - 1}"})
pd.DataFrame(live).to_parquet(C.project(RUN, "backtest") / "exp25_insider_reads_live.parquet", index=False)
print("live lists:", len(L), "communities")
