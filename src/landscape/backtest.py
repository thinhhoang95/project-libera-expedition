"""Phase 8: backtest the Phase 6 signals on the papers already in the database.

For each cutoff T:
1. A derived run `<base>@T<T>` holds only the base run's papers published in or before T, so
   only citations made in or before T are visible. Phase 5 clusters are re-fitted on that data
   alone (never reusing the full run's cluster ids), and the signals are computed at T with
   the LLM judge off (it knows the future).
2. Outcomes come from the base run's papers published in T+1..T+3. Every paper, past and
   future, is assigned to the cutoff's clusters by nearest centroid, so shares before and after
   T are measured the same way.
3. Each signal's ranking is scored by precision@k against the recipe's hit definitions and
   compared with random (the base rate), popularity and a citation-only baseline.

No new data is fetched: the corpus is the base run's.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from . import embed
from .config import DATA, VEC_DIR, RunParams

log = logging.getLogger(__name__)
KS_CLUSTER = (5, 10, 20)
KS = (10, 20, 50)


# ------------------------------------------------------------------ cutoff runs

def cutoff_run_id(base: str, T: int) -> str:
    return f"{base}@T{T}"


def build_cutoff_run(con, base: str, T: int, p: RunParams, rebuild: bool = False) -> str:
    from .core import build_core
    from .signals import build_signals

    run = cutoff_run_id(base, T)
    exists = con.execute("SELECT count(*) FROM signals WHERE run_id = ?", [run]).fetchone()[0]
    if exists and not rebuild:
        return run
    bp = replace(p)
    bp.clusters = replace(p.clusters, gloss_alphas=[0.3, 0.5])  # alphas the signals use
    con.execute("DELETE FROM runs WHERE run_id = ?", [run])
    con.execute("""INSERT INTO runs SELECT ?, seed_ids, ?, ?, model_versions, '{}', now(),
                   'backtest cutoff of ' || run_id FROM runs WHERE run_id = ?""",
                [run, json.dumps(bp.to_dict()), T, base])
    con.execute("DELETE FROM provenance WHERE run_id = ?", [run])
    con.execute("""INSERT INTO provenance SELECT ?, v.paper_id, v.route, v.hop, v.via_id, v.score
                   FROM provenance v JOIN papers p USING (paper_id)
                   WHERE v.run_id = ? AND p.year <= ?""", [run, base, T])
    log.info("cutoff %d: %d papers", T, con.execute("SELECT count(*) FROM provenance WHERE run_id = ?", [run]).fetchone()[0])
    build_core(con, run, bp, skip_labels=True, author_centroids=False)
    build_signals(con, run, bp, skip_judge=True)
    return run


# ------------------------------------------------------------------ assignment

class Assigner:
    """Nearest-centroid assignment of any base-run paper to a cutoff run's medium clusters."""

    def __init__(self, con, run: str, version: str):
        self.con, self.run, self.version = con, run, version
        cents = pd.read_parquet(VEC_DIR / f"cluster_centroids_{run}.parquet")
        self.cent = dict(zip(cents.cluster_id, cents.centroid))
        self.clusters = con.execute("""SELECT cluster_id, facet, alpha, size, kind FROM clusters
                                       WHERE run_id = ? AND resolution = 'medium'
                                         AND kind IN ('stable', 'hub')""", [run]).df()
        self._cache = {}

    def assign(self, facet: str, alpha: float, ids: list[str]) -> pd.Series:
        key = (facet, alpha)
        if key not in self._cache:
            cl = self.clusters[(self.clusters.facet == facet) & (self.clusters.alpha == alpha)]
            cids = cl.cluster_id.tolist()
            C = np.stack([np.asarray(self.cent[c], dtype=np.float32) for c in cids])
            pid, v = embed.blended(facet, alpha, None, dim=768, version=self.version)
            lab = np.array(cids)[np.argmax(v @ C.T, axis=1)]
            self._cache[key] = pd.Series(lab, index=pid)
        s = self._cache[key]
        return s[s.index.isin(ids)]


# ------------------------------------------------------------------ metrics

def precision_at(ranked: list, hits: set, ks) -> dict:
    out = {}
    for k in ks:
        top = ranked[:k]
        out[f"p@{k}"] = (sum(1 for x in top if x in hits) / len(top)) if len(top) >= k else np.nan
    return out


def score_rankings(rankings: dict[str, list], candidates: list, hits: set, ks) -> list[dict]:
    """rankings: name -> ranked candidate list. Adds the random baseline as the base rate."""
    base = len(hits & set(candidates)) / max(len(candidates), 1)
    rows = [{"ranking": "random (base rate)", **{f"p@{k}": base for k in ks}}]
    for name, r in rankings.items():
        rows.append({"ranking": name, **precision_at([c for c in r if c in set(candidates)], hits, ks)})
    for r in rows:
        r["n_candidates"], r["n_hits"] = len(candidates), len(hits & set(candidates))
    return rows


# ------------------------------------------------------------------ evaluation of one cutoff

def evaluate_cutoff(con, base: str, T: int, p: RunParams) -> pd.DataFrame:
    run = cutoff_run_id(base, T)
    sp = p.signals
    ver = p.embedding.version
    A = Assigner(con, run, ver)
    papers = con.execute("""SELECT v.paper_id, p.year FROM provenance v JOIN papers p USING (paper_id)
                            WHERE v.run_id = ?""", [base]).df()
    year = papers.set_index("paper_id").year
    all_ids = papers.paper_id.tolist()
    future = papers[(papers.year > T) & (papers.year <= T + 3)].paper_id
    edges = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r
                           JOIN provenance a ON a.paper_id = r.citing_id AND a.run_id = ?
                           JOIN provenance b ON b.paper_id = r.cited_id AND b.run_id = ?""", [base, base]).df()
    edges["cy"] = edges.citing_id.map(year)
    sig = con.execute("SELECT * FROM signals WHERE run_id = ? AND variant = 'all'", [run]).df()
    feats = con.execute("""SELECT cluster_id, scores FROM viz_cluster_year WHERE run_id = ? AND year = ?
                           AND scores IS NOT NULL""", [run, T]).df()
    feat = {c: json.loads(s) for c, s in zip(feats.cluster_id, feats.scores)}
    rows = []

    def add(signal, scope, res):
        for r in res:
            rows.append({"cutoff": T, "signal": signal, "scope": scope, **r})

    # ---- cluster-level: emerging, growing, moving away
    # Two hit definitions: the recipe's (single year T vs later years) and a smoothed one (3-year
    # mean share before vs after T). Single-year shares at T regress to the mean, which penalises
    # exactly the clusters a momentum signal ranks high, so both are reported side by side.
    names = ["emerging", "growing", "shrinking"]
    pooled = {(nm, h): [] for nm in names for h in ("recipe hit", "smoothed hit")}
    for facet, alpha in sp.cluster_sets:
        lab = A.assign(facet, alpha, all_ids)
        df = pd.DataFrame({"c": lab, "year": lab.index.map(year)})
        n = df.groupby(["c", "year"]).size().unstack(fill_value=0)
        share = n / n.sum(axis=0)
        S = lambda c, y: share.loc[c, y] if y in share.columns else np.nan  # noqa: E731
        M3 = lambda c, ys: float(np.nanmean([S(c, y) for y in ys]))  # noqa: E731
        size_T = df[df.year <= T].groupby("c").size()
        cands = [c for c in share.index if c in feat]
        rec = edges.assign(to=edges.cited_id.map(lab))
        rec_T = rec[rec.cy == T].groupby("to").size()
        rec_T2 = rec[rec.cy == T - 2].groupby("to").size()
        cit_growth = {c: np.log((rec_T.get(c, 0) + 1) / (rec_T2.get(c, 0) + 1)) for c in cands}
        momentum = {c: S(c, T) - S(c, T - 1) for c in cands}
        past, fut = [T - 2, T - 1, T], [T + 1, T + 2, T + 3]
        hits = {
            ("emerging", "recipe hit"): {c for c in cands if S(c, T + 3) >= 1.5 * S(c, T)},
            ("growing", "recipe hit"): {c for c in cands if M3(c, fut) > S(c, T)},
            ("shrinking", "recipe hit"): {c for c in cands if S(c, T + 1) < S(c, T) and S(c, T + 2) < S(c, T + 1)},
            ("emerging", "smoothed hit"): {c for c in cands if M3(c, fut) >= 1.5 * M3(c, past)},
            ("growing", "smoothed hit"): {c for c in cands if M3(c, fut) > M3(c, past)},
            ("shrinking", "smoothed hit"): {c for c in cands if M3(c, fut) < M3(c, past) and S(c, T + 2) < S(c, T + 1)},
        }
        elig = {"emerging": [c for c in cands if size_T.get(c, 0) >= sp.emergence_min_size],
                "growing": [c for c in cands if size_T.get(c, 0) >= sp.direction_min_size],
                "shrinking": [c for c in cands if size_T.get(c, 0) >= sp.direction_min_size]}
        val = lambda d: (lambda c: -np.inf if d.get(c) is None or np.isnan(d.get(c)) else d[c])  # noqa: E731
        f_ = lambda k: {c: feat[c].get(k) for c in cands}  # noqa: E731
        sig_score = {"emerging": f_("E"), "growing": f_("D"), "shrinking": f_("D")}
        recipe_score = {"emerging": f_("E_recipe"), "growing": f_("g"), "shrinking": f_("g")}
        for name in names:
            sgn = 1 if name != "shrinking" else -1
            key = lambda d: (lambda c: sgn * val(d)(c) if np.isfinite(val(d)(c)) else -np.inf)  # noqa: E731
            rk = {"signal": sorted(elig[name], key=key(sig_score[name]), reverse=True),
                  "recipe score": sorted(elig[name], key=key(recipe_score[name]), reverse=True),
                  "popularity (size at T)": sorted(elig[name], key=lambda c: size_T.get(c, 0), reverse=True),
                  "momentum (last-year share change)": sorted(elig[name], key=key(momentum), reverse=True),
                  "citation-only (received-citation growth)": sorted(elig[name], key=key(cit_growth), reverse=True)}
            for h in ("recipe hit", "smoothed hit"):
                add(f"{name} ({h})", f"{facet}@{alpha}", score_rankings(rk, elig[name], hits[(name, h)], KS_CLUSTER))
                for c in elig[name]:
                    pooled[(name, h)].append({
                        "c": c, "hit": c in hits[(name, h)], "size": size_T.get(c, 0), "scope": facet,
                        "signal": key(sig_score[name])(c), "recipe": key(recipe_score[name])(c),
                        "momentum": key(momentum)(c), "cit": key(cit_growth)(c)})
    for (name, h), lst in pooled.items():
        d = pd.DataFrame(lst)
        if d.empty:
            continue
        for col in ("signal", "recipe", "momentum", "cit", "size"):
            d[col + "_z"] = d.groupby("scope")[col].transform(
                lambda s: (s.replace(-np.inf, np.nan) - s.replace(-np.inf, np.nan).mean()) / (s.replace(-np.inf, np.nan).std() or 1)).fillna(-9)
        rk = {"signal": d.sort_values("signal_z", ascending=False).c.tolist(),
              "recipe score": d.sort_values("recipe_z", ascending=False).c.tolist(),
              "popularity (size at T)": d.sort_values("size_z", ascending=False).c.tolist(),
              "momentum (last-year share change)": d.sort_values("momentum_z", ascending=False).c.tolist(),
              "citation-only (received-citation growth)": d.sort_values("cit_z", ascending=False).c.tolist()}
        add(f"{name} ({h})", "pooled", score_rankings(rk, d.c.tolist(), set(d[d.hit].c), KS_CLUSTER))

    # ---- method transfer (P, m) at the bridge alpha
    ba = sp.bridge_alpha
    pa = A.assign("problem", ba, all_ids)
    ma = A.assign("method", ba, all_ids)
    both = pd.DataFrame({"P": pa, "m": ma}).dropna()
    both["year"] = both.index.map(year)
    past = both[both.year <= T]
    fut = both[both.index.isin(future)]
    M = past.groupby(["P", "m"]).size()
    Ps, ms = sorted(both.P.unique()), sorted(both.m.unique())
    cands = [(P, m) for P in Ps for m in ms if M.get((P, m), 0) == 0]
    fut_n = fut.groupby(["P", "m"]).size()
    hits = {c for c in cands if fut_n.get(c, 0) >= 2}
    tr = sig[sig.signal_type == "bridge_method_transfer"].sort_values("rank")
    sig_list = [(json.loads(s)["problem_cluster"], json.loads(s)["method_cluster"]) for s in tr.subject]
    sizeP, sizem = past.P.value_counts(), past.m.value_counts()
    # citation-only link prediction: citations from P's papers to papers that use method m
    e = edges[edges.cy <= T].assign(fP=lambda d: d.citing_id.map(pa), tm=lambda d: d.cited_id.map(ma))
    link = e.groupby(["fP", "tm"]).size()
    rk = {"signal": sig_list,
          "popularity (size P × size m)": sorted(cands, key=lambda c: sizeP.get(c[0], 0) * sizem.get(c[1], 0), reverse=True),
          "citation-only (P cites m-papers)": sorted(cands, key=lambda c: link.get(c, 0), reverse=True)}
    add("method_transfer", f"problem×method@{ba}", score_rankings(rk, cands, hits, KS))

    # ---- limitation -> innovation (X -> Y) at the bridge alpha
    li = sig[sig.signal_type == "bridge_lim_inn"].sort_values("rank")
    subj = [json.loads(s) for s in li.subject]
    feats_li = [json.loads(f) for f in li.features]
    Xs = sorted({s["from_cluster"] for s in subj}) or Ps
    cands = [(X, Y) for X in Xs for Y in Ps if X != Y]
    e = edges.assign(f=edges.citing_id.map(pa), t=edges.cited_id.map(pa))
    before = e[(e.cy > T - 3) & (e.cy <= T)].groupby(["f", "t"]).size() / 3
    after = e[(e.cy > T) & (e.cy <= T + 3)].groupby(["f", "t"]).size() / 3
    inno = {(s["from_cluster"], s["to_cluster"]): {pr["innovation"] for pr in f.get("pairs", [])}
            for s, f in zip(subj, feats_li)}
    fut_cites = e[(e.cy > T) & (e.cy <= T + 3)]
    hits = set()
    for X, Y in cands:
        if after.get((X, Y), 0) >= 3 * max(before.get((X, Y), 0), 1 / 3):
            hits.add((X, Y))
        elif inno.get((X, Y)) and fut_cites[(fut_cites.f == X) & fut_cites.cited_id.isin(inno[(X, Y)])].shape[0]:
            hits.add((X, Y))
    # citation-only common neighbours: works cited by both X and Y papers before T
    allrefs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v
                             ON v.paper_id = r.citing_id AND v.run_id = ?""", [base]).df()
    allrefs = allrefs[allrefs.citing_id.map(year) <= T]
    allrefs["c"] = allrefs.citing_id.map(pa)
    refsets = allrefs.dropna(subset=["c"]).groupby("c").cited_id.apply(set)
    cn = {(X, Y): len(refsets.get(X, set()) & refsets.get(Y, set())) for X, Y in cands}
    sizeX = past.P.value_counts()
    rk = {"signal": [(s["from_cluster"], s["to_cluster"]) for s in subj],
          "popularity (size X × size Y)": sorted(cands, key=lambda c: sizeX.get(c[0], 0) * sizeX.get(c[1], 0), reverse=True),
          "citation-only (common cited works)": sorted(cands, key=lambda c: cn[c], reverse=True)}
    add("lim_to_innovation", f"problem@{ba}", score_rankings(rk, cands, hits, KS))

    # ---- seeders and brokers: received citations in T+1..T+3 in the top decile
    au = con.execute("""SELECT s.paper_id, s.author_id FROM authorships s JOIN provenance v
                        ON v.paper_id = s.paper_id AND v.run_id = ?""", [run]).df()
    fut_rec = edges[(edges.cy > T) & (edges.cy <= T + 3)].groupby("cited_id").size()
    past_rec = edges[edges.cy <= T].groupby("cited_id").size()
    a_fut = au.assign(n=au.paper_id.map(fut_rec).fillna(0)).groupby("author_id").n.sum()
    a_past = au.assign(n=au.paper_id.map(past_rec).fillna(0)).groupby("author_id").n.sum()
    thr = a_fut.quantile(0.9)
    hit_auth = set(a_fut[a_fut > thr].index) if thr > 0 else set(a_fut[a_fut > 0].index)
    n_pap = au.groupby("author_id").paper_id.nunique()
    for typ, min_p in (("seeder_author", 1), ("broker_author", sp.broker_min_papers)):
        cands = n_pap[n_pap >= min_p].index.tolist()
        s_list = [json.loads(s)["author_id"] for s in sig[sig.signal_type == typ].sort_values("rank").subject]
        rk = {"signal": s_list,
              "popularity (corpus citations ≤ T)": sorted(cands, key=lambda a: a_past.get(a, 0), reverse=True),
              "productivity (papers ≤ T)": sorted(cands, key=lambda a: n_pap.get(a, 0), reverse=True)}
        add(typ.replace("_author", ""), "authors", score_rankings(rk, cands, hit_auth, KS))

    # ---- novel papers (extra; not in the recipe's table): top-quartile future citations
    ps = con.execute("""SELECT paper_id, facet, value FROM paper_scores WHERE run_id = ? AND metric = 'novelty'""",
                     [run]).df()
    for f, grp in ps.groupby("facet"):
        cand = grp[grp.paper_id.map(year).between(T - 2, T)]
        if cand.empty:
            continue
        fr = cand.paper_id.map(fut_rec).fillna(0)
        yq = cand.assign(fr=fr.values, y=cand.paper_id.map(year).values).groupby("y").fr.transform(lambda s: s.quantile(0.75))
        hits = set(cand.paper_id[(fr.values > yq.values) & (fr.values > 0)])
        rk = {"signal": cand.sort_values("value", ascending=False).paper_id.tolist(),
              "popularity (corpus citations ≤ T)": sorted(cand.paper_id, key=lambda x: past_rec.get(x, 0), reverse=True)}
        add("novel_paper (extra)", f, score_rankings(rk, cand.paper_id.tolist(), hits, KS))
    return pd.DataFrame(rows)


def cluster_feature_table(con, base: str, T: int, p: RunParams, outcome_routes: set | None = None) -> pd.DataFrame:
    """Every cluster-level feature at cutoff T next to its realised future share change.

    Used to see which ingredients of the emergence and direction scores actually predict, and
    to tune their weights across cutoffs (leave-one-cutoff-out). `outcome_routes` restricts the
    papers used for shares (e.g. to check that a result is not an artefact of one route)."""
    from .signals import Ctx, cluster_signals_at

    run = cutoff_run_id(base, T)
    ctx = Ctx(con, run, p)
    ps = con.execute("""SELECT paper_id, facet, value FROM paper_scores WHERE run_id = ?
                        AND metric = 'novelty'""", [run]).df()
    A = Assigner(con, run, p.embedding.version)
    papers = con.execute("""SELECT v.paper_id, p.year, v.route FROM provenance v JOIN papers p USING (paper_id)
                            WHERE v.run_id = ?""", [base]).df()
    year = papers.set_index("paper_id").year
    edges = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r
                           JOIN provenance a ON a.paper_id = r.citing_id AND a.run_id = ?
                           JOIN provenance b ON b.paper_id = r.cited_id AND b.run_id = ?""", [base, base]).df()
    edges["cy"] = edges.citing_id.map(year)
    share_ids = papers.paper_id if outcome_routes is None else papers[papers.route.isin(outcome_routes)].paper_id
    out = []
    for facet, alpha in p.signals.cluster_sets:
        nov = ps[ps.facet == facet].set_index("paper_id").value
        df, _ = cluster_signals_at(ctx, facet, alpha, T, "all", nov if len(nov) else None)
        lab = A.assign(facet, alpha, papers.paper_id.tolist())
        sl = lab[lab.index.isin(share_ids)]
        d = pd.DataFrame({"c": sl, "year": sl.index.map(year)})
        n = d.groupby(["c", "year"]).size().unstack(fill_value=0)
        share = n / n.sum(axis=0)
        S = lambda c, ys: float(np.nanmean([share.loc[c, y] if y in share.columns else np.nan for y in ys]))  # noqa: E731
        rec = edges.assign(to=edges.cited_id.map(lab))
        rT = rec[rec.cy == T].groupby("to").size()
        rT2 = rec[rec.cy == T - 2].groupby("to").size()
        df = df[df.cluster_id.isin(share.index)].copy()
        df["share_T"] = [S(c, [T]) for c in df.cluster_id]
        df["share_past3"] = [S(c, [T - 2, T - 1, T]) for c in df.cluster_id]
        df["momentum"] = [S(c, [T]) - S(c, [T - 1]) for c in df.cluster_id]
        df["cit_growth"] = [np.log((rT.get(c, 0) + 1) / (rT2.get(c, 0) + 1)) for c in df.cluster_id]
        df["share_T3"] = [S(c, [T + 3]) for c in df.cluster_id]
        df["share_fut3"] = [S(c, [T + 1, T + 2, T + 3]) for c in df.cluster_id]
        df["s1"] = [S(c, [T + 1]) for c in df.cluster_id]
        df["s2"] = [S(c, [T + 2]) for c in df.cluster_id]
        df["hit_emerging"] = df.share_T3 >= 1.5 * df.share_T
        df["hit_growing"] = df.share_fut3 > df.share_T
        df["hit_shrinking"] = (df.s1 < df.share_T) & (df.s2 < df.s1)
        # smoothed outcome: log of future 3-year share over past 3-year share
        df["y_logratio"] = np.log((df.share_fut3 + 1e-4) / (df.share_past3 + 1e-4))
        df["scope"] = facet
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    res["cutoff"] = T
    return res


def run_backtest(con, base: str, p: RunParams, cutoffs=(2020, 2021, 2022, 2023), rebuild=False) -> pd.DataFrame:
    res = []
    for T in cutoffs:
        build_cutoff_run(con, base, T, p, rebuild=rebuild)
        res.append(evaluate_cutoff(con, base, T, p))
    df = pd.concat(res, ignore_index=True)
    out = DATA / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    df.assign(run_id=base).to_parquet(out / f"{base}.parquet")
    summarize(df).to_csv(out / f"{base}_summary.csv", index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    pcols = [c for c in df.columns if c.startswith("p@")]
    return (df.groupby(["signal", "scope", "ranking"], sort=False)[pcols + ["n_candidates", "n_hits"]]
            .mean().round(3).reset_index())
