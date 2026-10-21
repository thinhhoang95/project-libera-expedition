"""Orientation artefacts for the product portfolio (Phase 10; evidence in docs/features/10-orientation-experiments.md).

    build_orientation(run_id)  ->  data/projects/<run>/portfolio/
        reading_paths.parquet   "Read this first" per community: the community's most cited papers interleaved with
                                the sources of its current methods (experiment 13: newcomers cite the first, build on
                                the second; alternating covers most of both)
        entry_routes.parquet    how people entered each community: entries of the latest year with complete
                                three-year outcomes, by strategy, with outcome rates and example entries; plus the
                                strategy mix of recent entries (experiment 12)
        entry_effects.json      the like-for-like effects of each strategy on the pilot (from experiment 12's table)
        lasting.parquet         per paper: breadth at publication (how many communities' earlier work is already close
                                to its idea) and a "lasting" flag = top third of breadth among equally generic ideas
                                (experiment 14)

Self-contained on purpose: it reads the data layout directly (data/shared/ for the database and vectors,
data/projects/<run>/ for per-run files), so it does not depend on other modules' internals. Communities are the
run's live problem communities (alpha 0.5, medium, stable and hub), every paper assigned to the nearest centroid,
as in the backtest. "Close" = cosine >= 0.894 between an innovation and a method or innovation (256-d), the 99th
percentile of random pairs (experiment 9).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "data" / "shared"
RESEARCH = ("article", "preprint", "conference-paper", "proceedings-article")
THR = 0.894
DIM = 256


def _project(run: str, *parts: str) -> Path:
    p = ROOT.joinpath("data", "projects", run, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def _vectors(facet: str, dim: int | None = DIM):
    t = pq.read_table(SHARED / "vectors" / f"{facet}.parquet", columns=["paper_id", "vector"])
    pid = t.column("paper_id").to_pylist()
    v = np.asarray(t.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                   dtype=np.float32).reshape(len(pid), -1)
    return {x: k for k, x in enumerate(pid)}, (_unit(v[:, :dim]) if dim else v)


class _Field:
    def __init__(self, con, run: str):
        self.run = run
        self.papers = con.execute("""SELECT v.paper_id, v.route, p.year, p.type, p.title FROM provenance v
                                     JOIN papers p USING (paper_id) WHERE v.run_id = ? AND p.year IS NOT NULL""",
                                  [run]).df()
        self.year = self.papers.set_index("paper_id").year
        self.title = self.papers.set_index("paper_id").title
        self.ptype = self.papers.set_index("paper_id").type
        self.route = self.papers.set_index("paper_id").route
        corpus = set(self.papers.paper_id)
        au = con.execute("""SELECT s.paper_id, s.author_id FROM authorships s JOIN provenance v
                            ON v.paper_id = s.paper_id AND v.run_id = ?""", [run]).df()
        au["year"] = au.paper_id.map(self.year)
        self.au = au
        self.team = au.groupby("paper_id").author_id.apply(frozenset)
        refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v ON v.paper_id = r.citing_id
                              AND v.run_id = ?""", [run]).df()
        refs["cy"] = refs.citing_id.map(self.year)
        self.refs = refs
        self.refs_in = refs[refs.cited_id.isin(corpus)]
        self.t_live = int(self.papers.year.max()) - 1   # the latest complete publication year
        self.PP, self.PV = _vectors("problem")
        self.MP, self.MV = _vectors("method")
        self.IP, self.IV = _vectors("innovation")
        # live communities, as the Assigner: nearest stable / hub medium centroid on alpha-0.5 blended vectors
        cents = pd.read_parquet(_project(run, "vectors") / f"cluster_centroids_{run}.parquet")
        cl = con.execute("""SELECT cluster_id, label, kind FROM clusters WHERE run_id = ? AND facet = 'problem'
                            AND alpha = 0.5 AND resolution = 'medium' AND kind IN ('stable', 'hub')""", [run]).df()
        c = cents[cents.cluster_id.isin(set(cl.cluster_id))]
        C = np.stack([np.asarray(x, dtype=np.float32) for x in c.centroid])
        fpos, fv = _vectors("problem", None)
        gpos, gv = _vectors("problem_gloss", None)
        g = np.array([gv[gpos[k]] if k in gpos else fv[i] for k, i in fpos.items()], dtype=np.float32)
        v = _unit(0.5 * fv + 0.5 * g)
        lab = pd.Series(np.array(c.cluster_id)[np.argmax(v @ C.T, axis=1)], index=list(fpos))
        self.lab = lab[lab.index.isin(corpus)]
        self.labels = cl.set_index("cluster_id").label
        self.kind = cl.set_index("cluster_id").kind

    def stack(self, pids):
        a = [self.MV[self.MP[x]] for x in pids if x in self.MP] + [self.IV[self.IP[x]] for x in pids if x in self.IP]
        return np.vstack(a) if a else np.zeros((0, DIM), np.float32)


# ------------------------------------------------------------------ "lasting" (experiment 14)

def lasting(f: _Field) -> pd.DataFrame:
    st_ids = list(f.MP) + list(f.IP)
    ST = np.vstack([f.MV[[f.MP[x] for x in f.MP]], f.IV[[f.IP[x] for x in f.IP]]])
    st_year = np.array([f.year.get(x, 9999) for x in st_ids])
    st_comm = np.array([f.lab.get(x, "") for x in st_ids])
    rand = ST[np.random.default_rng(0).choice(len(ST), min(1000, len(ST)), replace=False)]
    rows = []
    ids = [x for x in f.IP if x in f.year.index]
    V = f.IV[[f.IP[x] for x in ids]]
    for start in range(0, len(ids), 500):
        S = V[start:start + 500] @ ST.T
        for k, x in enumerate(ids[start:start + 500]):
            close = np.where((S[k] >= THR) & (st_year <= f.year[x]))[0]
            comms = {st_comm[j] for j in close if st_ids[j] != x and st_comm[j]}
            rows.append({"paper_id": x, "year": int(f.year[x]), "breadth": len(comms),
                         "genericness": float((rand @ V[start + k]).mean())})
    D = pd.DataFrame(rows)
    D["genericness_quintile"] = pd.qcut(D.genericness.rank(method="first"), 5, labels=False)
    D["breadth_rank_within_genericness"] = D.groupby("genericness_quintile").breadth.rank(pct=True, method="average")
    # validated on ideas from 2018-2019 (experiment 14); before 2018 the map holds too few earlier papers for
    # breadth to mean anything, so older papers are "not assessed" rather than "not lasting"
    D["assessed"] = D.year >= 2018
    D["lasting"] = D.assessed & (D.breadth_rank_within_genericness >= 2 / 3) & (D.breadth >= 2)
    D["evidence"] = "breadth at publication, compared among equally generic ideas: AUC 0.58 for still being used " \
                    "5+ years later (experiment 14); novel ideas fade faster (0.39); early citations say nothing"
    return D


# ------------------------------------------------------------------ "read this first" (experiment 13)

def reading_paths(f: _Field, L: pd.DataFrame | None, k: int = 10) -> pd.DataFrame:
    T = f.t_live
    inn_ids = list(f.IP)
    inn_year = np.array([f.year.get(x, 9999) for x in inn_ids])
    lasting_ids = set(L[L.lasting].paper_id) if L is not None else set()
    rows = []
    for X in sorted(set(f.lab)):
        mem = f.lab.index[f.lab.values == X]
        mem_set = set(mem)
        cited = f.refs_in[f.refs_in.citing_id.isin(mem_set) & (f.refs_in.cy <= T)].groupby("cited_id").size()
        classics = [p for p in cited.sort_values(ascending=False).index if f.year.get(p, 9999) <= T][:k]
        recent = [x for x in mem if T - 2 <= f.year[x] <= T]
        vecs = f.stack(recent)
        ok = np.where(inn_year < T - 1)[0]
        src = Counter()
        if len(vecs) and len(ok):
            S = vecs @ f.IV[ok].T
            for row in S:
                hit = ok[row >= THR]
                if len(hit):
                    src[inn_ids[hit[np.argmin(inn_year[hit])]]] += 1
        lineage = [p for p, _ in src.most_common()][:k]
        order, seen = [], set()
        for a, b in zip(classics + [None] * k, lineage + [None] * k):
            for p, role in ((a, "classic"), (b, "source of current methods")):
                if p and p not in seen:
                    seen.add(p)
                    order.append((p, role))
        for rank, (p, role) in enumerate(order[:2 * k], 1):
            rows.append({"cluster_id": X, "label": f.labels.get(X, X), "rank": rank, "paper_id": p,
                         "title": f.title.get(p), "year": f.year.get(p), "role": role,
                         "cited_by_community": int(cited.get(p, 0)), "source_of_methods": int(src.get(p, 0)),
                         "lasting": p in lasting_ids, "lasting_assessed": f.year.get(p, 0) >= 2018})
    return pd.DataFrame(rows).assign(as_of=T)


# ------------------------------------------------------------------ "how people got in" (experiment 12)

def _entries(f: _Field, T: int) -> pd.DataFrame:
    lab = f.lab
    mem_pre = {c: [x for x in lab.index[lab.values == c] if f.year[x] <= T - 1] for c in set(lab)}
    core = {c: _unit(np.mean([f.PV[f.PP[x]] for x in m if x in f.PP], axis=0)) for c, m in mem_pre.items()
            if sum(x in f.PP for x in m) >= 5}
    mcent = {c: _unit(np.mean([f.MV[f.MP[x]] for x in m if x in f.MP], axis=0)) for c, m in mem_pre.items()
             if sum(x in f.MP for x in m) >= 5}
    own_q = {c: np.quantile([float(f.MV[f.MP[x]] @ mcent[c]) for x in mem_pre[c] if x in f.MP], 0.25) for c in mcent}
    mc_ids = list(mcent)
    MC = np.vstack([mcent[c] for c in mc_ids])
    a2 = f.au.assign(X=f.au.paper_id.map(lab)).dropna(subset=["X"])
    first = a2[a2.year <= T].sort_values("year").groupby(["author_id", "X"]).head(1)
    ent = first[(first.year == T) & first.paper_id.map(f.ptype).isin(RESEARCH) & first.X.isin(core)]
    n_in_X_before = a2[a2.year < T].groupby(["author_id", "X"]).paper_id.nunique()
    later = a2[a2.year.between(T + 1, T + 3)].groupby(["author_id", "X"]).paper_id.nunique()
    yT = f.papers[(f.papers.year == T) & f.papers.type.isin(RESEARCH)].paper_id
    ns = f.refs[f.refs.cy.between(T + 1, T + 3) & f.refs.cited_id.isin(set(yT))]
    ns = ns[[not (f.team.get(a, frozenset()) & f.team.get(b, frozenset())) for a, b in zip(ns.citing_id, ns.cited_id)]]
    n_ns = ns.groupby("cited_id").size().reindex(yT).fillna(0)
    up_cut = n_ns.quantile(0.75)
    feat = {}
    for x, X in ent[["paper_id", "X"]].drop_duplicates().itertuples(index=False):
        if x not in f.PP or x not in f.MP:
            continue
        origin = mc_ids[int(np.argmax(MC @ f.MV[f.MP[x]]))]
        cited = [lab.get(r) for r in f.refs[f.refs.citing_id == x].cited_id if f.year.get(r, 9999) <= T - 1]
        cited = [c for c in cited if isinstance(c, str)]
        p = pd.Series(cited).value_counts(normalize=True).to_numpy() if len(cited) >= 3 else None
        feat[(x, X)] = {"core_distance": 1 - float(f.PV[f.PP[x]] @ core[X]),
                        "import": (origin != X) and X in mcent and float(f.MV[f.MP[x]] @ mcent[X]) < own_q[X],
                        "ref_entropy": float(-(p * np.log(p)).sum()) if p is not None else np.nan,
                        "insider": any(n_in_X_before.get((a, X), 0) >= 2 for a in f.team.get(x, frozenset())),
                        "taken_up": n_ns.get(x, 0) > up_cut}
    rows = []
    for r in ent.itertuples():
        g = feat.get((r.paper_id, r.X))
        if g is not None:
            rows.append({"T": T, "author_id": r.author_id, "paper_id": r.paper_id, "X": r.X,
                         "stayed": later.get((r.author_id, r.X), 0) >= 1, **g})
    D = pd.DataFrame(rows)
    if D.empty:
        return D
    D["far"] = D.core_distance >= D.core_distance.quantile(0.75)
    D["bridge"] = D.ref_entropy.notna() & (D.ref_entropy >= D.ref_entropy.quantile(0.75))
    D["strategy"] = np.select([D["import"], D.far, D.bridge], ["method import", "new problem for the community", "bridge"],
                              "core incremental")
    return D


def entry_routes(f: _Field) -> pd.DataFrame:
    """Entries of the latest year with three complete years of outcomes, by community and strategy; plus the
    strategy mix of the most recent year (no outcomes yet)."""
    T_out = f.t_live - 3
    D = _entries(f, T_out)
    R = _entries(f, f.t_live)
    rows = []
    for X in sorted(set(f.lab)):
        d, r = D[D.X == X] if len(D) else D, R[R.X == X] if len(R) else R
        for s in ("core incremental", "bridge", "method import", "new problem for the community"):
            ds = d[d.strategy == s] if len(d) else d
            rs = r[r.strategy == s] if len(r) else r
            papers_s = ds.drop_duplicates("paper_id") if len(ds) else ds
            good = papers_s[papers_s.taken_up].paper_id.tolist()[:3] if len(papers_s) else []
            rows.append({"cluster_id": X, "label": f.labels.get(X, X), "strategy": s,
                         "entries": int(len(ds)), "stayed_rate": float(ds.stayed.mean()) if len(ds) else np.nan,
                         "taken_up_rate": float(papers_s.taken_up.mean()) if len(papers_s) else np.nan,
                         "insider_coauthor_rate": float(ds.insider.mean()) if len(ds) else np.nan,
                         "examples_taken_up": [{"paper_id": p, "title": f.title.get(p)} for p in good],
                         "recent_entries": int(len(rs)), "outcome_year": T_out, "recent_year": f.t_live})
    out = pd.DataFrame(rows)
    out["examples_taken_up"] = out.examples_taken_up.map(json.dumps)
    return out


def _entry_effects(run: str) -> dict:
    p = _project(run, "backtest", "experiments") / "exp12_entry__strategies.parquet"
    if not p.exists():
        return {"available": False, "note": "run scripts/phase10/exp12_entry.py for the like-for-like effects"}
    R = pd.read_parquet(p)
    return {"available": True, "source": "experiment 12 (scripts/phase10/exp12_entry.py)",
            "note": "Adjusted differences vs core incremental entries (same community size, team, insider co-author, "
                    "newcomer, route, references in the map, year); 90% intervals resample communities. What happened "
                    "on this map over three years, not a guarantee.",
            "effects": R[["outcome", "strategy / trait", "adjusted difference", "90% CI",
                          "adjusted rate, core incremental"]].to_dict("records")}


# ------------------------------------------------------------------ build

def build_orientation(run: str, db_path: str | None = None) -> dict:
    con = duckdb.connect(db_path or os.environ.get("LANDSCAPE_DB", str(SHARED / "landscape.duckdb")), read_only=True)
    f = _Field(con, run)
    out = _project(run, "portfolio")
    L = lasting(f)
    L.to_parquet(out / "lasting.parquet", index=False)
    P = reading_paths(f, L)
    P.to_parquet(out / "reading_paths.parquet", index=False)
    E = entry_routes(f)
    E.to_parquet(out / "entry_routes.parquet", index=False)
    (out / "entry_effects.json").write_text(json.dumps(_entry_effects(run), indent=2, default=float) + "\n")
    stats = {"as_of": f.t_live, "built_at": datetime.now().isoformat(timespec="seconds"),
             "lasting_papers": int(L.lasting.sum()), "papers_scored": len(L),
             "reading_paths": int(P.cluster_id.nunique()), "entry_rows": len(E),
             "entries_with_outcomes": int(E.entries.sum()), "outcome_year": f.t_live - 3}
    (out / "orientation_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the orientation artefacts (Phase 10) for a run.")
    ap.add_argument("--run-id", default="dro-pilot")
    ap.add_argument("--db", help="database path (default: LANDSCAPE_DB or data/shared/landscape.duckdb)")
    a = ap.parse_args()
    print(json.dumps(build_orientation(a.run_id, a.db), indent=2))
