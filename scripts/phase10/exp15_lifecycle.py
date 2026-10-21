"""Experiment 15: community lifecycle. What kind of contribution does a community need now?

Each paper's kind, from its extracted facets (spot-check: about 89% agreement with a manual read of 36 abstracts;
"application" is rare in this corpus because applied engineering papers are labelled "method"):

    theory           contribution_type = theory
    applied method   method / application / dataset / benchmark, and an application domain or data is named
    generic method   method without a named domain or data
    survey           survey
    (other / position papers are left out)
    empirical        has_quantitative_result = true (a separate flag)

Communities: problem communities (alpha 0.5, medium) of each cutoff run, every paper assigned to the nearest
stable or hub centroid (common.assign). Cutoffs T = 2019..2022; "before" = T-2..T, "after" = T+1..T+3.

A. Does a community's stage predict its next mix, beyond persistence and the field-wide drift?
   Predict each kind's share after from the shares before: persistence (no change), field drift (every community
   moves like the whole field), and a lifecycle model (the change depends on the community's own mix; ridge,
   fitted leave-one-cutoff-out). Mean absolute error, with a 90% interval resampling communities.
B. "Theory wanted": method communities (alpha 0.5, medium) whose use is growing and empirical but which have few
   theory papers. Does their theory share rise more by T+3 than for method communities with the same theory share
   and size? (Matching on the theory share removes regression to the mean.)
C. Is the scarce kind of contribution taken up more? A paper's kind is scarce in its community when the community's
   share of that kind (T-2..T) is under half the field's. Compared within each kind (a scarce theory paper against
   theory papers where theory is common), on: taken up by others (non-self citations in the top quarter of year-T
   papers), taken up relative to its 15 nearest same-year papers (peer-normalized), and the community moving
   towards it (experiment 9: a later paper of the community by other authors closer than 0.894 to its innovation
   and closer than any of the community's papers from T-2..T).

Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a database copy.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import common as C

RUN = C.run_arg()
con = C.connect()
papers, au, refs, year, team = C.base_tables(con, RUN)
ids = papers.paper_id.tolist()
f = con.execute("""SELECT paper_id, facet, text FROM facets WHERE facet IN ('contribution_type', 'has_quantitative_result',
                   'domain_data')""").df().pivot_table(index="paper_id", columns="facet", values="text", aggfunc="first")
ct = f.contribution_type.reindex(ids)
dd = f.domain_data.reindex(ids).fillna("").astype(str).str.strip()
has_dd = (dd.str.len() > 0) & ~dd.str.lower().isin(["null", "none", "n/a", "not stated"])
KINDS = ["theory", "applied method", "generic method", "survey"]
kind = pd.Series(np.select([ct == "theory", ct == "survey",
                            ct.isin(["method", "application", "dataset", "benchmark"]) & has_dd,
                            ct.isin(["method", "application", "dataset", "benchmark"])],
                           ["theory", "survey", "applied method", "generic method"], default=""), index=ids)
emp = (f.has_quantitative_result.reindex(ids) == "true")
kind = kind[kind != ""]
print("kinds:", kind.value_counts().to_dict(), "| empirical share:", round(float(emp.mean()), 3))

PA, AV = C.vectors("abstract", ids)
MP, MV = C.vectors("method", ids)
IP, IV = C.vectors("innovation", ids)


def stack(pids):
    a = [MV[MP[x]] for x in pids if x in MP] + [IV[IP[x]] for x in pids if x in IP]
    return np.vstack(a) if a else np.zeros((0, MV.shape[1]), np.float32)


def assign_method(cutoff_run):
    cents = pd.read_parquet(C.project(RUN, "vectors") / f"cluster_centroids_{cutoff_run}.parquet")
    keep = set(con.execute("""SELECT cluster_id FROM clusters WHERE run_id = ? AND facet = 'method' AND alpha = 0.5
                              AND resolution = 'medium' AND kind IN ('stable', 'hub')""", [cutoff_run]).df().cluster_id)
    c = cents[cents.cluster_id.isin(keep)]
    Cm = np.stack([np.asarray(x, dtype=np.float32) for x in c.centroid])
    pos, v = C.blended("method", "method_gloss", 0.5, ids)
    return pd.Series(np.array(c.cluster_id)[np.argmax(v @ Cm.T, axis=1)], index=list(pos))


def mix(pids):
    k = kind.reindex(pids).dropna()
    s = k.value_counts(normalize=True).reindex(KINDS).fillna(0)
    return s, len(k), float(emp.reindex(pids).fillna(False).mean()) if len(pids) else np.nan


comm_rows, meth_rows, paper_rows = [], [], []
nonself = refs.assign(is_self=[bool(team.get(a, frozenset()) & team.get(b, frozenset()))
                            for a, b in zip(refs.citing_id, refs.cited_id)])
for T in range(2019, 2023):
    lab = C.assign(con, RUN, f"{RUN}@T{T}", ids)
    yr = lab.index.map(year)
    before = lab[(yr >= T - 2) & (yr <= T)]
    after = lab[(yr > T) & (yr <= T + 3)]
    fb, _, _ = mix(before.index.tolist())
    fa, _, _ = mix(after.index.tolist())
    for X in sorted(set(lab)):
        sb, nb, eb = mix(before.index[before.values == X].tolist())
        sa, na, ea = mix(after.index[after.values == X].tolist())
        if nb >= 10 and na >= 10:
            comm_rows.append({"T": T, "X": X, "n_before": nb, "n_after": na, "emp_before": eb,
                              **{f"b_{k}": sb[k] for k in KINDS}, **{f"a_{k}": sa[k] for k in KINDS},
                              **{f"field_b_{k}": fb[k] for k in KINDS}, **{f"field_a_{k}": fa[k] for k in KINDS}})

    # ---- B. theory wanted, on method communities
    ml = assign_method(f"{RUN}@T{T}")
    myr = ml.index.map(year)
    for m in sorted(set(ml)):
        g = ml[ml.values == m]
        gy = g.index.map(year)
        b = g.index[(gy >= T - 2) & (gy <= T)].tolist()
        a = g.index[(gy > T) & (gy <= T + 3)].tolist()
        kb, ka = kind.reindex(b).dropna(), kind.reindex(a).dropna()
        if len(kb) < 10 or len(ka) < 5:
            continue
        recent = ((gy >= T - 1) & (gy <= T)).sum()
        older = ((gy >= T - 3) & (gy <= T - 2)).sum()
        meth_rows.append({"T": T, "m": m, "n_before": len(kb), "growth": np.log((recent + 1) / (older + 1)),
                          "emp_before": float(emp.reindex(b).fillna(False).mean()),
                          "theory_before": float((kb == "theory").mean()), "theory_after": float((ka == "theory").mean())})

    # ---- C. scarce kinds and their uptake (papers of year T, research types)
    rp = set(papers[papers.type.isin(C.RESEARCH)].paper_id)
    pT = [x for x in lab.index[yr == T] if x in rp and x in kind.index]
    fut = nonself[(nonself.cy > T) & (nonself.cy <= T + 3) & ~nonself.is_self].groupby("cited_id").size()
    # peer-normalized uptake: log citations minus the mean over the 15 nearest same-year papers
    sameyear = [x for x in lab.index[yr == T] if x in PA]
    Sv = AV[[PA[x] for x in sameyear]]
    lc = np.log1p(np.array([fut.get(x, 0) for x in sameyear], dtype=float))
    S = Sv @ Sv.T
    np.fill_diagonal(S, -2)
    peer = lc[np.argsort(-S, axis=1)[:, :15]].mean(axis=1)
    rel = pd.Series(lc - peer, index=sameyear)
    for X in sorted(set(lab)):
        sb, nb, _ = mix(before.index[before.values == X].tolist())
        if nb < 10:
            continue
        later = [x for x in after.index[after.values == X]]
        recentX = [x for x in before.index[before.values == X]]
        Fv, Rv = stack(later), stack(recentX)
        for x in [p for p in pT if lab[p] == X]:
            k = kind[x]
            ratio = sb[k] / max(fb[k], 1e-9)
            moved = False
            if x in IP and len(Fv):
                tm = team.get(x, frozenset())
                oth = [y for y in later if not (team.get(y, frozenset()) & tm)]
                Fo = stack(oth)
                if len(Fo):
                    fmax = float((Fo @ IV[IP[x]]).max())
                    Rw = stack([y for y in recentX if y != x])  # the community before, without the paper itself
                    rmax = float((Rw @ IV[IP[x]]).max()) if len(Rw) else -1.0
                    moved = (fmax >= C.THR_REUSE) and (fmax > rmax)
            paper_rows.append({"T": T, "X": X, "paper_id": x, "kind": k, "share_ratio": ratio, "scarce": ratio < 0.5,
                               "n_community": nb, "others_cites": fut.get(x, 0), "peer_rel": rel.get(x, np.nan),
                               "moved_towards": moved})
    print(T, len(comm_rows), len(meth_rows), len(paper_rows), flush=True)

CM = pd.DataFrame(comm_rows)
ME = pd.DataFrame(meth_rows)
PA_ = pd.DataFrame(paper_rows)

# ---------------------------------------------------------------- A
pred = {"persistence": [], "field drift": [], "lifecycle (own mix)": []}
errs = []
for T in sorted(CM["T"].unique()):
    tr, te = CM[CM["T"] != T], CM[CM["T"] == T]
    Xtr = tr[[f"b_{k}" for k in KINDS] + ["emp_before"]].to_numpy()
    Xte = te[[f"b_{k}" for k in KINDS] + ["emp_before"]].to_numpy()
    for k in KINDS:
        y_tr = (tr[f"a_{k}"] - tr[f"b_{k}"]) - (tr[f"field_a_{k}"] - tr[f"field_b_{k}"])
        r = Ridge(alpha=1.0).fit(Xtr, y_tr)
        drift = te[f"field_a_{k}"] - te[f"field_b_{k}"]
        p_life = te[f"b_{k}"] + drift + r.predict(Xte)
        for name, p in (("persistence", te[f"b_{k}"]), ("field drift", te[f"b_{k}"] + drift), ("lifecycle (own mix)", p_life)):
            errs.append(pd.DataFrame({"T": T, "X": te.X.to_numpy(), "kind": k, "model": name,
                                      "abs_error": (te[f"a_{k}"] - p).abs().to_numpy()}))
E = pd.concat(errs)
groups = {x: g.index.to_numpy() for x, g in E.reset_index(drop=True).groupby("X")}
Er = E.reset_index(drop=True)
tabA = []
for k in KINDS + ["all kinds"]:
    d = Er if k == "all kinds" else Er[Er.kind == k]
    m = d.groupby("model").abs_error.mean()
    piv = d.pivot_table(index=["T", "X", "kind"], columns="model", values="abs_error").reset_index()
    diff = (piv["lifecycle (own mix)"] - piv["field drift"]).to_numpy()
    gidx = {x: g.index.to_numpy() for x, g in piv.groupby("X")}
    rng = np.random.default_rng(0)
    bs = [diff[np.concatenate([gidx[x] for x in rng.choice(list(gidx), len(gidx))])].mean() for _ in range(500)]
    lo, hi = np.percentile(bs, [5, 95])
    tabA.append({"kind": k, "MAE persistence": m["persistence"], "MAE field drift": m["field drift"],
                 "MAE lifecycle": m["lifecycle (own mix)"], "lifecycle - field drift": diff.mean(),
                 "90% CI": f"[{lo:+.4f}, {hi:+.4f}]", "community-cutoffs": piv[["T", "X"]].drop_duplicates().shape[0]})
TA = pd.DataFrame(tabA)
# the lifecycle pattern itself, fitted on all cutoffs: how each kind's share moves, given the mix
coef = []
Xall = CM[[f"b_{k}" for k in KINDS] + ["emp_before"]].to_numpy()
for k in KINDS:
    y = (CM[f"a_{k}"] - CM[f"b_{k}"]) - (CM[f"field_a_{k}"] - CM[f"field_b_{k}"])
    r = Ridge(alpha=1.0).fit(Xall, y)
    coef.append({"change in share of": k, **{f"if more {j}": c for j, c in zip(KINDS + ["empirical"], r.coef_)}})
TC = pd.DataFrame(coef)

# ---------------------------------------------------------------- B
ME["gap"] = np.nan
for T, g in ME.groupby("T"):
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)  # noqa: E731
    ME.loc[g.index, "gap"] = z(g.emp_before) + z(g.growth) - z(g.theory_before)
ME["wanted"] = ME.gap >= ME.groupby("T").gap.transform(lambda s: s.quantile(0.75))
ME["d_theory"] = ME.theory_after - ME.theory_before
ME["th_bin"] = ME.groupby("T").theory_before.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
ME["size_bin"] = ME.groupby("T").n_before.transform(lambda s: pd.qcut(s.rank(method="first"), 2, labels=False))
strata = ME.groupby(["T", "th_bin", "size_bin"])
diffs, w = [], []
for _, g in strata:
    a, b = g[g.wanted], g[~g.wanted]
    if len(a) and len(b):
        diffs.append(a.d_theory.mean() - b.d_theory.mean())
        w.append(len(a))
rng = np.random.default_rng(0)
keys = list(strata.groups)
bs = []
for _ in range(500):
    pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
    dd_, ww = [], []
    for kk in pick:
        g = strata.get_group(kk)
        a, b = g[g.wanted], g[~g.wanted]
        if len(a) and len(b):
            dd_.append(a.d_theory.mean() - b.d_theory.mean())
            ww.append(len(a))
    if ww:
        bs.append(np.average(dd_, weights=ww))
TB = pd.DataFrame([{"method communities x cutoffs": len(ME), "theory wanted": int(ME.wanted.sum()),
                    "theory share change, wanted": ME[ME.wanted].d_theory.mean(),
                    "theory share change, others": ME[~ME.wanted].d_theory.mean(),
                    "matched difference (same theory share and size)": np.average(diffs, weights=w) if w else np.nan,
                    "90% CI": f"[{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}]" if bs else "",
                    "share where theory rose, wanted": float((ME[ME.wanted].d_theory > 0).mean()),
                    "share where theory rose, others": float((ME[~ME.wanted].d_theory > 0).mean())}])

# ---------------------------------------------------------------- C
PA_["taken_up"] = PA_.others_cites > PA_.groupby("T").others_cites.transform(lambda s: s.quantile(0.75))
PA_["peer_hit"] = PA_.peer_rel > PA_.groupby("T").peer_rel.transform(lambda s: s.quantile(0.75))
gidx = {x: g.index.to_numpy() for x, g in PA_.reset_index(drop=True).groupby("X")}
P2 = PA_.reset_index(drop=True)
tabC = []
for k in KINDS + ["all kinds (within-kind)"]:
    d = P2 if k.startswith("all") else P2[P2.kind == k]
    if d.scarce.sum() < 5 or (~d.scarce).sum() < 5:
        continue
    row = {"kind": k, "papers": len(d), "scarce": int(d.scarce.sum())}
    for y in ("taken_up", "peer_hit", "moved_towards"):
        v = d[y].astype(float).to_numpy()
        if k.startswith("all"):  # within-kind difference: centre each paper's outcome on its kind's mean
            v = v - d.groupby("kind")[y].transform("mean").to_numpy()
        a, b = d.scarce.to_numpy(), ~d.scarce.to_numpy()
        gi = [g for g in ({x: np.where(d.X.to_numpy() == x)[0] for x in d.X.unique()}).values()]
        lo, hi = C.boot_diff(gi, v, a, b)
        row[f"{y}: scarce"] = float(d[y][d.scarce].mean())
        row[f"{y}: common"] = float(d[y][~d.scarce].mean())
        row[f"{y}: difference"] = float(v[a].mean() - v[b].mean())
        row[f"{y}: 90% CI"] = f"[{lo:+.3f}, {hi:+.3f}]"
    tabC.append(row)
TCc = pd.DataFrame(tabC)

pd.set_option("display.width", 250)
print(TA.round(4).to_string(index=False))
print(TC.round(3).to_string(index=False))
print(TB.round(4).T.to_string())
print(TCc.round(3).T.to_string())
C.save(RUN, "exp15_lifecycle", "a_stage_predicts_mix", TA, title="Lifecycle: does a community's mix predict its next mix?",
       how_to_read="Each community at each cutoff (2019-2022) with at least 10 papers before and after. The share of each "
       "kind of contribution three years later is predicted three ways: **persistence** (no change), **field drift** "
       "(the community moves like the whole field) and **lifecycle** (its change depends on its own mix; fitted on the "
       "other cutoffs). Lower mean absolute error is better; a negative *lifecycle - field drift* with an interval "
       "below zero means a community's own stage carries information.")
C.save(RUN, "exp15_lifecycle", "a_pattern", TC, title="Lifecycle: how mixes move (all cutoffs)",
       how_to_read="Ridge coefficients of the change in each kind's share (beyond the field's drift) on the community's "
       "current shares. A negative diagonal is mean reversion; off-diagonal terms show which kinds follow which.")
C.save(RUN, "exp15_lifecycle", "b_theory_wanted", TB, title="Lifecycle: do 'theory wanted' gaps get theory?",
       how_to_read="Method communities whose use is growing and empirical but which have few theory papers (top quarter "
       "of that gap) against other method communities with the same theory share and size: the change in theory share "
       "from T-2..T to T+1..T+3. Matching on the current theory share removes regression to the mean.")
C.save(RUN, "exp15_lifecycle", "c_scarce_kind_uptake", TCc, title="Lifecycle: is the scarce kind of work taken up more?",
       how_to_read="Papers of year T whose kind of contribution is scarce in their community (under half the field's "
       "share) against papers of the same kind where it is common. Outcomes: **taken_up** (non-self citations in the "
       "top quarter of year-T papers), **peer_hit** (top quarter relative to the 15 nearest same-year papers) and "
       "**moved_towards** (the community's later work by others moved towards the paper's innovation, experiment 9). "
       "Intervals resample communities.")

# ---------------------------------------------------------------- C, sensitivity to the "scarce" cut
sens = []
for cut in (0.25, 0.5, 0.75, 1.0):
    for k in KINDS:
        d = P2[P2.kind == k]
        sc = d.share_ratio < cut
        if sc.sum() < 5 or (~sc).sum() < 5:
            continue
        gi = [np.where(d.X.to_numpy() == x)[0] for x in d.X.unique()]
        for y in ("taken_up", "peer_hit", "moved_towards"):
            v = d[y].astype(float).to_numpy()
            lo, hi = C.boot_diff(gi, v, sc.to_numpy(), ~sc.to_numpy())
            sens.append({"scarce if share below (x field)": cut, "kind": k, "outcome": y, "scarce papers": int(sc.sum()),
                         "difference": float(v[sc.to_numpy()].mean() - v[~sc.to_numpy()].mean()),
                         "90% CI": f"[{lo:+.3f}, {hi:+.3f}]"})
TS = pd.DataFrame(sens)
print(TS[TS.kind == "survey"].round(3).to_string(index=False))
C.save(RUN, "exp15_lifecycle", "c_sensitivity", TS, title="Lifecycle: scarce-kind uptake, by where 'scarce' is cut",
       how_to_read="The comparison of table C repeated with 'scarce' meaning a community share below 0.25, 0.5, 0.75 or 1 "
       "times the field's share of that kind. A finding that only appears at one cut is fragile.")
