"""Experiment 11: do big labs steer the field? When a prominent author or institution steps somewhere, does a
follow-up wave by *other* people come after?

Prominence is measured as of the step year T only (OpenAlex citations received by each author up to T,
fetched by fetch_counts_by_year.py): **big author** = the team's most-cited author; **big lab** = the team's
institution whose authors in the map have the most citations up to T. "Big" = top 10% of that year's papers.

A. Paper level (T = 2018..2022, outcomes through 2025). For each paper x of year T, its neighbourhood is the
   ball around it reaching its 15th nearest earlier paper (abstract vectors). The **wave** is the share of
   T+1..T+3 papers inside the ball written by none of x's authors, relative to the share of T-2..T papers
   inside it (log ratio); hit = top quarter for T and at least 3 such papers. A second outcome counts
   citations to x from papers sharing no author with it. The base model knows x's novelty, local density
   and momentum, early global citations, team size, references and route; the question is what prominence
   adds, overall and for novel papers (a "step" into a new direction).
B. Community level (cutoff runs 2019..2022). Entrants of community X are authors whose first paper in X
   is from T-1..T after earlier papers elsewhere in the map. Does the prominence of entrants predict X's
   growth (mean share T+1..T+3 vs T-2..T) beyond momentum, size and the number of entrants?

Usage: --run-id (default dro-pilot); LANDSCAPE_DB may point at a copy of the database.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from landscape import embed, experiments
from landscape.backtest import Assigner
from landscape.cli import _params

RUN = experiments.run_arg()
WORK = experiments.work_dir(RUN)
con = experiments.connect_ro()
P = _params(con, RUN)
VER = P.embedding.version

papers = con.execute("""SELECT v.paper_id, v.route, p.year, p.type FROM provenance v JOIN papers p USING (paper_id)
                        WHERE v.run_id = ? AND p.year IS NOT NULL""", [RUN]).df()
year = papers.set_index("paper_id").year
au = con.execute("""SELECT s.paper_id, s.author_id, s.institution_ids FROM authorships s JOIN provenance v
                    ON v.paper_id = s.paper_id AND v.run_id = ?""", [RUN]).df()
au["year"] = au.paper_id.map(year)
team = au.groupby("paper_id").author_id.apply(frozenset)
refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v ON v.paper_id = r.citing_id
                      AND v.run_id = ?""", [RUN]).df()
nrefs = refs.groupby("citing_id").size()
corpus = set(papers.paper_id)
rc = refs[refs.cited_id.isin(corpus)].assign(cy=lambda d: d.citing_id.map(year))
rc["self"] = [bool(team.get(a, frozenset()) & team.get(b, frozenset())) for a, b in zip(rc.citing_id, rc.cited_id)]

ac = pd.read_parquet(WORK / "oa_author_counts.parquet")
ac = ac[(ac.year >= 2010) & (ac.year <= 2026)]
A = ac.pivot_table(index="author_id", columns="year", values="cites", aggfunc="sum").fillna(0)
wc = pd.read_parquet(WORK / "oa_work_counts.parquet").pivot_table(index="paper_id", columns="year", values="cites",
                                                                    aggfunc="sum").fillna(0)


def upto(M, ids, T, lo=0):
    cols = [c for c in M.columns if lo <= c <= T]
    return M.reindex(ids)[cols].sum(axis=1).fillna(0)


ids, V = embed.load_vectors("abstract", ids=papers.paper_id.tolist(), version=VER)
V = embed.truncate(V, 256)
pos = {x: k for k, x in enumerate(ids)}
yrs = np.array([year[x] for x in ids])


def prominence(T, pids):
    """Big author and big lab scores of each paper, as of T."""
    a_cum = upto(A, au.author_id.unique().tolist(), T)
    x = au[au.paper_id.isin(pids)].assign(c=lambda d: d.author_id.map(a_cum).fillna(0))
    author = x.groupby("paper_id").c.max()
    # institution strength: citations up to T of the institution's authors seen in the map up to T
    hist = au[au.year <= T].explode("institution_ids").dropna(subset=["institution_ids"])
    hist = hist.drop_duplicates(["institution_ids", "author_id"]).assign(c=lambda d: d.author_id.map(a_cum).fillna(0))
    inst = hist.groupby("institution_ids").c.sum()
    xi = x.explode("institution_ids").dropna(subset=["institution_ids"])
    lab = xi.assign(s=xi.institution_ids.map(inst).fillna(0)).groupby("paper_id").s.max()
    return np.log1p(pd.Series(pids).map(author).fillna(0).to_numpy()), np.log1p(pd.Series(pids).map(lab).fillna(0).to_numpy())


# ---------------------------------------------------------------- A. paper-level steps
rows = []
for T in range(2018, 2023):
    xs = papers[(papers.year == T) & papers.type.isin(["article", "preprint", "conference-paper", "proceedings-article"])
                & papers.paper_id.isin(pos)].paper_id.tolist()
    xi = np.array([pos[x] for x in xs])
    pre = np.where(yrs <= T)[0]
    S = V[xi] @ V[pre].T
    S[np.arange(len(xi)), np.searchsorted(pre, xi)] = -2
    radius = np.sort(S, axis=1)[:, -15]
    before_idx = np.where((yrs >= T - 2) & (yrs <= T))[0]
    after_idx = np.where((yrs > T) & (yrs <= T + 3))[0]
    in_b = (V[xi] @ V[before_idx].T) >= radius[:, None]
    in_a = (V[xi] @ V[after_idx].T) >= radius[:, None]
    F = pd.DataFrame({"paper_id": xs, "T": T})
    others_a, others_b = [], []
    for k, x in enumerate(xs):
        tm = team.get(x, frozenset())
        a_ids = after_idx[in_a[k]]
        b_ids = before_idx[in_b[k]]
        others_a.append(sum(1 for j in a_ids if not (team.get(ids[j], frozenset()) & tm)))
        others_b.append(sum(1 for j in b_ids if not (team.get(ids[j], frozenset()) & tm) and ids[j] != x))
    F["n_after_others"], F["n_before_others"] = others_a, others_b
    F["wave"] = np.log((F.n_after_others / len(after_idx) + 1e-4) / (F.n_before_others / len(before_idx) + 1e-4))
    fut = rc[(rc.cy > T) & (rc.cy <= T + 3) & ~rc.self].groupby("cited_id").size()
    F["others_citing"] = F.paper_id.map(fut).fillna(0)
    # base
    F["novelty"] = 1 - np.max(np.where(yrs[pre][None, :] < T, S, -2), axis=1)
    F["l_density"] = np.log1p(F.n_before_others)
    yT, yT1 = np.where(yrs == T)[0], np.where(yrs == T - 1)[0]
    F["local_momentum"] = np.log(((V[xi] @ V[yT].T >= radius[:, None]).mean(axis=1) + 1e-3)
                                 / ((V[xi] @ V[yT1].T >= radius[:, None]).mean(axis=1) + 1e-3))
    F["l_early_global"] = np.log1p(upto(wc, xs, T, T).to_numpy())
    F["l_team"] = np.log1p(F.paper_id.map(lambda x: len(team.get(x, ()))))
    F["l_refs"] = np.log1p(F.paper_id.map(nrefs).fillna(0))
    F["route_forward"] = (F.paper_id.map(papers.set_index("paper_id").route) == "forward").astype(float)
    F["big_author_score"], F["big_lab_score"] = prominence(T, xs)
    # a step *for the team*: distance from the most prominent author's own earlier papers in the map
    a_cum = upto(A, au.author_id.unique().tolist(), T)
    prior = au[au.year < T]
    lead = au[au.paper_id.isin(xs)].assign(c=lambda d: d.author_id.map(a_cum).fillna(0)) \
        .sort_values("c", ascending=False).drop_duplicates("paper_id").set_index("paper_id").author_id
    own = []
    for x in xs:
        a = lead.get(x)
        hist = [pos[p] for p in prior[prior.author_id == a].paper_id if p in pos] if a is not None else []
        if len(hist) >= 2:
            c = V[hist].mean(axis=0)
            own.append(1 - float(V[pos[x]] @ (c / np.linalg.norm(c))))
        else:
            own.append(np.nan)
    F["own_step"] = own
    rows.append(F)
    print(T, len(F), flush=True)
D = pd.concat(rows, ignore_index=True)
D["big_author"] = D.big_author_score >= D.groupby("T").big_author_score.transform(lambda s: s.quantile(0.9))
D["big_lab"] = D.big_lab_score >= D.groupby("T").big_lab_score.transform(lambda s: s.quantile(0.9))
D["novel"] = D.novelty >= D.groupby("T").novelty.transform(lambda s: s.quantile(0.75))
D["has_history"] = D.own_step.notna()
D["own_step"] = D.own_step.fillna(D.own_step.median())
step_cut = D[D.has_history].groupby("T").own_step.quantile(0.75)
D["new_for_team"] = D.has_history & (D.own_step >= D["T"].map(step_cut))
D["hit_wave"] = (D.wave >= D.groupby("T").wave.transform(lambda s: s.quantile(0.75))) & (D.n_after_others >= 3)
D["hit_cited"] = D.others_citing > D.groupby("T").others_citing.transform(lambda s: s.quantile(0.75))
for c in ("big_author", "big_lab", "novel"):
    D[c + "_f"] = D[c].astype(float)
D["step_author"] = D.big_author_f * D.novel_f
D["step_lab"] = D.big_lab_f * D.novel_f
D["new_for_team_f"] = D.new_for_team.astype(float)
D["has_history_f"] = D.has_history.astype(float)
D["own_step_author"] = D.big_author_f * D.new_for_team_f
D.to_parquet(WORK / "exp11_papers.parquet", index=False)

BASE = ["novelty", "l_density", "local_momentum", "l_early_global", "l_team", "l_refs", "route_forward"]
MODELS = {"base": BASE,
          "+ big author": BASE + ["big_author_score"],
          "+ big lab": BASE + ["big_lab_score"],
          "+ big author x novel (a step)": BASE + ["big_author_score", "novel_f", "step_author"],
          "+ big lab x novel (a step)": BASE + ["big_lab_score", "novel_f", "step_lab"],
          "+ big author x new for the team": BASE + ["big_author_score", "has_history_f", "new_for_team_f", "own_step_author"]}


def loco(d, feats, y):
    out = pd.Series(np.nan, index=d.index)
    for T in sorted(d["T"].unique()):
        tr, te = d["T"] != T, d["T"] == T
        s = StandardScaler().fit(d.loc[tr, feats])
        m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(s.transform(d.loc[tr, feats]), d.loc[tr, y])
        out[te] = m.decision_function(s.transform(d.loc[te, feats]))
    return out


def boot(y, a, b, n=300):
    rng = np.random.default_rng(0)
    g = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            g.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(g, [5, 95])


res = []
for y, label in (("hit_wave", "follow-up wave by others"), ("hit_cited", "citations from others")):
    for pop_name, d in (("all papers", D), ("forward route", D[D.route_forward == 1])):
        d = d.reset_index(drop=True)
        yy = d[y].to_numpy().astype(int)
        sc = {k: loco(d, f, y) for k, f in MODELS.items()}
        b = sc["base"].to_numpy()
        for k, s in sc.items():
            lo, hi = boot(yy, s.to_numpy(), b) if k != "base" else (0.0, 0.0)
            res.append({"outcome": label, "population": pop_name, "model": k, "AUC": roc_auc_score(yy, s),
                        "gain vs base": roc_auc_score(yy, s) - roc_auc_score(yy, b), "gain 90% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                        "papers": len(d), "base_rate": yy.mean()})
R = pd.DataFrame(res)

# the contrast a reader wants: novel steps by big vs other authors / labs, same year
tab = []
for who, step, step_name in (("big_author", "novel", "novel for the field"), ("big_lab", "novel", "novel for the field"),
                             ("big_author", "new_for_team", "new for the lead author")):
    for nov in (True, False):
        for big in (True, False):
            g = D[(D[step] == nov) & (D[who] == big) & (D.has_history if step == "new_for_team" else True)]
            tab.append({"who": who.replace("_", " "), "paper": step_name if nov else "not " + step_name,
                        "group": "top 10%" if big else "others", "papers": len(g), "wave hit rate": g.hit_wave.mean(),
                        "median others' papers nearby after": g.n_after_others.median(),
                        "cited-by-others hit rate": g.hit_cited.mean()})
TAB = pd.DataFrame(tab)

# ---------------------------------------------------------------- B. community level
crow = []
for T in range(2019, 2023):
    Asg = Assigner(con, f"{RUN}@T{T}", VER)
    lab = Asg.assign("problem", 0.5, papers.paper_id.tolist())
    a2 = au.assign(X=au.paper_id.map(lab)).dropna(subset=["X"])
    tot = a2.drop_duplicates("paper_id").groupby("year").size()
    py = a2.drop_duplicates("paper_id").groupby(["X", "year"]).size().unstack(fill_value=0)
    share = py.div(tot, axis=1).fillna(0)
    first_in = a2[a2.year <= T].groupby(["author_id", "X"]).year.min().rename("first").reset_index()
    first_any = a2[a2.year <= T].groupby("author_id").year.min()
    ent = first_in[(first_in["first"] >= T - 1) & (first_in.author_id.map(first_any) < T - 1)]
    a_cum = np.log1p(upto(A, ent.author_id.unique().tolist(), T))
    ent = ent.assign(p=ent.author_id.map(a_cum))
    big_cut = np.log1p(upto(A, a2[a2.year <= T].author_id.unique().tolist(), T)).quantile(0.9)
    for X in share.index:
        e = ent[ent.X == X]
        s = share.loc[X]
        crow.append({"T": T, "X": X, "size": int(py.loc[X, [c for c in py.columns if c <= T]].sum()),
                     "momentum": s.get(T, 0) - s.get(T - 1, 0), "entrants": len(e),
                     "big_entrants": int((e.p >= big_cut).sum()), "entrant_prominence": float(e.p.max()) if len(e) else 0.0,
                     "growth": np.log((s[[c for c in s.index if T < c <= T + 3]].mean() + 1e-4)
                                      / (s[[c for c in s.index if T - 2 <= c <= T]].mean() + 1e-4))})
C = pd.DataFrame(crow)
C = C[C["size"] >= 10]
C["l_size"], C["l_entrants"] = np.log(C["size"]), np.log1p(C.entrants)
# partial association: residualize growth and the entrant measures on momentum, size and number of entrants
Z = C[["momentum", "l_size", "l_entrants"]].to_numpy()
Z = np.column_stack([np.ones(len(Z)), Z])
resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
g_r = resid(C.growth.to_numpy())
crows = []
rng = np.random.default_rng(0)
for f in ("big_entrants", "entrant_prominence", "entrants", "momentum"):
    raw = spearmanr(C[f], C.growth).statistic
    part = spearmanr(resid(C[f].to_numpy().astype(float)), g_r).statistic if f in ("big_entrants", "entrant_prominence") else np.nan
    bs = []
    for _ in range(500):
        i = rng.integers(0, len(C), len(C))
        if f in ("big_entrants", "entrant_prominence"):
            bs.append(spearmanr(resid(C[f].to_numpy().astype(float))[i], g_r[i]).statistic)
    ci = np.nanpercentile(bs, [5, 95]) if bs else (np.nan, np.nan)
    crows.append({"feature": f, "rho with growth": raw, "partial rho (momentum, size, entrants held)": part,
                  "partial 90% CI": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if bs else "", "communities x cutoffs": len(C)})
CR = pd.DataFrame(crows)

pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
print(TAB.round(3).to_string(index=False))
print(CR.round(3).to_string(index=False))
experiments.save(RUN, "exp11_leaders", "paper_steps_models", R, title="Big labs: does a follow-up wave come after their papers?",
                 how_to_read="Each paper from 2018-2022. **Follow-up wave**: the share of the next three years' papers "
                 "in the paper's neighbourhood written by *other* people, against the share in the three years before "
                 "(top quarter and at least 3 such papers). **Citations from others**: citations from papers sharing no "
                 "author. The **base** model knows the paper's novelty, how crowded and how fast-growing its "
                 "neighbourhood is, its early global citations, team size, references and route. **big author / big "
                 "lab** add the team's prominence as of the paper's year; **x novel** adds whether a prominent team "
                 "stepped somewhere new (novelty top quarter). Gains with a 90% interval above zero are real here.")
experiments.save(RUN, "exp11_leaders", "paper_steps_contrast", TAB, title="Big labs: novel steps by big vs other teams",
                 how_to_read="Hit rates for the follow-up wave and for citations from others, comparing papers by the "
                 "top 10% most prominent authors or labs of their year with all others, for novel papers (a step into "
                 "a new direction) and the rest. Raw rates, without controls: read with the model table.")
experiments.save(RUN, "exp11_leaders", "community_entries", CR, title="Big labs: do communities grow after big names enter?",
                 how_to_read="Communities at each cutoff (2019-2022). Entrants: authors whose first paper in the "
                 "community is from the last two years, after earlier papers elsewhere in the map. **big_entrants** "
                 "counts entrants in the top 10% by citations up to T. The **partial rho** holds momentum, size and the "
                 "number of entrants fixed: does the prominence of who enters predict growth beyond how many enter?")
