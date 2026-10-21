"""Offline Phase 8c experiments and review artifacts. No downloads, LLMs or DB writes.

Run with ``landscape research-lab --run-id dro-pilot``. Candidate scores remain experimental;
the only production change is conservative alias exclusion in paper bridges.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import embed
from .backtest import Assigner
from .config import project_dir
from .identity import work_aliases


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def lab_dir(run_id: str) -> Path:
    return project_dir(run_id, "research_lab", mkdir=False)


class Lab:
    def __init__(self, con, run_id, params):
        self.con, self.run, self.p = con, run_id, params
        self.papers = con.execute("""SELECT p.paper_id, p.doi, p.title, p.year, p.type, v.route,
            p.abstract_raw IS NOT NULL AS has_abstract FROM papers p JOIN provenance v USING(paper_id)
            WHERE v.run_id = ? ORDER BY p.paper_id""", [run_id]).df()
        self.year = self.papers.set_index("paper_id").year
        self.ids = self.papers.paper_id.tolist()
        self.authors = con.execute("""SELECT a.paper_id, a.author_id FROM authorships a
            JOIN provenance v USING(paper_id) WHERE v.run_id = ?""", [run_id]).df()
        self.refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r
            JOIN provenance v ON v.paper_id = r.citing_id WHERE v.run_id = ?""", [run_id]).df()
        self.refs["year"] = self.refs.citing_id.map(self.year)
        self.facets = con.execute("""SELECT f.paper_id, f.facet, f.text FROM facets f
            JOIN provenance v USING(paper_id) WHERE v.run_id = ? AND f.model_id = ?
            AND f.prompt_version = ?""", [run_id, params.extraction.model,
                                        params.extraction.prompt_version]).df()
        self.text = self.facets.pivot(index="paper_id", columns="facet", values="text")
        self.vectors = {}

    def vectors_at(self, facet, cutoff):
        if facet not in self.vectors:
            ids, v = embed.load_vectors(facet, self.ids, self.p.embedding.version)
            self.vectors[facet] = pd.Index(ids), v
        ids, v = self.vectors[facet]
        keep = self.year.reindex(ids).to_numpy() <= cutoff
        return ids[keep], v[keep]

    def assignment(self, cutoff, facet="problem", alpha=0.5, live=False):
        run = self.run if live else f"{self.run}@T{cutoff}"
        return Assigner(self.con, run, self.p.embedding.version).assign(facet, alpha, self.ids)


def corpus_audit(lab, end_year):
    aliases = work_aliases(lab.papers, lab.authors)
    groups = pd.Series(aliases).value_counts()
    counts = lab.papers.groupby(["year", "route"]).size().rename("n").reset_index()
    limits = lab.text.get("limitations", pd.Series(dtype=object)).notna().sum()
    audit = {"papers": len(lab.papers), "unique_works": len(groups),
             "alias_groups": int((groups > 1).sum()), "redundant_records": int((groups - 1).sum()),
             "abstract_coverage": float(lab.papers.has_abstract.mean()),
             "explicit_limitations": int(limits), "complete_through": end_year}
    geometry = []
    rng = np.random.default_rng(42)
    for facet in ("abstract", "problem", "method", "innovation", "limitations"):
        ids, v = lab.vectors_at(facet, end_year)
        if len(ids) < 2:
            continue
        a, b = rng.integers(len(v), size=(2, 10000))
        keep = a != b
        a, b = a[keep], b[keep]
        for dim in (256, 768):
            x = unit(v[:, :dim])
            mu = x.mean(0)
            for name, z in (("raw", x), ("centered", unit(x - mu))):
                cos = (z[a] * z[b]).sum(1)
                geometry.append({"facet": facet, "dim": dim, "geometry": name,
                                 "q05": float(np.quantile(cos, .05)), "median": float(np.median(cos)),
                                 "q95": float(np.quantile(cos, .95)), "mean_vector_norm": float(np.linalg.norm(mu))})
    # Check the actual shipped list, using aliases known at that list's cutoff.
    signals = lab.con.execute("""SELECT subject, cutoff_year FROM signals WHERE run_id = ?
        AND signal_type = 'bridge_paper' AND variant = 'all'""", [lab.run]).df()
    false_missing, listed = [], []
    for cutoff, rows in signals.groupby("cutoff_year"):
        available = lab.papers[lab.papers.year <= cutoff]
        identity = work_aliases(available, lab.authors)
        membership = lab.con.execute("""SELECT paper_id, cluster_id FROM paper_cluster WHERE run_id = ?
            AND facet = 'problem' AND alpha = ? AND resolution = 'medium'""",
            [lab.run, lab.p.signals.bridge_alpha]).df().set_index("paper_id").cluster_id
        e = lab.refs[lab.refs.year <= cutoff].copy()
        e["cluster"] = e.citing_id.map(membership)
        e["work"] = e.cited_id.map(lambda i: identity.get(i, i))
        cited = e.groupby("cluster").work.agg(set)
        for row in rows.itertuples():
            s = json.loads(row.subject)
            pid = s["paper_id"]
            listed.append(pid)
            if identity.get(pid, pid) in cited.get(s["cluster_id"], set()):
                false_missing.append({"cluster_id": s["cluster_id"], "paper_id": pid})
    audit.update(bridge_rows=len(listed), bridge_distinct_papers=len(set(listed)),
                 already_cited_alias_recommendations=len(false_missing))
    return audit, counts, pd.DataFrame(geometry), pd.DataFrame(false_missing)


def dynamics(lab, end_year):
    """Exact symmetric decomposition: within-route change + route-mix change.

    This is a sensitivity analysis, never an inverse-probability correction. Inclusion
    probabilities are unknown; provenance only records the first route that found a work.
    """
    p = lab.papers[lab.papers.year.between(end_year - 5, end_year)].copy()
    aliases = work_aliases(lab.papers[lab.papers.year <= end_year], lab.authors)
    # One record per work; prefer earliest appearance, keeping that record's route.
    p["work"] = p.paper_id.map(aliases)
    p = p.sort_values(["year", "paper_id"]).drop_duplicates("work")
    assigned = lab.con.execute("""SELECT paper_id, cluster_id FROM paper_cluster WHERE run_id = ?
        AND facet = 'problem' AND alpha = 0.3 AND resolution = 'medium'""", [lab.run]).df()
    p = p.merge(assigned, on="paper_id").dropna(subset=["cluster_id"])
    p["period"] = np.where(p.year <= end_year - 3, "before", "after")
    totals = pd.crosstab(p.route, p.period).reindex(columns=["before", "after"], fill_value=0)
    routes = totals.index[(totals.min(axis=1) >= 20)].tolist()
    p = p[p.route.isin(routes)]
    totals = pd.crosstab(p.route, p.period).reindex(index=routes, columns=["before", "after"], fill_value=0)
    weights = totals / totals.sum(0)
    labels = lab.con.execute("SELECT cluster_id, label, kind FROM clusters WHERE run_id = ?",
                             [lab.run]).df().set_index("cluster_id")
    rows, series = [], []
    for cluster, members in p.groupby("cluster_id"):
        n = pd.crosstab(members.route, members.period).reindex(index=routes, columns=totals.columns, fill_value=0)
        shares = n / totals
        raw_before = float((weights.before * shares.before).sum())
        raw_after = float((weights.after * shares.after).sum())
        within = float(((weights.before + weights.after) / 2 * (shares.after - shares.before)).sum())
        composition = float(((shares.before + shares.after) / 2 * (weights.after - weights.before)).sum())
        changes = shares.after - shares.before
        rows.append({"cluster_id": cluster, "label": labels.loc[cluster, "label"],
                     "kind": labels.loc[cluster, "kind"], "before": raw_before, "after": raw_after,
                     "raw_change": raw_after - raw_before, "within_route_change": within,
                     "composition_change": composition, "sign_flip": (raw_after - raw_before) * within < 0,
                     "route_sign_agreement": max(float((changes > 0).mean()), float((changes < 0).mean())),
                     "before_n": int(n.before.sum()), "after_n": int(n.after.sum()),
                     "routes": ", ".join(routes), "evidence": members.sort_values("year", ascending=False).paper_id.head(8).tolist()})
        for year in sorted(p.year.unique()):
            total = p[p.year == year]
            m = members[members.year == year]
            series.append({"cluster_id": cluster, "year": int(year), "route": "all",
                           "share": len(m) / len(total), "n": len(m), "denominator": len(total)})
            for route in routes:
                denom = int((total.route == route).sum())
                count = int((m.route == route).sum())
                series.append({"cluster_id": cluster, "year": int(year), "route": route,
                               "share": count / denom if denom else np.nan, "n": count, "denominator": denom})
    return pd.DataFrame(rows), pd.DataFrame(series), weights.reset_index()


def bridge_candidates(lab, cutoff, live=False):
    """Cutoff-fitted geometry, alias-aware candidates and distinct-work adoption outcomes."""
    assignment = lab.assignment(cutoff, live=live)
    past = assignment[assignment.index.map(lab.year) <= cutoff]
    available = lab.papers[lab.papers.year <= cutoff]
    aliases = work_aliases(available, lab.authors)
    # Outcomes may use future aliases, but feature construction and filtering above cannot.
    future_aliases = work_aliases(lab.papers[lab.papers.year <= cutoff + 3], lab.authors)
    refs = lab.refs[lab.refs.year <= cutoff].copy()
    refs["citing_work"] = refs.citing_id.map(lambda x: aliases.get(x, x))
    refs["cited_work"] = refs.cited_id.map(lambda x: aliases.get(x, x))
    edges = refs.drop_duplicates(["citing_work", "cited_work"])
    edges = edges[edges.citing_work != edges.cited_work]
    pop = edges.groupby("cited_work").size()
    cited = edges.assign(X=edges.citing_id.map(past)).groupby("X").cited_work.agg(set)
    future = lab.refs[lab.refs.year.between(cutoff + 1, cutoff + 3)].copy()
    future["work"] = future.cited_id.map(lambda x: future_aliases.get(x, x))
    future = future[future.citing_id.map(lambda x: future_aliases.get(x, x)) != future.work]
    adopted = future.assign(X=future.citing_id.map(assignment)).groupby("X").work.agg(set)

    iid, iv = lab.vectors_at("innovation", cutoff)
    qid, qv = lab.vectors_at("problem", cutoff)
    lid, lv = lab.vectors_at("limitations", cutoff)
    if len(iv) == 0 or len(qv) == 0:
        return pd.DataFrame()
    # Remove only one pooled/facet mean, estimated on the past. This is an experiment,
    # not a claim that centering is universally appropriate for this embedding model.
    ic, qc, lc = unit(iv - iv.mean(0)), unit(qv - qv.mean(0)), unit(lv - lv.mean(0)) if len(lv) else lv
    iq, qq, lq = ({x: i for i, x in enumerate(ids)} for ids in (iid, qid, lid))
    methods = lab.assignment(cutoff, facet="method", live=live)
    # Canonicalize reference targets for coupling, while retaining source paper identity.
    rid = {x: i for i, x in enumerate(available.paper_id)}
    wid = {x: i for i, x in enumerate(refs.cited_work.unique())}
    r = sparse.csr_matrix((np.ones(len(refs)), (refs.citing_id.map(rid), refs.cited_work.map(wid))),
                          shape=(len(rid), max(1, len(wid))))
    r.data[:] = 1
    irows = np.array([rid[x] for x in iid])
    global_lim = unit(lv.mean(0)) if len(lv) else np.zeros(iv.shape[1])
    generic_cut = np.quantile(lv @ global_lim, .8) if len(lv) else 0
    # Density of each innovation's limitation matches: the CSLS hub penalty.
    all_sim = iv @ lv.T if len(lv) else np.zeros((len(iv), 0))
    inn_density = np.sort(all_sim, axis=1)[:, -min(10, len(lv)):].mean(1) if len(lv) else np.zeros(len(iv))
    lim_density = np.sort(all_sim, axis=0)[-min(10, len(iv)):, :].mean(0) if len(lv) else np.array([])
    rows = []
    for x in sorted(past.unique()):
        members = past.index[past == x]
        # A work may straddle clusters because its versions have different abstracts.
        member_works = {aliases.get(i, i) for i in members}
        q = [qq[i] for i in members if i in qq]
        if len(q) < 5:
            continue
        l = [lq[i] for i in members if i in lq and lv[lq[i]] @ global_lim <= generic_cut]
        centroid, centered_centroid = unit(qv[q].mean(0)), unit(qc[q].mean(0))
        forbidden = member_works | cited.get(x, set())
        # Keep a single available representative per work, preferring earlier versions.
        candidates = available[available.paper_id.isin(iid)].copy()
        candidates["work"] = candidates.paper_id.map(aliases)
        candidates = candidates[~candidates.work.isin(forbidden)]
        candidates = candidates.sort_values(["year", "paper_id"]).drop_duplicates("work")
        idx = np.array([iq[i] for i in candidates.paper_id], dtype=int)
        if not len(idx):
            continue
        source = [rid[i] for i in members]
        # Duplicate versions contribute once per source work.
        source = list(dict((aliases.get(available.paper_id.iloc[k], available.paper_id.iloc[k]), k) for k in source).values())
        coupling = np.asarray(r[source].sum(0) @ r[irows[idx]].T).ravel() / max(1, len(source))
        raw_prob = iv[idx] @ centroid
        centered_prob = ic[idx] @ centered_centroid
        raw_lim, centered_lim, csls = raw_prob.copy(), centered_prob.copy(), np.zeros(len(idx))
        best = [None] * len(idx)
        if l:
            s = all_sim[np.ix_(idx, l)]
            raw_lim = np.sort(s, axis=1)[:, -3:].mean(1)
            centered_lim = np.sort(ic[idx] @ lc[l].T, axis=1)[:, -3:].mean(1)
            contrast = 2 * s - inn_density[idx, None] - lim_density[l][None, :]
            csls = np.sort(contrast, axis=1)[:, -3:].mean(1)
            best = [lid[l[k]] for k in np.argmax(contrast, axis=1)]
        for j, (pid, work) in enumerate(zip(candidates.paper_id, candidates.work)):
            future_work = future_aliases.get(pid, pid)
            rows.append({"T": cutoff, "X": x, "i": pid, "work": work,
                         "method": methods.get(pid, "unassigned"), "pop": int(pop.get(work, 0)),
                         "lpop": float(np.log1p(pop.get(work, 0))), "lcoup": float(np.log1p(coupling[j] * 100)),
                         "prob_sim": float(raw_prob[j]), "lim_sim3_f": float(raw_lim[j]),
                         "age": int(cutoff - lab.year[pid]), "centered_prob": float(centered_prob[j]),
                         "centered_lim": float(centered_lim[j]), "csls": float(csls[j]), "n_lim": len(l),
                         "best_limitation": best[j], "hit": future_work in adopted.get(x, set())})
    return pd.DataFrame(rows)


BASE_FEATURES = ["lpop", "lcoup", "prob_sim", "lim_sim3_f", "age"]
EXTRA_FEATURES = BASE_FEATURES + ["centered_prob", "centered_lim", "csls"]


def fitted_scores(train, test, features):
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    model.fit(scaler.transform(train[features]), train.hit)
    return model.decision_function(scaler.transform(test[features]))


def select_diverse(group, score, k=10, max_per_method=3):
    """Transparent quota: at most 3 from one method cluster; never fill with quota violations."""
    chosen, counts = [], {}
    for idx, row in group.sort_values(score, ascending=False, kind="stable").iterrows():
        method = row.method if isinstance(row.method, str) else "unassigned"
        if counts.get(method, 0) >= max_per_method:
            continue
        chosen.append(idx)
        counts[method] = counts.get(method, 0) + 1
        if len(chosen) == k:
            break
    return group.loc[chosen]


def evaluate_bridges(data, complete_through):
    data = data.copy()
    data["baseline_model"] = np.nan
    data["contrastive_model"] = np.nan
    metrics, comparisons = [], []

    def measure(test, protocol, cutoff):
        for score in ("lpop", "lcoup", "prob_sim", "centered_prob", "lim_sim3_f", "centered_lim", "csls",
                      "baseline_model", "contrastive_model", "diverse_baseline"):
            feature = "baseline_model" if score == "diverse_baseline" else score
            chosen = []
            p_by_cluster, coverage = [], []
            for x, group in test.groupby("X"):
                top = select_diverse(group, feature) if score == "diverse_baseline" else group.nlargest(10, feature)
                # Slots left empty count as misses; list-shortening cannot inflate precision.
                precision = int(top.hit.sum()) / min(10, len(group))
                chosen.extend(top.i)
                p_by_cluster.append(precision)
                coverage.append(top.method.nunique())
                comparisons.append({"protocol": protocol, "T": cutoff, "X": x, "ranking": score,
                                    "p@10": precision, "methods@10": top.method.nunique(), "filled": len(top)})
            y, s = test.hit, test[feature]
            tail = test[test["pop"] <= 1]
            metrics.append({"protocol": protocol, "T": cutoff, "ranking": score, "n": len(test),
                            "hits": int(y.sum()), "base_rate": float(y.mean()), "p@10": float(np.mean(p_by_cluster)),
                            "methods@10": float(np.mean(coverage)), "distinct_papers": len(set(chosen)),
                            "auc": float(roc_auc_score(y, s)) if y.nunique() > 1 else None,
                            "average_precision": float(average_precision_score(y, s)),
                            "tail_auc": float(roc_auc_score(tail.hit, tail[feature])) if tail.hit.nunique() > 1 else None})

    cutoffs = sorted(data["T"].unique())
    for cutoff in cutoffs:
        train, test = data[data["T"] != cutoff], data[data["T"] == cutoff].copy()
        for name, features in (("baseline_model", BASE_FEATURES), ("contrastive_model", EXTRA_FEATURES)):
            test[name] = fitted_scores(train, test, features)
            data.loc[test.index, name] = test[name]
        measure(test, "LOCO retrospective", int(cutoff))
        # Strict availability of labels: the entire 3-year outcome window must have elapsed.
        train = data[data["T"] + 3 <= cutoff]
        if len(train) and train.hit.nunique() > 1:
            for name, features in (("baseline_model", BASE_FEATURES), ("contrastive_model", EXTRA_FEATURES)):
                test[name] = fitted_scores(train, test, features)
            maturity = "complete outcome" if cutoff + 3 <= complete_through else "partial outcome"
            measure(test, f"purged chronological ({maturity})", int(cutoff))
    return data, pd.DataFrame(metrics), pd.DataFrame(comparisons)


def flow_opportunities(lab, end_year):
    """Find semantically near communities with fewer citations than a degree null expects.

    An expected edge is exposure, not proof a connection should exist. Require expected
    count >= 5 and report actual/expected alongside evidence for human review.
    """
    a = lab.con.execute("""SELECT paper_id, cluster_id FROM paper_cluster WHERE run_id = ?
        AND facet = 'problem' AND alpha = 0.3 AND resolution = 'medium'
        AND cluster_id IS NOT NULL""", [lab.run]).df().set_index("paper_id").cluster_id
    ids, v = lab.vectors_at("problem", end_year)
    x = unit(v - v.mean(0))
    frame = pd.DataFrame({"paper_id": ids, "cluster_id": a.reindex(ids).values}).dropna()
    pos = {pid: k for k, pid in enumerate(ids)}
    clusters = sorted(frame.cluster_id.unique())
    centroids = unit(np.stack([x[[pos[i] for i in frame[frame.cluster_id == c].paper_id]].mean(0) for c in clusters]))
    similarity = centroids @ centroids.T
    e = lab.refs[lab.refs.year.between(end_year - 2, end_year)].copy()
    aliases = work_aliases(lab.papers[lab.papers.year <= end_year], lab.authors)
    e = e[e.cited_id.map(lab.year) <= end_year]
    e["source_work"] = e.citing_id.map(lambda i: aliases.get(i, i))
    e["target_work"] = e.cited_id.map(lambda i: aliases.get(i, i))
    e = e.drop_duplicates(["source_work", "target_work"])
    e = e[e.source_work != e.target_work]
    e["source"], e["target"] = e.citing_id.map(a), e.cited_id.map(a)
    flows = pd.crosstab(e.source, e.target).reindex(index=clusters, columns=clusters, fill_value=0).to_numpy()
    # Include within-community edges in the null. Excluding the diagonal would redistribute
    # its mass to other pairs and manufacture apparent underexposure everywhere.
    expected = np.outer(flows.sum(1), flows.sum(0)) / max(1, flows.sum())
    labels = lab.con.execute("SELECT cluster_id, label, kind FROM clusters WHERE run_id = ?",
                             [lab.run]).df().set_index("cluster_id")
    paper_info = lab.papers.set_index("paper_id")
    research_types = {"article", "preprint", "conference-paper", "proceedings-article"}
    rows = []
    offdiag = ~np.eye(len(clusters), dtype=bool)
    semantic_threshold = np.quantile(similarity[offdiag], .75)
    for j, source in enumerate(clusters):
        for k, target in enumerate(clusters):
            if j == k or expected[j, k] < 5 or similarity[j, k] < semantic_threshold:
                continue
            observed, exp = int(flows[j, k]), float(expected[j, k])
            deficit = max(0, (exp - observed) / np.sqrt(exp + 1))
            if deficit == 0:
                continue
            def representatives(cluster, center):
                members = frame[frame.cluster_id == cluster].paper_id
                members = members[members.map(paper_info["type"]).isin(research_types)]
                if not len(members):
                    return members
                # Prevent an incidental, generic member (e.g. a textbook chapter) from
                # becoming the entire explanation for a community-to-community bridge.
                sim = x[[pos[i] for i in members]] @ center
                return members.iloc[np.argsort(-sim)[:30]]

            source_members = representatives(source, centroids[j])
            target_members = representatives(target, centroids[k])
            if not len(source_members) or not len(target_members):
                continue
            si, ti = [pos[i] for i in source_members], [pos[i] for i in target_members]
            cross = x[si] @ x[ti].T
            same_work = np.array([aliases.get(i, i) for i in source_members])[:, None] == np.array(
                [aliases.get(i, i) for i in target_members])[None, :]
            cross[same_work] = -np.inf
            if not np.isfinite(cross).any():
                continue
            pair = np.unravel_index(np.argmax(cross), cross.shape)
            rows.append({"source": source, "target": target, "source_label": labels.loc[source, "label"],
                         "target_label": labels.loc[target, "label"], "source_kind": labels.loc[source, "kind"],
                         "target_kind": labels.loc[target, "kind"], "semantic_similarity": float(similarity[j, k]),
                         "observed": observed, "expected": exp, "deficit": deficit,
                         "score": float(deficit * max(similarity[j, k], 0)),
                         "evidence": [source_members.iloc[pair[0]], target_members.iloc[pair[1]]]})
    return pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame()


def contextual_people(lab, end_year):
    """Role profiles by topic; fractional credit and external citing teams, no global fame score."""
    a = lab.con.execute("""SELECT paper_id, cluster_id FROM paper_cluster WHERE run_id = ?
        AND facet = 'problem' AND alpha = 0.3 AND resolution = 'medium'
        AND cluster_id IS NOT NULL""", [lab.run]).df().set_index("paper_id").cluster_id
    papers = lab.papers[lab.papers.year <= end_year].copy()
    aliases = work_aliases(papers, lab.authors)
    au = lab.authors[lab.authors.paper_id.isin(papers.paper_id)].copy()
    au["work"] = au.paper_id.map(aliases)
    counts = au.groupby("work").author_id.nunique()
    author_sets = au.groupby("work").author_id.agg(set).to_dict()
    au["fraction"] = 1 / au.work.map(counts)
    au["cluster_id"], au["year"] = au.paper_id.map(a), au.paper_id.map(lab.year)
    au = au.dropna(subset=["cluster_id"]).drop_duplicates(["work", "author_id"])
    edges = lab.refs[lab.refs.year.between(end_year - 2, end_year)].copy()
    edges = edges[edges.cited_id.isin(papers.paper_id)]
    edges["source_work"] = edges.citing_id.map(aliases)
    edges["target_work"] = edges.cited_id.map(aliases)
    edges = edges.drop_duplicates(["source_work", "target_work"])
    edges = edges[[not (author_sets.get(s, set()) & author_sets.get(t, set()))
                   for s, t in zip(edges.source_work, edges.target_work)]]
    adoption = edges.groupby("target_work").source_work.nunique()
    names = lab.con.execute("SELECT author_id, display_name FROM authors").df().set_index("author_id").display_name
    labels = lab.con.execute("SELECT cluster_id, label FROM clusters WHERE run_id = ?", [lab.run]).df().set_index("cluster_id").label
    recent = au[au.year >= end_year - 2].copy()
    recent["adoption"] = recent.work.map(adoption).fillna(0) * recent.fraction
    rows = []
    for (cluster, aid), g in recent.groupby(["cluster_id", "author_id"]):
        all_work = au[au.author_id == aid]
        topic_weights = all_work.groupby("cluster_id").fraction.sum()
        dist = topic_weights / topic_weights.sum()
        evidence = g.sort_values("adoption", ascending=False).paper_id.tolist()
        rows.append({"cluster_id": cluster, "label": labels.get(cluster), "author_id": aid, "name": names.get(aid),
                     "recent_fractional_works": float(g.fraction.sum()), "external_adoption": float(g.adoption.sum()),
                     "participation": float(1 - (dist ** 2).sum()), "topics": len(dist),
                     "first_topic_year": int(au[(au.author_id == aid) & (au.cluster_id == cluster)].year.min()),
                     "evidence": evidence[:8]})
    return pd.DataFrame(rows)


def run_lab(con, run_id, params, *, as_of=None, cutoffs=(2019, 2020, 2021, 2022, 2023)):
    today = date.fromisoformat(as_of) if as_of else date.today()
    lab = Lab(con, run_id, params)
    end_year = min(today.year - 1, int(lab.papers.year.max()))
    if len(set(cutoffs)) < 2:
        raise ValueError("At least two distinct cutoffs are needed for leave-one-cutoff-out evaluation.")
    cutoffs = sorted(set(cutoffs))
    for cutoff in cutoffs:
        if not con.execute("SELECT count(*) FROM clusters WHERE run_id = ?",
                           [f"{run_id}@T{cutoff}"]).fetchone()[0]:
            raise ValueError(f"Missing {run_id}@T{cutoff}. Build offline core objects with "
                             f"scripts/phase8c/prepare_cutoffs.py --run-id {run_id} --cutoffs {cutoff}")
    path = lab_dir(run_id)
    path.mkdir(parents=True, exist_ok=True)
    audit, routes, geometry, false_missing = corpus_audit(lab, end_year)
    trends, timeseries, weights = dynamics(lab, end_year)
    for name, frame in (("routes", routes), ("geometry", geometry), ("alias_false_bridges", false_missing),
                        ("trends", trends), ("timeseries", timeseries), ("route_weights", weights)):
        frame.to_parquet(path / f"{name}.parquet", index=False)
    history = []
    for cutoff in cutoffs:
        if cutoff + 3 > int(lab.papers.year.max()):
            continue
        print(f"research lab: cutoff {cutoff}", flush=True)
        history.append(bridge_candidates(lab, cutoff))
    history, metrics, comparisons = evaluate_bridges(pd.concat(history, ignore_index=True), end_year)
    history.to_parquet(path / "bridge_candidates.parquet", index=False)
    metrics.to_csv(path / "bridge_metrics.csv", index=False)
    comparisons.to_parquet(path / "bridge_comparisons.parquet", index=False)
    # Live review uses completed publication years. Fit only outcomes available by end_year.
    live = bridge_candidates(lab, end_year, live=True)
    train = history[history["T"] + 3 <= end_year]
    live["baseline_model"] = fitted_scores(train, live, BASE_FEATURES)
    live["contrastive_model"] = fitted_scores(train, live, EXTRA_FEATURES)
    live["diverse_pick"] = False
    live["baseline_pick"] = False
    for _, group in live.groupby("X"):
        live.loc[select_diverse(group, "baseline_model").index, "diverse_pick"] = True
        live.loc[group.nlargest(10, "baseline_model").index, "baseline_pick"] = True
    # Do not persist future outcomes for the live candidates as if they were evaluated.
    live = live.drop(columns="hit")
    live["title"] = live.i.map(lab.papers.set_index("paper_id").title)
    live["limitation_text"] = live.best_limitation.map(lab.text.get("limitations", pd.Series(dtype=object)))
    live["innovation_text"] = live.i.map(lab.text.get("innovation", pd.Series(dtype=object)))
    live.to_parquet(path / "live_bridges.parquet", index=False)
    opportunities = flow_opportunities(lab, end_year)
    opportunities.to_parquet(path / "opportunities.parquet", index=False)
    contextual_people(lab, end_year).to_parquet(path / "people.parquet", index=False)
    errors = abs(trends.raw_change - trends.within_route_change - trends.composition_change)
    assert errors.max() < 1e-10, "route decomposition must reconstruct observed share change"
    assert not history.duplicated(["T", "X", "work"]).any(), "duplicate work in a candidate list"
    assert np.isfinite(history[EXTRA_FEATURES].to_numpy()).all(), "non-finite model feature"
    audit.update(as_of=today.isoformat(), direction_sign_flips=int(trends.sign_flip.sum()),
                 direction_clusters=len(trends), flow_opportunities=len(opportunities),
                 cutoffs=cutoffs,
                 protocols={"LOCO retrospective": "Other cutoffs can contain future outcomes; diagnostic only.",
                            "purged chronological (complete outcome)": "Train only on cutoffs S with S+3 <= T. 2019 -> 2022; outcomes 2023-2025 are complete calendar years.",
                            "purged chronological (partial outcome)": "Train only on cutoffs S with S+3 <= T. 2019/2020 -> 2023; outcomes include partial 2026."},
                 limitations=["Corpus selected in 2026; retrospective sampling bias remains.",
                              "2023's outcome window includes partial 2026; not a mature test.",
                              "Live topic partitions use all years and are descriptive, not historical predictions.",
                              "Route decomposition tests sensitivity; it cannot recover unknown inclusion probabilities.",
                              "Flow deficits and people roles are descriptive and need expert review.",
                              "Semantic similarity does not establish that an innovation solves a limitation."])
    supported_trends = trends[(trends.before_n >= 20) & (trends.after_n >= 20)]
    audit["supported_direction_sign_flips"] = int(supported_trends.sign_flip.sum())
    audit["supported_direction_clusters"] = len(supported_trends)
    audit["models"] = {"embedding": params.embedding.version,
                       "extraction": f"{params.extraction.model}|{params.extraction.prompt_version}"}
    baseline = path / "audit_before_alias_fix.json"
    if baseline.exists():
        old = json.loads(baseline.read_text())
        audit["alias_fix"] = {"before_false_missing": old["already_cited_alias_recommendations"],
                              "after_false_missing": audit["already_cited_alias_recommendations"]}
    (path / "summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(metrics.groupby(["protocol", "ranking"])[["p@10", "methods@10", "auc", "tail_auc"]].mean().round(4).to_string())
    print(json.dumps(audit, indent=2))
    return path
