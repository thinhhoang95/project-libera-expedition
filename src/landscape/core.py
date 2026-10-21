"""Phase 5: core analytical objects.

5.1 facet clusters (kNN graph + multi-seed Leiden at three resolutions, stability filter, labels)
5.2 problem x method matrix   5.3 time slices   5.4 citation flows   5.5 author profiles
"""

from __future__ import annotations

import asyncio
import colorsys
import json
import logging
import re
import zlib
from collections import Counter, defaultdict

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy import sparse

from . import embed, store
from .config import ClusterParams, LabelParams, RunParams, run_vectors
from .llm import LLMBudgetExceeded, OpenRouter, quantizations_for

log = logging.getLogger(__name__)
GLOSS_FACETS = {"problem", "method"}
RES_ORDER = ["coarse", "medium", "fine"]


# ------------------------------------------------------------------ 5.1 clustering

def knn_graph(v: np.ndarray, k: int, drop_pct: float) -> ig.Graph:
    import faiss

    n = len(v)
    k = min(k, n - 1)
    index = faiss.IndexFlatIP(v.shape[1])
    index.add(v)
    sims, nbrs = index.search(v, k + 1)
    src = np.repeat(np.arange(n), k + 1)
    dst, w = nbrs.reshape(-1), sims.reshape(-1)
    keep = (dst != src) & (dst >= 0)
    src, dst, w = src[keep], dst[keep], w[keep]
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    df = pd.DataFrame({"a": lo, "b": hi, "w": w}).groupby(["a", "b"], as_index=False).w.max()
    cut = np.percentile(df.w, drop_pct)
    df = df[df.w >= cut]
    g = ig.Graph(n=n, edges=list(zip(df.a, df.b)), directed=False)
    g.es["weight"] = np.clip(df.w.to_numpy(), 1e-6, None).tolist()
    return g


def leiden(g: ig.Graph, resolution: float, seed: int) -> np.ndarray:
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                             resolution_parameter=resolution, seed=seed, n_iterations=-1)
    return np.asarray(part.membership)


def best_jaccard(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """For each cluster in ref, the best Jaccard overlap with any cluster in other."""
    n = len(ref)
    c = sparse.coo_matrix((np.ones(n), (ref, other))).tocsr()
    size_a = np.asarray(c.sum(axis=1)).ravel()
    size_b = np.asarray(c.sum(axis=0)).ravel()
    c = c.tocoo()
    jac = c.data / (size_a[c.row] + size_b[c.col] - c.data)
    best = np.zeros(c.shape[0])
    np.maximum.at(best, c.row, jac)
    return best


def stable_partition(g: ig.Graph, res: float, seeds: list[int], thr: float, min_size: int):
    parts = [leiden(g, res, s) for s in seeds]
    ref = parts[0]
    stab = np.mean([best_jaccard(ref, o) for o in parts[1:]], axis=0) if len(parts) > 1 \
        else np.ones(ref.max() + 1)
    sizes = np.bincount(ref)
    return ref, stab, sizes, (stab >= thr) & (sizes >= min_size)


def cluster_coherence(v: np.ndarray, membership: np.ndarray, sizes: np.ndarray, min_size: int,
                      pairs: int = 500, seed: int = 0) -> np.ndarray:
    """Mean pairwise cosine within each cluster (sampled pairs); NaN for fragments."""
    rng = np.random.default_rng(seed)
    out = np.full(len(sizes), np.nan)
    for c in np.where(sizes >= min_size)[0]:
        m = np.where(membership == c)[0]
        i, j = rng.choice(m, pairs), rng.choice(m, pairs)
        keep = i != j
        out[c] = float(np.mean(np.sum(v[i[keep]] * v[j[keep]], axis=1)))
    return out


def palette(n: int, offset: float = 0.0) -> list[str]:
    out = []
    for i in range(n):
        h = (offset + i * 0.618033988749895) % 1.0
        l = 0.45 if i % 2 == 0 else 0.58
        r, gg, b = colorsys.hls_to_rgb(h, l, 0.62)
        out.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(gg * 255), int(b * 255)))
    return out


def ctfidf_terms(texts_by_cluster: dict[str, list[str]], top: int = 15) -> dict[str, list[str]]:
    from sklearn.feature_extraction.text import CountVectorizer

    keys = list(texts_by_cluster)
    docs = [" ".join(texts_by_cluster[k]) for k in keys]
    if len(docs) < 2:
        return {k: [] for k in keys}
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
    try:
        x = cv.fit_transform(docs).astype(float)
    except ValueError:
        return {k: [] for k in keys}
    words = np.asarray(cv.get_feature_names_out())
    tf = sparse.diags(1 / np.maximum(1, np.asarray(x.sum(axis=1)).ravel())) @ x
    avg = np.asarray(x.sum(axis=1)).mean()
    idf = np.log(1 + avg / np.maximum(1, np.asarray(x.sum(axis=0)).ravel()))
    score = (tf @ sparse.diags(idf)).toarray()
    return {k: words[np.argsort(-score[i])[:top]].tolist() for i, k in enumerate(keys)}


BOILERPLATE = {
    # statements that say a facet is absent rather than state one; the extraction prompt asks for null here
    "limitations": [
        re.compile(r"^\s*(null|none|n/?a)\.?\s*$", re.I),
        re.compile(r"\bno (specific |explicit |particular )?(limitations?|weakness(es)?|drawbacks?)\b[^.]*"
                   r"\b(stated|mentioned|given|reported|discussed|identified|specified)\b", re.I),
        re.compile(r"\b(specific )?limitations? (are|is) not (explicitly )?(stated|mentioned|given|discussed)\b",
                   re.I),
    ],
}
# "future work is discussed" with nothing specific: only short statements, so a named open problem stays
_BOILERPLATE_SHORT = re.compile(r"\b(open (questions|problems|challenges)|future (research|work|directions|"
                                r"extensions|prospects)|research gaps|challenges)\b", re.I)


def is_boilerplate(facet: str, text: str | None) -> bool:
    if not isinstance(text, str) or facet not in BOILERPLATE:
        return False
    if any(p.search(text) for p in BOILERPLATE[facet]):
        return True
    return len(text.split()) <= 12 and bool(_BOILERPLATE_SHORT.search(text))


def seed_titles(con, run_id: str) -> str:
    """Distinct seed titles of a run (cutoff runs share their base run's seeds)."""
    row = con.execute("SELECT seed_ids FROM runs WHERE run_id = ?", [run_id]).fetchone()
    titles = con.execute("SELECT DISTINCT lower(title) FROM papers WHERE paper_id IN (SELECT unnest(?))",
                         [list(row[0]) if row and row[0] is not None else []]).fetchall()
    return "; ".join(f'"{t[0]}"' for t in titles if t[0]) or "unknown"


def cluster_facet(con, run_id: str, facet: str, alpha: float, cp: ClusterParams, version: str,
                  corpus_ids: list[str], graph_dim: int = 256, lp: LabelParams | None = None):
    """Cluster one facet at one alpha at all resolutions. Returns (clusters_df, assignments_df,
    centroids {cluster_id: vec768}, label_inputs, consolidation_spend)."""
    texts = con.execute("""SELECT paper_id, text FROM facets WHERE facet = ? AND paper_id IN
                           (SELECT paper_id FROM provenance WHERE run_id = ?)""",
                        [facet, run_id]).df().drop_duplicates("paper_id").set_index("paper_id").text
    if cp.drop_boilerplate and facet in BOILERPLATE:
        drop = {k for k, t in texts.items() if is_boilerplate(facet, t)}
        corpus_ids = [i for i in corpus_ids if i not in drop]
        log.info("%s: dropped %d boilerplate statements before clustering", facet, len(drop))
    ids, v256 = embed.blended(facet, alpha, corpus_ids, dim=graph_dim, version=version)
    if len(ids) < 50:
        log.warning("%s: only %d vectors, skipping", facet, len(ids))
        return None
    _, v768 = embed.blended(facet, alpha, ids, dim=768, version=version)
    g = knn_graph(v256, cp.k, cp.drop_edge_pct)
    seeds = cp.seed_values or list(range(cp.n_seeds))
    levels = {}
    for res_name in RES_ORDER:
        res = cp.resolutions[res_name]
        ref, stab, sizes, keep = stable_partition(g, res, seeds, cp.stability_jaccard,
                                                  cp.min_cluster_size)
        coh = cluster_coherence(v256, ref, sizes, cp.min_cluster_size)
        # hub: a coherent body of work whose boundary moves between Leiden seeds (on the pilot,
        # the DRO core). Kept and assigned, but reported separately from stable clusters.
        stable_coh = coh[keep & ~np.isnan(coh)]
        coh_floor = np.percentile(stable_coh, 25) if len(stable_coh) else np.inf
        hub = (~keep) & (sizes >= max(cp.hub_min_size, cp.hub_min_share * len(ids))) & (coh >= coh_floor)
        kind = np.where(keep, "stable", np.where(hub, "hub", "unstable")).astype(object)
        levels[res_name] = {"ref": ref, "stab": stab, "sizes": sizes, "keep": keep, "coh": coh, "kind": kind}
    decisions, spend = {}, 0.0
    if cp.consolidate and lp is not None:
        decisions, spend = consolidate_levels(facet, levels, ids, v768, texts, cp, lp, seed_titles(con, run_id))

    clusters, assigns, cents, label_inputs = [], [], {}, {}
    membership_by_res = {}
    for res_name in RES_ORDER:
        L = levels[res_name]
        ref, stab, sizes, keep, coh, kind = L["ref"], L["stab"], L["sizes"], L["keep"], L["coh"], L["kind"]
        merge, generic, notes = decisions.get(res_name, ({}, {}, {}))
        # ids follow the raw partition, so a cluster keeps its id with or without consolidation
        order = np.argsort(-sizes)
        rank = {int(c): i for i, c in enumerate(order)}
        cid = lambda c: f"{facet}:{alpha:.1f}:{res_name}:{rank[int(c)]:03d}"  # noqa: E731
        final = ref.copy()
        for c, s in merge.items():
            final[ref == c] = s
            kind[c] = "merged"
            if keep[c]:
                kind[s] = "stable"  # a hub that absorbs a stable cluster reports as stable
        if cp.drop_generic:
            for c in generic:
                kind[c] = "generic"
        fsizes = np.bincount(final, minlength=len(sizes))
        if merge:
            coh = coh.copy()
            fcoh = cluster_coherence(v256, final, fsizes, cp.min_cluster_size)
            for s in set(merge.values()):
                coh[s] = fcoh[s]
        assigned = np.isin(kind, ["stable", "hub"])
        lab = np.array([cid(c) if assigned[c] else None for c in final], dtype=object)
        membership_by_res[res_name] = lab
        sims = np.full(len(ids), np.nan)
        by_cluster_texts = {}
        level_cents = {}
        for c in order:
            absorbed = kind[c] == "merged"
            members = np.where((ref if absorbed else final) == c)[0]
            if not len(members):
                continue
            cent = v768[members].mean(axis=0)
            cent /= np.linalg.norm(cent) or 1
            s = v768[members] @ cent
            if not absorbed:
                sims[members] = s
            key = cid(c)
            cents[key] = cent
            if len(members) >= cp.min_cluster_size and not absorbed:
                level_cents[key] = cent
            top10 = members[np.argsort(-s)[:10]]
            label_inputs[key] = [texts.get(ids[i], "") for i in top10]
            by_cluster_texts[key] = [texts.get(ids[i], "") for i in members]
            clusters.append({"cluster_id": key, "facet": facet, "resolution": res_name,
                             "alpha": alpha, "stability": float(stab[c]), "is_stable": kind[c] == "stable",
                             "kind": str(kind[c]), "coherence": float(coh[c]), "size": len(members),
                             "merged_into": cid(merge[c]) if c in merge else None,
                             "is_generic": c in generic,
                             "consolidation_note": notes.get(c) or generic.get(c)})
        # centrality: z-score of a centroid's mean cosine to the other centroids of this level
        if len(level_cents) > 2:
            keys = list(level_cents)
            m = np.stack([level_cents[k] for k in keys])
            sim = m @ m.T
            np.fill_diagonal(sim, np.nan)
            mean = np.nanmean(sim, axis=1)
            z = (mean - mean.mean()) / (mean.std() or 1)
            cz = dict(zip(keys, z))
            for row in clusters:
                if row["resolution"] == res_name:
                    row["centrality"] = float(cz.get(row["cluster_id"], np.nan))
        terms = ctfidf_terms(by_cluster_texts)
        for row in clusters:
            if row["resolution"] == res_name:
                row["top_terms"] = terms.get(row["cluster_id"], [])
        assigns.append(pd.DataFrame({"paper_id": ids, "facet": facet, "resolution": res_name,
                                     "alpha": alpha, "cluster_id": lab, "sim_to_centroid": sims,
                                     "raw_cluster_id": [cid(c) for c in ref]}))
        log.info("%s a=%.1f %s: %d clusters (+%d fragments < %d papers), %d stable, %d hubs, "
                 "%d merged, %d flagged generic, %.0f%% papers assigned", facet, alpha, res_name,
                 int((sizes >= cp.min_cluster_size).sum()), int((sizes < cp.min_cluster_size).sum()),
                 cp.min_cluster_size, int((kind == "stable").sum()), int((kind == "hub").sum()),
                 len(merge), len(generic), 100 * np.mean([x is not None for x in lab]))
    cdf = pd.DataFrame(clusters)
    # parent links: majority overlap with the next coarser level
    parents = {}
    for child, parent in (("medium", "coarse"), ("fine", "medium")):
        ch, pa_ = membership_by_res[child], membership_by_res[parent]
        pairs = Counter((c, p_) for c, p_ in zip(ch, pa_) if c is not None and p_ is not None)
        best = {}
        for (c, p_), n in pairs.items():
            if n > best.get(c, (None, 0))[1]:
                best[c] = (p_, n)
        parents.update({c: p_ for c, (p_, _) in best.items()})
    cdf["parent_cluster_id"] = cdf.cluster_id.map(parents)
    return cdf, pd.concat(assigns, ignore_index=True), cents, label_inputs, spend


# ------------------------------------------------------------------ labels

LABEL_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["label", "description"],
    "properties": {"label": {"type": "string", "description": "at most 6 words"},
                   "description": {"type": "string", "description": "one line, at most 25 words"}},
}
LABEL_PROMPT = """You label a cluster of research papers that were grouped by their "{facet}" facet.
Below are the {facet} statements of the {n} papers closest to the cluster centre, then the 15 most
distinctive terms of the cluster (c-TF-IDF).
Return:
- "label": at most 6 words naming what these papers share, in generic technical vocabulary.
- "description": one line (at most 25 words) describing the cluster.

Statements:
{statements}

Distinctive terms: {terms}"""


async def label_clusters(rows: list[dict], inputs: dict, lp: LabelParams) -> tuple[dict, float]:
    client = OpenRouter(lp.model, ["fp8", "bf16", "fp16", "fp32"], lp.concurrency, lp.llm_spend_cap_usd)
    out = {}

    async def one(r):
        stm = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(inputs[r["cluster_id"]]) if t)
        prompt = LABEL_PROMPT.format(facet=r["facet"], n=len(inputs[r["cluster_id"]]),
                                     statements=stm, terms=", ".join(r.get("top_terms") or []))
        try:
            obj, _, err = await client.chat_json(prompt, LABEL_SCHEMA, "cluster_label", max_tokens=300)
        except LLMBudgetExceeded:
            return
        if obj:
            out[r["cluster_id"]] = (" ".join(obj["label"].split()[:6]), obj["description"].strip())
        else:
            log.warning("label failed for %s: %s", r["cluster_id"], err)

    try:
        await asyncio.gather(*(one(r) for r in rows))
    finally:
        await client.close()
    return out, client.spent


# ------------------------------------------------------------------ consolidation

_ITEMS = lambda props, req: {"type": "array", "items": {  # noqa: E731
    "type": "object", "additionalProperties": False, "required": req, "properties": props}}
CONSOLIDATE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["merge", "generic"],
    "properties": {
        "merge": _ITEMS({"ids": {"type": "array", "items": {"type": "integer"}},
                         "reason": {"type": "string"}, "difference": {"type": "string"}},
                        ["ids", "reason", "difference"]),
        "generic": _ITEMS({"id": {"type": "integer"}, "reason": {"type": "string"},
                           "shared_subject": {"type": "string"}}, ["id", "reason", "shared_subject"]),
    },
}
CONSOLIDATE_PROMPT = """You review the full catalogue of "{facet}" categories found by clustering the {facet}
statements of {n} research papers. The corpus was collected around these seed papers, so their subject is
the corpus's central topic: {seeds}. Each category shows its id, its number of papers, its most distinctive
terms and the statements of the papers closest to its centre.

Return:
- "merge": groups of ids that are the same {facet} category described in different words: their statements
  are interchangeable, so an expert would file their papers under one heading. If you can name any difference
  in focus, subject, technique or application between two categories, they are not the same category; nor
  are a broad category and a narrower one inside it. Belonging to the same wider area is not enough. Most
  catalogues need no merge at all.
- "generic": ids too general or vague to be a useful category next to the others: boilerplate that almost
  any paper could state (e.g. "no specific limitation is stated", "future work is discussed", "outperforms
  existing methods", "validated by simulation"), a document genre with no specific subject (e.g. "reviews
  of optimization methods"), or a catch-all whose statements share no specific subject. A broad category
  with a real subject is not generic, and neither is a review or survey of a specific subject, nor a
  category about the corpus's central topic, however common it is.
Give a short reason for each entry. For each merge group also give "difference": a difference in focus,
subject, technique or application that would make an expert keep its members apart, or "" if there is none. For each generic id also give
"shared_subject": the specific subject its statements share, or "" if there is none. Return empty lists when
nothing applies.

Catalogue:
{catalogue}"""


def _clip(text: str, n: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def _stated(text: str) -> bool:
    return text.strip().strip(".").lower() not in ("", "none", "n/a", "no difference", "nothing")


def _catalogue(ids, v768, texts, ref, cands: list[int], n_stmt: int = 5):
    """Prompt text for one clustering's catalogue, and the raw centroid of each listed cluster."""
    members = {c: np.where(ref == c)[0] for c in cands}
    terms = ctfidf_terms({c: [texts.get(ids[i], "") for i in m] for c, m in members.items()})
    cents, entries = {}, []
    for c, m in members.items():
        cent = v768[m].mean(axis=0)
        cent /= np.linalg.norm(cent) or 1
        cents[c] = cent
        top = m[np.argsort(-(v768[m] @ cent))[:n_stmt]]
        stm = "\n".join(f"  - {_clip(texts.get(ids[i]))}" for i in top if isinstance(texts.get(ids[i]), str))
        entries.append(f"[id {c}] {len(m)} papers · terms: {', '.join(terms[c][:8])}\n{stm}")
    return "\n\n".join(entries), cents


async def _review_catalogues(jobs: list[tuple[str, str]], lp: LabelParams) -> tuple[dict, float]:
    client = OpenRouter(lp.consolidate_model, quantizations_for(lp.consolidate_model), lp.concurrency,
                        lp.consolidate_spend_cap_usd)
    out = {}

    async def one(key, prompt):
        try:
            obj, _, err = await client.chat_json(prompt, CONSOLIDATE_SCHEMA, "catalogue_review", max_tokens=8000)
        except LLMBudgetExceeded:
            return
        if obj:
            out[key] = obj
        else:
            log.warning("catalogue review failed for %s: %s", key, err)

    try:
        await asyncio.gather(*(one(k, pr) for k, pr in jobs))
    finally:
        await client.close()
    return out, client.spent


def consolidate_levels(facet: str, levels: dict, ids, v768, texts, cp: ClusterParams, lp: LabelParams,
                       seeds: str = "unknown"):
    """One catalogue review per resolution. Returns ({res: (merge {c: survivor}, generic {c: reason},
    notes {c: reason})}, spend). A group merges into its largest member; a member joins only if its
    centroid cosine to that member is in the top (1 - merge_pair_quantile) of sibling pairs."""
    jobs, ctx = [], {}
    for res_name, L in levels.items():
        if res_name not in cp.consolidate_levels:
            continue
        cands = [int(c) for c in np.where(np.isin(L["kind"], ["stable", "hub"]))[0]]
        if len(cands) < 3:
            continue
        cat, cents = _catalogue(ids, v768, texts, L["ref"], cands)
        n = int(L["sizes"][cands].sum())
        jobs.append((res_name, CONSOLIDATE_PROMPT.format(facet=facet, n=n, seeds=seeds, catalogue=cat)))
        ctx[res_name] = cents
    replies, spend = asyncio.run(_review_catalogues(jobs, lp))
    out = {}
    for res_name, obj in replies.items():
        cents, sizes = ctx[res_name], levels[res_name]["sizes"]
        keys = list(cents)
        pos = {k: i for i, k in enumerate(keys)}
        m = np.stack([cents[k] for k in keys])
        sim = m @ m.T
        thr = float(np.quantile(sim[np.triu_indices(len(keys), 1)], cp.merge_pair_quantile))
        # self-consistency: a merge whose members differ, or a generic category with a subject, is dropped
        generic = {g["id"]: g["reason"] for g in obj["generic"]
                   if g["id"] in cents and not _stated(g["shared_subject"])}
        blocked = generic if cp.drop_generic else {}
        merge, notes, used = {}, {}, set()
        for grp in obj["merge"]:
            if _stated(grp["difference"]):
                log.info("%s %s: merge %s rejected (difference: %s)", facet, res_name, grp["ids"],
                         _clip(grp["difference"], 160))
                continue
            group = [c for c in dict.fromkeys(grp["ids"]) if c in cents and c not in blocked and c not in used]
            if len(group) < 2:
                continue
            s = max(group, key=lambda c: (sizes[c], -c))
            for c in group:
                if c == s:
                    continue
                cos = float(sim[pos[c], pos[s]])
                if cos < thr:
                    log.info("%s %s: merge %d -> %d rejected (cos %.3f < %.3f): %s",
                             facet, res_name, c, s, cos, thr, grp["reason"])
                    continue
                merge[c], notes[c] = s, grp["reason"]
                used.update((c, s))
        out[res_name] = (merge, generic, notes)
        log.info("%s %s: %d merged, %d flagged generic (pair threshold %.3f)", facet, res_name, len(merge),
                 len(generic), thr)
    return out, spend


# ------------------------------------------------------------------ orchestration

def build_core(con, run_id: str, p: RunParams, skip_labels: bool = False,
               author_centroids: bool = True) -> dict:
    cp, version = p.clusters, p.embedding.version
    corpus = con.execute("""SELECT v.paper_id, p.year FROM provenance v JOIN papers p USING (paper_id)
                            WHERE v.run_id = ?""", [run_id]).df()
    corpus_ids = corpus.paper_id.tolist()
    all_clusters, all_assign, all_cents, all_inputs = [], [], {}, {}
    consolidation_spend = 0.0
    for facet in cp.facets:
        alphas = cp.gloss_alphas if facet in GLOSS_FACETS else [0.0]
        for alpha in alphas:
            r = cluster_facet(con, run_id, facet, alpha, cp, version, corpus_ids, p.embedding.graph_dim,
                              lp=p.labels)
            if r is None:
                continue
            cdf, adf, cents, inputs, spent = r
            consolidation_spend += spent
            all_clusters.append(cdf)
            all_assign.append(adf)
            all_cents.update(cents)
            all_inputs.update(inputs)
    clusters = pd.concat(all_clusters, ignore_index=True)
    assign = pd.concat(all_assign, ignore_index=True)

    # colors: one per cluster, assigned once per run
    colors = {}
    for (f, a, res), grp in clusters.groupby(["facet", "alpha", "resolution"]):
        for cid, col in zip(grp.cluster_id, palette(len(grp), offset=zlib.crc32(f.encode()) % 100 / 100)):
            colors[cid] = col
    clusters["color"] = clusters.cluster_id.map(colors)

    # labels: LLM for stable/hub/generic coarse+medium (and fine if configured), c-TF-IDF terms otherwise
    clusters["label"] = clusters.top_terms.map(lambda t: ", ".join(t[:3]) if t is not None else None)
    clusters["description"] = None
    clusters["label_model"] = "ctfidf"
    spend = 0.0
    if not skip_labels:
        levels = RES_ORDER if p.labels.label_fine else ["coarse", "medium"]
        todo = clusters[clusters.kind.isin(["stable", "hub", "generic"]) & clusters.resolution.isin(levels)].to_dict("records")
        log.info("labelling %d clusters with %s", len(todo), p.labels.model)
        labels, spend = asyncio.run(label_clusters(todo, all_inputs, p.labels))
        for cid, (lab, desc) in labels.items():
            i = clusters.index[clusters.cluster_id == cid]
            clusters.loc[i, "label"] = lab
            clusters.loc[i, "description"] = desc
            clusters.loc[i, "label_model"] = p.labels.model

    # 5.3 time slices ---------------------------------------------------------------------
    years = corpus.set_index("paper_id").year
    assign["year"] = assign.paper_id.map(years)
    totals = (assign.groupby(["facet", "resolution", "alpha", "year"]).paper_id.nunique()
              .rename("n_total").reset_index())
    cy = (assign[assign.cluster_id.notna()].groupby(["facet", "resolution", "alpha", "cluster_id", "year"])
          .size().rename("n").reset_index())
    cy = cy.merge(totals, on=["facet", "resolution", "alpha", "year"])
    cy["share"] = cy.n / cy.n_total
    span = cy.groupby("cluster_id").agg(n=("n", "sum"), y0=("year", "min"), y1=("year", "max"))
    per_year = span.n / (span.y1 - span.y0 + 1)
    clusters["time_bucket_years"] = clusters.cluster_id.map(lambda c: 1 if per_year.get(c, 0) >= 20 else 2)

    yt = totals[(totals.resolution == "medium")][["facet", "alpha", "year", "n_total"]]
    year_totals = pd.concat([
        corpus.groupby("year").size().rename("n").reset_index().assign(facet="corpus"),
        yt[yt.alpha == 0.0].rename(columns={"n_total": "n"})[["facet", "year", "n"]],
    ])

    # write 5.1 + 5.3 -----------------------------------------------------------------------
    cols = ["cluster_id", "facet", "resolution", "alpha", "label", "description", "top_terms",
            "stability", "is_stable", "kind", "coherence", "centrality", "size", "parent_cluster_id",
            "color", "time_bucket_years",
            "label_model", "merged_into", "is_generic", "consolidation_note"]
    store.replace_run_rows(con, "clusters", run_id, clusters[cols].assign(run_id=run_id))
    store.replace_run_rows(con, "paper_cluster", run_id,
                           assign.drop(columns="year").assign(run_id=run_id))
    store.replace_run_rows(con, "cluster_year", run_id, cy.assign(run_id=run_id))
    store.replace_run_rows(con, "year_totals", run_id, year_totals.assign(run_id=run_id))
    pd.DataFrame({"run_id": run_id, "cluster_id": list(all_cents),
                  "centroid": [v.tolist() for v in all_cents.values()]}).to_parquet(
        run_vectors(run_id) / f"cluster_centroids_{run_id}.parquet")

    med = assign[(assign.resolution == "medium") & assign.cluster_id.notna()]

    # 5.2 problem x method matrix -----------------------------------------------------------
    pm_rows = []
    for alpha in cp.gloss_alphas:
        pr = med[(med.facet == "problem") & (med.alpha == alpha)][["paper_id", "cluster_id", "year"]]
        me = med[(med.facet == "method") & (med.alpha == alpha)][["paper_id", "cluster_id"]]
        m = pr.merge(me, on="paper_id", suffixes=("_p", "_m"))
        if m.empty:
            continue
        cell = m.groupby(["cluster_id_p", "cluster_id_m", "year"]).size().rename("count").reset_index()
        first = cell.groupby(["cluster_id_p", "cluster_id_m"]).year.min().rename("first_year")
        cell = cell.join(first, on=["cluster_id_p", "cluster_id_m"])
        pm_rows.append(cell.rename(columns={"cluster_id_p": "problem_cluster",
                                            "cluster_id_m": "method_cluster"}).assign(alpha=alpha))
    pm = pd.concat(pm_rows, ignore_index=True) if pm_rows else pd.DataFrame()
    store.replace_run_rows(con, "pm_matrix", run_id, pm.assign(run_id=run_id) if not pm.empty else pm)

    tags = con.execute("""
        WITH t AS (
            SELECT f.paper_id, f.facet, coalesce(v.canonical, lower(trim(f.text))) AS tag
            FROM facets f JOIN provenance pv ON pv.paper_id = f.paper_id AND pv.run_id = ?
            LEFT JOIN tag_vocab v ON v.run_id = pv.run_id AND v.facet = f.facet AND v.tag = lower(trim(f.text))
            WHERE f.facet IN ('problem_class', 'method_family') AND f.model_id = ? AND f.prompt_version = ?)
        SELECT a.tag AS problem_class, b.tag AS method_family, p.year, count(*) AS count
        FROM t a JOIN t b ON a.paper_id = b.paper_id AND a.facet = 'problem_class' AND b.facet = 'method_family'
        JOIN papers p ON p.paper_id = a.paper_id GROUP BY ALL""",
                       [run_id, p.extraction.model, p.extraction.prompt_version]).df()
    if not tags.empty:
        tags["first_year"] = tags.groupby(["problem_class", "method_family"]).year.transform("min")
    store.replace_run_rows(con, "pm_tag_matrix", run_id, tags.assign(run_id=run_id) if not tags.empty else tags)

    # 5.4 citation flows ----------------------------------------------------------------------
    edges = con.execute("""
        SELECT r.citing_id, r.cited_id FROM refs r
        JOIN provenance a ON a.paper_id = r.citing_id AND a.run_id = ?
        JOIN provenance b ON b.paper_id = r.cited_id AND b.run_id = ?""", [run_id, run_id]).df()
    flows = []
    for (facet, alpha), grp in med.groupby(["facet", "alpha"]):
        cl = grp.set_index("paper_id").cluster_id
        e = edges.assign(from_cluster=edges.citing_id.map(cl), to_cluster=edges.cited_id.map(cl),
                         year=edges.citing_id.map(years)).dropna(subset=["from_cluster", "to_cluster", "year"])
        if e.empty:
            continue
        f = e.groupby(["year", "from_cluster", "to_cluster"]).size().rename("count").reset_index()
        f["row_share"] = f["count"] / f.groupby(["year", "from_cluster"])["count"].transform("sum")
        flows.append(f.assign(facet=facet, alpha=alpha))
    flows = pd.concat(flows, ignore_index=True) if flows else pd.DataFrame()
    if not flows.empty:
        flows["year"] = flows.year.astype(int)
    store.replace_run_rows(con, "citation_flows", run_id, flows.assign(run_id=run_id) if not flows.empty else flows)

    # 5.5 author profiles ---------------------------------------------------------------------
    au = con.execute("""
        SELECT a.paper_id, a.author_id, a.institution_ids, p.year FROM authorships a
        JOIN provenance v ON v.paper_id = a.paper_id AND v.run_id = ?
        JOIN papers p ON p.paper_id = a.paper_id""", [run_id]).df()
    ay_rows = []
    profile_sets = [(f, 0.0) for f in cp.facets] + [(f, 0.3) for f in GLOSS_FACETS if 0.3 in cp.gloss_alphas]
    for facet, alpha in profile_sets:
        cl = med[(med.facet == facet) & (med.alpha == alpha)][["paper_id", "cluster_id"]]
        m = au.merge(cl, on="paper_id")
        if m.empty:
            continue
        g = m.groupby(["author_id", "year"])
        dist = g.cluster_id.agg(lambda s: dict(Counter(s)))
        inst = g.institution_ids.agg(lambda s: sorted({i for x in s if x is not None for i in x}))
        df = pd.DataFrame({"cluster_dist": dist, "institution_ids": inst,
                           "n_papers": g.paper_id.nunique()}).reset_index()
        df["dominant_cluster"] = df.cluster_dist.map(lambda d: max(d.items(), key=lambda kv: (kv[1], kv[0]))[0])
        df["cluster_dist"] = df.cluster_dist.map(json.dumps)
        df["facet"], df["alpha"] = facet, alpha
        df["centroid_ref"] = (f"author_year_{facet}.parquet#" + df.author_id + "|" + df.year.astype(str)) \
            if alpha == 0.0 else None
        ay_rows.append(df)
    ay = pd.concat(ay_rows, ignore_index=True) if ay_rows else pd.DataFrame()
    store.replace_run_rows(con, "author_year", run_id, ay.assign(run_id=run_id) if not ay.empty else ay)
    if author_centroids:
        write_author_year_centroids(au, cp.facets, version, run_id)

    return {
        "clusterings": int(clusters.groupby(["facet", "alpha", "resolution"]).ngroups),
        "clusters": int(len(clusters)), "stable_clusters": int(clusters.is_stable.sum()),
        "hub_clusters": int((clusters.kind == "hub").sum()),
        "by_level": clusters.groupby(["facet", "alpha", "resolution"]).agg(
            n=("cluster_id", "size"), stable=("is_stable", "sum"),
            hubs=("kind", lambda k: int((k == "hub").sum()))).reset_index().to_dict("records"),
        "pm_cells": int(len(pm.groupby(["alpha", "problem_cluster", "method_cluster"])) if not pm.empty else 0),
        "corpus_citation_edges": int(len(edges)),
        "author_year_rows": int(len(ay)), "label_spend_usd": spend,
        "merged_clusters": int((clusters.kind == "merged").sum()),
        "generic_clusters": int(clusters.is_generic.fillna(False).sum()),
        "consolidation_spend_usd": consolidation_spend,
    }


def write_author_year_centroids(au: pd.DataFrame, facets: list[str], version: str, run_id: str):
    """Author-year centroid per facet (mean of that author's paper vectors that year, alpha = 0)."""
    for facet in facets:
        ids, v = embed.load_vectors(facet, version=version)
        if not ids:
            continue
        pos = {k: i for i, k in enumerate(ids)}
        m = au[au.paper_id.isin(pos)]
        keys, cents = [], []
        for (a, y), grp in m.groupby(["author_id", "year"]):
            c = v[[pos[p_] for p_ in grp.paper_id]].mean(axis=0)
            keys.append(f"{a}|{y}")
            cents.append(c / (np.linalg.norm(c) or 1))
        if keys:
            embed.append_vectors_to(run_vectors(run_id) / f"author_year_{facet}_{run_id}.parquet", keys,
                                    np.asarray(cents, dtype=np.float32), version)
