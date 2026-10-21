"""Phase 2: grow a seed set into a corpus through capped routes, with provenance.

Routes are filled in order (seed, ai_report, back, forward, author, semantic) until the
paper budget is reached. The semantic route runs after a first abstract-embedding pass.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import ingest, store
from .clients import BudgetExceeded, OpenAlex, SemanticScholar, short_id
from .config import DATA, ExpansionParams

log = logging.getLogger(__name__)
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>\]\)]+", re.I)


class Corpus:
    def __init__(self, con, run_id: str, p: ExpansionParams):
        self.con, self.run_id, self.p = con, run_id, p
        prev = con.execute("SELECT paper_id, route, hop, via_id, score FROM provenance WHERE run_id = ?",
                           [run_id]).df()
        self.rows = {r["paper_id"]: r for r in prev.to_dict("records")}

    def __contains__(self, pid):
        return pid in self.rows

    def __len__(self):
        return len(self.rows)

    def route_count(self, route):
        return sum(1 for r in self.rows.values() if r["route"] == route)

    def full(self, route=None) -> bool:
        if len(self.rows) >= self.p.paper_budget:
            return True
        cap = self.p.route_caps.get(route)
        return cap is not None and self.route_count(route) >= cap

    def add(self, pid, route, hop, via=None, score=None) -> bool:
        if pid in self.rows or self.full(route):
            return False
        self.rows[pid] = {"paper_id": pid, "route": route, "hop": hop, "via_id": via, "score": score}
        return True

    def ids(self, route=None, hop=None):
        return [k for k, r in self.rows.items()
                if (route is None or r["route"] == route) and (hop is None or r["hop"] == hop)]

    def save(self):
        df = pd.DataFrame(list(self.rows.values()))
        if df.empty:
            return
        df.insert(0, "run_id", self.run_id)
        store.replace_run_rows(self.con, "provenance", self.run_id, df)


def _refs_of(con, ids: list[str]) -> dict[str, list[str]]:
    if not ids:
        return {}
    con.register("_ids", pd.DataFrame({"paper_id": ids}))
    df = con.execute("""SELECT paper_id, referenced_works FROM papers
                        WHERE paper_id IN (SELECT paper_id FROM _ids)""").df()
    con.unregister("_ids")
    return {r.paper_id: list(r.referenced_works if r.referenced_works is not None else [])
            for r in df.itertuples()}


def _substantive_edges(con, citing_ids: list[str]) -> dict[str, set[str] | None]:
    """Per citing paper: cited ids S2 marks influential or 'methodology'; None if no S2 data."""
    if not citing_ids:
        return {}
    con.register("_ids", pd.DataFrame({"paper_id": citing_ids}))
    df = con.execute("""SELECT citing_id, cited_id, is_influential, intents FROM refs
                        WHERE citing_id IN (SELECT paper_id FROM _ids)""").df()
    con.unregister("_ids")
    out: dict[str, set[str] | None] = {c: None for c in citing_ids}
    for r in df.itertuples():
        if r.is_influential is None or pd.isna(r.is_influential):
            continue
        s = out[r.citing_id] if out[r.citing_id] is not None else set()
        intents = list(r.intents) if r.intents is not None else []
        if r.is_influential or "methodology" in intents:
            s.add(r.cited_id)
        out[r.citing_id] = s
    return out


async def resolve_ai_report(oa: OpenAlex, con, run_id: str, report_path: str) -> list[str]:
    """Resolve a deep-research report's references to OpenAlex works by DOI only.

    Title+year matching would need OpenAlex search calls; unresolved entries are dropped and
    counted, as the recipe requires. The report file is stored with the run."""
    run_dir = DATA / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(report_path, run_dir / ("ai_report" + "".join(Path(report_path).suffixes)))
    text = open(report_path, encoding="utf-8", errors="ignore").read()
    dois = sorted({d.rstrip(".,;").lower() for d in DOI_RE.findall(text)})
    works = await oa.works_by_dois(dois) if dois else []
    ingest.load_works(con, works)
    log.info("ai_report: %d DOIs found, %d resolved", len(dois), len(works))
    return [short_id(w["id"]) for w in works]


async def expand_citation_routes(con, run_id: str, p: ExpansionParams, seed_dois: list[str],
                                 seed_ids: list[str] | None = None, ai_report: str | None = None,
                                 budget=None) -> dict:
    oa = OpenAlex(budget, p.freshness_days)
    s2 = SemanticScholar(p.freshness_days)
    corpus = Corpus(con, run_id, p)
    stats: dict = {}
    this_year = date.today().year
    try:
        # 1. seeds -------------------------------------------------------------------------
        seed_works = await oa.works_by_dois(seed_dois) if seed_dois else []
        if seed_ids:
            seed_works += list((await oa.works_by_ids(seed_ids)).values())
        ingest.load_works(con, seed_works)
        seeds = list(dict.fromkeys(short_id(w["id"]) for w in seed_works))
        for s in seeds:
            corpus.add(s, "seed", 0)
        if ai_report:
            for pid in await resolve_ai_report(oa, con, run_id, ai_report):
                corpus.add(pid, "ai_report", 0)
        seed_set = corpus.ids("seed") + corpus.ids("ai_report")
        con.execute("UPDATE runs SET seed_ids = ? WHERE run_id = ?", [seed_set, run_id])
        stats["seeds"] = seed_set
        corpus.save()

        # S2 contexts for the seeds' references (needed for substantive hop-1 authors)
        if p.s2_contexts_for != "none":
            stats["s2_seed_refs"] = await ingest.s2_annotate_refs(con, s2, seed_set, p.s2_time_budget_s)

        # 2a. backward hop 1 ---------------------------------------------------------------
        seed_refs = _refs_of(con, seed_set)
        hop1_ids = list(dict.fromkeys(r for rs in seed_refs.values() for r in rs))
        works = await oa.works_by_ids(hop1_ids)
        ingest.load_works(con, list(works.values()))
        via = {r: s for s, rs in seed_refs.items() for r in rs}
        for r in hop1_ids:
            if r in works:
                corpus.add(r, "back", 1, via.get(r))
        hop1 = corpus.ids("back", 1)
        corpus.save()

        # 2b. backward hop 2: substantive edges, else top-k by co-citation in corpus --------
        if p.max_hops >= 2 and hop1:
            hop1_refs = _refs_of(con, hop1)
            cand = list(dict.fromkeys(r for rs in hop1_refs.values() for r in rs if r not in corpus))
            cand_works = await oa.works_by_ids(cand)
            ingest.load_works(con, list(cand_works.values()))
            if p.s2_contexts_for in ("seed_hop1", "all"):
                stats["s2_hop1_refs"] = await ingest.s2_annotate_refs(con, s2, hop1, p.s2_time_budget_s)
            subst = _substantive_edges(con, hop1)
            # co-citation: how many seed/hop-1 papers cite the candidate
            cocite = Counter(r for rs in list(hop1_refs.values()) + list(seed_refs.values())
                             for r in set(rs))
            cbc = {k: (w.get("cited_by_count") or 0) for k, w in cand_works.items()}
            picks = []
            for h, rs in hop1_refs.items():
                rs = [r for r in rs if r in cand_works]
                if subst.get(h) is not None:
                    rs = [r for r in rs if r in subst[h]]
                rs.sort(key=lambda r: (cocite[r], cbc.get(r, 0)), reverse=True)
                picks += [(cocite[r], r, h) for r in rs[:p.hop2_keep_per_paper]]
            for sc, r, h in sorted(picks, reverse=True):
                corpus.add(r, "back", 2, h, float(sc))
            corpus.save()
        stats["back"] = corpus.route_count("back")

        # 3. forward hops ------------------------------------------------------------------
        since = f"{this_year - p.forward_years}-01-01"
        anchors = set(seed_set) | set(hop1)
        cand_score: dict[str, float] = {}
        cand_via: dict[str, str] = {}
        sources = [(s, p.forward_per_seed) for s in seed_set] + [(h, p.forward_per_paper) for h in hop1]
        for src, cap in sources:
            ws = await oa.list_works(f"cites:{src},from_publication_date:{since}",
                                     sort="cited_by_count:desc", max_results=cap, kind="cites")
            ingest.load_works(con, ws)
            for w in ws:
                wid = short_id(w["id"])
                if wid in corpus:
                    continue
                cited = {short_id(r) for r in w.get("referenced_works") or []}
                score = len(cited & anchors) + 10 * len(cited & set(seed_set))
                if score > cand_score.get(wid, -1):
                    cand_score[wid], cand_via[wid] = score, src
            budget.flush(con, run_id)
        for wid, sc in sorted(cand_score.items(), key=lambda kv: -kv[1]):
            if not corpus.add(wid, "forward", 1, cand_via[wid], sc) and corpus.full("forward"):
                break
        corpus.save()
        stats["forward"] = corpus.route_count("forward")

        # 4. author fan-out ----------------------------------------------------------------
        subst_seed = _substantive_edges(con, seed_set)
        subst_hop1 = set().union(*[s for s in subst_seed.values() if s]) & set(hop1)
        if not subst_hop1:  # no S2 data: use the hop-1 papers most co-cited within the corpus
            refs_all = _refs_of(con, corpus.ids())
            cc = Counter(r for rs in refs_all.values() for r in set(rs) if r in set(hop1))
            subst_hop1 = {r for r, _ in cc.most_common(15)}
        con.register("_ids", pd.DataFrame({"paper_id": seed_set + sorted(subst_hop1)}))
        au = con.execute("""SELECT a.author_id, min(CASE WHEN a.paper_id IN ({}) THEN 0 ELSE 1 END) AS pri,
                                   count(*) AS n
                            FROM authorships a WHERE a.paper_id IN (SELECT paper_id FROM _ids)
                            GROUP BY 1 ORDER BY pri, n DESC""".format(
            ",".join(f"'{s}'" for s in seed_set) or "''")).df()
        con.unregister("_ids")
        authors = au["author_id"].tolist()[:p.max_authors]
        stats["fanout_authors"] = len(authors)
        a_since = f"{this_year - p.author_years}-01-01"
        for a in authors:
            if corpus.full("author"):
                break
            ws = await oa.list_works(f"author.id:{a},from_publication_date:{a_since}",
                                     sort="publication_date:desc", max_results=p.works_per_author,
                                     kind="author_works")
            ingest.load_works(con, ws)
            for w in ws:
                corpus.add(short_id(w["id"]), "author", 1, a)
        corpus.save()
        stats["author"] = corpus.route_count("author")
    except BudgetExceeded as e:
        log.warning("stopping expansion: %s", e)
        stats["stopped"] = str(e)
        corpus.save()
    finally:
        budget.flush(con, run_id)
        # S2 abstract fill for the corpus
        try:
            stats["s2_abstract_fill"] = await ingest.s2_fill_abstracts(con, s2, corpus.ids())
        finally:
            await oa.close()
            await s2.close()
        # arXiv copies of journal papers (OpenAlex locations -> sibling record or arXiv API)
        from .preprints import fill_from_preprints

        stats["arxiv_abstract_fill"] = await fill_from_preprints(con, run_id, budget, p.freshness_days)
        budget.flush(con, run_id)
    stats["corpus_size"] = len(corpus)
    return stats


async def expand_semantic(con, run_id: str, p: ExpansionParams, budget, embedder) -> dict:
    """Route 5: pull recent works under the corpus's dominant topics and keep those close to a
    corpus centroid (KMeans over abstract vectors)."""
    from sklearn.cluster import KMeans

    from . import embed

    con.execute("DELETE FROM provenance WHERE run_id = ? AND route = 'semantic'", [run_id])
    corpus = Corpus(con, run_id, p)
    if corpus.full("semantic"):
        return {"semantic": corpus.route_count("semantic"), "note": "route full"}
    topics = con.execute("""
        SELECT json_extract_string(topics, '$[0].id') AS tid,
               json_extract_string(topics, '$[0].name') AS name, count(*) AS n
        FROM papers p JOIN provenance v USING (paper_id)
        WHERE v.run_id = ? AND v.route != 'semantic' AND json_array_length(topics) > 0
        GROUP BY 1, 2 ORDER BY n DESC LIMIT ?""", [run_id, p.semantic_topics]).df()
    max_year = con.execute("""SELECT max(year) FROM papers JOIN provenance USING (paper_id)
                              WHERE run_id = ?""", [run_id]).fetchone()[0]
    since = f"{int(max_year) - p.forward_years}-01-01"
    oa = OpenAlex(budget, p.freshness_days)
    cands: dict[str, str] = {}
    try:
        for t in topics.itertuples():
            ws = await oa.list_works(f"topics.id:{t.tid},from_publication_date:{since},has_abstract:true",
                                     sort="cited_by_count:desc",
                                     max_results=p.semantic_candidates_per_topic, kind="topic_works")
            ingest.load_works(con, ws)
            for w in ws:
                wid = short_id(w["id"])
                if wid not in corpus:
                    cands.setdefault(wid, t.tid)
    except BudgetExceeded as e:
        log.warning("semantic route stopped: %s", e)
    finally:
        budget.flush(con, run_id)
        await oa.close()
    if not cands:
        return {"semantic": 0}

    base_ids, base_vecs = embed.load_vectors("abstract", ids=corpus.ids())
    base = embed.truncate(base_vecs, 256)
    k = int(np.clip(len(base_ids) // 150, 10, 60))
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(base)
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)

    if p.semantic_threshold == "auto":
        own = (base @ cents.T).max(axis=1)
        threshold = float(np.quantile(own, p.semantic_auto_quantile))
    else:
        threshold = float(p.semantic_threshold)
    log.info("semantic threshold %.3f (%s)", threshold, p.semantic_threshold)

    embed.embed_abstracts(con, embedder, list(cands))
    cand_ids, cand_vecs = embed.load_vectors("abstract", ids=list(cands))
    sims = embed.truncate(cand_vecs, 256) @ cents.T
    best = sims.max(axis=1)
    order = np.argsort(-best)
    added = 0
    for i in order:
        if best[i] < threshold:
            break
        if corpus.add(cand_ids[i], "semantic", 1, cands[cand_ids[i]], float(best[i])):
            added += 1
        elif corpus.full("semantic"):
            break
    corpus.save()
    return {"semantic_candidates": len(cands), "semantic_added": added, "threshold": threshold,
            "sim_quantiles": {q: float(np.quantile(best, q)) for q in (0.1, 0.5, 0.9)},
            "topics": topics[["tid", "name", "n"]].to_dict("records")}
