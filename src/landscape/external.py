"""External works: what the corpus cites outside itself (Phase 8b, experiment 6; product: Start here,
Bridges › From outside the field, My paper).

80-94% of a year's references point outside the corpus. This step brings in the external works cited
by at least `min_cites` corpus papers, kept apart from the corpus tables:

    fetch    OpenAlex metadata, abstract, field/subfield/topic and yearly citation counts
    extract  the corpus facet prompt (same model, schema and validation as Phase 3); LLM spend
    embed    abstracts and faithful facets with the corpus embedding model

Files: data/external/<run>/{edges,works,facets,facet_status}.parquet and vectors_<facet>.parquet.
Each stage skips what is already there, so a stopped run resumes.

    uv run landscape external --run-id dro-pilot [--stages fetch extract embed] [--min-cites 5]
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import embed, store
from .clients import OA_SELECT, Budget, OpenAlex, short_id
from .config import RunParams
from .extract import FIELDS, PROMPT_V1, SCHEMA, TAGS, validate
from .ingest import rebuild_abstract
from .llm import LLMBudgetExceeded, OpenRouter

log = logging.getLogger(__name__)
FACETS = ("problem", "method", "innovation", "result", "limitations")


def ext_dir(run_id: str) -> Path:
    d = store.DATA / "external" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _name(v):
    return v.get("display_name") if isinstance(v, dict) else v


def edges(con, run_id: str) -> pd.DataFrame:
    """Citations from corpus papers to works outside the corpus, with the citing year."""
    return con.execute("""
        WITH c AS (SELECT paper_id FROM provenance WHERE run_id = ?)
        SELECT r.citing_id, r.cited_id, p.year AS citing_year FROM refs r
        JOIN c ON c.paper_id = r.citing_id JOIN papers p ON p.paper_id = r.citing_id
        WHERE r.cited_id NOT IN (SELECT paper_id FROM c)""", [run_id]).df()


async def fetch(con, run_id: str, min_cites: int = 5, cap_usd: float = 1.0) -> dict:
    out_dir = ext_dir(run_id)
    e = edges(con, run_id)
    e.to_parquet(out_dir / "edges.parquet", index=False)
    n = e.groupby("cited_id").citing_id.nunique()
    ids = n[n >= min_cites].index.tolist()
    budget = Budget(cap_usd=cap_usd)
    oa = OpenAlex(budget)
    out, missing = {}, []
    for i in ids:
        c = store.raw_get("openalex", "external_work", i, 365)
        if c is not None:
            out[i] = c
        elif store.raw_get("openalex", "external_work_missing", i, 365) is None:
            missing.append(i)

    async def one(chunk):
        data = await oa.get("/works", {"filter": "openalex:" + "|".join(chunk), "per_page": 100,
                                       "select": OA_SELECT + ",counts_by_year"})
        for w in data.get("results", []):
            k = short_id(w["id"]) or w["id"]
            store.raw_put("openalex", "external_work", k, w)
            out[k] = w
        # remember ids OpenAlex no longer serves (merged or deleted) so they are not requested again
        for k in set(chunk) - set(out):
            store.raw_put("openalex", "external_work_missing", k, {"missing": True})

    try:
        await asyncio.gather(*(one(missing[i:i + 100]) for i in range(0, len(missing), 100)))
    finally:
        await oa.close()
        budget.flush(con, run_id)
    rows = []
    for k, w in out.items():
        t = (w.get("topics") or [{}])[0] or {}
        src = ((w.get("primary_location") or {}).get("source") or {})
        rows.append({"paper_id": k, "title": w.get("title"),
                     "abstract": rebuild_abstract(w.get("abstract_inverted_index")),
                     "year": w.get("publication_year"), "type": w.get("type"), "cited_by_count": w.get("cited_by_count"),
                     "field": _name(t.get("field")), "subfield": _name(t.get("subfield")),
                     "topic": t.get("name") or t.get("display_name"), "venue": src.get("display_name"),
                     "n_refs": len(w.get("referenced_works") or []),
                     "counts_by_year": json.dumps(w.get("counts_by_year") or []), "corpus_citers": int(n.get(k, 0))})
    pd.DataFrame(rows).to_parquet(out_dir / "works.parquet", index=False)
    return {"external_citations": len(e), "external_works": int(e.cited_id.nunique()), "fetched_ids": len(ids),
            "requested": len(missing), "have": len(out), "spend_usd": round(budget.spent, 4)}


async def extract(run_id: str, p: RunParams, cap_usd: float = 2.0) -> dict:
    out_dir = ext_dir(run_id)
    works = pd.read_parquet(out_dir / "works.parquet")
    fpath = out_dir / "facets.parquet"
    have = set(pd.read_parquet(fpath).paper_id) if fpath.exists() else set()
    todo = works[(works.abstract.fillna("").str.len() >= 100) & ~works.paper_id.isin(have)]
    ep = p.extraction
    client = OpenRouter(ep.model, ep.quantizations, ep.concurrency, cap_usd)
    rows, status = [], []

    def flush():
        if rows:
            new = pd.DataFrame(rows, columns=["paper_id", "facet", "text", "model_id", "prompt_version"])
            (pd.concat([pd.read_parquet(fpath), new]) if fpath.exists() else new).to_parquet(fpath, index=False)
            rows.clear()

    async def _one(r):
        title = r.title if isinstance(r.title, str) else ""
        prompt = PROMPT_V1.replace("{title}", title).replace("{abstract}", r.abstract)
        obj, cost, err = await client.chat_json(prompt, SCHEMA, "facets", max_tokens=1200)
        ok, flags = validate(obj, r.abstract) if obj is not None else (False, {})
        if not ok:
            obj, c2, err = await client.chat_json(prompt, SCHEMA, "facets", max_tokens=1200)
            cost += c2
            ok, flags = validate(obj, r.abstract) if obj is not None else (False, {})
        status.append({"paper_id": r.paper_id, "ok": ok, "flags": json.dumps(flags), "error": err, "cost": cost})
        if ok:
            for k in FIELDS:
                v = obj[k]
                if isinstance(v, bool):
                    v = "true" if v else "false"
                elif isinstance(v, str):
                    v = v.strip().lower() if k in TAGS else v.strip()
                rows.append((r.paper_id, k, v, ep.model, ep.prompt_version))

    async def one(r):
        try:
            await _one(r)
        except LLMBudgetExceeded:
            raise
        except Exception as ex:  # one bad record must not stop the batch
            status.append({"paper_id": r.paper_id, "ok": False, "flags": "{}", "error": repr(ex)[:200], "cost": 0.0})

    tasks = [asyncio.create_task(one(r)) for r in todo.itertuples()]
    try:
        for k, t in enumerate(asyncio.as_completed(tasks), 1):
            await t
            if k % 300 == 0:
                flush()
                log.info("external facets %d/%d, $%.3f", k, len(tasks), client.spent)
    except LLMBudgetExceeded as ex:
        log.warning("external extraction stopped: %s", ex)
        for t in tasks:
            t.cancel()
    finally:
        flush()
        await client.close()
    st = pd.DataFrame(status)
    if len(st):
        old = out_dir / "facet_status.parquet"
        (pd.concat([pd.read_parquet(old), st]) if old.exists() else st).to_parquet(old, index=False)
    return {"todo": len(todo), "ok": int(st.ok.sum()) if len(st) else 0,
            "failed": int((~st.ok).sum()) if len(st) else 0, "spend_usd": round(client.spent, 4)}


def embed_all(run_id: str, p: RunParams) -> dict:
    out_dir = ext_dir(run_id)
    works = pd.read_parquet(out_dir / "works.parquet")
    fpath = out_dir / "facets.parquet"
    facets = pd.read_parquet(fpath) if fpath.exists() else pd.DataFrame(columns=["paper_id", "facet", "text"])
    titles = works.title.where(works.title.map(lambda t: isinstance(t, str)), "")
    jobs = {"abstract": works.assign(text=[f"{t}. {b}" if isinstance(b, str) and b else t
                                           for t, b in zip(titles, works.abstract)])[["paper_id", "text"]]}
    for f in FACETS:
        d = facets[(facets.facet == f) & facets.text.notna()]
        jobs[f] = d[d.text.astype(str).str.strip().str.len() > 0][["paper_id", "text"]]
    E, done = None, {}
    for f, d in jobs.items():
        path = out_dir / f"vectors_{f}.parquet"
        have = set(pd.read_parquet(path, columns=["paper_id"]).paper_id) if path.exists() else set()
        d = d[~d.paper_id.isin(have) & (d.text.str.len() > 0)]
        if len(d):
            E = E or embed.Embedder(p.embedding)
            embed.append_vectors_to(path, d.paper_id.tolist(), E.encode(d.text.tolist()), E.version)
        done[f] = len(d)
    return done


def vectors(run_id: str, facet: str, dim: int | None = None) -> tuple[dict, np.ndarray]:
    """{paper_id: row} and the (optionally truncated, renormalized) vectors of one external facet."""
    path = ext_dir(run_id) / f"vectors_{facet}.parquet"
    if not path.exists():
        return {}, np.zeros((0, dim or 768), np.float32)
    t = pq.read_table(path)
    pid = t.column("paper_id").to_pylist()
    v = np.asarray(t.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                   dtype=np.float32).reshape(len(pid), -1)
    return {x: k for k, x in enumerate(pid)}, (embed.truncate(v, dim) if dim else v)


def run_external(con, run_id: str, p: RunParams, stages=("fetch", "extract", "embed"), min_cites: int = 5,
                 llm_cap_usd: float = 2.0) -> dict:
    out = {}
    if "fetch" in stages:
        out["fetch"] = asyncio.run(fetch(con, run_id, min_cites))
    if "extract" in stages:
        out["extract"] = asyncio.run(extract(run_id, p, llm_cap_usd))
    if "embed" in stages:
        out["embed"] = embed_all(run_id, p)
    return out
