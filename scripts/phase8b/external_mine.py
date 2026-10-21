"""Mine the works that corpus papers cite but that are outside the corpus ("external imports").

80-94% of each year's references point outside the corpus. This script brings in the external
works cited at least MIN_CITES times by corpus papers, kept apart from the corpus tables:

    fetch    OpenAlex metadata, abstract, field/subfield/topic and yearly citation counts
    extract  the corpus facet prompt (same model, schema and validation as Phase 3)
    embed    abstracts and faithful facets with the corpus embedding model

Everything is written to data/backtest/<run>/external/ (works, edges, facets, vectors_<facet>).
Each stage skips what is already there, so a stopped run resumes.

Usage: uv run python scripts/phase8b/external_mine.py [fetch] [extract] [embed] [--run-id R]
       [--min-cites 5] [--db PATH]   (--db: a read-only copy of the database, e.g. while the
       explorer holds the lock)
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime

import duckdb
import pandas as pd

from landscape import embed, experiments, store
from landscape.cli import _params
from landscape.clients import OA_SELECT, Budget, OpenAlex, short_id
from landscape.extract import FIELDS, PROMPT_V1, SCHEMA, TAGS, validate
from landscape.ingest import rebuild_abstract
from landscape.llm import LLMBudgetExceeded, OpenRouter

ap = argparse.ArgumentParser()
ap.add_argument("stages", nargs="*", default=["fetch", "extract", "embed"])
ap.add_argument("--run-id", default="dro-pilot")
ap.add_argument("--min-cites", type=int, default=5)
ap.add_argument("--db", default=str(store.DB_PATH))
ap.add_argument("--llm-cap", type=float, default=1.0)
a = ap.parse_args()
RUN = a.run_id
OUT = experiments.work_dir(RUN) / "external"
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(a.db, read_only=True)
P = _params(con, RUN)


def edges():
    """Citations from corpus papers to works outside the corpus, with the citing year."""
    return con.execute("""
        WITH c AS (SELECT paper_id FROM provenance WHERE run_id = ?)
        SELECT r.citing_id, r.cited_id, p.year AS citing_year FROM refs r
        JOIN c ON c.paper_id = r.citing_id JOIN papers p ON p.paper_id = r.citing_id
        WHERE r.cited_id NOT IN (SELECT paper_id FROM c)""", [RUN]).df()


def _name(v):
    return v.get("display_name") if isinstance(v, dict) else v


async def fetch():
    e = edges()
    e.to_parquet(OUT / "edges.parquet", index=False)
    n = e.groupby("cited_id").citing_id.nunique()
    ids = n[n >= a.min_cites].index.tolist()
    print(f"{len(e)} external citations to {e.cited_id.nunique()} works; {len(ids)} cited >= {a.min_cites} times")
    budget = Budget(cap_usd=1.0)
    oa = OpenAlex(budget)
    out, missing = {}, []
    for i in ids:
        c = store.raw_get("openalex", "external_work", i, 365)
        (missing.append(i) if c is None else out.__setitem__(i, c))

    async def one(chunk):
        data = await oa.get("/works", {"filter": "openalex:" + "|".join(chunk), "per_page": 100,
                                       "select": OA_SELECT + ",counts_by_year"})
        for w in data.get("results", []):
            k = short_id(w["id"]) or w["id"]
            store.raw_put("openalex", "external_work", k, w)
            out[k] = w

    try:
        await asyncio.gather(*(one(missing[i:i + 100]) for i in range(0, len(missing), 100)))
    finally:
        await oa.close()
        if budget.calls:  # log the spend in the run's api_calls when the database is writable
            try:
                w = store.connect()
                budget.flush(w, RUN)
                w.close()
            except duckdb.IOException:
                pd.DataFrame(budget.calls, columns=["source", "endpoint", "cost", "status", "ts"]).to_parquet(
                    OUT / f"api_calls_{datetime.now():%Y%m%d%H%M%S}.parquet")
    print(f"fetched {len(missing)}, have {len(out)}, spend ${budget.spent:.4f}")
    rows = []
    for k, w in out.items():
        t = (w.get("topics") or [{}])[0] or {}
        src = ((w.get("primary_location") or {}).get("source") or {})
        rows.append({"paper_id": k, "title": w.get("title"), "abstract": rebuild_abstract(w.get("abstract_inverted_index")),
                     "year": w.get("publication_year"), "type": w.get("type"), "cited_by_count": w.get("cited_by_count"),
                     "field": _name(t.get("field")), "subfield": _name(t.get("subfield")), "topic": t.get("name")
                     or t.get("display_name"),
                     "venue": src.get("display_name"), "n_refs": len(w.get("referenced_works") or []),
                     "counts_by_year": json.dumps(w.get("counts_by_year") or []),
                     "corpus_citers": int(n.get(k, 0))})
    pd.DataFrame(rows).to_parquet(OUT / "works.parquet", index=False)


async def extract():
    works = pd.read_parquet(OUT / "works.parquet")
    have = set(pd.read_parquet(OUT / "facets.parquet").paper_id) if (OUT / "facets.parquet").exists() else set()
    todo = works[(works.abstract.fillna("").str.len() >= 100) & ~works.paper_id.isin(have)]
    print(f"extracting {len(todo)} abstracts ({(works.abstract.fillna('').str.len() < 100).sum()} without one)")
    p = P.extraction
    client = OpenRouter(p.model, p.quantizations, p.concurrency, a.llm_cap)
    rows, status = [], []

    def flush():
        if rows:
            new = pd.DataFrame(rows, columns=["paper_id", "facet", "text", "model_id", "prompt_version"])
            if (OUT / "facets.parquet").exists():
                new = pd.concat([pd.read_parquet(OUT / "facets.parquet"), new])
            new.to_parquet(OUT / "facets.parquet", index=False)
            rows.clear()

    async def one(r):
        try:
            await _one(r)
        except LLMBudgetExceeded:
            raise
        except Exception as ex:  # one bad record must not stop the batch
            status.append({"paper_id": r.paper_id, "ok": False, "flags": "{}", "error": repr(ex)[:200], "cost": 0.0})

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
        if not ok:
            return
        for k in FIELDS:
            v = obj[k]
            if isinstance(v, bool):
                v = "true" if v else "false"
            elif isinstance(v, str):
                v = v.strip().lower() if k in TAGS else v.strip()
            rows.append((r.paper_id, k, v, p.model, p.prompt_version))

    tasks = [asyncio.create_task(one(r)) for r in todo.itertuples()]
    try:
        for k, t in enumerate(asyncio.as_completed(tasks), 1):
            await t
            if k % 300 == 0:
                flush()
                print(f"  {k}/{len(tasks)} · ${client.spent:.3f}", flush=True)
    except LLMBudgetExceeded as ex:
        print("stopped:", ex)
        for t in tasks:
            t.cancel()
    finally:
        flush()
        await client.close()
    st = pd.DataFrame(status)
    if len(st):
        old = OUT / "facet_status.parquet"
        (pd.concat([pd.read_parquet(old), st]) if old.exists() else st).to_parquet(old, index=False)
    print(f"ok {int(st.ok.sum()) if len(st) else 0}, failed {int((~st.ok).sum()) if len(st) else 0}, "
          f"spend ${client.spent:.4f}")


def embed_all():
    works = pd.read_parquet(OUT / "works.parquet")
    facets = pd.read_parquet(OUT / "facets.parquet") if (OUT / "facets.parquet").exists() else pd.DataFrame(
        columns=["paper_id", "facet", "text"])
    E = embed.Embedder(P.embedding)
    titles = works.title.where(works.title.map(lambda t: isinstance(t, str)), "")
    jobs = {"abstract": works.assign(text=[f"{t}. {b}" if isinstance(b, str) and b else t
                                           for t, b in zip(titles, works.abstract)])[["paper_id", "text"]]}
    for f in ("problem", "method", "innovation", "result", "limitations"):
        d = facets[(facets.facet == f) & facets.text.notna()]
        jobs[f] = d[d.text.astype(str).str.strip().str.len() > 0][["paper_id", "text"]]
    for f, d in jobs.items():
        path = OUT / f"vectors_{f}.parquet"
        have = set()
        if path.exists():
            have = set(pd.read_parquet(path, columns=["paper_id"]).paper_id)
        d = d[~d.paper_id.isin(have) & (d.text.str.len() > 0)]
        if len(d):
            print(f"embedding {len(d)} {f} texts", flush=True)
            embed.append_vectors_to(path, d.paper_id.tolist(), E.encode(d.text.tolist()), E.version)


for s in a.stages:
    print(f"== {s}", flush=True)
    if s == "fetch":
        asyncio.run(fetch())
    elif s == "extract":
        asyncio.run(extract())
    elif s == "embed":
        embed_all()
    else:
        sys.exit(f"unknown stage {s}")
