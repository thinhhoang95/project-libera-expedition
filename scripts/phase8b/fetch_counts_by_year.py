"""Fetch OpenAlex counts_by_year for every corpus work and author (experiment 5, second round).

Yearly counts make "as of T" measurements possible without leaking the future: a work's global
citations received in each year, and an author's global citations received and works published in
each year. OpenAlex keeps the last ten years. Responses are cached under data/shared/raw/openalex/
{work_counts,author_counts}/ and written to data/projects/<run>/backtest/oa_{work,author}_counts.parquet. The calls
are logged in api_calls under the base run.
"""
import asyncio

import pandas as pd

from landscape import experiments, store
from landscape.clients import Budget, OpenAlex, short_id

BASE = experiments.run_arg()


async def fetch(oa, path, kind, ids, select):
    out, missing = {}, []
    for i in dict.fromkeys(ids):
        c = store.raw_get("openalex", kind, i, 365)
        if c is not None:
            out[i] = c
        else:
            missing.append(i)

    async def one(chunk):
        data = await oa.get(path, {"filter": "openalex:" + "|".join(chunk), "per_page": 100, "select": select})
        for r in data.get("results", []):
            k = short_id(r["id"]) or r["id"]
            store.raw_put("openalex", kind, k, r)
            out[k] = r

    await asyncio.gather(*(one(missing[i:i + 100]) for i in range(0, len(missing), 100)))
    print(f"{kind}: {len(ids)} ids, {len(missing)} fetched, {len(out)} returned")
    return out


async def main():
    con = store.connect(read_only=True)
    works = con.execute("SELECT paper_id FROM provenance WHERE run_id = ?", [BASE]).df().paper_id.tolist()
    authors = con.execute("""SELECT DISTINCT author_id FROM authorships s JOIN provenance v USING (paper_id)
                             WHERE v.run_id = ?""", [BASE]).df().author_id.tolist()
    con.close()
    budget = Budget(cap_usd=1.0)
    oa = OpenAlex(budget)
    try:
        w = await fetch(oa, "/works", "work_counts", works, "id,cited_by_count,counts_by_year")
        a = await fetch(oa, "/authors", "author_counts", authors, "id,works_count,cited_by_count,counts_by_year")
    finally:
        await oa.close()
        con = store.connect()
        budget.flush(con, BASE)
        con.close()
    print(f"spend ${budget.spent:.4f}")
    wr = [{"paper_id": k, "year": c["year"], "cites": c.get("cited_by_count", 0)}
          for k, r in w.items() for c in r.get("counts_by_year") or []]
    ar = [{"author_id": k, "year": c["year"], "cites": c.get("cited_by_count", 0), "works": c.get("works_count", 0)}
          for k, r in a.items() for c in r.get("counts_by_year") or []]
    pd.DataFrame(wr).to_parquet(experiments.work_dir(BASE) / "oa_work_counts.parquet")
    pd.DataFrame(ar).to_parquet(experiments.work_dir(BASE) / "oa_author_counts.parquet")
    print(f"{len(wr)} work-year rows, {len(ar)} author-year rows")


asyncio.run(main())
