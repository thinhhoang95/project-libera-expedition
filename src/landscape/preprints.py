"""Fill missing abstracts from arXiv copies of the same work.

1. Ask OpenAlex for the `locations` of papers still missing an abstract (cheap: 100 ids per
   call, cached under raw/openalex/locations) and pull arXiv ids out of them.
2. If a work with that arXiv id is already in `papers` with an abstract (OpenAlex often holds
   the journal and the preprint as separate works), copy it.
3. Otherwise fetch the abstract from the arXiv API (free, batched, ~1 request per 3 s).

Filled abstracts get abstract_source = 'arxiv'. A preprint abstract can differ slightly from
the published one; the source column keeps that visible.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET

import httpx
import pandas as pd

from . import store
from .clients import OpenAlex, short_id
from .ingest import ARXIV_DOI_RE, ARXIV_RE

log = logging.getLogger(__name__)
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
_VERSION = re.compile(r"v\d+$")


def arxiv_from_locations(locations: list[dict] | None) -> str | None:
    for loc in locations or []:
        doi = (loc.get("doi") or "").lower()
        if m := ARXIV_DOI_RE.search(doi):
            return _VERSION.sub("", m.group(1))
        for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
            if url and (m := ARXIV_RE.search(url)):
                return _VERSION.sub("", m.group(1))
    return None


async def _locations(oa: OpenAlex, ids: list[str]) -> dict[str, list]:
    out, missing = {}, []
    for wid in ids:
        c = store.raw_get("openalex", "locations", wid, oa.fresh)
        if c is not None:
            out[wid] = c["locations"]
        else:
            missing.append(wid)

    async def one(chunk):
        data = await oa.get("/works", {"filter": "openalex:" + "|".join(chunk), "per_page": 100,
                                       "select": "id,locations"})
        got = {short_id(w["id"]): w.get("locations") or [] for w in data.get("results", [])}
        for wid in chunk:
            store.raw_put("openalex", "locations", wid, {"locations": got.get(wid, [])})
            out[wid] = got.get(wid, [])

    await asyncio.gather(*(one(missing[i:i + 100]) for i in range(0, len(missing), 100)))
    return out


async def _arxiv_abstracts(arxiv_ids: list[str], fresh: int) -> dict[str, str]:
    out, missing = {}, []
    for a in arxiv_ids:
        c = store.raw_get("arxiv", "abs", a, fresh)
        if c is not None:
            if c.get("abstract"):
                out[a] = c["abstract"]
        else:
            missing.append(a)
    async with httpx.AsyncClient(timeout=60.0, headers={"User-Agent": "landscape-engine/0.1"}) as http:
        for i in range(0, len(missing), 50):
            chunk = missing[i:i + 50]
            for attempt in range(4):
                try:
                    r = await http.get(ARXIV_API, params={"id_list": ",".join(chunk),
                                                          "max_results": len(chunk)})
                    r.raise_for_status()
                    break
                except httpx.HTTPError as e:
                    log.warning("arXiv API error %s (attempt %d)", e, attempt + 1)
                    await asyncio.sleep(3 * 2**attempt)
            else:
                continue
            found = {}
            for entry in ET.fromstring(r.text).findall("a:entry", ATOM):
                eid = entry.findtext("a:id", default="", namespaces=ATOM)
                aid = _VERSION.sub("", eid.split("/abs/")[-1])
                summary = " ".join((entry.findtext("a:summary", default="", namespaces=ATOM)).split())
                if aid and summary:
                    found[aid] = summary
            for a in chunk:
                store.raw_put("arxiv", "abs", a, {"abstract": found.get(a)})
                if found.get(a):
                    out[a] = found[a]
            await asyncio.sleep(3.1)  # arXiv asks for one request every 3 seconds
    return out


async def fill_from_preprints(con, run_id: str, budget, fresh: int = 30) -> dict:
    rows = con.execute("""
        SELECT p.paper_id, p.arxiv_id FROM papers p JOIN provenance v USING (paper_id)
        WHERE v.run_id = ? AND (p.abstract_raw IS NULL OR length(p.abstract_raw) < 100)""",
                       [run_id]).df()
    if rows.empty:
        return {"missing": 0}
    oa = OpenAlex(budget, fresh)
    try:
        locs = await _locations(oa, rows.paper_id.tolist())
    finally:
        await oa.close()
    link = {}
    for r in rows.itertuples():
        aid = r.arxiv_id if isinstance(r.arxiv_id, str) and r.arxiv_id else arxiv_from_locations(locs.get(r.paper_id))
        if aid:
            link[r.paper_id] = aid
    # copy from a sibling record (e.g. the preprint work) already in the database
    local = {}
    if link:
        sib = con.execute("""SELECT arxiv_id, abstract_raw FROM papers
                             WHERE arxiv_id IS NOT NULL AND length(abstract_raw) >= 100""").df()
        local = dict(zip(sib.arxiv_id, sib.abstract_raw))
    need_api = sorted({a for a in link.values() if a not in local})
    remote = await _arxiv_abstracts(need_api, fresh)
    filled_local = filled_api = 0
    for pid, aid in link.items():
        con.execute("UPDATE papers SET arxiv_id = coalesce(arxiv_id, ?) WHERE paper_id = ?", [aid, pid])
        text = local.get(aid) or remote.get(aid)
        if text:
            con.execute("UPDATE papers SET abstract_raw = ?, abstract_source = 'arxiv' WHERE paper_id = ?",
                        [text, pid])
            filled_local += aid in local
            filled_api += aid not in local
    return {"missing_before": len(rows), "with_arxiv_copy": len(link),
            "filled_from_local_record": filled_local, "filled_from_arxiv_api": filled_api,
            "missing_after": len(rows) - filled_local - filled_api}
