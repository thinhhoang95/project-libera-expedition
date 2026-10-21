"""Async HTTP clients for OpenAlex and Semantic Scholar with caching, backoff and a budget stop."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime

import httpx

from . import store
from .config import OPENALEX_API_KEY, S2_API_KEY

log = logging.getLogger(__name__)

OA_BASE = "https://api.openalex.org"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
OA_SELECT = (
    "id,doi,title,publication_year,publication_date,authorships,referenced_works,"
    "abstract_inverted_index,primary_location,cited_by_count,is_retracted,topics,type,language"
)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Running spend per source; raises once the cap is reached so a run stops cleanly."""

    def __init__(self, cap_usd: float, spent_usd: float = 0.0):
        self.cap = cap_usd
        self.spent = spent_usd
        self.calls: list[tuple] = []  # (source, endpoint, cost, status, ts) -> api_calls table

    def charge(self, source: str, endpoint: str, cost: float, status: int) -> None:
        self.spent += cost
        self.calls.append((source, endpoint, cost, status, datetime.now()))

    def check(self) -> None:
        if self.spent >= self.cap:
            raise BudgetExceeded(f"metadata spend ${self.spent:.4f} reached cap ${self.cap:.2f}")

    def flush(self, con, run_id: str) -> None:
        if not self.calls:
            return
        con.executemany(
            "INSERT INTO api_calls VALUES (?, ?, ?, ?, ?, ?)",
            [(run_id, *c) for c in self.calls],
        )
        self.calls.clear()


def short_id(oa_id: str | None) -> str | None:
    if not oa_id:
        return None
    return oa_id.rsplit("/", 1)[-1]


async def _request_with_backoff(client: httpx.AsyncClient, method: str, url: str, *,
                                max_tries: int = 8, base: float = 1.0, cap: float = 60.0, **kw):
    for attempt in range(max_tries):
        try:
            r = await client.request(method, url, **kw)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            if attempt == max_tries - 1:
                raise
            log.warning("transport error %s on %s, retrying", e, url[:120])
        else:
            if r.status_code < 400:
                return r
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == max_tries - 1:
                r.raise_for_status()
        await asyncio.sleep(min(cap, base * 2**attempt) * (0.5 + random.random()))
    raise RuntimeError("unreachable")


class OpenAlex:
    def __init__(self, budget: Budget, freshness_days: int = 30, concurrency: int = 16,
                 max_rps: float = 50.0):
        if not OPENALEX_API_KEY:
            raise RuntimeError("OPENALEX_API_KEY missing from .env.local")
        self.budget = budget
        self.fresh = freshness_days
        self.sem = asyncio.Semaphore(concurrency)
        self._min_gap = 1.0 / max_rps
        self._last = 0.0
        self._lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=60.0, headers={"User-Agent": "landscape-engine/0.1"})

    async def close(self):
        await self.http.aclose()

    async def _throttle(self):
        async with self._lock:
            wait = self._last + self._min_gap - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

    async def get(self, path: str, params: dict) -> dict:
        self.budget.check()
        params = {**params, "api_key": OPENALEX_API_KEY}
        async with self.sem:
            await self._throttle()
            r = await _request_with_backoff(self.http, "GET", OA_BASE + path, params=params)
        data = r.json()
        cost = float((data.get("meta") or {}).get("cost_usd") or 0.0)
        self.budget.charge("openalex", path + "?" + str(params.get("filter", ""))[:200], cost,
                           r.status_code)
        return data

    # -- single entities, cached per work

    def cached_work(self, wid: str):
        return store.raw_get("openalex", "work", wid, self.fresh)

    async def works_by_ids(self, ids: list[str]) -> dict[str, dict]:
        """Fetch works by OpenAlex W id, 100 per call; cached ids are not re-fetched."""
        out, missing = {}, []
        for wid in dict.fromkeys(ids):
            w = self.cached_work(wid)
            if w is not None:
                out[wid] = w
            else:
                missing.append(wid)
        chunks = [missing[i:i + 100] for i in range(0, len(missing), 100)]

        async def one(chunk):
            data = await self.get("/works", {
                "filter": "openalex:" + "|".join(chunk), "per_page": 100, "select": OA_SELECT})
            for w in data.get("results", []):
                wid = short_id(w["id"])
                store.raw_put("openalex", "work", wid, w)
                out[wid] = w
            # Remember ids OpenAlex no longer serves (merged/deleted) so they are not retried.
            for wid in set(chunk) - {short_id(w["id"]) for w in data.get("results", [])}:
                store.raw_put("openalex", "work_missing", wid, {"missing": True})

        await asyncio.gather(*(one(c) for c in chunks))
        return out

    async def works_by_dois(self, dois: list[str]) -> list[dict]:
        dois = [d.lower().removeprefix("https://doi.org/") for d in dois]
        res = []
        for i in range(0, len(dois), 100):
            chunk = dois[i:i + 100]
            data = await self.get("/works", {
                "filter": "doi:" + "|".join(chunk), "per_page": 100, "select": OA_SELECT})
            for w in data.get("results", []):
                store.raw_put("openalex", "work", short_id(w["id"]), w)
                res.append(w)
        return res

    async def list_works(self, filt: str, *, sort: str | None = None, max_results: int = 200,
                         kind: str = "query") -> list[dict]:
        """Cursor-paged /works listing, cached as a whole under a query hash."""
        key = store.query_key(filt, sort, max_results)
        cached = store.raw_get("openalex", kind, key, self.fresh)
        if cached is not None:
            return cached["results"]
        results, cursor = [], "*"
        per_page = min(100, max_results)
        while cursor and len(results) < max_results:
            params = {"filter": filt, "per_page": per_page, "cursor": cursor, "select": OA_SELECT}
            if sort:
                params["sort"] = sort
            data = await self.get("/works", params)
            batch = data.get("results", [])
            results.extend(batch)
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not batch:
                break
        results = results[:max_results]
        for w in results:
            store.raw_put("openalex", "work", short_id(w["id"]), w)
        store.raw_put("openalex", kind, key, {"filter": filt, "sort": sort, "results": results})
        return results


class SemanticScholar:
    """S2 Graph API. Works without a key, but the shared pool returns many 429s."""

    def __init__(self, freshness_days: int = 30):
        self.fresh = freshness_days
        headers = {"User-Agent": "landscape-engine/0.1"}
        if S2_API_KEY:
            headers["x-api-key"] = S2_API_KEY
        self.http = httpx.AsyncClient(timeout=60.0, headers=headers)
        self._lock = asyncio.Lock()  # S2 keys are typically 1 rps: serialize calls
        self._min_gap = 1.05
        self._last = 0.0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3  # circuit breaker: give up on S2 for this run

    @property
    def disabled(self) -> bool:
        return self.consecutive_failures >= self.max_consecutive_failures

    async def close(self):
        await self.http.aclose()

    async def _call(self, method, path, **kw):
        if self.disabled:
            raise httpx.HTTPError("S2 disabled after repeated failures")
        async with self._lock:
            wait = self._last + self._min_gap - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                r = await _request_with_backoff(self.http, method, S2_BASE + path, max_tries=6,
                                                base=2.0, **kw)
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:
                    self.consecutive_failures += 1
                raise
            except httpx.HTTPError:
                self.consecutive_failures += 1
                raise
            finally:
                self._last = time.monotonic()
        self.consecutive_failures = 0
        return r.json()

    async def batch_papers(self, ids: list[str], fields: str = "abstract,externalIds,year,title"):
        """ids like 'DOI:10.x/y' or 'ARXIV:2101.00001'. Returns {id: record or None}."""
        out, missing = {}, []
        for i in ids:
            c = store.raw_get("s2", "paper", i, self.fresh)
            if c is not None:
                out[i] = c.get("record")
            else:
                missing.append(i)
        for k in range(0, len(missing), 400):
            chunk = missing[k:k + 400]
            try:
                data = await self._call("POST", "/paper/batch", params={"fields": fields},
                                        json={"ids": chunk})
            except httpx.HTTPError as e:
                log.warning("S2 batch failed (%s); leaving %d ids unfilled", e, len(chunk))
                continue
            for i, rec in zip(chunk, data):
                store.raw_put("s2", "paper", i, {"record": rec})
                out[i] = rec
        return out

    async def references(self, s2_query_id: str, cache_only: bool = False) -> list[dict] | None:
        """All references of one paper with contexts, intents and isInfluential."""
        cached = store.raw_get("s2", "refs", s2_query_id, self.fresh)
        if cached is not None:
            return cached["data"]
        if cache_only or self.disabled:
            return None
        fields = "contexts,intents,isInfluential,externalIds"
        data, offset = [], 0
        try:
            while True:
                page = await self._call("GET", f"/paper/{s2_query_id}/references",
                                        params={"fields": fields, "limit": 1000, "offset": offset})
                data.extend(page.get("data") or [])
                if page.get("next") is None:
                    break
                offset = page["next"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                store.raw_put("s2", "refs", s2_query_id, {"data": []})
                return []
            log.warning("S2 references failed for %s: %s", s2_query_id, e)
            return None
        except httpx.HTTPError as e:
            log.warning("S2 references failed for %s: %s", s2_query_id, e)
            return None
        store.raw_put("s2", "refs", s2_query_id, {"data": data})
        return data
