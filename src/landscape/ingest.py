"""Phase 1: parse raw OpenAlex / S2 responses into the papers, authors, authorships,
institutions and refs tables."""

from __future__ import annotations

import json
import re
from datetime import datetime

import pandas as pd

from . import store
from .clients import short_id

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})",
                      re.I)
ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)


def rebuild_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


def _doi(w: dict) -> str | None:
    d = w.get("doi")
    return d.lower().removeprefix("https://doi.org/") if d else None


def arxiv_id(w: dict) -> str | None:
    doi = _doi(w)
    if doi and (m := ARXIV_DOI_RE.search(doi)):
        return m.group(1)
    loc = w.get("primary_location") or {}
    for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
        if url and (m := ARXIV_RE.search(url)):
            return m.group(1)
    return None


def parse_works(works: list[dict]) -> dict[str, pd.DataFrame]:
    now = datetime.now()
    papers, authors, authorships, insts, refs = [], {}, [], {}, []
    for w in works:
        pid = short_id(w["id"])
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        abstract = rebuild_abstract(w.get("abstract_inverted_index"))
        ref_ids = [short_id(r) for r in (w.get("referenced_works") or [])]
        topics = [
            {"id": short_id(t.get("id")), "name": t.get("display_name"), "score": t.get("score"),
             "field": (t.get("field") or {}).get("display_name"),
             "subfield": (t.get("subfield") or {}).get("display_name")}
            for t in (w.get("topics") or [])
        ]
        pub_date = w.get("publication_date")
        papers.append({
            "paper_id": pid, "doi": _doi(w), "s2_id": None, "arxiv_id": arxiv_id(w),
            "title": w.get("title"), "abstract_raw": abstract,
            "abstract_source": "openalex" if abstract else None,
            "year": w.get("publication_year"),
            "pub_date": pd.to_datetime(pub_date, errors="coerce") if pub_date else None,
            "venue_id": short_id(src.get("id")), "venue_name": src.get("display_name"),
            "type": w.get("type"), "language": w.get("language"),
            "is_retracted": bool(w.get("is_retracted")),
            "cited_by_count": w.get("cited_by_count"), "topics": json.dumps(topics),
            "referenced_works": ref_ids, "fetched_at": now,
        })
        for pos, a in enumerate(w.get("authorships") or []):
            au = a.get("author") or {}
            aid = short_id(au.get("id"))
            if not aid:
                continue
            authors[aid] = {"author_id": aid, "s2_author_id": None,
                            "orcid": (au.get("orcid") or "").removeprefix("https://orcid.org/") or None,
                            "display_name": au.get("display_name")}
            inst_ids = []
            for ins in a.get("institutions") or []:
                iid = short_id(ins.get("id"))
                if iid:
                    inst_ids.append(iid)
                    insts[iid] = {"institution_id": iid, "name": ins.get("display_name"),
                                  "country": ins.get("country_code"), "type": ins.get("type")}
            authorships.append({"paper_id": pid, "author_id": aid, "position": pos,
                                "institution_ids": inst_ids})
        for rid in ref_ids:
            refs.append({"citing_id": pid, "cited_id": rid, "source": "openalex",
                         "contexts": None, "intents": None, "is_influential": None})
    df_auth = pd.DataFrame(authorships)
    if not df_auth.empty:
        df_auth = df_auth.drop_duplicates(["paper_id", "author_id"])
    return {
        "papers": pd.DataFrame(papers).drop_duplicates("paper_id") if papers else pd.DataFrame(),
        "authors": pd.DataFrame(list(authors.values())),
        "authorships": df_auth,
        "institutions": pd.DataFrame(list(insts.values())),
        "refs": pd.DataFrame(refs).drop_duplicates(["citing_id", "cited_id"]) if refs else pd.DataFrame(),
    }


def load_works(con, works: list[dict]) -> None:
    """Upsert works. Existing S2-filled abstracts and S2 ref annotations are preserved."""
    if not works:
        return
    t = parse_works(works)
    p = t["papers"]
    if not p.empty:
        # keep abstracts previously filled from S2 when OpenAlex has none
        con.register("_new", p[["paper_id"]])
        old = con.execute("""SELECT paper_id, s2_id, abstract_raw, abstract_source FROM papers
                             WHERE paper_id IN (SELECT paper_id FROM _new)""").df()
        con.unregister("_new")
        if not old.empty:
            p = p.merge(old, on="paper_id", how="left", suffixes=("", "_old"))
            fill = p["abstract_raw"].isna() & p["abstract_raw_old"].notna()
            p.loc[fill, "abstract_raw"] = p.loc[fill, "abstract_raw_old"]
            p.loc[fill, "abstract_source"] = p.loc[fill, "abstract_source_old"]
            p["s2_id"] = p["s2_id"].fillna(p["s2_id_old"])
            p = p.drop(columns=["abstract_raw_old", "abstract_source_old", "s2_id_old"])
        store.upsert_df(con, "papers", p)
    store.upsert_df(con, "authors", t["authors"])
    store.upsert_df(con, "authorships", t["authorships"])
    store.upsert_df(con, "institutions", t["institutions"])
    r = t["refs"]
    if not r.empty:
        con.register("_r", r)
        con.execute("""INSERT INTO refs SELECT * FROM _r
                       ON CONFLICT (citing_id, cited_id) DO NOTHING""")
        con.unregister("_r")


def s2_query_id(row) -> str | None:
    if isinstance(row.get("doi"), str) and row["doi"]:
        return "DOI:" + row["doi"]
    if isinstance(row.get("arxiv_id"), str) and row["arxiv_id"]:
        return "ARXIV:" + row["arxiv_id"]
    return None


async def s2_fill_abstracts(con, s2, paper_ids: list[str]) -> dict:
    """Fill missing abstracts from S2 (by DOI, arXiv fallback) and record s2 ids."""
    con.register("_ids", pd.DataFrame({"paper_id": paper_ids}))
    rows = con.execute("""SELECT paper_id, doi, arxiv_id FROM papers
                          WHERE paper_id IN (SELECT paper_id FROM _ids)
                            AND (abstract_raw IS NULL OR length(abstract_raw) < 50)""").df()
    con.unregister("_ids")
    rows["q"] = [s2_query_id(r) for r in rows.to_dict("records")]
    rows = rows[rows["q"].notna()]
    if rows.empty:
        return {"tried": 0, "filled": 0}
    res = await s2.batch_papers(rows["q"].tolist())
    filled = 0
    for r in rows.itertuples():
        rec = res.get(r.q)
        if not rec:
            continue
        con.execute("UPDATE papers SET s2_id = ? WHERE paper_id = ?", [rec.get("paperId"), r.paper_id])
        if rec.get("abstract"):
            con.execute("""UPDATE papers SET abstract_raw = ?, abstract_source = 's2'
                           WHERE paper_id = ?""", [rec["abstract"], r.paper_id])
            filled += 1
    return {"tried": len(rows), "filled": filled}


async def s2_annotate_refs(con, s2, paper_ids: list[str], time_budget_s: float = 300) -> dict:
    """Attach S2 contexts / intents / isInfluential to refs rows of the given citing papers.

    S2 references are matched to OpenAlex works by DOI (arXiv id as fallback). After
    `time_budget_s` only cached S2 responses are used, so a slow S2 pool cannot stall a run."""
    import time

    deadline = time.monotonic() + time_budget_s
    con.register("_ids", pd.DataFrame({"paper_id": paper_ids}))
    rows = con.execute("""SELECT paper_id, doi, arxiv_id FROM papers
                          WHERE paper_id IN (SELECT paper_id FROM _ids)""").df()
    con.unregister("_ids")
    done = annotated = 0
    for r in rows.to_dict("records"):
        q = s2_query_id(r)
        if not q:
            continue
        data = await s2.references(q, cache_only=time.monotonic() > deadline)
        if data is None:
            continue
        done += 1
        cited = con.execute("""SELECT r.cited_id, p.doi, p.arxiv_id FROM refs r
                               JOIN papers p ON p.paper_id = r.cited_id
                               WHERE r.citing_id = ?""", [r["paper_id"]]).df()
        by_doi = {d: c for c, d in zip(cited.cited_id, cited.doi) if d}
        by_ax = {a: c for c, a in zip(cited.cited_id, cited.arxiv_id) if a}
        for item in data:
            ext = (item.get("citedPaper") or {}).get("externalIds") or {}
            cid = by_doi.get((ext.get("DOI") or "").lower()) or by_ax.get(ext.get("ArXiv") or "")
            if not cid:
                continue
            con.execute("""UPDATE refs SET contexts = ?, intents = ?, is_influential = ?
                           WHERE citing_id = ? AND cited_id = ?""",
                        [item.get("contexts") or [], item.get("intents") or [],
                         bool(item.get("isInfluential")), r["paper_id"], cid])
            annotated += 1
    return {"papers_with_s2_refs": done, "edges_annotated": annotated}
