"""Conservative work identity for evidence lists; stored OpenAlex records stay untouched.

Only records available at the caller's cutoff are considered. Equal normalized DOIs are
aliases; equal substantial titles also need at least one shared OpenAlex author ID.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def title_key(title: str | None) -> str:
    text = unicodedata.normalize("NFKD", title or "").casefold()
    return " ".join(re.findall(r"[^\W_]+", text))


def work_aliases(papers: pd.DataFrame, authorships: pd.DataFrame) -> dict[str, str]:
    """Map each supplied record to its earliest supplied alias (then paper ID).

    This deliberately misses paraphrased titles. No fuzzy semantic merges, and no future
    records used to choose a representative. Missing authors cannot establish title identity.
    """
    records = papers.drop_duplicates("paper_id").copy()
    parent = {pid: pid for pid in records.paper_id}

    def root(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    def join(a, b):
        parent[root(b)] = root(a)

    authors = authorships.groupby("paper_id").author_id.agg(set).to_dict()
    dois, titles = {}, {}
    for row in records.itertuples():
        doi = getattr(row, "doi", None)
        if isinstance(doi, str) and doi.strip():
            key = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi.strip().lower())
            if key in dois:
                join(row.paper_id, dois[key])
            else:
                dois[key] = row.paper_id
        key = title_key(row.title if isinstance(row.title, str) else "")
        if len(key.split()) >= 5:
            for other in titles.get(key, []):
                if authors.get(row.paper_id, set()) & authors.get(other, set()):
                    join(row.paper_id, other)
            titles.setdefault(key, []).append(row.paper_id)
    records["_root"] = records.paper_id.map(root)
    representatives = (records.sort_values(["year", "paper_id"], na_position="last")
                       .drop_duplicates("_root").set_index("_root").paper_id)
    return {pid: representatives[root(pid)] for pid in parent}
