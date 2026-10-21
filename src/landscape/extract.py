"""Phase 3: LLM facet extraction with automatic QA checks."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from .config import ExtractionParams
from .llm import LLMBudgetExceeded, OpenRouter

log = logging.getLogger(__name__)

FAITHFUL = ["problem", "domain_data", "method", "innovation", "result", "limitations"]
GLOSSES = ["problem_gloss", "method_gloss"]
TAGS = ["problem_class", "method_family"]
CONTRIB = ["method", "dataset", "benchmark", "theory", "survey", "application", "position", "other"]
FIELDS = ["problem", "problem_class", "problem_gloss", "domain_data", "method", "method_family",
          "method_gloss", "innovation", "result", "limitations", "contribution_type",
          "has_quantitative_result"]

_s = {"type": "string"}
_sn = {"type": ["string", "null"]}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": FIELDS,
    "properties": {
        "problem": {**_s, "description": "the task or question addressed, stated generically (faithful)"},
        "problem_class": {**_s, "description": "1-4 word canonical tag, e.g. 'stochastic optimization'"},
        "problem_gloss": {**_s, "description": "1-2 sentences of background on this problem class as it appears here (model knowledge)"},
        "domain_data": {**_sn, "description": "application domain and data used, or null (faithful)"},
        "method": {**_s, "description": "the core technique or approach (faithful)"},
        "method_family": {**_s, "description": "1-4 word canonical tag, e.g. 'chance-constrained programming'"},
        "method_gloss": {**_s, "description": "1-2 sentences on how this technique works, in the variant this abstract describes (model knowledge)"},
        "innovation": {**_s, "description": "what is new relative to prior work (faithful)"},
        "result": {**_s, "description": "main finding, with numbers if stated (faithful)"},
        "limitations": {**_sn, "description": "ONLY if explicitly stated in the abstract (faithful)"},
        "contribution_type": {"type": "string", "enum": CONTRIB},
        "has_quantitative_result": {"type": "boolean"},
    },
}

PROMPT_V1 = """You extract structured facets from a scientific abstract.
Faithful fields (problem, domain_data, method, innovation, result, limitations):
- Use only information in the abstract. Never infer or add.
- One or two standalone sentences each, at most 40 words, no pronouns referring outside the field.
- Use generic technical vocabulary rather than the paper's coined names, but keep the coined name in parentheses.
- "limitations": null unless the abstract explicitly states a limitation, weakness, or open problem.
- "domain_data": null if not stated.
Gloss fields (problem_gloss, method_gloss):
- Explain the general concept in at most 50 words, as a textbook would, matched to the specific variant this abstract describes.
- If the variant is unclear, describe the general family and say "variant unclear".
- Never copy gloss content into faithful fields.
Tags (problem_class, method_family):
- 1-4 words, lowercase, the most common textbook term for the class or family.
Write in English even if the abstract is not.
Return JSON matching the schema exactly.

Title: {title}
Abstract: {abstract}"""

_W = re.compile(r"[a-z0-9]+")


def _words(s: str) -> list[str]:
    return _W.findall(s.lower())


def copy_ratio(field: str, abstract: str, min_run: int = 3) -> float:
    """Share of a field's words that appear in the abstract as in-order runs of >= min_run words."""
    fw, aw = _words(field), _words(abstract)
    if not fw:
        return 0.0
    sm = difflib.SequenceMatcher(None, fw, aw, autojunk=False)
    copied = sum(b.size for b in sm.get_matching_blocks() if b.size >= min_run)
    return copied / len(fw)


def validate(obj: dict, abstract: str) -> tuple[bool, dict]:
    flags: dict = {}
    ok = isinstance(obj, dict) and all(k in obj for k in FIELDS)
    if not ok:
        return False, {"schema": "missing fields"}
    for k in FIELDS:
        v = obj[k]
        if k == "has_quantitative_result":
            ok &= isinstance(v, bool)
        elif k == "contribution_type":
            ok &= v in CONTRIB
        elif k in ("domain_data", "limitations"):
            ok &= v is None or isinstance(v, str)
        else:
            ok &= isinstance(v, str) and bool(v.strip())
    if not ok:
        flags["schema"] = "type mismatch"
    for k in FAITHFUL + GLOSSES:
        v = obj.get(k)
        if isinstance(v, str):
            n = len(v.split())
            if n > 60:
                flags.setdefault("too_long", []).append(k)
            if k in FAITHFUL and copy_ratio(v, abstract) > 0.8:
                flags.setdefault("copied", []).append(k)
    for k in TAGS:
        v = obj.get(k)
        if isinstance(v, str) and len(v.split()) > 4:
            flags.setdefault("tag_too_long", []).append(k)
    return ok, flags


async def run_extraction(con, run_id: str, p: ExtractionParams, limit: int | None = None) -> dict:
    """Extract facets for corpus papers lacking a row for (model, prompt_version)."""
    # papers marked title-only that have since gained an abstract (S2 / arXiv fill) are retried
    con.execute("""DELETE FROM facet_status WHERE status = 'title_only' AND model_id = ?
                   AND prompt_version = ? AND paper_id IN (
                       SELECT paper_id FROM papers WHERE length(abstract_raw) >= 100)""",
                [p.model, p.prompt_version])
    todo = con.execute("""
        SELECT p.paper_id, p.title, p.abstract_raw FROM papers p
        JOIN provenance v USING (paper_id)
        LEFT JOIN facet_status s ON s.paper_id = p.paper_id AND s.model_id = ?
             AND s.prompt_version = ? AND s.status IN ('ok', 'title_only')
        WHERE v.run_id = ? AND s.paper_id IS NULL
        ORDER BY v.route = 'seed' DESC, p.paper_id""", [p.model, p.prompt_version, run_id]).df()
    if limit:
        todo = todo.head(limit)
    now = datetime.now()
    no_abs = todo[todo.abstract_raw.isna() | (todo.abstract_raw.fillna("").str.len() < 100)]
    con.executemany("INSERT OR REPLACE INTO facet_status VALUES (?, ?, ?, 'title_only', NULL, NULL, 0, ?)",
                    [(pid, p.model, p.prompt_version, now) for pid in no_abs.paper_id])
    todo = todo.drop(no_abs.index)
    log.info("extracting %d abstracts (%d title-only skipped)", len(todo), len(no_abs))

    client = OpenRouter(p.model, p.quantizations, p.concurrency, p.llm_spend_cap_usd)
    stats = {"ok": 0, "failed": 0, "title_only": len(no_abs), "flagged": 0}
    pending_rows, pending_status = [], []

    def flush():
        if pending_rows:
            con.executemany("INSERT OR REPLACE INTO facets VALUES (?, ?, ?, ?, ?, ?)", pending_rows)
            pending_rows.clear()
        if pending_status:
            con.executemany("INSERT OR REPLACE INTO facet_status VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            pending_status)
            pending_status.clear()

    async def one(row):
        prompt = PROMPT_V1.replace("{title}", row.title or "").replace("{abstract}", row.abstract_raw)
        obj, cost, err = await client.chat_json(prompt, SCHEMA, "facets", max_tokens=1200)
        ts = datetime.now()
        if obj is not None:
            ok, flags = validate(obj, row.abstract_raw)
            if not ok:
                obj2, cost2, err2 = await client.chat_json(prompt, SCHEMA, "facets", max_tokens=1200)
                cost += cost2
                if obj2 is not None:
                    ok, flags = validate(obj2, row.abstract_raw)
                    obj = obj2
                err = err2 or err
        if obj is None or not ok:
            stats["failed"] += 1
            pending_status.append((row.paper_id, p.model, p.prompt_version, "failed",
                                   json.dumps(flags if obj else {}), err or "schema invalid", cost, ts))
            return
        stats["ok"] += 1
        stats["flagged"] += bool(flags)
        for k in FIELDS:
            v = obj[k]
            if isinstance(v, bool):
                v = "true" if v else "false"
            elif isinstance(v, str):
                v = v.strip()
                if k in TAGS:
                    v = v.lower()
            pending_rows.append((row.paper_id, k, v, p.model, p.prompt_version, ts))
        pending_status.append((row.paper_id, p.model, p.prompt_version, "ok", json.dumps(flags),
                               None, cost, ts))

    rows = list(todo.itertuples())
    bar = tqdm(total=len(rows), desc="facets")
    # All tasks start at once; the client's semaphore bounds concurrency, and results are
    # flushed every 200 completions so a slow request never holds up a whole batch.
    tasks = [asyncio.create_task(one(r)) for r in rows]
    try:
        for n, t in enumerate(asyncio.as_completed(tasks), 1):
            await t
            bar.update(1)
            if n % 200 == 0:
                flush()
                bar.set_postfix(spent=f"${client.spent:.3f}")
        flush()
    except LLMBudgetExceeded as e:
        log.warning("extraction stopped: %s", e)
        stats["stopped"] = str(e)
        for t in tasks:
            t.cancel()
        flush()
    finally:
        bar.close()
        await client.close()
    stats["spend_usd"] = client.spent
    stats["providers"] = client.providers_used
    return stats


def qa_report(con, run_id: str, model_id: str, prompt_version: str) -> dict:
    """Automatic QA summary: status counts, flag rates and limitations null rate per field."""
    st = con.execute("""
        SELECT s.status, count(*) n, sum(s.cost_usd) AS cost_usd FROM facet_status s
        JOIN provenance v USING (paper_id)
        WHERE v.run_id = ? AND s.model_id = ? AND s.prompt_version = ? GROUP BY 1""",
                     [run_id, model_id, prompt_version]).df()
    flags = con.execute("""
        SELECT s.qa_flags FROM facet_status s JOIN provenance v USING (paper_id)
        WHERE v.run_id = ? AND s.model_id = ? AND s.prompt_version = ? AND s.status = 'ok'""",
                        [run_id, model_id, prompt_version]).df()
    counts: dict = {}
    for f in flags.qa_flags:
        for k, v in (json.loads(f) if f else {}).items():
            for field in (v if isinstance(v, list) else [v]):
                counts[f"{k}:{field}"] = counts.get(f"{k}:{field}", 0) + 1
    null_rate = con.execute("""
        SELECT coalesce(json_extract_string(p.topics, '$[0].field'), 'unknown') AS field,
               count(*) AS n, avg(CASE WHEN f.text IS NULL THEN 1 ELSE 0 END) AS limitations_null_rate
        FROM facets f JOIN papers p USING (paper_id) JOIN provenance v USING (paper_id)
        WHERE v.run_id = ? AND f.facet = 'limitations' AND f.model_id = ? AND f.prompt_version = ?
        GROUP BY 1 ORDER BY n DESC""", [run_id, model_id, prompt_version]).df()
    return {"status": st.to_dict("records"), "n_ok": int(len(flags)),
            "flag_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "limitations_null_rate_by_field": null_rate.to_dict("records")}
