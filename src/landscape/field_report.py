"""Field intelligence report: one strong model reasons over every portfolio product of a project.

    uv run landscape field-report --run-id dro-pilot --model claude [--focus "..."]

The evidence pack is built from the same functions that serve the web app's views (Start here, Map & grid,
Facets, People, Research lab, Experiments), compacted to the rows that carry signal, and sent with the
evidence ledger (what each product's backtest supports). The model chooses the report's structure and length;
the prompt fixes only the rules of evidence and the citation format. Reports land in
projects/<project>/reports/<id>.json with the evidence pack beside them, so every claim can be audited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

from . import model_catalog
from .config import OPENROUTER_API_KEY, project_dir

log = logging.getLogger(__name__)
URL = "https://openrouter.ai/api/v1/chat/completions"

# Presets for the report; any OpenRouter slug works too (the app's model picker lists the whole catalogue).
# LIBERA_REPORT_MODEL_<KEY> overrides a preset's slug (e.g. when a provider renames a model).
MODELS = {
    "claude": {"label": "Claude Opus 5.5", "slug": "anthropic/claude-opus-5.5"},
    "gpt": {"label": "GPT-6.1 Sol", "slug": "openai/gpt-6.1-sol"},
}
MAX_OUTPUT_TOKENS = 64_000  # reasoning included


def model_slug(key: str) -> str:
    """A preset key ('claude', 'gpt') or an OpenRouter slug -> the slug."""
    if key in MODELS:
        return os.environ.get(f"LIBERA_REPORT_MODEL_{key.upper()}") or MODELS[key]["slug"]
    if not model_catalog.SLUG_RE.match(key):
        raise ValueError(f"not a preset ({', '.join(MODELS)}) or an OpenRouter slug: {key}")
    return key


def reports_dir(p: str, mkdir: bool = True) -> Path:
    return project_dir(p, "reports", mkdir=mkdir)


# ================================================================== evidence pack

def _r(x, nd: int = 3):
    """Round floats to a few significant digits, recursively; drop NaN."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(f"{x:.{nd}g}")
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items() if v is not None and v != [] and v != ""}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


def _line(ln: dict | None) -> str | None:
    """A category's line of work as one phrase: 'growing (68 recent vs 115 before, share x1.2)'."""
    if not ln:
        return None
    return f"{ln.get('trend')} ({ln.get('recent')} papers T-2..T vs {ln.get('before')} T-5..T-3, share ×{ln.get('ratio')})"


def _when(w: dict | None) -> dict | None:
    if not w:
        return None
    return {"year": w.get("year"), "timing": w.get("timing"), "last_cited_in_field": w.get("last_cited"),
            "line": _line(w.get("line"))}


def _ev(rows, n: int = 3) -> list[str]:
    """Evidence papers as 'W123 (2024) Title'."""
    return [f"{e.get('paper_id')} ({e.get('year')}) {e.get('title')}" for e in (rows or [])[:n]]


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # a product not built for this project is reported as missing, not fatal
        log.warning("evidence: %s unavailable (%s)", getattr(fn, "__name__", fn), str(e)[:200])
        return None


def evidence_pack(p: str) -> dict:
    """Every portfolio product of project `p`, compacted. Reads the web app's replica (or LANDSCAPE_DB)."""
    from .web import views_facets as vf
    from .web import views_lab as vl
    from .web import views_map as vm
    from .web import views_portfolio as vp
    from . import projects

    o = vp.overview(p)
    if not o.get("built"):
        raise RuntimeError(f"{p}: the portfolio is not built (run the Core map template first)")
    T = o["as_of"]
    meta = projects.read_meta(p)
    pack: dict = {
        "project": {"id": p, "name": meta.get("name") or p, "description": meta.get("description"),
                    "seed_dois": meta.get("seed_dois"), "as_of_year": T,
                    "note": f"T = {T}, the latest complete publication year. 'Recent' = {T - 2}–{T}, 'before' = "
                            f"{T - 5}–{T - 3}. Year {T + 1} is partial."},
        "map": {**o["metrics"], "papers_by_route": o["routes"]},
    }

    # --- the communities (problem facet, alpha 0.5) and every facet's categories
    pack["communities"] = [_r({"id": c["cluster_id"], "label": c["label"], "kind": c["kind"], "size": c["size"],
                               "recent_papers": c["recent"], "share_now": c["share_now"], "share_change": c["change"],
                               f"papers_per_year_{T - 9}_{T}": c["series"]}) for c in o["communities"]]
    cat = _safe(vf.catalogue, p)
    if cat and cat.get("built"):
        facets = {}
        for f in cat["facets"]:
            if f["facet"] == "problem":
                continue  # the communities above
            cats = sorted(f["categories"], key=lambda c: -(c.get("size") or 0))[:25]
            facets[f["facet"]] = {
                "alpha": f["alpha"], f"papers_per_year_{cat['y0']}_{cat['y1']}": f["totals"],
                "categories": [_r({"id": c["cluster_id"], "label": c["label"], "kind": c.get("kind"), "size": c.get("size"),
                                   "about": (c.get("description") or "")[:220], "terms": (c.get("top_terms") or [])[:6],
                                   "per_year": c.get("counts")}) for c in cats]}
        pack["facet_categories"] = {"years": f"{cat['y0']}–{cat['y1']} (last year partial)", "facets": facets}

    # --- where the field borrows from, what it imports now, what the map misses
    pack["import_map"] = o.get("import_map")
    ext = _safe(vm.external_imports, p)
    if ext and ext.get("built"):
        rows = sorted(ext["rows"], key=lambda r: -(r.get("score") or 0))
        keep = ["paper_id", "title", "year", "subfield", "recent_citers", "global_trend", "cluster_id", "cluster_label", "score"]
        pack["imports"] = {
            "outside_field": [_r({k: r.get(k) for k in keep}) for r in rows if r.get("outside_field")][:30],
            "adjacent": [_r({k: r.get(k) for k in keep}) for r in rows if not r.get("outside_field")][:20]}
    pack["coverage_gaps"] = _r(o.get("coverage_gaps"))

    # --- reading
    ppl = _safe(vm.people, p) or {}
    lasting = {r["paper_id"]: r.get("lasting") for r in o["read_now"]}
    pack["read_now"] = [_r({**r, "lasting": lasting.get(r["paper_id"])}) for r in ppl.get("early_work", [])[:25]] \
        or o["read_now"]
    if o.get("reading_paths"):
        by: dict = {}
        for r in o["reading_paths"]:
            lst = by.setdefault(r["label"], [])
            if len(lst) < 5:
                lst.append(_r({"paper_id": r["paper_id"], "title": r["title"], "year": r.get("year"), "role": r["role"],
                               "cited_by_community": r.get("cited_by_community"),
                               "builders_recent": [f"{b['paper_id']} ({b['year']})" for b in r.get("builders", [])[:2]]}))
        pack["reading_paths"] = by
    if o.get("tools"):
        by = {}
        for r in o["tools"]["rows"]:
            lst = by.setdefault(r["label"], [])
            if len(lst) < 4:
                lst.append(_r({"method": r["method_label"], "method_id": r["method_cluster"], "status": r["status"],
                               "papers_here_recent": r.get("papers_here_recent")}))
        pack["tools_to_learn"] = {"evidence": o["tools"].get("evidence"), "by_community": by}
    if o.get("survey_gaps"):
        pack["survey_gaps"] = [_r({k: r.get(k) for k in ("cluster_id", "label", "status", "recent_papers", "survey_share",
                                                         "field_survey_share", "latest_survey_title", "latest_survey_year")})
                               for r in o["survey_gaps"] if r.get("status") in ("survey gap", "thin")]
        pack["survey_gaps_evidence"] = next((r.get("evidence") for r in o["survey_gaps"] if r.get("evidence")), None)

    # --- entering the field
    if o.get("entry_routes"):
        pack["entry_routes"] = [_r({"community": r["label"], "strategy": r["strategy"], "entries": r["entries"],
                                    "stayed_rate": r["stayed_rate"], "taken_up_rate": r["taken_up_rate"],
                                    "recent_entries": r.get("recent_entries")})
                                for r in o["entry_routes"] if r["entries"] >= 5]
    if o.get("entry_effects"):
        pack["entry_effects"] = _r(o["entry_effects"])

    # --- opportunities, each with its timing
    op = o["opportunities"]
    opps: dict = {
        "hidden_gems": [_r({"paper_id": g["paper_id"], "title": g["title"], "for_community": g["community"],
                            "community_id": g["cluster_id"], "when": _when(g.get("when")),
                            "community_line": _line(g.get("to_line"))}) for g in op["gems"]],
        "limitation_to_innovation": [_r({"limitation_side": x["from"], "innovation_side": x["to"],
                                         "ids": [x["from_cluster"], x["to_cluster"]], "judge_1to5": x["judge"],
                                         "judge_rationale": x["why"], "timing": (x.get("when") or {}).get("timing"),
                                         "citation_flow": x.get("flow")}) for x in op["lim_inn"]],
        "method_transfer": [_r({"method": x["method"], "to_problem": x["problem"],
                                "ids": [x["method_cluster"], x["problem_cluster"]], "judge_1to5": x["judge"],
                                "method_line": _line(x.get("method_line")), "problem_line": _line(x.get("problem_line")),
                                "used_by_similar_problems_since": x.get("elsewhere_since"),
                                "similar_problems_recent_vs_before": [x.get("elsewhere_recent"), x.get("elsewhere_before")]})
                            for x in op["method_transfer"]],
        "underconnected_communities": [_r({"a": x["source_label"], "b": x["target_label"], "ids": [x["source"], x["target"]],
                                           "observed_citations": x["observed"], "expected": x["expected"],
                                           "flow": x.get("flow")}) for x in (op.get("underconnected") or [])],
    }
    pack["opportunities"] = opps
    br = _safe(vm.bridges, p)
    if br:
        opps["more_cluster_pairs"] = {
            t: [_r({k: r.get(k) for k in ("rank", "candidate", "judge", "timing", "trend", "flow")}) for r in rows[10:30]]
            for t, rows in br["lists"].items() if len(rows) > 10}
    pb = _safe(vm.paper_bridges, p)
    if pb and pb.get("lists"):
        lab = {c["cluster_id"]: c["label"] for c in pb["clusters"]}
        pack["papers_to_bring_in"] = {
            lab.get(cid, cid): {
                "community_id": cid,
                "likely": [_r({"paper_id": r["paper_id"], "title": r["title"], "year": r.get("year"), "cites": r.get("cites"),
                               "timing": r.get("timing")}) for r in L["bridge_paper"][:4]],
                "hidden_gems": [_r({"paper_id": r["paper_id"], "title": r["title"], "year": r.get("year"),
                                    "timing": r.get("timing")}) for r in L["bridge_paper_gem"][:3]]}
            for cid, L in list(pb["lists"].items())[:25]}

    # --- diffusion under way
    waves = sorted(o.get("second_waves") or [], key=lambda w: -(w.get("momentum") or 0))
    pack["second_waves"] = {
        kind: [_r({"idea": w["idea"], "idea_id": w.get("idea_id"), "origin": w.get("origin"), "year": w.get("year"),
                   "citing_recent_vs_earlier": [w["recent"], w["earlier"]], "new_audiences": w["new_audiences"],
                   "quiet_uptake": w.get("quiet_uptake")[:4], "likely_next": w["likely_next"][:3],
                   "evidence": _ev(w.get("evidence"), 2)}) for w in waves if w["kind"] == kind][:15]
        for kind in ("paper", "external", "method")}

    # --- people (descriptive only)
    pack["people_descriptive_only"] = {k: [f"{r['name']} ({r['score']})" for r in ppl.get(k, [])[:10]]
                                       for k in ("seeder_author", "broker_author", "institution_seeder", "institution_broker")}

    # --- sampling audit and direction robustness (Research lab)
    lab_ = _safe(vl.lab, p)
    if lab_ and lab_.get("built"):
        pack["sampling_audit"] = _r(lab_["audit"])
        pack["direction_robustness"] = {
            "how_to_read": "raw_change = share after − before; within_route_change removes the change in how papers "
                           "were sampled (expansion routes). sign_flip = the raw direction reverses once sampling is "
                           "separated out: do not call those topics growing or shrinking.",
            "topics": [_r({k: r.get(k) for k in ("cluster_id", "label", "kind", "before", "after", "raw_change",
                                                 "within_route_change", "sign_flip", "before_n", "after_n")})
                       for r in lab_["trends"]]}
        pack["underconnected_all"] = [_r({k: r.get(k) for k in ("source_label", "target_label", "observed", "expected",
                                                                "semantic_similarity", "score")})
                                      for r in sorted(lab_["opportunities"], key=lambda r: -(r.get("score") or 0))[:20]]

    # --- what the evidence supports (the ledger the model must respect)
    ex = _safe(vl.experiments_index, p)
    if ex and ex.get("portfolio"):
        pack["evidence_ledger"] = ex["portfolio"]
    return pack


# ================================================================== prompt

SYSTEM = """You are a senior research-intelligence analyst. You write field intelligence reports for working researchers: a PhD student choosing a thesis direction, a postdoc deciding what to work on next, a PI allocating a group's next two years, an industry lead scouting a field. Your reader is smart and busy. They want what they cannot get by reading twenty abstracts: the shape of the field, where it is actually moving, where the openings are, and what to do about it — each claim tied to evidence they can open.

You are given an evidence pack: every product of a research-landscape engine built on one field's citation graph (communities, facet categories, trends, imports, reading paths, opportunities, diffusion watchlists, entry routes, sampling audits) and an evidence ledger saying how far each product's backtest supports it. You have no other source. Do not bring in outside facts about papers, people or trends; your general knowledge of science may inform interpretation (what a method is, why two problems are related) but never the evidence.

# How to think before writing

Work through this privately; the report shows only conclusions and their grounds.

1. Understand the field as the map sees it: what the seed is, what the communities are, which are hubs, how big and how recent. Note what the map structurally cannot see (sampling route, coverage gaps, share of references outside the map).
2. Ask which questions matter most to researchers in *this* field right now. Typical ones:
   - What is this field, really, and how is it organized?
   - Where is the energy now, and is that real or a sampling artefact?
   - What must I read first, and what is everyone about to read?
   - Which ideas are moving between communities, and which are arriving from outside?
   - Where are concrete, under-occupied openings — and how fresh, how contested, how plausible are they?
   - How do newcomers get in successfully here?
   - Where are surveys or tools missing?
   - What is this map blind to?
   Rank them for this case. Some will not matter or will have no usable evidence: drop them.
3. Cross-reference products. The best findings come from convergence: an opportunity whose method line is growing, whose target community is underconnected, and whose ingredients are a second wave already under way is worth far more than any one list's top row. Equally, flag divergence (a "growing" topic whose direction flips under the sampling audit).
4. For every candidate claim, check the ledger. A "Use" product can support a recommendation. "Use with care" supports a lead to check. "Describe" products describe the map; they never predict. "Dropped" or "Not demonstrated" products are not evidence. Topic growth and decline are sampling-sensitive: only state a direction when within_route_change agrees, and say so.
5. Keep only what changes a decision. Cut anything a researcher in the field already knows unless the evidence contradicts or sharpens it.

# How to write

- There is no fixed template. Choose the structure, the sections and the length that this field's evidence earns. A rich, moving field may warrant a long report with many findings; a thin or static map may warrant a short one that says so. Never pad, never fill a section to complete a pattern, never list a product because it exists. Silence on a topic is fine; say "nothing notable" only if a reader would otherwise expect something there.
- Open with the bottom line: a title (one `#` heading that states the main finding, not "Field report"), then 3–6 sentences a reader could stop after.
- Every substantive claim carries its evidence inline: the numbers and the papers or categories behind it. Prefer specific ("18 papers in 2023–25 vs 7 in 2020–22") over adjectives ("rapidly growing").
- Mark confidence in plain words tied to the ledger: *backtested*, *a lead to check*, *descriptive*. Use these words consistently.
- Recommendations are concrete actions with a reason and a first step: what to read, what to try, whom to watch, what to check before committing. Where it helps, say who the action suits (newcomer, established group).
- End with what the map cannot tell the reader and what would change the conclusions (missing coverage, partial years, unvalidated lists).
- Markdown only: headings, short paragraphs, bullet lists, and tables where comparison helps. No HTML, no code blocks, no emojis.

# Links (required)

Make papers and categories clickable so the reader can open the evidence:
- a paper or outside work by its id: [Short title](paper:W1234567890)
- a community or facet category by its id: [Label](cat:problem:0.5:medium:003)
Use only ids that appear in the evidence pack; never invent one. If an item has no id, name it without a link."""


def build_prompt(pack: dict, focus: str | None) -> str:
    focus_txt = (f"The reader asked for this focus; let it shape what you prioritize, but still report anything "
                 f"in the evidence that they would be wrong to miss:\n\n> {focus.strip()}\n\n" if focus and focus.strip()
                 else "No particular focus was given: write for a capable researcher deciding what to do in this field.\n\n")
    return (f"Write the field intelligence report for **{pack['project']['name']}**.\n\n{focus_txt}"
            "The evidence pack follows as JSON. Read all of it before deciding what matters.\n\n"
            "<evidence_pack>\n" + json.dumps(pack, ensure_ascii=False, separators=(",", ":"), default=str) +
            "\n</evidence_pack>")


# ================================================================== the call

def _stream(slug: str, system: str, prompt: str) -> tuple[str, dict, float]:
    """Streamed chat completion with high reasoning effort; prints progress lines the job queue reads."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter key missing (OPENAI_API_KEY / OPENROUTER_API_KEY in .env.local)")
    body = {"model": slug, "stream": True, "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "usage": {"include": True}}
    rs = model_catalog.reasoning_for(slug, "high")
    if rs:
        body["reasoning"] = rs
    if slug.startswith("openai/"):
        # sent once: explicit mode with no breakpoint skips OpenAI's automatic cache write (1.25x on GPT-5.6+)
        body["prompt_cache_options"] = {"mode": "explicit"}
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "X-Title": "libera-expedition"}
    text, reasoning_chars, usage, last, finish = [], 0, {}, 0.0, None
    t0 = time.time()
    for attempt in range(4):
        try:
            with httpx.stream("POST", URL, json=body, headers=headers,
                              timeout=httpx.Timeout(600.0, connect=20.0)) as r:
                if r.status_code >= 400:
                    msg = r.read().decode("utf-8", "replace")[:500]
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                        raise httpx.HTTPStatusError(msg, request=r.request, response=r)
                    raise RuntimeError(f"OpenRouter {r.status_code}: {msg}")
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue  # keep-alive comments
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    if "error" in ev:
                        raise RuntimeError(f"OpenRouter: {str(ev['error'])[:400]}")
                    usage = ev.get("usage") or usage
                    for ch in ev.get("choices") or []:
                        finish = ch.get("finish_reason") or finish
                        d = ch.get("delta") or {}
                        reasoning_chars += len(d.get("reasoning") or "")
                        if d.get("content"):
                            text.append(d["content"])
                    if time.time() - last > 3:
                        last = time.time()
                        n = sum(map(len, text))
                        phase = "writing" if n else "reasoning"
                        # a fraction for the queue's progress bar: reasoning fills the first half, the report the rest
                        frac = min(0.5, reasoning_chars / 60_000) if not n else 0.5 + min(0.49, n / 40_000)
                        print(f"\r{phase}: {reasoning_chars:,} reasoning chars, {n:,} report chars, "
                              f"{time.time() - t0:.0f}s  {round(frac * 100)}%|", end="", flush=True)
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if text or attempt == 3:
                raise RuntimeError(f"stream failed: {e}") from e
            log.warning("retrying after %s", str(e)[:200])
            time.sleep(5 * 2 ** attempt)
    print()
    if finish == "length":
        log.warning("the report hit the output limit (%d tokens) and may end mid-section", MAX_OUTPUT_TOKENS)
    return "".join(text).strip(), usage, time.time() - t0


def generate(p: str, model: str = "claude", focus: str | None = None) -> Path:
    slug = model_slug(model)
    print("Building the evidence pack 5%|", flush=True)
    pack = evidence_pack(p)
    prompt = build_prompt(pack, focus)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = re.sub(r"[^a-z0-9]+", "-", (model if model in MODELS else slug.split("/")[-1]).lower()).strip("-")[:40]
    rid = f"{stamp}-{short}"
    out = reports_dir(p)
    pack_json = json.dumps(pack, ensure_ascii=False, indent=1, default=str)
    print(f"Evidence pack: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens). Asking {slug} 10%|", flush=True)
    text, usage, secs = _stream(slug, SYSTEM, prompt)
    if not text:
        raise RuntimeError("the model returned an empty report")
    title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), "Field report")
    rec = {"id": rid, "project": p, "title": title, "created_at": datetime.now().isoformat(timespec="seconds"),
           "model": model, "model_label": MODELS[model]["label"] if model in MODELS else model_catalog.label(slug),
           "model_slug": slug, "focus": (focus or "").strip() or None,
           "as_of": pack["project"]["as_of_year"], "markdown": text, "seconds": round(secs),
           "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage},
           "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "cost_usd": usage.get("cost"), "evidence_chars": len(pack_json),
           "evidence_sha256": hashlib.sha256(pack_json.encode()).hexdigest()[:16],
           "replica": Path(os.environ.get("LANDSCAPE_DB", "")).name or None}
    (out / f"{rid}.evidence.json").write_text(pack_json)  # only beside a finished report
    path = out / f"{rid}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"Report: {title} -> {path} ({secs:.0f}s, ${rec['cost_usd'] or 0:.2f}) 100%|", flush=True)
    return path


def list_reports(p: str) -> list[dict]:
    d = reports_dir(p, mkdir=False)
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.json"), reverse=True):
        if f.name.startswith("._") or f.name.count(".") > 1:  # <id>.evidence.json, <id>.chats.json
            continue
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({k: r.get(k) for k in ("id", "title", "created_at", "model", "model_label", "focus", "as_of",
                                         "cost_usd", "seconds")})
    return out


RID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9-]{1,40}$")


def report_path(p: str, rid: str, suffix: str = "") -> Path | None:
    if not RID_RE.match(rid):
        return None
    return reports_dir(p, mkdir=False) / f"{rid}{suffix}.json"


def read_report(p: str, rid: str) -> dict | None:
    f = report_path(p, rid)
    return json.loads(f.read_text()) if f and f.exists() else None


def edit_report(p: str, rid: str, find: str, replace: str) -> dict:
    """Replace one exact passage of a report's text (an edit proposed in the chat and applied by the reader).
    The model's original text is kept the first time; every edit is logged, so it can be undone."""
    rec = read_report(p, rid)
    if rec is None:
        raise KeyError(rid)
    md = rec["markdown"]
    n = md.count(find) if find else 0
    if n != 1:
        raise ValueError("The passage to replace is not in the report any more" if n == 0 else
                         f"The passage to replace occurs {n} times; it must be unique")
    rec.setdefault("markdown_original", md)
    rec["markdown"] = md.replace(find, replace, 1)
    rec.setdefault("edits", []).append({"at": datetime.now().isoformat(timespec="seconds"), "find": find,
                                        "replace": replace})
    if rec["markdown"] == rec["markdown_original"]:  # undone back to the model's text: nothing to flag
        rec.pop("markdown_original")
        rec.pop("edits")
    title = next((ln.lstrip("# ").strip() for ln in rec["markdown"].splitlines() if ln.startswith("# ")), rec["title"])
    rec["title"] = title
    report_path(p, rid).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec
