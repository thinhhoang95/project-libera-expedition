"""Command line entry point: one sub-command per phase, all keyed by --run-id.

    landscape new-run --run-id dro-pilot --seed-doi 10.1007/s10107-017-1172-1 --paper-budget 8000
    landscape expand --run-id dro-pilot          # Phase 1-2, routes seed/back/forward/author
    landscape semantic --run-id dro-pilot        # Phase 2 route 5 (embeds abstracts first)
    landscape extract --run-id dro-pilot         # Phase 3
    landscape embed --run-id dro-pilot           # Phase 4 (+ tag normalization)
    landscape core --run-id dro-pilot            # Phase 5
    landscape status --run-id dro-pilot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime

from . import store
from .config import RunParams

log = logging.getLogger("landscape")


def _params(con, run_id: str) -> RunParams:
    row = con.execute("SELECT params FROM runs WHERE run_id = ?", [run_id]).fetchone()
    if not row:
        sys.exit(f"unknown run {run_id}; create it with new-run")
    return RunParams.from_dict(json.loads(row[0]))


def _spend(con, run_id: str) -> dict:
    row = con.execute("SELECT spend_usd FROM runs WHERE run_id = ?", [run_id]).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


def _add_spend(con, run_id: str, key: str, usd: float) -> None:
    s = _spend(con, run_id)
    s[key] = round(s.get(key, 0.0) + usd, 6)
    con.execute("UPDATE runs SET spend_usd = ? WHERE run_id = ?", [json.dumps(s), run_id])


def _set_model_version(con, run_id: str, key: str, value: str) -> None:
    row = con.execute("SELECT model_versions FROM runs WHERE run_id = ?", [run_id]).fetchone()
    mv = json.loads(row[0]) if row and row[0] else {}
    mv[key] = value
    con.execute("UPDATE runs SET model_versions = ? WHERE run_id = ?", [json.dumps(mv), run_id])


def _openalex_spent(con, run_id: str) -> float:
    return con.execute("SELECT coalesce(sum(cost_usd), 0) FROM api_calls WHERE run_id = ? AND source = 'openalex'",
                       [run_id]).fetchone()[0]


def cmd_new_run(a, con):
    p = RunParams()
    if a.paper_budget:
        p.expansion.paper_budget = a.paper_budget
    if a.params:
        p = RunParams.from_dict(_deep_merge(p.to_dict(), json.loads(open(a.params).read())))
    if con.execute("SELECT count(*) FROM runs WHERE run_id = ?", [a.run_id]).fetchone()[0]:
        print(f"run {a.run_id} exists: replacing its parameters (spend log and data are kept)")
    con.execute("DELETE FROM runs WHERE run_id = ?", [a.run_id])
    con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        a.run_id, [], json.dumps(p.to_dict()), a.cutoff_year, json.dumps({}), json.dumps({}),
        datetime.now(), json.dumps({"seed_dois": a.seed_doi or [], "seed_ids": a.seed_id or [],
                                    "ai_report": a.ai_report})])
    print(f"created run {a.run_id}")


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        base[k] = _deep_merge(base.get(k, {}), v) if isinstance(v, dict) and k != "route_caps" else v
    return base


def cmd_expand(a, con):
    from .clients import Budget
    from .expand import expand_citation_routes

    p = _params(con, a.run_id)
    if a.reset:
        con.execute("DELETE FROM provenance WHERE run_id = ?", [a.run_id])
    notes = json.loads(con.execute("SELECT notes FROM runs WHERE run_id = ?", [a.run_id]).fetchone()[0])
    budget = Budget(p.expansion.metadata_spend_cap_usd, _openalex_spent(con, a.run_id))
    stats = asyncio.run(expand_citation_routes(
        con, a.run_id, p.expansion, notes["seed_dois"], notes["seed_ids"], notes["ai_report"],
        budget=budget))
    stats["openalex_spend_usd"] = _openalex_spent(con, a.run_id)
    _set_spend_abs(con, a.run_id, "openalex", stats["openalex_spend_usd"])
    print(json.dumps(stats, indent=2, default=str))


def _set_spend_abs(con, run_id, key, usd):
    s = _spend(con, run_id)
    s[key] = round(usd, 6)
    con.execute("UPDATE runs SET spend_usd = ? WHERE run_id = ?", [json.dumps(s), run_id])


def cmd_preprints(a, con):
    from .clients import Budget
    from .preprints import fill_from_preprints

    p = _params(con, a.run_id)
    budget = Budget(p.expansion.metadata_spend_cap_usd, _openalex_spent(con, a.run_id))
    stats = asyncio.run(fill_from_preprints(con, a.run_id, budget, p.expansion.freshness_days))
    budget.flush(con, a.run_id)
    _set_spend_abs(con, a.run_id, "openalex", _openalex_spent(con, a.run_id))
    print(json.dumps(stats, indent=2))


def cmd_semantic(a, con):
    from .clients import Budget
    from .embed import Embedder, embed_abstracts
    from .expand import expand_semantic

    p = _params(con, a.run_id)
    emb = Embedder(p.embedding)
    _set_model_version(con, a.run_id, "embedding", emb.version)
    ids = con.execute("SELECT paper_id FROM provenance WHERE run_id = ?", [a.run_id]).df().paper_id
    embed_abstracts(con, emb, ids.tolist())
    budget = Budget(p.expansion.metadata_spend_cap_usd, _openalex_spent(con, a.run_id))
    stats = asyncio.run(expand_semantic(con, a.run_id, p.expansion, budget, emb))
    _set_spend_abs(con, a.run_id, "openalex", _openalex_spent(con, a.run_id))
    print(json.dumps(stats, indent=2, default=str))


def cmd_extract(a, con):
    from .extract import qa_report, run_extraction

    p = _params(con, a.run_id)
    _set_model_version(con, a.run_id, "extraction", f"{p.extraction.model}|{p.extraction.prompt_version}")
    stats = asyncio.run(run_extraction(con, a.run_id, p.extraction, limit=a.limit))
    _add_spend(con, a.run_id, "llm_extraction", stats.get("spend_usd", 0.0))
    print(json.dumps(stats, indent=2, default=str))
    print(json.dumps(qa_report(con, a.run_id, p.extraction.model, p.extraction.prompt_version),
                     indent=2, default=str))


def cmd_qa(a, con):
    from .extract import qa_report

    p = _params(con, a.run_id)
    print(json.dumps(qa_report(con, a.run_id, p.extraction.model, p.extraction.prompt_version),
                     indent=2, default=str))


def cmd_embed(a, con):
    from .embed import Embedder, embed_facets, normalize_tags

    p = _params(con, a.run_id)
    emb = Embedder(p.embedding)
    _set_model_version(con, a.run_id, "embedding", emb.version)
    out = embed_facets(con, a.run_id, emb, p.extraction.model, p.extraction.prompt_version)
    out["tags"] = normalize_tags(con, a.run_id, emb, p.extraction.model, p.extraction.prompt_version)
    print(json.dumps(out, indent=2))


def cmd_core(a, con):
    from .core import build_core

    p = _params(con, a.run_id)
    _set_model_version(con, a.run_id, "labels", p.labels.model)
    stats = build_core(con, a.run_id, p, skip_labels=a.skip_labels)
    _add_spend(con, a.run_id, "llm_labels", stats.get("label_spend_usd", 0.0))
    print(json.dumps(stats, indent=2, default=str))


def cmd_signals(a, con):
    from .signals import build_signals

    stats = build_signals(con, a.run_id, _params(con, a.run_id), skip_judge=a.skip_judge)
    _add_spend(con, a.run_id, "llm_judge", stats.get("judge_spend_usd", 0.0))
    print(json.dumps(stats, indent=2, default=str))


def cmd_backtest(a, con):
    from .backtest import run_backtest, summarize

    cutoffs = tuple(int(x) for x in a.cutoffs.split(","))
    df = run_backtest(con, a.run_id, _params(con, a.run_id), cutoffs=cutoffs, rebuild=a.rebuild)
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 45):
        print(summarize(df).to_string(index=False))


def cmd_viz(a, con):
    from .viz import build_viz

    print(json.dumps(build_viz(con, a.run_id, _params(con, a.run_id)), indent=2))


def cmd_status(a, con):
    q = lambda sql: con.execute(sql, [a.run_id]).df().to_string(index=False)  # noqa: E731
    print(q("SELECT run_id, seed_ids, spend_usd, model_versions FROM runs WHERE run_id = ?"))
    print(q("""SELECT route, hop, count(*) n FROM provenance WHERE run_id = ? GROUP BY ALL ORDER BY ALL"""))
    print(q("""SELECT count(*) papers, avg(CASE WHEN abstract_raw IS NULL THEN 1 ELSE 0 END) missing_abstract,
                      sum(CASE WHEN abstract_source = 's2' THEN 1 ELSE 0 END) s2_filled
               FROM papers JOIN provenance USING (paper_id) WHERE run_id = ?"""))
    print(q("""SELECT facet, resolution, alpha, count(*) clusters, sum(CASE WHEN is_stable THEN 1 ELSE 0 END) "stable",
                      sum(CASE WHEN kind = 'hub' THEN 1 ELSE 0 END) hubs
               FROM clusters WHERE run_id = ? GROUP BY ALL ORDER BY ALL"""))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="landscape")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("new-run")
    s.add_argument("--run-id", required=True)
    s.add_argument("--seed-doi", action="append")
    s.add_argument("--seed-id", action="append", help="OpenAlex W id")
    s.add_argument("--ai-report", help="deep research report file (DOIs are extracted)")
    s.add_argument("--paper-budget", type=int)
    s.add_argument("--cutoff-year", type=int)
    s.add_argument("--params", help="JSON file overriding RunParams fields")
    s.set_defaults(fn=cmd_new_run)

    for name, fn in [("expand", cmd_expand), ("preprints", cmd_preprints), ("semantic", cmd_semantic), ("extract", cmd_extract),
                     ("qa", cmd_qa), ("embed", cmd_embed), ("core", cmd_core), ("viz", cmd_viz), ("signals", cmd_signals), ("backtest", cmd_backtest), ("status", cmd_status)]:
        s = sub.add_parser(name)
        s.add_argument("--run-id", required=True)
        if name == "extract":
            s.add_argument("--limit", type=int)
        if name == "expand":
            s.add_argument("--reset", action="store_true",
                           help="drop this run's provenance and refill all routes (responses are cached)")
        if name == "backtest":
            s.add_argument("--cutoffs", default="2020,2021,2022,2023")
            s.add_argument("--rebuild", action="store_true", help="re-fit the cutoff runs")
        if name == "signals":
            s.add_argument("--skip-judge", action="store_true")
        if name == "core":
            s.add_argument("--skip-labels", action="store_true")
        s.set_defaults(fn=fn)

    s = sub.add_parser("explorer", help="open the Streamlit explorer (close it before running pipeline steps)")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(fn=None)

    a = ap.parse_args(argv)
    if a.cmd == "explorer":
        import subprocess
        from pathlib import Path

        app = Path(__file__).with_name("app.py")
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                                  "--server.port", str(a.port), "--server.headless", "true"]))
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    con = store.connect()
    try:
        a.fn(a, con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
