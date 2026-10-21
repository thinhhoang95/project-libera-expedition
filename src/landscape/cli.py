"""Command line entry point: one sub-command per phase, all keyed by --run-id.

    landscape new-run --run-id dro-pilot --seed-doi 10.1007/s10107-017-1172-1 --paper-budget 8000
    landscape expand --run-id dro-pilot          # Phase 1-2, routes seed/back/forward/author
    landscape semantic --run-id dro-pilot        # Phase 2 route 5 (embeds abstracts first)
    landscape extract --run-id dro-pilot         # Phase 3
    landscape embed --run-id dro-pilot           # Phase 4 (+ tag normalization)
    landscape core --run-id dro-pilot            # Phase 5
    landscape status --run-id dro-pilot

    landscape serve                              # the Libera Expedition web app and its job worker
    landscape queue --run-id dro-pilot --template full   # queue a whole pipeline (resumable)
    landscape jobs                               # the queue; landscape worker runs it without the web app
    landscape migrate-layout                     # move a flat data/ folder into shared/ + projects/
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
    from .projects import write_meta

    write_meta(a.run_id, name=a.name, description=a.description, seed_dois=a.seed_doi or [],
               seed_ids=a.seed_id or [])
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
    from .embed import Embedder, canonicalize_tags_llm, embed_facets, normalize_tags

    p = _params(con, a.run_id)
    emb = Embedder(p.embedding)
    _set_model_version(con, a.run_id, "embedding", emb.version)
    out = embed_facets(con, a.run_id, emb, p.extraction.model, p.extraction.prompt_version)
    out["tags"] = normalize_tags(con, a.run_id, emb, p.extraction.model, p.extraction.prompt_version)
    if p.labels.canonicalize_tags:
        out["tags_llm"] = canonicalize_tags_llm(con, a.run_id, emb, p.labels)
        _add_spend(con, a.run_id, "llm_tags", out["tags_llm"].get("spend_usd", 0.0))
    print(json.dumps(out, indent=2))


def cmd_core(a, con):
    from .core import build_core

    p = _params(con, a.run_id)
    _set_model_version(con, a.run_id, "labels", p.labels.model)
    if p.clusters.consolidate:
        _set_model_version(con, a.run_id, "consolidation", p.labels.consolidate_model)
    stats = build_core(con, a.run_id, p, skip_labels=a.skip_labels)
    _add_spend(con, a.run_id, "llm_labels", stats.get("label_spend_usd", 0.0))
    _add_spend(con, a.run_id, "llm_consolidation", stats.get("consolidation_spend_usd", 0.0))
    print(json.dumps(stats, indent=2, default=str))


def cmd_signals(a, con):
    from .signals import build_signals, refresh_paper_bridges

    if a.paper_bridges_only:
        stats = refresh_paper_bridges(con, a.run_id, _params(con, a.run_id))
    else:
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


def cmd_research_lab(a, con):
    from .research_lab import run_lab

    path = run_lab(con, a.run_id, _params(con, a.run_id), as_of=a.as_of,
                   cutoffs=tuple(int(t) for t in a.cutoffs.split(",")))
    print(f"Review artifacts: {path}")


def cmd_external(a, con):
    from .external import run_external

    stats = run_external(con, a.run_id, _params(con, a.run_id), stages=tuple(a.stages), min_cites=a.min_cites,
                         llm_cap_usd=a.llm_cap)
    _add_spend(con, a.run_id, "openalex_external", stats.get("fetch", {}).get("spend_usd", 0.0))
    _add_spend(con, a.run_id, "llm_external_facets", stats.get("extract", {}).get("spend_usd", 0.0))
    print(json.dumps(stats, indent=2, default=str))


def cmd_portfolio(a, con):
    from .orientation import build_orientation
    from .portfolio import build_portfolio

    stats = build_portfolio(con, a.run_id, _params(con, a.run_id))
    # orientation opens the database read-only; DuckDB refuses that while this process holds it read-write
    con.close()
    stats["orientation"] = build_orientation(a.run_id, db_path=str(store.DB_PATH))
    print(json.dumps(stats, indent=2, default=str))


def cmd_my_paper(a, con):
    from .portfolio import analyse_doi

    path = analyse_doi(con, a.run_id, _params(con, a.run_id), a.doi)
    rec = json.loads(path.read_text())
    if not a.read_only:  # on the web app's replica the cost stays in the paper's record only
        _add_spend(con, a.run_id, "llm_my_paper", rec.get("llm_cost_usd", 0.0))
    warm = sorted((w for w in rec["warmth"] if w.get("rank") == w.get("rank") and w.get("rank")), key=lambda w: w["rank"])
    print(f"{rec['title']} ({rec['year']}) -> {path}")
    labels = dict(con.execute("SELECT cluster_id, label FROM clusters WHERE run_id = ?", [a.run_id]).fetchall())
    for w in warm[:5]:
        print(f"  {int(w['rank'])}. {labels.get(w['cluster_id'], w['cluster_id'])}  (warmth {w['warmth']:.2f})")
    print("Open it in the explorer: My paper.")


def cmd_score_snapshot(a, con):
    from .snapshot import score_snapshot

    print(json.dumps(score_snapshot(con, a.run_id, _params(con, a.run_id), a.date, a.months), indent=2, default=str))


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
    s.add_argument("--name", help="display name of the project")
    s.add_argument("--description")
    s.set_defaults(fn=cmd_new_run)

    s = sub.add_parser("serve", help="the Libera Expedition web app (with the job worker)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8600)
    s.add_argument("--no-worker", action="store_true", help="do not run queued jobs in this process")
    s.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    s.set_defaults(fn=None)

    s = sub.add_parser("worker", help="run queued jobs without the web app")
    s.set_defaults(fn=None)

    s = sub.add_parser("queue", help="queue pipeline steps for a project")
    s.add_argument("--run-id", required=True)
    from .jobs import TEMPLATES

    s.add_argument("--template", choices=list(TEMPLATES), default="map")
    s.add_argument("--steps", help="comma-separated step names instead of a template")
    s.set_defaults(fn=None)

    s = sub.add_parser("jobs", help="show the job queue")
    s.add_argument("--run-id")
    s.set_defaults(fn=None)

    s = sub.add_parser("migrate-layout", help="move data/ from the flat layout into shared/ + projects/")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=None)

    for name, fn in [("expand", cmd_expand), ("preprints", cmd_preprints), ("semantic", cmd_semantic), ("extract", cmd_extract),
                     ("qa", cmd_qa), ("embed", cmd_embed), ("core", cmd_core), ("viz", cmd_viz), ("signals", cmd_signals), ("backtest", cmd_backtest), ("research-lab", cmd_research_lab), ("external", cmd_external),
                     ("portfolio", cmd_portfolio), ("my-paper", cmd_my_paper), ("score-snapshot", cmd_score_snapshot),
                     ("status", cmd_status)]:
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
        if name == "research-lab":
            s.add_argument("--cutoffs", default="2019,2020,2021,2022,2023")
            s.add_argument("--as-of", help="local calendar date YYYY-MM-DD; latest full year is used for live views")
        if name == "signals":
            s.add_argument("--skip-judge", action="store_true")
            s.add_argument("--paper-bridges-only", action="store_true",
                           help="refresh the two paper-bridge lists without recomputing other signals or calling the judge")
        if name == "core":
            s.add_argument("--skip-labels", action="store_true")
        if name == "external":
            s.add_argument("--stages", nargs="+", default=["fetch", "extract", "embed"],
                           choices=["fetch", "extract", "embed"])
            s.add_argument("--min-cites", type=int, default=5, help="fetch external works cited by at least this many corpus papers")
            s.add_argument("--llm-cap", type=float, default=2.0, help="USD cap for external facet extraction")
        if name == "my-paper":
            s.add_argument("--doi", required=True)
            s.add_argument("--read-only", action="store_true", help="open the database read-only (spend is not logged)")
        if name == "score-snapshot":
            s.add_argument("--date", required=True, help="snapshot date YYYY-MM-DD (docs/snapshots/<run>-<date>.json)")
            s.add_argument("--months", type=int, default=12, help="horizon after the snapshot date")
        s.set_defaults(fn=fn)

    s = sub.add_parser("field-report", help="a field intelligence report: a strong model reasons over the portfolio")
    s.add_argument("--run-id", required=True)
    s.add_argument("--model", default="claude", help="claude, gpt, or any OpenRouter slug (e.g. google/gemini-3.5-flash)")
    s.add_argument("--focus", help="what the reader wants the report to prioritize")
    s.set_defaults(fn=None)

    s = sub.add_parser("explorer", help="open the Streamlit explorer (close it before running pipeline steps)")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(fn=None)

    a = ap.parse_args(argv)
    if a.cmd in ("serve", "worker", "queue", "jobs", "migrate-layout"):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        return _app_command(a)
    if a.cmd == "field-report":  # reads the web app's replica through its view functions; no database writes
        from .field_report import generate

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        generate(a.run_id, a.model, a.focus)
        return
    if a.cmd == "explorer":
        import subprocess
        from pathlib import Path

        app = Path(__file__).with_name("app.py")
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                                  "--server.port", str(a.port), "--server.headless", "true"]))
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _check_layout()
    con = store.connect(read_only=a.cmd == "research-lab" or getattr(a, "read_only", False))
    try:
        a.fn(a, con)
    finally:
        con.close()


def _check_layout() -> None:
    from .projects import needs_migration

    if needs_migration():
        sys.exit("data/ uses the old flat layout: run `landscape migrate-layout` once (close the explorer first).")


def _app_command(a) -> None:
    from . import jobs

    if a.cmd == "migrate-layout":
        from .projects import migrate_layout

        for s, t in migrate_layout(dry_run=a.dry_run):
            print(f"{s}  ->  {t}")
        print("dry run: nothing moved" if a.dry_run else "done; the moves are listed in data/migration.json")
    elif a.cmd == "serve":
        _check_layout()
        import uvicorn

        from .web.server import create_app

        if a.reload:
            import os

            os.environ["LIBERA_NO_WORKER"] = "1" if a.no_worker else ""
            uvicorn.run("landscape.web.server:create_app", factory=True, host=a.host, port=a.port, reload=True)
        else:
            uvicorn.run(create_app(worker=not a.no_worker), host=a.host, port=a.port)
    elif a.cmd == "worker":
        _check_layout()
        import time

        from .web.server import make_worker

        w = make_worker()
        lanes = w.start()
        if not lanes:
            sys.exit("every lane is already run by another worker (the web app?)")
        print(f"running lanes: {', '.join(lanes)} (Ctrl-C interrupts; interrupted steps resume on the next start)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            w.shutdown()
    elif a.cmd == "queue":
        con = jobs.connect()
        names = a.steps.split(",") if a.steps else jobs.TEMPLATES[a.template]
        unknown = [n for n in names if n not in jobs.STEPS]
        if unknown:
            sys.exit(f"unknown steps {unknown}; known: {', '.join(jobs.STEPS)}")
        jid = jobs.enqueue(con, a.run_id, names, title=f"{a.template if not a.steps else 'Steps'}: {a.run_id}")
        print(f"queued job {jid}: {' -> '.join(names)}")
    elif a.cmd == "jobs":
        con = jobs.connect()
        for j in jobs.list_jobs(con, a.run_id):
            print(f"#{j['id']:<4} {j['status']:<10} {j['progress']:>4.0%}  {j['project']:<20} {j['title']}"
                  + (f"  · {j['current']}" if j["status"] not in ("done",) and j["current"] else ""))


if __name__ == "__main__":
    main()
