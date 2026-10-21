"""Libera Expedition: the web app (FastAPI) and its job worker.

    uv run landscape serve            # http://127.0.0.1:8600

The API lives under /api; the built frontend (web/, `npm run build`) is served from web/dist. The server
never opens the main database: it reads the replica (replica.py) and queues every write as a job (jobs.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import experiments, jobs, portfolio, projects, replica
from ..config import ROOT, RunParams, project_dir
from ..research_lab import lab_dir
from .data import Router, _CACHE, DB, jl, memo, q
from .views_lab import router as lab_router
from .views_map import router as map_router
from .views_portfolio import router as portfolio_router
from .views_review import router as review_router
from .views_review import sync_reviews

log = logging.getLogger(__name__)
DIST = ROOT / "web" / "dist"
router = Router()
_WORKER: jobs.Worker | None = None
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")


# ------------------------------------------------------------------ worker

def _after_step(job: dict, step: dict) -> None:
    if job["lane"] == "pipeline":
        replica.publish()


def _before_job(job: dict) -> None:
    n = sync_reviews()
    if n:
        log.info("copied %d saved judgments into review_actions", n)


def make_worker() -> jobs.Worker:
    return jobs.Worker(on_step_done=_after_step, before_job=_before_job)


def create_app(worker: bool | None = None) -> FastAPI:
    if worker is None:
        worker = not os.environ.get("LIBERA_NO_WORKER")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _WORKER
        # a stale replica is refreshed in the background (copying takes a few seconds)
        threading.Thread(target=replica.ensure_fresh, daemon=True).start()
        if worker:
            _WORKER = make_worker()
            lanes = _WORKER.start()
            log.info("worker lanes: %s", lanes or "none (another worker runs them)")
        yield
        if _WORKER:
            _WORKER.shutdown()

    app = FastAPI(title="Libera Expedition", lifespan=lifespan)
    for r in (router, portfolio_router, map_router, review_router, lab_router):
        app.include_router(r, prefix="/api")

    @app.exception_handler(HTTPException)
    async def http_error(_, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    if DIST.exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = DIST / path
            if path and f.is_file() and DIST in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(DIST / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_build():
            return JSONResponse({"detail": "Frontend not built: cd web && npm install && npm run build"}, 503)
    return app


# ------------------------------------------------------------------ system

@router.get("/system")
def system() -> dict:
    rep = replica.current()
    return {"app": "Libera Expedition", "replica": rep.name if rep else None,
            "replica_at": rep.stat().st_mtime if rep else None, "replica_stale": replica.stale(),
            "worker_lanes": _WORKER.owned if _WORKER else []}


@router.post("/system/refresh")
def refresh() -> dict:
    """Publish a fresh replica now (only possible while no step writes the database)."""
    p = replica.publish()
    if p is None:
        raise HTTPException(409, "The database is busy (a pipeline step is writing); it refreshes after the step.")
    return {"replica": p.name}


# ------------------------------------------------------------------ projects

def _artefacts(p: str) -> dict:
    """What has been built for a project, by pipeline step."""
    have: dict[str, object] = {}
    try:
        r = DB.one("""SELECT (SELECT count(*) FROM provenance WHERE run_id = ?),
                             (SELECT count(*) FROM provenance WHERE run_id = ? AND route = 'semantic'),
                             (SELECT count(*) FROM facet_status s JOIN provenance v USING (paper_id)
                               WHERE v.run_id = ? AND s.status = 'ok'),
                             (SELECT count(*) FROM clusters WHERE run_id = ?),
                             (SELECT count(*) FROM viz_layout WHERE run_id = ?),
                             (SELECT count(*) FROM signals WHERE run_id = ?),
                             (SELECT count(*) FROM runs WHERE run_id LIKE ? || '@T%'),
                             (SELECT count(*) FROM clusters WHERE run_id = ? || '@T2019')""", (p,) * 8)
        papers, semantic, facets, clusters, layout, signals, cutoffs, t2019 = r
        have.update(expand=papers, semantic=semantic, extract=facets, core=clusters, viz=layout, signals=signals,
                    backtest_runs=cutoffs)
        have["prepare-cutoffs"] = bool(t2019)
    except HTTPException:
        pass
    have["embed"] = bool(have.get("core"))
    have["external"] = (project_dir(p, "external", mkdir=False) / "works.parquet").exists()
    have["portfolio"] = (project_dir(p, "portfolio", mkdir=False) / "summary.json").exists()
    have["backtest"] = (project_dir(p, "backtest", mkdir=False) / "summary.csv").exists()
    have["research-lab"] = (lab_dir(p) / "summary.json").exists()
    # experiment steps: how many of their experiments have saved tables for this project
    saved = {m["experiment"] for m in experiments.load_index(p).values()}
    for step in experiments.STEP_SCRIPTS:
        have[step] = len({e for e, s in experiments.STEP_OF.items() if s == step} & saved)
    have["fit-models"] = any(project_dir(p, "models", mkdir=False).glob("*.json"))
    have["freeze-snapshot"] = any(f for f in (ROOT / "docs" / "snapshots").glob(f"{p}-*.json") if not f.name.startswith("._"))
    return have


@memo(64)
def _project_summary(p: str) -> dict:
    meta = projects.read_meta(p)
    run = q("SELECT params, spend_usd, CAST(started_at AS VARCHAR) started_at, notes FROM runs WHERE run_id = ?", (p,))
    out = {**meta, "in_database": len(run) > 0, "artefacts": _artefacts(p)}
    if len(run):
        out["spend"] = jl(run.spend_usd.iloc[0])
        out["started_at"] = run.started_at.iloc[0]
        notes = jl(run.notes.iloc[0])
        out.setdefault("seed_dois", notes.get("seed_dois"))
        out["paper_budget"] = jl(run.params.iloc[0]).get("expansion", {}).get("paper_budget")
        seeds = notes.get("seed_dois") or []
        if seeds:
            t = q(f"SELECT DISTINCT paper_id, title, year FROM papers WHERE lower(doi) IN ({','.join('?' * len(seeds))})",
                  [d.lower() for d in seeds]).drop(columns="paper_id")
            out["seed_titles"] = [{"title": a, "year": None if b != b else int(b)} for a, b in t.itertuples(index=False)]
    summ = portfolio.out_dir(p) / "summary.json"
    if summ.exists():
        out["as_of"] = json.loads(summ.read_text()).get("as_of")
    return out


def _projects() -> list[str]:
    ids = set(projects.list_projects())
    try:
        ids |= set(q("SELECT run_id FROM runs WHERE run_id NOT LIKE '%@T%'").run_id)
    except HTTPException:
        pass
    return sorted(ids)


@router.get("/projects")
def list_projects() -> list[dict]:
    con = jobs.connect()
    out = []
    for p in _projects():
        s = _project_summary(p)
        active = con.execute("""SELECT id, status, title FROM jobs WHERE project = ? AND status IN
                                ('queued','running','pausing','cancelling','paused','failed') ORDER BY id DESC LIMIT 1""",
                             (p,)).fetchone()
        out.append({**s, "job": jobs.job_dict(con, active["id"]) if active else None})
    return sorted(out, key=lambda d: d.get("last_opened") or d.get("created_at") or "", reverse=True)


@router.get("/projects/{p}")
def get_project(p: str) -> dict:
    if p not in _projects():
        raise HTTPException(404, f"No project {p}")
    s = dict(_project_summary(p))
    if s.get("in_database"):
        s["params"] = jl(q("SELECT params FROM runs WHERE run_id = ?", (p,)).params.iloc[0])
    return s


class ProjectPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    last_opened: str | None = None


@router.patch("/projects/{p}")
def patch_project(p: str, body: ProjectPatch) -> dict:
    if p not in _projects():
        raise HTTPException(404)
    meta = projects.write_meta(p, **body.model_dump())
    _CACHE.clear()  # summaries are cached per replica version; the name lives in project.json
    return meta


class NewProject(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    seed_dois: list[str] = Field(default_factory=list)
    seed_ids: list[str] = Field(default_factory=list)
    paper_budget: int | None = None
    params: dict | None = None
    steps: list[str] | None = None
    template: str = "map"
    start: bool = True


@router.post("/projects")
def create_project(body: NewProject) -> dict:
    pid = body.id.strip().lower()
    if not ID_RE.match(pid) or "@" in pid:
        raise HTTPException(422, "Project id: 2–63 lowercase letters, digits, '.', '_' or '-'")
    if pid in _projects():
        raise HTTPException(409, f"Project {pid} exists")
    dois = [d.strip().removeprefix("https://doi.org/") for d in body.seed_dois if d.strip()]
    if not dois and not body.seed_ids:
        raise HTTPException(422, "Give at least one seed DOI or OpenAlex work id")
    if body.params:
        try:
            RunParams.from_dict({**RunParams().to_dict(), **body.params})
        except TypeError as e:
            raise HTTPException(422, f"Invalid parameters: {e}") from e
    projects.write_meta(pid, name=body.name or pid, description=body.description, seed_dois=dois,
                        seed_ids=body.seed_ids)
    argv = ["cli", "new-run", "--run-id", pid]
    for d in dois:
        argv += ["--seed-doi", d]
    for s in body.seed_ids:
        argv += ["--seed-id", s]
    if body.paper_budget:
        argv += ["--paper-budget", str(body.paper_budget)]
    if body.params:
        f = project_dir(pid, "inputs") / "params.json"
        f.write_text(json.dumps(body.params, indent=2) + "\n")
        argv += ["--params", str(f)]
    if body.name:
        argv += ["--name", body.name]
    names = body.steps or jobs.TEMPLATES[body.template]
    con = jobs.connect()
    jid = jobs.enqueue(con, pid, [("init", "Create the run", argv), *names],
                       title=f"New project: {body.name or pid}", params={"template": body.template})
    if not body.start:
        jobs.control(con, jid, "pause")
    return {"id": pid, "job": jobs.job_dict(con, jid)}


# ------------------------------------------------------------------ jobs

@router.get("/pipeline/steps")
def steps() -> dict:
    return {"steps": [{"name": s.name, "label": s.label, "phase": s.phase, "about": s.about, "optional": s.optional}
                      for s in jobs.CATALOG], "templates": jobs.TEMPLATES}


@router.get("/jobs")
def list_jobs(project: str | None = None, limit: int = 100) -> list[dict]:
    return jobs.list_jobs(jobs.connect(), project, limit)


class NewJob(BaseModel):
    project: str
    steps: list[str] | None = None
    template: str | None = None
    title: str | None = None


@router.post("/jobs")
def create_job(body: NewJob) -> dict:
    if body.project not in _projects():
        raise HTTPException(404)
    names = body.steps or jobs.TEMPLATES.get(body.template or "map")
    bad = [n for n in names or [] if n not in jobs.STEPS]
    if not names or bad:
        raise HTTPException(422, f"Unknown steps: {bad}")
    con = jobs.connect()
    title = body.title or (jobs.TEMPLATE_TITLES.get(body.template, body.template) if body.template and not body.steps
                           else " → ".join(jobs.STEPS[n].label for n in names) if len(names) <= 3
                           else f"{len(names)} steps")
    return jobs.job_dict(con, jobs.enqueue(con, body.project, names, title=title))


@router.get("/jobs/{jid}")
def get_job(jid: int) -> dict:
    d = jobs.job_dict(jobs.connect(), jid)
    if not d:
        raise HTTPException(404)
    return d


@router.post("/jobs/{jid}/{action}")
def job_action(jid: int, action: str) -> dict:
    try:
        return jobs.control(jobs.connect(), jid, action)
    except KeyError:
        raise HTTPException(404) from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None


@router.get("/jobs/{jid}/steps/{idx}/log")
def step_log(jid: int, idx: int) -> dict:
    row = jobs.connect().execute("SELECT log FROM steps WHERE job_id = ? AND idx = ?", (jid, idx)).fetchone()
    if not row:
        raise HTTPException(404)
    return {"log": jobs.read_log(row["log"]) if row["log"] else ""}


@router.post("/p/{p}/my-paper/analyse")
def analyse(p: str, body: dict) -> dict:
    doi = str(body.get("doi", "")).strip().removeprefix("https://doi.org/")
    if not re.match(r"^10\.\d{4,9}/\S+$", doi):
        raise HTTPException(422, "Give a DOI such as 10.1109/TAC.2020.3006812")
    con = jobs.connect()
    jid = jobs.enqueue(con, p, [("my-paper", "Analyse a paper by DOI",
                                 ["cli", "my-paper", "--run-id", p, "--doi", doi, "--read-only"])],
                       title=f"Analyse {doi}", kind="my-paper", lane="quick", params={"doi": doi})
    return jobs.job_dict(con, jid)


class ScoreSnapshot(BaseModel):
    months: int = Field(12, ge=1, le=120)


@router.post("/p/{p}/snapshots/{day}/score")
def score_snapshot(p: str, day: str, body: ScoreSnapshot) -> dict:
    """Queue the scoring of a frozen snapshot at one horizon (`landscape score-snapshot`)."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", day) or not (ROOT / "docs" / "snapshots" / f"{p}-{day}.json").exists():
        raise HTTPException(404, f"No frozen snapshot of {p} on {day}")
    con = jobs.connect()
    jid = jobs.enqueue(con, p, [("score-snapshot", f"Score the {day} snapshot at {body.months} months",
                                 ["cli", "score-snapshot", "--run-id", p, "--date", day, "--months", str(body.months)])],
                       title=f"Score snapshot {day} · {body.months} months", kind="score-snapshot",
                       params={"date": day, "months": body.months})
    return jobs.job_dict(con, jid)


@router.get("/events")
def events(since: int = 0, project: str | None = None) -> dict:
    con = jobs.connect()
    sql = "SELECT * FROM events WHERE id > ?" + (" AND (project = ? OR project IS NULL)" if project else "")
    rows = [dict(r) for r in con.execute(sql + " ORDER BY id DESC LIMIT 50", (since, project) if project else (since,))]
    unseen = con.execute("SELECT count(*) FROM events WHERE seen = 0 AND level IN ('success', 'error', 'warning')").fetchone()[0]
    return {"events": rows, "unseen": unseen}


@router.post("/events/seen")
def events_seen() -> dict:
    jobs.connect().execute("UPDATE events SET seen = 1 WHERE seen = 0")
    return {"ok": True}


@router.get("/p/{p}/snapshots")
def snapshots(p: str) -> list[dict]:
    """Frozen prospective snapshots (docs/snapshots/<project>-<date>.json): lists, scoring date, file integrity."""
    import hashlib
    from datetime import date

    out = []
    for man in sorted((ROOT / "docs" / "snapshots").glob(f"{p}-*.json")):
        if man.name.startswith("._"):
            continue
        m = json.loads(man.read_text())
        d = project_dir(p, "snapshots", m["snapshot_date"], mkdir=False)
        bad = [f for f, h in m.get("sha256", {}).items()
               if not (d / f).exists() or hashlib.sha256((d / f).read_bytes()).hexdigest() != h]
        y, mo, dd = map(int, m["snapshot_date"].split("-"))
        horizons = []
        for months in m.get("protocol", {}).get("horizons_months") or [12]:
            due = date(y + (mo - 1 + months) // 12, (mo - 1 + months) % 12 + 1, min(dd, 28))
            horizons.append({"months": months, "due": due.isoformat(), "due_now": date.today() >= due,
                             "command": f"uv run landscape score-snapshot --run-id {p} --date {m['snapshot_date']} --months {months}"})
        scored = sorted(d.glob("score_*.json")) if d.exists() else []
        out.append({"date": m["snapshot_date"], "created_at": m.get("created_at"), "git_commit": (m.get("git_commit") or "")[:9],
                    "lists": len(m.get("lists", [])), "files": len(m.get("sha256", {})), "integrity": "ok" if not bad else "mismatch",
                    "mismatched": bad, "corpus_papers": m.get("corpus_papers"), "horizons": horizons,
                    "scored": [f.stem for f in scored if not f.name.startswith("._")]})
    return out
