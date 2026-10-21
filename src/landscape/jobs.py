"""Job queue for the pipeline: queue whole pipelines (or single steps), run them, resume after interruption.

State lives in SQLite (data/app/app.sqlite), not in DuckDB, so the queue can be read and edited while a
step holds the database. A job is an ordered list of steps; each step is one `landscape <command>` (or
analysis script) run in its own process with its output in projects/<project>/logs/.

Two lanes run side by side:
- pipeline: steps that write the database, strictly one at a time across all projects;
- quick: read-only work against the replica (a paper analysed by DOI), so it never waits for a pipeline.

Resuming. Every pipeline step is incremental and cached (it redoes only missing work), so a job resumes by
re-running its first step that is not done. A step interrupted by Pause, a crash or a restart is marked
`interrupted` and re-run when the job runs again; finished steps are skipped. One worker per lane holds a
lock file, so a second server or `landscape worker` cannot run the same lane twice.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import select
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import APP_DIR, ROOT, project_dir

log = logging.getLogger(__name__)
DB_FILE = APP_DIR / "app.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL, kind TEXT NOT NULL, title TEXT,
    lane TEXT NOT NULL DEFAULT 'pipeline', status TEXT NOT NULL, params TEXT,
    created_at TEXT, started_at TEXT, finished_at TEXT, error TEXT, position REAL
);
CREATE TABLE IF NOT EXISTS steps (
    job_id INTEGER NOT NULL, idx INTEGER NOT NULL, name TEXT NOT NULL, label TEXT, argv TEXT NOT NULL,
    status TEXT NOT NULL, started_at TEXT, finished_at TEXT, exit_code INTEGER, log TEXT,
    progress REAL, message TEXT, attempts INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (job_id, idx)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, project TEXT, job_id INTEGER, level TEXT, message TEXT,
    seen INTEGER NOT NULL DEFAULT 0
);
-- judgments saved in the web app; copied into DuckDB review_actions before the next pipeline step
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, reviewer TEXT, view TEXT, subject TEXT, action TEXT,
    note TEXT, alpha TEXT, ts TEXT, synced INTEGER NOT NULL DEFAULT 0
);
"""

ACTIVE = ("queued", "running", "pausing", "cancelling")


@dataclass(frozen=True)
class Step:
    name: str
    label: str
    phase: str
    command: tuple  # ("cli", <subcommand>, *args) or ("script", <path>, *args)
    about: str
    optional: bool = False


# The full pipeline, in order (README "Pipeline"). Every step takes --run-id <project>.
CATALOG: list[Step] = [
    Step("expand", "Expand the corpus", "1–2", ("cli", "expand"),
         "Seed, references, citing papers and authors' recent work; S2 and arXiv abstract fill."),
    Step("extract", "Extract facets", "3", ("cli", "extract"),
         "LLM facets (problem, method, innovation, result, limitations) for every paper so far."),
    Step("semantic", "Semantic expansion", "2", ("cli", "semantic"),
         "Embeds abstracts, then adds semantically close papers in the corpus's topics."),
    Step("extract-semantic", "Extract facets for the additions", "3", ("cli", "extract"),
         "Incremental: only the papers the semantic route added."),
    Step("embed", "Embed facets", "4", ("cli", "embed"), "EmbeddingGemma vectors per facet and tag normalisation."),
    Step("core", "Communities and matrices", "5", ("cli", "core"),
         "Clusters at three resolutions, labels, problem × method matrix, citation flows, author profiles."),
    Step("viz", "Map layouts", "6b", ("cli", "viz"), "Fixed UMAP layouts for the map."),
    Step("signals", "Signals", "6", ("cli", "signals"),
         "Direction, emergence, niches, bridges with the LLM judge, influence, early work."),
    Step("external", "Outside works", "9", ("cli", "external"),
         "External works the field cites at least 5 times: metadata, facets, embeddings (about $1 once)."),
    Step("portfolio", "Portfolio", "9", ("cli", "portfolio"),
         "Start here and My paper: import map, imports now, community warmth, second waves."),
    Step("backtest", "Backtest", "8", ("cli", "backtest"),
         "Cutoff runs 2020–2023 and precision@k against baselines (long).", optional=True),
    Step("prepare-cutoffs", "Cutoff 2019", "8c", ("script", "scripts/phase8c/prepare_cutoffs.py", "--cutoffs", "2019"),
         "The extra cutoff run the research lab's chronological test needs.", optional=True),
    Step("research-lab", "Research lab", "8c", ("cli", "research-lab"),
         "Sampling decomposition, broader bridge lists, underconnected communities, people by topic.",
         optional=True),
]
STEPS = {s.name: s for s in CATALOG}
TEMPLATES = {
    "full": [s.name for s in CATALOG],
    "map": [s.name for s in CATALOG if not s.optional],
    "refresh": ["signals", "external", "portfolio"],
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_SCHEMA_READY: set[Path] = set()
_SCHEMA_LOCK = threading.Lock()


def connect() -> sqlite3.Connection:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_FILE, timeout=30, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    with _SCHEMA_LOCK:  # once per process: concurrent DDL on a fresh file can fail
        if DB_FILE not in _SCHEMA_READY:
            con.executescript(SCHEMA)
            if "pid" not in {c[1] for c in con.execute("PRAGMA table_info(steps)")}:
                con.execute("ALTER TABLE steps ADD COLUMN pid INTEGER")
            _SCHEMA_READY.add(DB_FILE)
    return con


# ------------------------------------------------------------------ queueing

def _argv(command: tuple, project: str) -> list[str]:
    kind, target, *rest = command
    if kind == "cli":
        return ["cli", target, "--run-id", project, *rest]
    return ["script", target, "--run-id", project, *rest]


def enqueue(con, project: str, steps: list[str] | list[tuple[str, str, list[str]]], *, title: str,
            kind: str = "pipeline", lane: str = "pipeline", params: dict | None = None) -> int:
    """Queue a job. `steps` are catalogue names, or (name, label, argv) for one-off steps."""
    rows = []
    for s in steps:
        if isinstance(s, str):
            st = STEPS[s]
            rows.append((s, st.label, _argv(st.command, project)))
        else:
            rows.append(s)
    pos = (con.execute("SELECT max(position) FROM jobs").fetchone()[0] or 0) + 1
    cur = con.execute("""INSERT INTO jobs (project, kind, title, lane, status, params, created_at, position)
                         VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)""",
                      (project, kind, title, lane, json.dumps(params or {}), now(), pos))
    jid = cur.lastrowid
    for i, (name, label, argv) in enumerate(rows):
        con.execute("""INSERT INTO steps (job_id, idx, name, label, argv, status) VALUES (?, ?, ?, ?, ?, 'pending')""",
                    (jid, i, name, label, json.dumps(argv)))
    event(con, project, jid, "info", f"Queued: {title}")
    return jid


def event(con, project: str | None, job_id: int | None, level: str, message: str) -> None:
    con.execute("INSERT INTO events (ts, project, job_id, level, message) VALUES (?, ?, ?, ?, ?)",
                (now(), project, job_id, level, message))


def job_dict(con, jid: int) -> dict | None:
    j = con.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
    if not j:
        return None
    steps = [dict(s) for s in con.execute("SELECT * FROM steps WHERE job_id = ? ORDER BY idx", (jid,))]
    for s in steps:
        s["argv"] = json.loads(s["argv"])
    d = dict(j)
    d["params"] = json.loads(d["params"] or "{}")
    d["steps"] = steps
    n = len(steps)
    done = sum(s["status"] in ("done", "skipped") for s in steps)
    running = sum((s["progress"] or 0) for s in steps if s["status"] == "running")
    d["progress"] = (done + running) / n if n else 0.0
    d["current"] = next((s["label"] for s in steps if s["status"] in ("running", "interrupted", "pending", "failed")), None)
    if d["status"] == "queued":
        d["queue_position"] = con.execute(
            "SELECT count(*) FROM jobs WHERE lane = ? AND status = 'queued' AND position < ?",
            (d["lane"], d["position"])).fetchone()[0] + 1
    return d


def list_jobs(con, project: str | None = None, limit: int = 100) -> list[dict]:
    sql = "SELECT id FROM jobs" + (" WHERE project = ?" if project else "") + \
          " ORDER BY CASE WHEN status IN ('running','pausing','cancelling') THEN 0 WHEN status = 'queued' THEN 1 ELSE 2 END," \
          " CASE WHEN status = 'queued' THEN position END, id DESC LIMIT ?"
    ids = [r[0] for r in con.execute(sql, ((project, limit) if project else (limit,)))]
    return [job_dict(con, i) for i in ids]


def control(con, jid: int, action: str) -> dict:
    """pause | resume | cancel | retry | delete | up | down"""
    j = con.execute("SELECT status, project, title FROM jobs WHERE id = ?", (jid,)).fetchone()
    if not j:
        raise KeyError(jid)
    st = j["status"]
    if action == "pause":
        if st == "running":
            con.execute("UPDATE jobs SET status = 'pausing' WHERE id = ?", (jid,))
        elif st == "queued":
            con.execute("UPDATE jobs SET status = 'paused' WHERE id = ?", (jid,))
    elif action == "cancel":
        if st in ("running", "pausing"):
            con.execute("UPDATE jobs SET status = 'cancelling' WHERE id = ?", (jid,))
        elif st in ("queued", "paused", "failed"):
            con.execute("UPDATE jobs SET status = 'cancelled', finished_at = ? WHERE id = ?", (now(), jid))
    elif action in ("resume", "retry"):
        if st in ("paused", "failed", "cancelled", "done"):
            if st == "done":  # run again from the start (incremental steps redo only missing work)
                con.execute("UPDATE steps SET status = 'pending', progress = NULL, message = NULL WHERE job_id = ?", (jid,))
            con.execute("UPDATE steps SET status = 'pending' WHERE job_id = ? AND status IN ('failed', 'interrupted')", (jid,))
            pos = (con.execute("SELECT max(position) FROM jobs").fetchone()[0] or 0) + 1
            con.execute("UPDATE jobs SET status = 'queued', error = NULL, finished_at = NULL, position = ? WHERE id = ?",
                        (pos, jid))
            event(con, j["project"], jid, "info", f"Resumed: {j['title']}")
    elif action == "skip-step":
        if st in ("paused", "failed"):
            con.execute("""UPDATE steps SET status = 'skipped' WHERE job_id = ? AND idx = (SELECT min(idx) FROM steps
                           WHERE job_id = ? AND status NOT IN ('done', 'skipped'))""", (jid, jid))
    elif action == "delete":
        if st in ACTIVE[1:]:
            raise ValueError("pause or cancel the job first")
        con.execute("DELETE FROM steps WHERE job_id = ?", (jid,))
        con.execute("DELETE FROM jobs WHERE id = ?", (jid,))
        return {"deleted": jid}
    elif action in ("up", "down"):
        if st == "queued":
            rows = con.execute("SELECT id, position FROM jobs WHERE status = 'queued' ORDER BY position").fetchall()
            ids = [r["id"] for r in rows]
            k = ids.index(jid)
            o = k - 1 if action == "up" else k + 1
            if 0 <= o < len(ids):
                a, b = rows[k]["position"], rows[o]["position"]
                con.execute("UPDATE jobs SET position = ? WHERE id = ?", (b, jid))
                con.execute("UPDATE jobs SET position = ? WHERE id = ?", (a, ids[o]))
    else:
        raise ValueError(action)
    return job_dict(con, jid)


def recover(con, lane: str) -> int:
    """After a crash or restart: jobs left running go back to the queue; their running step is re-run."""
    rows = con.execute("SELECT id, status FROM jobs WHERE lane = ? AND status IN ('running', 'pausing', 'cancelling')",
                       (lane,)).fetchall()
    for r in rows:
        # a step of a worker that was killed outright can outlive it (own session) and still hold the database
        for s in con.execute("SELECT pid FROM steps WHERE job_id = ? AND status = 'running' AND pid IS NOT NULL", (r["id"],)):
            _kill_orphan(s["pid"])
        con.execute("UPDATE steps SET status = 'interrupted' WHERE job_id = ? AND status = 'running'", (r["id"],))
        new = {"running": "queued", "pausing": "paused", "cancelling": "cancelled"}[r["status"]]
        con.execute("UPDATE jobs SET status = ? WHERE id = ?", (new, r["id"]))
    return len(rows)


# ------------------------------------------------------------------ running

_PCT = re.compile(r"(\d{1,3})%\|")
_FRAC = re.compile(r"\b(\d+)/(\d+)\b")


def _progress(line: str) -> float | None:
    m = _PCT.findall(line)
    if m:
        return min(int(m[-1]), 100) / 100
    m = _FRAC.findall(line)
    if m:
        a, b = map(int, m[-1])
        if 0 < b and a <= b:
            return a / b
    return None


class _LaneLock:
    def __init__(self, lane: str):
        self.path = APP_DIR / f"worker-{lane}.lock"
        self.fh = None

    def acquire(self) -> bool:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            self.fh = None
            return False
        except OSError:  # file systems without flock: run without the cross-process guard
            log.warning("no file locking on %s; make sure only one worker runs", self.path)
        return True

    def release(self) -> None:
        if self.fh:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_UN)
            except OSError:
                pass
            self.fh.close()
            self.fh = None


class Worker:
    """Runs queued jobs, one lane per thread. `on_step_done(job, step)` is called after each pipeline step."""

    def __init__(self, lanes=("pipeline", "quick"), on_step_done=None, before_job=None, poll_s: float = 1.5):
        self.lanes, self.on_step_done, self.before_job, self.poll_s = lanes, on_step_done, before_job, poll_s
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self.owned: list[str] = []
        self.procs: dict[str, subprocess.Popen] = {}

    def start(self) -> list[str]:
        for lane in self.lanes:
            lock = _LaneLock(lane)
            if not lock.acquire():
                log.info("lane %s is run by another worker", lane)
                continue
            self.owned.append(lane)
            t = threading.Thread(target=self._loop, args=(lane, lock), name=f"worker-{lane}", daemon=True)
            t.start()
            self.threads.append(t)
        return self.owned

    def shutdown(self, timeout: float = 15) -> None:
        """Interrupt running steps (they are re-run on the next start) and stop."""
        self.stop.set()
        for p in list(self.procs.values()):
            _terminate(p)
        for t in self.threads:
            t.join(timeout)

    def _loop(self, lane: str, lock: _LaneLock) -> None:
        con = connect()
        n = recover(con, lane)
        if n:
            log.info("lane %s: %d interrupted job(s) back in the queue", lane, n)
        while not self.stop.is_set():
            row = con.execute("""UPDATE jobs SET status = 'running', started_at = coalesce(started_at, ?)
                                 WHERE id = (SELECT id FROM jobs WHERE lane = ? AND status = 'queued'
                                             ORDER BY position, id LIMIT 1) RETURNING id""", (now(), lane)).fetchone()
            if not row:
                self.stop.wait(self.poll_s)
                continue
            try:
                self._run_job(con, lane, row["id"])
            except Exception as e:  # never let one job kill the lane
                log.exception("job %s crashed", row["id"])
                con.execute("UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
                            (repr(e), now(), row["id"]))
        lock.release()
        con.close()

    def _run_job(self, con, lane: str, jid: int) -> None:
        job = job_dict(con, jid)
        project = job["project"]
        event(con, project, jid, "info", f"Started: {job['title']}")
        if self.before_job and lane == "pipeline":
            self.before_job(job)
        for s in job["steps"]:
            if s["status"] in ("done", "skipped"):
                continue
            outcome = self._run_step(con, lane, job, s)
            if outcome == "done":
                if self.on_step_done:
                    try:
                        self.on_step_done(job, s)
                    except Exception:
                        log.exception("after-step hook failed")
                continue
            status = con.execute("SELECT status FROM jobs WHERE id = ?", (jid,)).fetchone()["status"]
            if outcome == "stopped" and self.stop.is_set() and status == "running":
                return  # shutdown: recover() puts it back in the queue on the next start
            if status == "pausing":
                con.execute("UPDATE jobs SET status = 'paused' WHERE id = ?", (jid,))
                event(con, project, jid, "info", f"Paused at '{s['label']}': {job['title']}")
            elif status == "cancelling":
                con.execute("UPDATE jobs SET status = 'cancelled', finished_at = ? WHERE id = ?", (now(), jid))
                event(con, project, jid, "warning", f"Cancelled: {job['title']}")
            else:
                tail = _tail(s.get("log") or con.execute("SELECT log FROM steps WHERE job_id = ? AND idx = ?",
                                                         (jid, s["idx"])).fetchone()["log"])
                con.execute("UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
                            (f"{s['label']} failed\n{tail}", now(), jid))
                event(con, project, jid, "error", f"Failed at '{s['label']}': {job['title']}")
            return
        con.execute("UPDATE jobs SET status = 'done', finished_at = ? WHERE id = ?", (now(), jid))
        event(con, project, jid, "success", f"Finished: {job['title']}")

    def _run_step(self, con, lane: str, job: dict, s: dict) -> str:
        jid, idx, project = job["id"], s["idx"], job["project"]
        logs = project_dir(project, "logs")
        log_path = logs / f"job{jid:04d}-{idx:02d}-{s['name']}.log"
        kind, *args = s["argv"]
        cmd = ([sys.executable, "-m", "landscape.cli", *args] if kind == "cli"
               else [sys.executable, str(ROOT / args[0]), *args[1:]])
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "2"}
        if lane == "quick":
            from . import replica
            rep = replica.current()
            if rep is None:
                with open(log_path, "a") as fh:
                    fh.write("No read replica of the database yet: run a pipeline step first.\n")
                con.execute("UPDATE steps SET status = 'failed', log = ?, finished_at = ? WHERE job_id = ? AND idx = ?",
                            (str(log_path), now(), jid, idx))
                return "failed"
            env["LANDSCAPE_DB"] = str(rep)
        con.execute("""UPDATE steps SET status = 'running', started_at = ?, finished_at = NULL, exit_code = NULL,
                       log = ?, progress = 0, message = NULL, attempts = attempts + 1 WHERE job_id = ? AND idx = ?""",
                    (now(), str(log_path), jid, idx))
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== {now()} · {' '.join(cmd[2:] if kind == 'cli' else cmd[1:])} "
                     f"(attempt {s['attempts'] + 1})\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            self.procs[lane] = proc
            con.execute("UPDATE steps SET pid = ? WHERE job_id = ? AND idx = ?", (proc.pid, jid, idx))
            stopped = self._pump(con, proc, fh, jid, idx)
            self.procs.pop(lane, None)
            code = proc.wait()
            proc.stdout.close()
            fh.write(f"=== {now()} · exit {code}\n")
        if stopped:
            con.execute("UPDATE steps SET status = 'interrupted', exit_code = ?, finished_at = ? WHERE job_id = ? AND idx = ?",
                        (code, now(), jid, idx))
            return "stopped"
        status = "done" if code == 0 else "failed"
        con.execute("""UPDATE steps SET status = ?, exit_code = ?, finished_at = ?, progress = CASE WHEN ? = 'done'
                       THEN 1 ELSE progress END WHERE job_id = ? AND idx = ?""", (status, code, now(), status, jid, idx))
        return status

    def _pump(self, con, proc: subprocess.Popen, fh, jid: int, idx: int) -> bool:
        """Copy the step's output to its log, keep progress and the last message current; stop on request."""
        fd = proc.stdout.fileno()
        buf, last_ui, last_ctl, msg, prog = b"", 0.0, 0.0, None, None
        while True:
            r, _, _ = select.select([fd], [], [], 1.0)
            if r:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf += chunk
                parts = re.split(rb"(\r|\n)", buf)
                buf = parts.pop()  # incomplete tail
                for text, sep in zip(parts[0::2], parts[1::2]):
                    line = text.decode("utf-8", "replace").rstrip()
                    if not line:
                        continue
                    if sep == b"\n":  # carriage-return updates (progress bars) are not written to the log
                        fh.write(line + "\n")
                    p = _progress(line)
                    if p is not None:
                        prog = p
                    msg = line[-300:]
                fh.flush()
            t = time.time()
            if t - last_ui > 1.0 and (msg is not None or prog is not None):
                con.execute("UPDATE steps SET message = coalesce(?, message), progress = coalesce(?, progress) "
                            "WHERE job_id = ? AND idx = ?", (msg, prog, jid, idx))
                last_ui = t
            if t - last_ctl > 1.0:
                last_ctl = t
                st = con.execute("SELECT status FROM jobs WHERE id = ?", (jid,)).fetchone()["status"]
                if st in ("pausing", "cancelling") or self.stop.is_set():
                    _terminate(proc)
                    return True
            if proc.poll() is not None and not r:
                break
        if buf.strip():
            fh.write(buf.decode("utf-8", "replace") + "\n")
        return False


def _terminate(proc: subprocess.Popen, grace_s: float = 20) -> None:
    """SIGINT first (Python unwinds and DuckDB closes cleanly), then SIGTERM, then SIGKILL."""
    for sig, wait in ((signal.SIGINT, grace_s), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
        except subprocess.TimeoutExpired:
            continue


def _kill_orphan(pid: int, wait_s: float = 30) -> None:
    """Stop a step process left behind by a dead worker (SIGINT, then SIGKILL), if it is still ours."""
    try:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    except OSError:
        return
    if "landscape" not in cmd and "scripts/" not in cmd:
        return  # the pid was reused by an unrelated process
    log.warning("stopping orphaned step process %s", pid)
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        t0 = time.time()
        while time.time() - t0 < wait_s:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)


def _tail(path: str | None, n: int = 25) -> str:
    if not path or not Path(path).exists():
        return ""
    lines = Path(path).read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def read_log(path: str, max_bytes: int = 200_000) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    size = p.stat().st_size
    with open(p, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", "replace")
