# Feature: Libera Expedition web app, projects and the job queue

Status: implemented · Code: `src/landscape/web/` (FastAPI), `web/` (React), `jobs.py`, `replica.py`, `projects.py` · CLI: `landscape serve`, `landscape worker`, `landscape queue`, `landscape jobs`, `landscape migrate-layout` · Tests: `tests/test_jobs.py`

The web app replaces the Streamlit explorer as the main way to use the engine. It shows the same views
(Start here, My paper, Map & grid, Research lab, Experiments, Corpus view, Facet review, Cluster review)
with the same numbers and the same evidence notes. It adds three things:
- several projects side by side;
- a queue that runs the whole pipeline, with progress you can pause and resume;
- reading while a step writes the database.

## Run

```bash
uv sync
cd web && npm install && npm run build && cd ..   # once, and after frontend changes
uv run landscape serve                            # http://127.0.0.1:8600
```

`landscape serve` runs the job worker in the same process. Use `--no-worker` and run
`uv run landscape worker` elsewhere to keep jobs going while the web server restarts.
For frontend work, `npm --prefix web run dev` serves on :5173 and proxies `/api` to :8600.

The Streamlit explorer (`landscape explorer`) still works. It opens the main database file, though, so
close it before pipeline steps run.

## Projects and the data layout

A project is a base run: `runs.run_id` without `@T`. Its cutoff runs `<project>@T<year>` belong to it.

```
data/
  shared/                     reused by every project
    landscape.duckdb          papers, facets, refs and every run-keyed table
    raw/                      raw API response cache
    vectors/                  paper-level facet vectors and FAISS indexes
  projects/<project>/
    project.json              name, description, seeds, created_at
    inputs/                   deep-research report, parameter overrides
    vectors/                  cluster and author-year centroids (the run and its cutoff runs)
    external/ portfolio/ backtest/ research_lab/ snapshots/
    logs/                     one log per pipeline step run by the queue
  app/
    app.sqlite                job queue, events, judgments saved in the app
    replica/                  read copies of the database (see below)
  migration.json              the moves made by `landscape migrate-layout`
```

Papers, facets and embeddings are shared because two fields that overlap reach the same papers. Each
paper is then fetched, extracted (LLM) and embedded only once. Everything a project builds lives in its
own folder, so you can archive or delete a project by removing the folder and its `runs` rows.

All paths come from `config.project_dir(run_id, ...)` and `config.run_vectors(run_id)`.
`landscape migrate-layout` moves an older flat `data/` folder into this layout:
- every move is a rename on the same volume, recorded in `data/migration.json`;
- `--dry-run` lists the moves first;
- every other command refuses to run on the old layout until you migrate.

## The job queue

A job is an ordered list of steps. Each step is one `landscape <command> --run-id <project>` (or an
analysis script) run in its own process. The step catalogue in `jobs.py` follows the README pipeline:
expand, extract, semantic, extract (for the additions), embed, core, viz, signals, external, portfolio,
then, optionally, backtest, cutoff 2019 and research-lab.

Templates:

| Template | Steps |
| --- | --- |
| Core map | everything except the optional steps: enough for Start here, My paper and Map & grid |
| Full pipeline | all steps |
| Refresh portfolio | signals, external, portfolio |

Single steps can be queued from the Pipeline page or `landscape queue --steps a,b`.

**Lanes.**
- *pipeline* runs steps that write the database, one at a time across all projects (DuckDB allows
  one writer).
- *quick* runs read-only work against the replica, so analysing a paper by DOI never waits for a
  pipeline.

One worker per lane holds `data/app/worker-<lane>.lock`, so a second server or worker cannot run a lane
twice.

**Resuming.** Every pipeline step is incremental and cached: running it again only does the missing
work. A job therefore resumes by re-running its first unfinished step and skipping the finished ones.
- **Pause** sends SIGINT to the step's process group (so DuckDB closes cleanly), then SIGTERM, then
  SIGKILL. The step is marked `interrupted` and the job `paused`. **Resume** queues it again.
- **A restart or crash** of the worker puts running jobs back in the queue. Their running step is marked
  `interrupted`, and a step process that outlived a killed worker is stopped first.
- **A failed** job keeps its log tail as the error. Resume retries the failed step, and *Skip step*
  moves past it.

**Progress.** The worker reads each step's output.
- Lines go to `projects/<project>/logs/job<id>-<idx>-<step>.log`.
- Carriage-return progress bars (tqdm) update the step's progress and message in the queue but are not
  written to the log.
- Job progress = finished steps plus the running step's fraction, over all steps.

Events (started, finished, failed, paused) feed the notification bell.

## Reading while a step writes

While a step writes the database, no other process may open it. The web server therefore never opens the
main file:
- After each pipeline step, the worker copies it to `data/app/replica/landscape-<stamp>.duckdb` and
  points `replica/current` at the new copy, then deletes copies older than 15 minutes.
- On start-up the server publishes a copy if the main file is newer and nobody is writing it. The
  Pipeline page has a *Refresh the copy* button for steps run from a terminal.
- The server switches to a new copy on the next request. Cached results are keyed by the copy, so they
  refresh with it.

Judgments saved in the app (facet ratings, cluster reviews, bridge verdicts, My paper feedback) go to
`app.sqlite` first. They are copied into DuckDB `review_actions` before the next pipeline job starts. The
views read both.

## API

Everything is under `/api`:
- `/projects`, `/jobs`, `/pipeline/steps`, `/events` and `/system`;
- per project, `/p/<project>/…`: `overview`, `my-paper/<id>`, `map`, `grid`, `bridges`,
  `paper-bridges`, `external-imports`, `people`, `lab`, `experiments`, `corpus`, `facets/…`,
  `clusters/…`, `paper/<id>`, `search` and `reviews`.

The view endpoints port the Streamlit computations one for one. Large tables (map points) are sent as
columns, and the time window, layers and selection are applied in the browser.
