# Feature: Libera Expedition web app, projects and the job queue

Status: implemented · Code: `src/landscape/web/` (FastAPI), `web/` (React), `jobs.py`, `replica.py`, `projects.py` · CLI: `landscape serve`, `landscape worker`, `landscape queue`, `landscape jobs`, `landscape migrate-layout` · Tests: `tests/test_jobs.py`

The web app replaces the Streamlit explorer as the main way to use the engine. It shows the same views
(Start here, My paper, Map & grid, Research lab, Experiments, Corpus view, Facet review, Cluster review)
with the same numbers and the same evidence notes. It adds four things:
- several projects side by side;
- a queue that runs the whole pipeline, with progress you can pause and resume;
- reading while a step writes the database;
- facet categories with a details sheet each, and navigation that links every view (below).

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
expand, extract, semantic, extract (for the additions), embed, core, viz, signals, external, portfolio.
The optional steps come after it:

| Step | What it runs |
| --- | --- |
| `backtest`, `prepare-cutoffs` | the cutoff runs 2020–2023, and 2019 for the chronological tests |
| `research-lab` | the Phase 8c lab, ending with the semantic ablation |
| `experiments-8` | the Phase 8 backtest analyses (`scripts/phase8/`) |
| `experiments-8b` | signal experiments 1–11 (`scripts/phase8b/`); OpenAlex yearly counts, about $0.02 once |
| `experiments-10` | orientation experiments 12–28 (`scripts/phase10/`); two LLM reads, about $0.55 once |
| `experiments-11` | time experiments 29–30 (`scripts/phase11/`): opportunity age, still open; no spend |
| `fit-models` | refits paper bridges, early work and community warmth on this project (below) |
| `freeze-snapshot` | freezes today's lists for the prospective test |

The four experiment steps run `scripts/run_experiments.py --step <step>`. It takes the step's scripts
in order from `experiments.RUNS` and runs each one in its own process. It records the scripts already
done in `backtest/experiments/.progress-<step>.json`, so a failed or interrupted step resumes at the
first unfinished script. It adds the spend that experiments 24 and 28 report to the run's spend. Every
table lands in the Experiments view. That page also queues a step whose experiments have no results yet,
or re-runs the step behind any table.

**Models.** Paper bridges, hidden gems, early work and community warmth are logistic models. The hidden-gem model needs experiment 30's table (`experiments-11`); without it, gems use the paper-bridges model.
- The committed models in `src/landscape/models/` were fitted on the pilot and serve every project by
  default.
- `fit-models` fits a project's own from its signal experiments into `projects/<project>/models/`.
  Those then win for that project and its cutoff runs (`config.model_file`).
- `scripts/phase8b/fit_models.py --as-default` replaces the committed defaults instead.

**Snapshots.**
- The Pipeline page's *Frozen snapshots* card queues `freeze-snapshot`, at most once a day.
- When a horizon is due, the card queues `landscape score-snapshot` for that horizon
  (`POST /p/<project>/snapshots/<date>/score`).

Templates:

| Template | Steps |
| --- | --- |
| Core map | everything except the optional steps: enough for Start here, My paper and Map & grid |
| Full pipeline | the core map, then backtest, cutoff 2019, research lab and the three experiment steps, then the portfolio again so it picks up the experiments' tables (experiment 12's entry effects) |
| Experiments | the same without the core map, for a project whose map is built |
| This field's models | fit-models, signals, portfolio |
| Refresh portfolio | signals, external, portfolio |

Neither Full pipeline nor Experiments refits the models or freezes a snapshot. Both are deliberate choices.

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

## Facets, details sheets and navigation

**Categories.** Each facet's clusters are its categories: problem and method at α = 0.5 (where bridges
and method transfer are computed), innovation, result and limitations at α = 0, all at medium resolution.
The problem categories are the communities. The **Facets** page lists every facet's categories with a
trend, the share in a time window and the change against the window before it. Tick categories to
compare their trends. **Across facets** crosses two facets as a matrix; a cell opens the row category
filtered to the papers in both.

**Details sheet.** Any category opens a sheet over the current page (`?cat=<cluster id>`), from Start
here, Facets, the map, the grid, Cluster review, My paper, the paper card or search. One time seeker
scopes the whole sheet. The sheet has:
- the trend: papers per year and the share of the facet, with the window's papers, share, Δ share,
  rank, first and peak year;
- how it relates to the other facets: where its papers fall in each other facet, with lift against the
  whole map in the same window, in the window or over time; a row filters the papers;
- citations with its sibling categories, and the signals that name it (bridges, papers to bring in,
  method transfer, new combinations; direction labels for other α);
- its papers: search over titles and facet statements, five sorts, filters by window, stated limitation
  and categories in the other facets, and a CSV download.

`?with=<cluster id>` opens a sheet already filtered (the grid and the matrix use it). Other clusters (other
α or resolution, from the map or Cluster review) open the same sheet, marked as not a reference category.

**Paper card.** The paper drawer (`?paper=<id>`) shows the paper's category in each facet, the corpus
papers it builds on and those citing it (each opens its own card), and links to the map and My paper.

**Map links.** The map keeps its view in the address (`facet`, `alpha`, `res`, `sel`, `tab`, `w`), so a
link or a reload reopens it. *Show on the map* from a sheet selects the category; from a paper it adds a
star at `focus=<id>` and zooms there. Pan and zoom now survive layer and window changes.

**Finding your way.** Opening a sheet or a paper adds a history entry, so Back closes it. The sidebar groups
pages into Explore, Evidence and Review. Search (⌘K or `/`) finds papers, categories of every facet,
pages and projects, and lists what you opened recently. `g` then a letter goes to a page; `?` lists the
shortcuts. Start here has a section bar, and its community names open their sheets.

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
  `clusters/…`, `paper/<id>`, `search` and `reviews`;
- the facet categories (`views_facets.py`): `catalogue` (every facet's categories with yearly counts),
  `category/<cluster id>` (the sheet: papers with their other categories, flows, signals),
  `cooccurrence?a=&b=` and `categories/search`.

The view endpoints port the Streamlit computations one for one. Large tables (map points) are sent as
columns, and the time window, layers and selection are applied in the browser.
