import clsx from "clsx";
import { ArrowRight, CheckCircle2, Circle, Compass, Plus } from "lucide-react";
import { Link } from "react-router-dom";
import { useApi, type Job, type Project } from "../api";
import { JobCard } from "../components/Jobs";
import { PageHeader } from "../components/Layout";
import { Card, Empty, Spinner, num } from "../components/ui";

export const STEP_LABELS: [string, string][] = [
  ["expand", "Corpus"], ["extract", "Facets"], ["semantic", "Semantic"], ["core", "Communities"], ["viz", "Map"],
  ["signals", "Signals"], ["external", "Outside works"], ["portfolio", "Portfolio"], ["backtest", "Backtest"],
  ["research-lab", "Lab"],
];

export function BuiltChips({ a }: { a: Record<string, number | boolean> }) {
  return (
    <div className="flex flex-wrap gap-1">
      {STEP_LABELS.map(([k, label]) => {
        const ok = !!a[k];
        return (
          <span
            key={k}
            className={clsx(
              "inline-flex items-center gap-1 rounded-md px-1.5 py-[1px] text-[11.5px]",
              ok ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300" : "bg-[var(--surface-2)] text-[var(--faint)]",
            )}
          >
            {ok ? <CheckCircle2 className="size-3" /> : <Circle className="size-3" />} {label}
          </span>
        );
      })}
    </div>
  );
}

export const spendTotal = (s?: Record<string, number>) => Object.values(s ?? {}).reduce((a, b) => a + (b || 0), 0);

export function Projects() {
  const { data, isLoading } = useApi<Project[]>("/projects", { refetchInterval: 5000 });
  const { data: jobs } = useApi<Job[]>("/jobs?limit=6", { refetchInterval: 4000 });
  const active = (jobs ?? []).filter((j) => ["running", "queued", "pausing", "paused", "failed"].includes(j.status));
  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader title="Projects" sub="Each project maps one field from its seed papers. Papers, facets and embeddings are shared between projects, so overlapping fields cost less.">
        <Link to="/new" className="btn btn-primary"><Plus className="size-4" /> New project</Link>
      </PageHeader>
      {isLoading && <Spinner />}
      {data && data.length === 0 && (
        <Empty title="No projects yet" icon={<Compass className="size-8 text-brand-500" />}>
          Start one from a seed paper's DOI. The pipeline expands the corpus, extracts facets, builds the map and the portfolio.
          <div className="mt-3"><Link to="/new" className="btn btn-primary">Create a project</Link></div>
        </Empty>
      )}
      <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
        {(data ?? []).map((p) => (
          <Link key={p.id} to={`/p/${p.id}/start`} className="card group flex flex-col gap-3 p-5 transition hover:border-brand-300 hover:shadow-md">
            <div className="flex items-start gap-3">
              <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-[var(--chip)] text-brand-600"><Compass className="size-5" /></span>
              <div className="min-w-0 flex-1">
                <div className="h-title truncate text-[16px]">{p.name}</div>
                <div className="faint truncate text-[12px]">{p.id}{p.as_of && ` · as of ${p.as_of}`}</div>
              </div>
              <ArrowRight className="faint size-4 transition group-hover:translate-x-0.5 group-hover:text-brand-600" />
            </div>
            {p.description && <p className="muted line-clamp-2 text-[13px]">{p.description}</p>}
            <div className="text-[12.5px]">
              <span className="muted">Seed: </span>
              {p.seed_titles?.[0]?.title ?? p.seed_dois?.join(", ") ?? "—"}
            </div>
            <div className="grid grid-cols-3 gap-2 rounded-lg bg-[var(--surface-2)] px-3 py-2 text-[12px]">
              <div><div className="muted">Papers</div><div className="h-title text-[15px]">{num(Number(p.artefacts.expand) || 0)}</div></div>
              <div><div className="muted">Communities</div><div className="h-title text-[15px]">{p.artefacts.core ? "built" : "—"}</div></div>
              <div><div className="muted">Spend</div><div className="h-title text-[15px]">${spendTotal(p.spend).toFixed(2)}</div></div>
            </div>
            <BuiltChips a={p.artefacts} />
            {p.job && (
              <div className="rounded-lg border border-[var(--line)] px-3 py-2 text-[12.5px]">
                <div className="flex justify-between"><span className="muted">{p.job.title}</span><span>{p.job.status} · {Math.round(p.job.progress * 100)}%</span></div>
                <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[var(--line)]"><div className="h-full bg-brand-500" style={{ width: `${p.job.progress * 100}%` }} /></div>
              </div>
            )}
          </Link>
        ))}
      </div>
      {active.length > 0 && (
        <div className="mt-8">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="h-title text-[17px]">Active jobs</h2>
            <Link to="/queue" className="text-[13px] text-brand-600">Open the queue →</Link>
          </div>
          <div className="flex flex-col gap-3">{active.map((j) => <JobCard key={j.id} job={j} showProject />)}</div>
        </div>
      )}
      <Card className="mt-8" title="How projects are stored">
        <p className="muted text-[13px] leading-relaxed">
          <code>data/shared/</code> holds what every project reuses (the paper database, the raw response cache and the paper vectors).
          Each project keeps its own outputs under <code>data/projects/&lt;id&gt;/</code>: portfolio, external works, backtest,
          research lab, snapshots, run-level vectors and one log per pipeline step.
        </p>
      </Card>
    </div>
  );
}
