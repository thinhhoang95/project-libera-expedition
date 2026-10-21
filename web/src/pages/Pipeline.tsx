import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { CheckCircle2, Circle, Lock, Play, RefreshCw, Rocket, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, useApi, type Job, type Project } from "../api";
import { JobCard } from "../components/Jobs";
import { PageHeader } from "../components/Layout";
import { Card, ErrorBox, Pill, Spinner } from "../components/ui";
import { useProjectId } from "../context";
import { spendTotal } from "./Projects";

type Catalog = { steps: { name: string; label: string; phase: string; about: string; optional: boolean }[]; templates: Record<string, string[]> };

const TEMPLATES = [
  { key: "map", label: "Core map", icon: Rocket, about: "Corpus → facets → communities → map → signals → portfolio. What Start, My paper and Map & grid need." },
  { key: "full", label: "Full pipeline", icon: Sparkles, about: "Everything, including the backtest cutoffs and the research lab (several hours)." },
  { key: "refresh", label: "Refresh portfolio", icon: RefreshCw, about: "Signals, outside works and portfolio again (after reviews or new outside data)." },
];

export function StepPicker({ catalog, selected, setSelected, built, wide }: {
  catalog: Catalog; selected: string[]; setSelected: (s: string[]) => void; built?: Record<string, number | boolean>; wide?: boolean;
}) {
  return (
    <div className={clsx("grid gap-1.5", wide && "sm:grid-cols-2")}>
      {catalog.steps.map((s) => {
        const on = selected.includes(s.name);
        const done = built && !!built[s.name === "extract-semantic" ? "extract" : s.name === "embed" ? "core" : s.name];
        return (
          <label key={s.name} className={clsx("flex cursor-pointer gap-2.5 rounded-lg border px-3 py-2", on ? "border-brand-300 bg-[var(--chip)]/50" : "border-[var(--line)]")}>
            <input
              type="checkbox"
              className="mt-0.5 accent-[var(--color-brand-600)]"
              checked={on}
              onChange={() => setSelected(on ? selected.filter((x) => x !== s.name) : catalog.steps.map((c) => c.name).filter((n) => n === s.name || selected.includes(n)))}
            />
            <span className="min-w-0">
              <span className="flex items-center gap-1.5 text-[13px] font-medium text-[var(--text-strong)]">
                {s.label}
                <span className="faint text-[11px] font-normal">phase {s.phase}</span>
                {s.optional && <Pill>optional</Pill>}
                {built && (done ? <CheckCircle2 className="size-3.5 text-emerald-600" /> : <Circle className="faint size-3.5" />)}
              </span>
              <span className="muted block text-[12px] leading-snug">{s.about}</span>
            </span>
          </label>
        );
      })}
    </div>
  );
}

function Snapshots({ p }: { p: string }) {
  const { data } = useApi<any[]>(`/p/${p}/snapshots`);
  if (!data?.length) return null;
  return (
    <Card title="Frozen snapshots" icon={<Lock className="size-4 text-brand-600" />} subtitle="Prospective test: lists frozen on a date, scored later against what happened">
      {data.map((s) => (
        <div key={s.date} className="text-[12.5px]">
          <div className="flex items-center gap-2">
            <span className="h-title text-[13.5px]">{s.date}</span>
            <Pill tone={s.integrity === "ok" ? "green" : "red"}>{s.integrity === "ok" ? "files intact" : "hash mismatch"}</Pill>
          </div>
          <div className="muted mt-0.5">{s.lists} lists · {s.files} files · {s.corpus_papers?.toLocaleString()} papers · commit {s.git_commit}</div>
          {s.mismatched.length > 0 && <div className="mt-1 text-red-600">Changed since freezing: {s.mismatched.join(", ")}</div>}
          <ul className="mt-2 flex flex-col gap-1">
            {s.horizons.map((h: any) => (
              <li key={h.months} className="flex items-center gap-2">
                <span className="w-16">{h.months} months</span>
                <span className="muted">score on {h.due}</span>
                {h.due_now ? <Pill tone="amber">due</Pill> : <Pill>not yet</Pill>}
              </li>
            ))}
          </ul>
          {s.horizons.some((h: any) => h.due_now) && <code className="mt-2 block text-[11.5px]">{s.horizons.find((h: any) => h.due_now).command}</code>}
          {s.scored.length > 0 && <div className="muted mt-1">Scored: {s.scored.join(", ")}</div>}
        </div>
      ))}
    </Card>
  );
}

export function Pipeline() {
  const p = useProjectId();
  const qc = useQueryClient();
  const nav = useNavigate();
  const { data: project } = useApi<Project>(`/projects/${p}`, { refetchInterval: 8000 });
  const { data: jobs, isLoading } = useApi<Job[]>(`/jobs?project=${p}`, { refetchInterval: 2500, staleTime: 0 });
  const { data: catalog } = useApi<Catalog>("/pipeline/steps");
  const { data: sys } = useApi<any>("/system", { refetchInterval: 10_000 });
  const [custom, setCustom] = useState<string[] | null>(null);
  const queue = useMutation({
    mutationFn: (body: any) => api<Job>("/jobs", { method: "POST", json: { project: p, ...body } }),
    onSuccess: () => {
      setCustom(null);
      qc.invalidateQueries({ predicate: (q) => String(q.queryKey[0]).startsWith("/jobs") || String(q.queryKey[0]).startsWith("/projects") });
    },
  });
  const refresh = useMutation({ mutationFn: () => api("/system/refresh", { method: "POST" }), onSuccess: () => qc.invalidateQueries() });
  const [active, history] = useMemo(() => {
    const a = (jobs ?? []).filter((j) => !["done", "cancelled"].includes(j.status));
    return [a, (jobs ?? []).filter((j) => !a.includes(j))];
  }, [jobs]);
  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader
        title="Pipeline"
        crumbs={<span>{project?.name ?? p}</span>}
        sub="Queue the whole pipeline or single steps. Jobs run one at a time across projects; a paused, failed or interrupted job resumes at its first unfinished step, and every step only redoes missing work."
      />
      <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
        <div className="flex min-w-0 flex-col gap-3">
          <h2 className="h-title text-[16px]">Queue and running</h2>
          {isLoading && <Spinner />}
          {active.length === 0 && !isLoading && <div className="card muted px-4 py-6 text-center text-[13px]">Nothing queued for this project.</div>}
          {active.map((j) => <JobCard key={j.id} job={j} />)}
          {history.length > 0 && <h2 className="h-title mt-4 text-[16px]">History</h2>}
          {history.map((j) => <JobCard key={j.id} job={j} />)}
        </div>
        <div className="flex flex-col gap-4">
          <Card title="Run">
            <div className="flex flex-col gap-2">
              {TEMPLATES.map((t) => (
                <button key={t.key} className="flex items-start gap-3 rounded-xl border border-[var(--line)] px-3.5 py-3 text-left hover:border-brand-300 hover:bg-[var(--surface-2)]"
                  disabled={queue.isPending} onClick={() => queue.mutate({ template: t.key })}>
                  <t.icon className="mt-0.5 size-4.5 text-brand-600" />
                  <span className="min-w-0 flex-1">
                    <span className="h-title block text-[13.5px]">{t.label}</span>
                    <span className="muted block text-[12px] leading-snug">{t.about}</span>
                  </span>
                  <Play className="faint mt-0.5 size-4" />
                </button>
              ))}
              {queue.error && <ErrorBox error={queue.error} />}
            </div>
          </Card>
          {catalog && (
            <Card title="Choose steps" subtitle="Ticks show what is already built for this project.">
              <StepPicker catalog={catalog} selected={custom ?? []} setSelected={setCustom} built={project?.artefacts} />
              <button className="btn btn-primary mt-3 w-full" disabled={!custom?.length || queue.isPending} onClick={() => queue.mutate({ steps: custom })}>
                Queue {custom?.length || ""} step{custom?.length === 1 ? "" : "s"}
              </button>
            </Card>
          )}
          <Snapshots p={p} />
          <Card title="Project">
            <dl className="grid grid-cols-[120px_1fr] gap-y-1.5 text-[12.5px]">
              <dt className="muted">Id</dt><dd><code>{p}</code></dd>
              <dt className="muted">Seeds</dt><dd>{project?.seed_dois?.join(", ") || "—"}</dd>
              <dt className="muted">Paper budget</dt><dd>{project?.paper_budget?.toLocaleString() ?? "—"}</dd>
              <dt className="muted">Spend so far</dt><dd>${spendTotal(project?.spend).toFixed(3)}</dd>
              <dt className="muted">Database copy</dt>
              <dd>
                {sys?.replica ? new Date(sys.replica_at * 1000).toLocaleString() : "none"}
                {sys?.replica_stale && <Pill tone="amber" className="ml-1">newer data waiting</Pill>}
              </dd>
            </dl>
            <button className="btn btn-sm mt-3" onClick={() => refresh.mutate()} disabled={refresh.isPending}><RefreshCw className="size-3.5" /> Refresh the copy</button>
            {refresh.error && <div className="mt-2 text-[12px] text-amber-700">{String((refresh.error as Error).message)}</div>}
            <p className="muted mt-3 text-[12px] leading-relaxed">
              The app reads a copy of the database published after each step, so every page stays usable while a step writes.
              Pipeline steps can also be run from a terminal (<code>landscape queue</code>, <code>landscape worker</code>).
            </p>
            <button className="btn btn-sm mt-1" onClick={() => nav(`/p/${p}/start`)}>Open the project</button>
          </Card>
        </div>
      </div>
    </div>
  );
}

export function Queue() {
  const { data: jobs, isLoading } = useApi<Job[]>("/jobs?limit=200", { refetchInterval: 2500, staleTime: 0 });
  const { data: sys } = useApi<any>("/system", { refetchInterval: 10_000 });
  const active = (jobs ?? []).filter((j) => !["done", "cancelled"].includes(j.status));
  const history = (jobs ?? []).filter((j) => ["done", "cancelled"].includes(j.status));
  return (
    <div className="mx-auto max-w-[1200px]">
      <PageHeader title="Job queue" sub={<>Pipeline lane: one database-writing step at a time across all projects. Quick lane: read-only analyses (papers by DOI) that never wait for the pipeline. {sys && (sys.worker_lanes.length ? `This server runs: ${sys.worker_lanes.join(", ")}.` : "No worker in this server: run `landscape worker`.")}</>} />
      {isLoading && <Spinner />}
      <div className="flex flex-col gap-3">
        {active.map((j) => <JobCard key={j.id} job={j} showProject />)}
        {!isLoading && active.length === 0 && <div className="card muted px-4 py-6 text-center text-[13px]">The queue is empty.</div>}
        {history.length > 0 && <h2 className="h-title mt-5 text-[16px]">Finished</h2>}
        {history.map((j) => <JobCard key={j.id} job={j} showProject />)}
      </div>
    </div>
  );
}
