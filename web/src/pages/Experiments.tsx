import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "../components/Toasts";
import { Play } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, useApi, type Job } from "../api";
import { DataTable } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { Card, ErrorBox, Md, Note, Pill, Spinner } from "../components/ui";
import { useProjectId } from "../context";

const HOW = `Each experiment asks whether a signal predicts what happened next, using only data available at a cutoff year T (2020–2023) and checking against T+1…T+3. Models are fitted on three cutoffs and scored on the fourth (*leave-one-cutoff-out*), so the numbers are honest out-of-sample estimates **for this corpus**.

- **AUC**: how well a ranking separates hits from misses (0.5 = chance, 1 = perfect; below 0.5 means the score points the wrong way).
- **p@k**: the share of hits among the top k; compare it with the base rate or *random* row.
- **Baselines** (popularity, citations so far, random) are always shown. A signal is only interesting where it beats them.

Results depend on the corpus: how it was sampled, the field and its size. A signal that lost here may win elsewhere, so all tables stay visible.`;

const FILTER_COLS = ["outcome", "population", "level", "hit", "model", "signal"];

/** One result table with the same per-column filters as the explorer. */
function FilteredTable({ rows }: { rows: any[] }) {
  const cols = FILTER_COLS.filter((c) => rows.length && c in rows[0] && new Set(rows.map((r) => r[c])).size > 1);
  const [f, setF] = useState<Record<string, any>>({});
  const out = useMemo(() => {
    const val = (c: string) => f[c] ?? rows[0]?.[c];
    return rows.filter((r) => cols.every((c) => r[c] === val(c))).map((r) => {
      const o = { ...r };
      cols.forEach((c) => delete o[c]);
      return o;
    });
  }, [rows, f]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div>
      {cols.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {cols.map((c) => (
            <label key={c} className="text-[12px]"><span className="label">{c}</span>
              <select className="select" value={String(f[c] ?? rows[0][c])} onChange={(e) => setF({ ...f, [c]: [...new Set(rows.map((r) => r[c]))].find((v) => String(v) === e.target.value) })}>
                {[...new Set(rows.map((r) => r[c]))].map((v) => <option key={String(v)}>{String(v)}</option>)}
              </select>
            </label>
          ))}
        </div>
      )}
      <DataTable rows={out} maxHeight={420} dense />
    </div>
  );
}

function Table({ name }: { name: string }) {
  const p = useProjectId();
  const { data } = useApi<any>(`/p/${p}/experiments/table/${name}`);
  if (!data) return <Spinner />;
  return (
    <div className="mb-6">
      <div className="h-title text-[14px]">{data.meta.title}</div>
      {data.meta.how_to_read && <Md className="muted mt-1 mb-2 text-[12.8px]">{data.meta.how_to_read}</Md>}
      <FilteredTable rows={data.rows} />
      <Note>{data.meta.rows} rows · saved {data.meta.saved_at}</Note>
    </div>
  );
}

type ExpStep = { name: string; label: string; about: string; experiments: number; saved: number };

/** Queue one experiment step (or several) for this project; the job shows on the Pipeline page. */
function useRunSteps() {
  const p = useProjectId();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (steps: string[]) => api<Job>("/jobs", { method: "POST", json: { project: p, steps } }),
    onSuccess: (j) => {
      toast({ title: `Queued: ${j.title}`, tone: "success", href: `/p/${j.project}/pipeline`, linkLabel: "Follow it on the Pipeline page" });
      qc.invalidateQueries({ predicate: (q) => String(q.queryKey[0]).startsWith("/jobs") });
    },
  });
}

function RunExperiments({ steps }: { steps: ExpStep[] }) {
  const p = useProjectId();
  const run = useRunSteps();
  const missing = steps.filter((s) => s.saved < s.experiments);
  return (
    <Card title="Run the experiments on this project" subtitle="Each step re-runs its scripts on this corpus and saves every table here. They need the backtest's cutoff runs, so queue the Experiments template on the Pipeline page if those are not built yet.">
      <div className="grid gap-2 sm:grid-cols-2">
        {steps.map((s) => (
          <div key={s.name} className="flex items-start gap-3 rounded-xl border border-[var(--line)] px-3.5 py-2.5">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 text-[13px] font-medium text-[var(--text-strong)]">
                {s.label}
                <Pill tone={s.saved === s.experiments ? "green" : s.saved ? "amber" : "neutral"}>{s.saved}/{s.experiments} with results</Pill>
              </div>
              <div className="muted text-[12px] leading-snug">{s.about}</div>
            </div>
            <button className="btn btn-sm" disabled={run.isPending} onClick={() => run.mutate([s.name])} title={`Queue ${s.label}`}><Play className="size-3.5" /> Run</button>
          </div>
        ))}
      </div>
      {missing.length > 1 && (
        <button className="btn btn-primary btn-sm mt-3" disabled={run.isPending} onClick={() => run.mutate(missing.map((s) => s.name))}>
          <Play className="size-3.5" /> Run the {missing.length} steps without full results
        </button>
      )}
      {run.data && <Note>Queued job #{run.data.id}: follow it on the <Link className="text-brand-600 underline" to={`/p/${p}/pipeline`}>Pipeline</Link> page.</Note>}
      {run.error && <ErrorBox error={run.error} />}
    </Card>
  );
}

function RerunStep({ step, label }: { step: string; label: string }) {
  const p = useProjectId();
  const run = useRunSteps();
  return (
    <div className="mt-1 flex flex-wrap items-center gap-2 text-[12px]">
      <span className="muted">Produced by the pipeline step <b>{label}</b></span>
      <button className="btn btn-sm" disabled={run.isPending} onClick={() => run.mutate([step])}><Play className="size-3.5" /> Re-run on {p}</button>
      {run.data && <span className="muted">queued job #{run.data.id} (<Link className="text-brand-600 underline" to={`/p/${p}/pipeline`}>Pipeline</Link>)</span>}
      {run.error && <ErrorBox error={run.error} />}
    </div>
  );
}

export function Experiments() {
  const p = useProjectId();
  const { data: d, isLoading } = useApi<any>(`/p/${p}/experiments`);
  const [tab, setTab] = useState<string | null>(null);
  if (isLoading || !d) return <Spinner />;
  const tabs = [...(d.backtest ? ["backtest"] : []), ...d.groups.map((g: any) => g.experiment)];
  const cur = tab ?? tabs[0];
  const g = d.groups.find((x: any) => x.experiment === cur);
  const P = d.portfolio;
  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader title="Experiments" sub={`Every result table the experiments saved for ${p}, adopted or not`} />
      <details className="card mb-4 px-5 py-3"><summary className="h-title cursor-pointer text-[14px]">How to read this page</summary><Md className="mt-2 text-[13px]">{HOW}</Md></details>
      {P["The portfolio"] && (
        <Card title="Findings portfolio (pilot)" subtitle="Every experiment so far, its verdict and where it lives in the product. Measured on dro-pilot only; the tables below hold the evidence for this project.">
          <Md className="text-[13px]">{P["The portfolio"]}</Md>
          <details className="mt-2"><summary className="muted cursor-pointer text-[12.5px] font-medium">Verdicts and protocols</summary><Md className="mt-2 text-[13px]">{P["Verdicts"] + "\n\n" + P["Protocols, for reading the numbers"]}</Md></details>
          <details className="mt-2"><summary className="muted cursor-pointer text-[12.5px] font-medium">What the evidence says about how attention works</summary>
            <Md className="mt-2 text-[13px]">{P["What the evidence says about how attention works (pilot)"] + "\n\n**Implications**\n\n" + P["Implications for the product"]}</Md></details>
        </Card>
      )}
      {d.steps?.length > 0 && (!tabs.length || d.steps.some((s: ExpStep) => s.saved < s.experiments)) && <div className="mt-5"><RunExperiments steps={d.steps} /></div>}
      {!tabs.length ? (
        <Card className="mt-5"><Note>No experiment results for {p} yet.</Note></Card>
      ) : (
        <div className="mt-5 grid gap-5 lg:grid-cols-[280px_1fr]">
          <div className="card self-start p-2">
            {d.backtest && <button className={"block w-full rounded-lg px-3 py-2 text-left text-[13px] " + (cur === "backtest" ? "bg-[var(--nav-active)] text-brand-600" : "hover:bg-[var(--surface-2)]")} onClick={() => setTab("backtest")}>Phase 8 backtest</button>}
            {d.groups.map((x: any) => (
              <button key={x.experiment} onClick={() => setTab(x.experiment)} className={"block w-full rounded-lg px-3 py-2 text-left text-[13px] " + (cur === x.experiment ? "bg-[var(--nav-active)] text-brand-600" : "hover:bg-[var(--surface-2)]")}>
                <div className="font-medium">{x.title}</div>
                <div className="muted line-clamp-2 text-[11.5px]">{x.decision}</div>
              </button>
            ))}
          </div>
          <Card>
            {cur === "backtest" ? (
              <>
                <Md className="mb-3 text-[13px]">{"The Phase 8 backtest of the recipe's signals. Each **signal** is scored against its baselines in each **scope** (clustering or pooled), averaged over cutoffs. Cluster lists are short (about 17–20 clusters), so p@5–p@20 differences of ±0.05 are noise; **n_hits** says how many hits there were to find."}</Md>
                <FilteredTable rows={d.backtest} />
              </>
            ) : g && (
              <>
                <Note>Pilot decision: {g.decision ?? "—"}</Note>
                <div className="mt-3">{g.tables.map((t: any) => <Table key={t.name} name={t.name} />)}</div>
                {g.step && <RerunStep step={g.step} label={g.step_label} />}
                {g.scripts.length > 0 && <Note>From a terminal: {g.scripts.map((s: string) => <code key={s} className="mr-2">uv run python {s} --run-id {p}</code>)}</Note>}
              </>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
