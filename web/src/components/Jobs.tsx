import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import {
  ArrowDown, ArrowUp, CheckCircle2, Circle, CircleDashed, CirclePause, Loader2, Play, RotateCcw, ScrollText, SkipForward,
  Square, Trash2, XCircle,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, useApi, type Job, type Step } from "../api";
import { Bar, Pill } from "./ui";

const STATUS_TONE: Record<string, "neutral" | "brand" | "green" | "red" | "amber" | "violet"> = {
  queued: "neutral", running: "brand", pausing: "amber", paused: "amber", cancelling: "red", cancelled: "neutral",
  done: "green", failed: "red",
};

export function JobStatus({ job }: { job: Job }) {
  return (
    <Pill tone={STATUS_TONE[job.status]}>
      {job.status === "running" && <Loader2 className="size-3 animate-spin" />}
      {job.status}
      {job.status === "queued" && job.queue_position ? ` · #${job.queue_position}` : ""}
    </Pill>
  );
}

function StepIcon({ s }: { s: Step }) {
  const c = "size-4 shrink-0";
  switch (s.status) {
    case "done": return <CheckCircle2 className={clsx(c, "text-emerald-600")} />;
    case "running": return <Loader2 className={clsx(c, "animate-spin text-brand-500")} />;
    case "failed": return <XCircle className={clsx(c, "text-red-600")} />;
    case "interrupted": return <CirclePause className={clsx(c, "text-amber-600")} />;
    case "skipped": return <CircleDashed className={clsx(c, "faint")} />;
    default: return <Circle className={clsx(c, "faint")} />;
  }
}

const dur = (a: string | null, b: string | null) => {
  if (!a) return "";
  const s = ((b ? new Date(b) : new Date()).getTime() - new Date(a).getTime()) / 1000;
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
};

function LogView({ job, idx }: { job: number; idx: number }) {
  const { data } = useApi<{ log: string }>(`/jobs/${job}/steps/${idx}/log`, { refetchInterval: 3000, staleTime: 0 });
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [data]);
  return (
    <pre ref={ref} className="scroll-thin mt-2 max-h-80 overflow-auto rounded-lg bg-[#0d1424] p-3 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap text-slate-200">
      {data?.log || "No output yet."}
    </pre>
  );
}

export function JobCard({ job, showProject, defaultOpen }: { job: Job; showProject?: boolean; defaultOpen?: boolean }) {
  const qc = useQueryClient();
  const [log, setLog] = useState<number | null>(null);
  const [open, setOpen] = useState(defaultOpen ?? ["running", "failed", "paused", "pausing"].includes(job.status));
  const act = useMutation({
    mutationFn: (a: string) => api(`/jobs/${job.id}/${a}`, { method: "POST" }),
    onSettled: () => {
      qc.invalidateQueries({ predicate: (q) => String(q.queryKey[0]).startsWith("/jobs") || String(q.queryKey[0]).startsWith("/projects") });
    },
  });
  const btn = (a: string, label: string, Icon: any, tone?: string) => (
    <button className={clsx("btn btn-sm", tone)} disabled={act.isPending} onClick={() => act.mutate(a)} title={label}>
      <Icon className="size-3.5" /> <span className="hidden sm:inline">{label}</span>
    </button>
  );
  const s = job.status;
  const doneSteps = job.steps.filter((x) => x.status === "done" || x.status === "skipped").length;
  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-center gap-3 px-4 py-3">
        <button className="min-w-0 flex-1 text-left" onClick={() => setOpen(!open)}>
          <div className="flex items-center gap-2">
            <span className="faint text-[12px] font-medium">#{job.id}</span>
            <span className="h-title truncate text-[14px]">{job.title}</span>
            <JobStatus job={job} />
            {job.lane === "quick" && <Pill tone="violet">quick lane</Pill>}
          </div>
          <div className="muted mt-0.5 flex flex-wrap gap-x-3 text-[12px]">
            {showProject && <Link to={`/p/${job.project}/pipeline`} onClick={(e) => e.stopPropagation()} className="hover:text-brand-600">{job.project}</Link>}
            <span>{doneSteps}/{job.steps.length} steps</span>
            {job.current && s !== "done" && <span>· {s === "running" ? "now" : "next"}: {job.current}</span>}
            {job.started_at && <span>· {dur(job.started_at, job.finished_at)}</span>}
            <span>· queued {job.created_at.replace("T", " ").slice(5, 16)}</span>
          </div>
        </button>
        <div className="flex flex-wrap items-center gap-1.5">
          {s === "queued" && <>{btn("up", "Up", ArrowUp)}{btn("down", "Down", ArrowDown)}{btn("pause", "Hold", CirclePause)}</>}
          {s === "running" && btn("pause", "Pause", CirclePause)}
          {(s === "running" || s === "queued") && btn("cancel", "Cancel", Square)}
          {(s === "paused" || s === "failed") && btn("resume", "Resume", Play, "btn-primary")}
          {(s === "paused" || s === "failed") && btn("skip-step", "Skip step", SkipForward)}
          {(s === "cancelled" || s === "done") && btn("retry", "Run again", RotateCcw)}
          {["done", "failed", "cancelled", "paused"].includes(s) && btn("delete", "Delete", Trash2)}
        </div>
      </div>
      <Bar value={job.progress} className="!h-[3px] !rounded-none" />
      {open && (
        <div className="px-4 py-3">
          {job.error && s === "failed" && (
            <pre className="mb-3 overflow-auto rounded-lg border border-red-300/50 bg-red-500/5 p-2.5 text-[11.5px] whitespace-pre-wrap text-red-700 dark:text-red-300">{job.error}</pre>
          )}
          <ol className="flex flex-col">
            {job.steps.map((st) => (
              <li key={st.idx} className="border-b border-[var(--line)] py-2 last:border-0">
                <div className="flex items-center gap-2.5">
                  <StepIcon s={st} />
                  <span className={clsx("text-[13px]", st.status === "running" ? "h-title" : "")}>{st.label}</span>
                  <code className="faint hidden text-[11px] md:inline">{st.argv.slice(1, 2).join(" ")}</code>
                  {st.attempts > 1 && <Pill tone="amber">attempt {st.attempts}</Pill>}
                  {st.status === "interrupted" && <Pill tone="amber">resumes here</Pill>}
                  <span className="faint ml-auto text-[11.5px]">{dur(st.started_at, st.finished_at)}</span>
                  {st.log && (
                    <button className="btn btn-ghost btn-sm !h-6 !px-1.5" title="Log" onClick={() => setLog(log === st.idx ? null : st.idx)}>
                      <ScrollText className="size-3.5" />
                    </button>
                  )}
                </div>
                {st.status === "running" && (
                  <div className="mt-1.5 ml-6.5 flex items-center gap-3">
                    <Bar value={st.progress ?? 0} className="max-w-xs" />
                    <span className="muted min-w-0 flex-1 truncate font-mono text-[11px]">{st.message}</span>
                  </div>
                )}
                {log === st.idx && <LogView job={job.id} idx={st.idx} />}
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
