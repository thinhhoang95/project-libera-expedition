import { useMutation } from "@tanstack/react-query";
import { Check, ExternalLink, FileText, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, useApi } from "../api";
import { usePaperDrawer, useReviewer } from "../context";
import { ErrorBox, Pill, Spinner } from "./ui";

type Card = {
  paper_id: string; title: string; year: number | null; venue_name: string | null; type: string | null; doi: string | null;
  cited_by_count: number | null; abstract_raw: string | null; abstract_source: string | null; route: string | null;
  hop: number | null; authors: string[]; facets: { facet: string; text: string }[]; link: string;
};

export function PaperCard({ pid, compact }: { pid: string; compact?: boolean }) {
  const { project } = useParams();
  const { data: p, error, isLoading } = useApi<Card>(`/p/${project}/paper/${pid}`);
  if (isLoading) return <Spinner />;
  if (error) return <ErrorBox error={error} />;
  if (!p) return null;
  return (
    <div className="flex flex-col gap-3">
      <div>
        <a href={p.link} target="_blank" rel="noreferrer" className="h-title text-[15.5px] leading-snug hover:text-brand-600">
          {p.title} <ExternalLink className="mb-0.5 inline size-3.5 opacity-60" />
        </a>
        <div className="muted mt-1 text-[12.5px]">
          {p.year ?? "?"} · {p.venue_name || "unknown venue"}
          {p.cited_by_count != null && <> · {p.cited_by_count.toLocaleString()} citations (OpenAlex)</>}
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {p.route && <Pill tone="brand">route {p.route}{p.hop != null && ` · hop ${p.hop}`}</Pill>}
          {p.type && <Pill>{p.type}</Pill>}
        </div>
        {p.authors.length > 0 && (
          <div className="muted mt-2 text-[12.5px]">{p.authors.slice(0, 10).join(", ")}{p.authors.length > 10 && " …"}</div>
        )}
      </div>
      {p.facets.length ? (
        <dl className="grid gap-2">
          {p.facets.filter((f) => !compact || ["problem", "method", "innovation", "limitations"].includes(f.facet)).map((f) => (
            <div key={f.facet} className="rounded-lg bg-[var(--surface-2)] px-3 py-2">
              <dt className="faint text-[11px] font-semibold tracking-wide uppercase">{f.facet.replace(/_/g, " ")}</dt>
              <dd className="mt-0.5 text-[13px] leading-relaxed">{f.text}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <div className="muted text-[12.5px]">Title only: no facets.</div>
      )}
      {!compact && (
        <details className="rounded-lg border border-[var(--line)] px-3 py-2">
          <summary className="muted cursor-pointer text-[12.5px] font-medium">Abstract ({p.abstract_source || "none"})</summary>
          <p className="mt-2 text-[13px] leading-relaxed">{p.abstract_raw || "—"}</p>
        </details>
      )}
    </div>
  );
}

export function PaperDrawer() {
  const { paper, open } = usePaperDrawer();
  const { project } = useParams();
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && open(null);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open]);
  if (!paper || !project) return null;
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-slate-900/20" onClick={() => open(null)}>
      <aside className="scroll-thin h-full w-full max-w-[560px] overflow-y-auto bg-[var(--surface)] px-6 py-5 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-center justify-between">
          <span className="muted flex items-center gap-1.5 text-[12.5px] font-medium"><FileText className="size-4" /> Paper</span>
          <div className="flex items-center gap-2">
            <Link to={`/p/${project}/paper/${paper}`} onClick={() => open(null)} className="btn btn-sm">Open in My paper</Link>
            <button className="btn btn-ghost btn-icon btn-sm" onClick={() => open(null)}><X className="size-4" /></button>
          </div>
        </div>
        <PaperCard pid={paper} />
      </aside>
    </div>
  );
}

/** A clickable paper title that opens the drawer. */
export function PaperLink({ pid, title, className }: { pid?: string | null; title: string; className?: string }) {
  const { open } = usePaperDrawer();
  if (!pid) return <span className={className}>{title}</span>;
  return (
    <button
      className={"text-left hover:text-brand-600 hover:underline " + (className ?? "")}
      onClick={(e) => {
        e.stopPropagation();
        open(pid);
      }}
    >
      {title}
    </button>
  );
}

/** Judgment buttons; every judgment is stored with the reviewer's name. */
export function ReviewBar({ view, subject, actions, alpha, label = "Your judgment" }: {
  view: string; subject: Record<string, any>; actions: string[]; alpha?: Record<string, number>; label?: string;
}) {
  const { project } = useParams();
  const { reviewer } = useReviewer();
  const [note, setNote] = useState("");
  const [saved, setSaved] = useState<string | null>(null);
  const key = JSON.stringify(subject);
  useEffect(() => {
    setSaved(null);
    setNote("");
  }, [key]);
  const m = useMutation({
    mutationFn: (action: string) =>
      api(`/p/${project}/reviews`, { method: "POST", json: { view, subject, action, note, alpha, reviewer: reviewer || "anonymous" } }),
    onSuccess: (_, action) => setSaved(action),
  });
  return (
    <div className="mt-3 rounded-lg border border-[var(--line)] bg-[var(--surface-2)] p-3">
      <div className="muted mb-2 text-[12px] font-semibold">{label}</div>
      <div className="flex flex-wrap items-center gap-2">
        <input className="input min-w-[200px] flex-1" placeholder="Note (optional): why?" value={note} onChange={(e) => setNote(e.target.value)} />
        {actions.map((a) => (
          <button key={a} className="btn btn-sm" disabled={m.isPending} onClick={() => m.mutate(a)}>{a}</button>
        ))}
      </div>
      {saved && <div className="mt-2 flex items-center gap-1 text-[12px] text-emerald-600"><Check className="size-3.5" /> Saved “{saved}”{!reviewer && " (as anonymous: set your name in the sidebar)"}.</div>}
      {m.error && <div className="mt-2 text-[12px] text-red-600">{String(m.error)}</div>}
    </div>
  );
}
