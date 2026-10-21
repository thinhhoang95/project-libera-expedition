import { useMutation } from "@tanstack/react-query";
import { Check, ExternalLink, FileText, Link2, Map as MapIcon, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, useApi } from "../api";
import { usePaperDrawer, useRecordRecent, useReviewer } from "../context";
import { FACET_META, mapHref, type FacetKey } from "../facets";
import { CatLink, Dot } from "./Facets";
import { ErrorBox, Pill, Spinner } from "./ui";

type Cited = { paper_id: string; title: string; year: number | null; cites_here: number };

type Card = {
  paper_id: string; title: string; year: number | null; venue_name: string | null; type: string | null; doi: string | null;
  cited_by_count: number | null; abstract_raw: string | null; abstract_source: string | null; route: string | null;
  hop: number | null; authors: string[]; facets: { facet: string; text: string }[]; link: string;
  categories?: { facet: FacetKey; cluster_id: string; label: string; color: string; kind: string }[];
  citations?: { cites_here: number; refs_here: number; citers_here: number; refs: Cited[]; citers: Cited[] } | null;
};

function CitedList({ title, rows, total }: { title: string; rows: Cited[]; total: number }) {
  return (
    <div>
      <div className="muted mb-1 text-[12px] font-semibold">{title} <span className="faint font-normal">({total})</span></div>
      {rows.length ? (
        <ul className="flex flex-col gap-0.5">
          {rows.map((r) => (
            <li key={r.paper_id} className="flex gap-2 text-[12.5px]">
              <span className="faint w-9 shrink-0 tabular-nums">{r.year ?? "—"}</span>
              <PaperLink pid={r.paper_id} title={r.title} className="min-w-0 flex-1" />
              <span className="faint shrink-0 tabular-nums" title="Citations from papers in this map">{r.cites_here}</span>
            </li>
          ))}
        </ul>
      ) : <div className="faint text-[12.5px]">None in this map.</div>}
    </div>
  );
}

export function PaperCard({ pid, compact }: { pid: string; compact?: boolean }) {
  const { project } = useParams();
  const { data: p, error, isLoading } = useApi<Card>(`/p/${project}/paper/${pid}`);
  useRecordRecent(compact ? undefined : project, p ? { kind: "paper", id: pid, label: p.title } : null);
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
          {p.citations && <Pill tone="green" title="Citations from papers in this map">{p.citations.cites_here} citations here</Pill>}
        </div>
        {p.authors.length > 0 && (
          <div className="muted mt-2 text-[12.5px]">{p.authors.slice(0, 10).join(", ")}{p.authors.length > 10 && " …"}</div>
        )}
      </div>
      {!!p.categories?.length && (
        <div className="flex flex-col gap-1">
          <div className="faint text-[11px] font-semibold tracking-wide uppercase">Categories</div>
          {p.categories.map((c) => (
            <div key={c.facet} className="flex items-center gap-2 text-[12.8px]">
              <span className="w-[76px] shrink-0 text-[11.5px] font-semibold" style={{ color: FACET_META[c.facet]?.color }}>{FACET_META[c.facet]?.label}</span>
              <Dot color={c.color} />
              <CatLink cid={c.cluster_id} label={c.label} className="min-w-0 truncate" />
            </div>
          ))}
        </div>
      )}
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
      {!compact && p.citations && (
        <div className="grid gap-3 rounded-lg border border-[var(--line)] px-3 py-2.5">
          <CitedList title="Builds on (in this map)" rows={p.citations.refs} total={p.citations.refs_here} />
          <CitedList title="Cited by (in this map)" rows={p.citations.citers} total={p.citations.citers_here} />
        </div>
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
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && !(e.target as HTMLElement)?.closest?.("input, textarea") && open(null);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open]);
  if (!paper || !project) return null;
  return (
    <div className="fixed inset-0 z-[45] flex justify-end bg-slate-900/20" onClick={() => open(null)}>
      <aside className="scroll-thin h-full w-full max-w-[580px] overflow-y-auto bg-[var(--surface)] px-6 py-5 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <span className="muted flex items-center gap-1.5 text-[12.5px] font-medium"><FileText className="size-4" /> Paper</span>
          <div className="flex flex-wrap items-center gap-1.5">
            <Link to={mapHref(project, { paper })} className="btn btn-sm" title="Problem map, α = 0.5"><MapIcon className="size-3.5" /> Show on the map</Link>
            <Link to={`/p/${project}/paper/${paper}`} className="btn btn-sm">Analyse in My paper</Link>
            <button className="btn btn-ghost btn-icon btn-sm" title={copied ? "Copied" : "Copy a link to this paper"}
              onClick={() => { navigator.clipboard?.writeText(window.location.href); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>
              {copied ? <Check className="size-4 text-emerald-600" /> : <Link2 className="size-4" />}
            </button>
            <button className="btn btn-ghost btn-icon btn-sm" title="Close (Esc)" onClick={() => open(null)}><X className="size-4" /></button>
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
