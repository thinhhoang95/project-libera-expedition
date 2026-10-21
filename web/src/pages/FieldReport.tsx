import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Download, FileSearch, ListTree, Loader2, MessageSquareText, MessageSquarePlus, RotateCcw, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { api, useApi, type Job } from "../api";
import { LS } from "../context";
import { PageHeader } from "../components/Layout";
import { ModelPicker } from "../components/ModelPicker";
import { ReportChat } from "../components/ReportChat";
import { ReportMarkdown, slug } from "../components/ReportMarkdown";
import { toast } from "../components/Toasts";
import { Card, Empty, ErrorBox, Note, PageSkeleton, Pill } from "../components/ui";

type Meta = { id: string; title: string; created_at: string; model: string; model_label: string; focus: string | null; as_of: number; cost_usd: number | null; seconds: number | null };
type Report = Meta & { markdown: string; markdown_original?: string; edits?: { at: string }[]; model_slug: string; usage: Record<string, number>; reasoning_tokens: number | null; evidence_chars: number; evidence_sha256: string };
type Models = { key_present: boolean; models: { key: string; label: string; slug: string }[]; default_report: string; default_chat: string };

const ACTIVE = ["queued", "running", "pausing", "cancelling"];
const when = (iso: string) => iso.slice(0, 16).replace("T", " ");
const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

function Generate({ p, models, job, onJob }: { p: string; models?: Models; job?: Job; onJob: (j: Job) => void }) {
  const [model, setModelRaw] = useState<string | null>(() => LS.get("libera.reportModel"));
  const setModel = (s: string) => { LS.set("libera.reportModel", s); setModelRaw(s); };
  const slugNow = model ?? models?.default_report;
  const [focus, setFocus] = useState("");
  const m = useMutation({
    mutationFn: () => api<Job>(`/p/${p}/reports`, { method: "POST", json: { model: slugNow, focus: focus || null } }),
    onSuccess: (j) => { onJob(j); toast({ title: "Field report queued", body: "Quick lane: a few minutes", tone: "info" }); },
  });
  const running = !!job && ACTIVE.includes(job.status);
  return (
    <Card title="New report" icon={<Sparkles className="size-4.5 text-brand-600" />}>
      <div className="flex flex-col gap-3">
        <div>
          <div className="faint mb-1 text-[11px] font-semibold tracking-wide uppercase">Model</div>
          <ModelPicker value={slugNow} onChange={setModel} minContext={80_000}
            suggested={[...(models?.models ?? []).map((x) => x.slug), "anthropic/claude-sonnet-5.5", "google/gemini-3.1-pro-preview", "deepseek/deepseek-v4-pro"]} />
        </div>
        <div>
          <div className="faint mb-1 text-[11px] font-semibold tracking-wide uppercase">Focus (optional)</div>
          <textarea className="input min-h-[76px] py-2 text-[13px]" maxLength={2000} value={focus} onChange={(e) => setFocus(e.target.value)}
            placeholder="e.g. I'm a control theorist looking for a 2-year research direction that uses DRO" />
        </div>
        <button className="btn btn-primary" disabled={running || m.isPending || !slugNow || models?.key_present === false} onClick={() => m.mutate()}>
          {running ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />} Write the report
        </button>
        {models?.key_present === false && <div className="text-[12.5px] text-red-600">No OpenRouter key in .env.local.</div>}
        {m.error && <ErrorBox error={m.error} />}
        {job && (job.status !== "done") && (
          <div className="muted flex items-start gap-2 text-[12.5px]">
            {running && <Loader2 className="mt-0.5 size-3.5 shrink-0 animate-spin" />}
            <span className="min-w-0 break-words">
              {job.status === "queued" ? "Queued (quick lane)…"
                : running ? job.steps[0]?.message?.replace(/\s*\d+%\|$/, "") || "Building the evidence pack…"
                : job.status === "failed" ? <span className="text-red-600">{job.error?.split("\n").slice(-2).join(" ")}</span> : job.status}
            </span>
          </div>
        )}
      </div>
      <Note>The model reads every product of this map (communities, facets, imports, reading paths, opportunities with their timing, second waves, entry routes, the sampling audit) and the evidence ledger, then decides what is worth saying. The prompt is about 60k tokens; with Claude Opus 5.5 a report costs roughly $0.5–1, with GPT-6.1 Sol about half that. Any OpenRouter model works.</Note>
    </Card>
  );
}

/** "Ask about this" next to a selection inside the report. */
function SelectionAsk({ root, onAsk }: { root: React.RefObject<HTMLElement>; onAsk: (text: string) => void }) {
  const [pos, setPos] = useState<{ x: number; y: number; text: string } | null>(null);
  useEffect(() => {
    const onUp = () => setTimeout(() => {
      const sel = window.getSelection();
      const text = sel?.toString().trim() ?? "";
      if (!sel || !text || !sel.rangeCount || !root.current?.contains(sel.anchorNode)) return setPos(null);
      const r = sel.getRangeAt(0).getBoundingClientRect();
      setPos({ x: Math.min(window.innerWidth - 150, r.right), y: r.top, text });
    }, 0);
    const onDown = (e: MouseEvent) => { if (!(e.target as HTMLElement).closest?.("[data-ask]")) setPos(null); };
    document.addEventListener("mouseup", onUp);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("scroll", () => setPos(null), { capture: true, passive: true, once: true } as any);
    return () => { document.removeEventListener("mouseup", onUp); document.removeEventListener("mousedown", onDown); };
  }, [root]);
  if (!pos) return null;
  return (
    <button data-ask style={{ left: pos.x, top: Math.max(8, pos.y - 38) }}
      className="fixed z-40 flex items-center gap-1.5 rounded-lg bg-[var(--text-strong)] px-2.5 py-1.5 text-[12px] font-medium text-[var(--surface)] shadow-lg"
      onMouseDown={(e) => e.preventDefault()} onClick={() => { onAsk(pos.text); setPos(null); window.getSelection()?.removeAllRanges(); }}>
      <MessageSquarePlus className="size-3.5" /> Ask about this <span className="opacity-60">{isMac ? "⌘L" : "Ctrl+L"}</span>
    </button>
  );
}

export function FieldReport() {
  const { project: p = "" } = useParams();
  const qc = useQueryClient();
  const [sp, setSp] = useSearchParams();
  const { data: models } = useApi<Models>("/report-models");
  const { data: list, isLoading, error } = useApi<Meta[]>(`/p/${p}/reports`, { staleTime: 0 });
  const [jobId, setJobId] = useState<number | null>(null);
  // pick up a report job started earlier (another tab, before a reload)
  const { data: jobs } = useApi<Job[]>(`/jobs?project=${p}&limit=30`, { staleTime: 0 });
  useEffect(() => {
    const j = jobs?.find((x) => x.kind === "field-report" && ACTIVE.includes(x.status));
    if (j && !jobId) setJobId(j.id);
  }, [jobs]); // eslint-disable-line react-hooks/exhaustive-deps
  const { data: job } = useApi<Job>(jobId ? `/jobs/${jobId}` : null, { refetchInterval: (q) => (ACTIVE.includes((q.state.data as Job | undefined)?.status ?? "queued") ? 1500 : false), staleTime: 0 });
  useEffect(() => {
    if (job?.status === "done") {
      qc.invalidateQueries({ queryKey: [`/p/${p}/reports`] });
      setSp((s) => { s.delete("r"); return s; }, { replace: true });
    }
  }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const rid = sp.get("r") ?? list?.[0]?.id ?? null;
  const { data: rep, isLoading: repLoading } = useApi<Report>(rid ? `/p/${p}/reports/${rid}` : null, { keep: true });
  const toc = useMemo(() => (rep?.markdown ?? "").split("\n").filter((l) => /^## /.test(l)).map((l) => l.slice(3).replace(/[*_`]/g, "").trim()), [rep?.markdown]);

  // the side panel: the chat (Cursor-style, ⌘L) or the report tools
  const [panel, setPanelRaw] = useState<"chat" | "tools">(() => (LS.get("libera.reportPanel") === "tools" ? "tools" : "chat"));
  const setPanel = (v: "chat" | "tools") => { LS.set("libera.reportPanel", v); setPanelRaw(v); };
  const [quote, setQuote] = useState<string | null>(null);
  const [focusTick, setFocusTick] = useState(0);
  const article = useRef<HTMLElement>(null);
  const ask = useCallback((text?: string) => {
    if (text) setQuote(text.length > 4000 ? text.slice(0, 4000) + "…" : text);
    setPanel("chat");
    setFocusTick((t) => t + 1);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "l") {
        e.preventDefault();
        const sel = window.getSelection();
        const text = sel?.toString().trim();
        if (text && article.current?.contains(sel!.anchorNode)) ask(text);
        else if (panel === "chat" && document.activeElement?.closest("[data-chat]")) setPanel("tools");
        else ask();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [panel, ask]); // eslint-disable-line react-hooks/exhaustive-deps

  const download = () => {
    if (!rep) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([rep.markdown], { type: "text/markdown" }));
    a.download = `${p}-field-report-${rep.id}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const restore = async () => {
    if (!rep?.markdown_original) return;
    try {
      await api(`/p/${p}/reports/${rep.id}/edit`, { method: "POST", json: { find: rep.markdown, replace: rep.markdown_original } });
      qc.invalidateQueries({ queryKey: [`/p/${p}/reports/${rep.id}`] });
      toast({ title: "Original report restored", tone: "success", ttl: 1800 });
    } catch (e: any) { toast({ title: "Could not restore", body: e.message, tone: "error" }); }
  };

  if (isLoading) return <PageSkeleton stats={false} />;
  if (error) return <ErrorBox error={error} />;
  const chatOpen = panel === "chat" && !!rep;
  // the window must hold the evidence pack, the report and a conversation
  const minContext = rep ? Math.round((rep.evidence_chars + rep.markdown.length) / 3.4) + 12_000 : 0;
  return (
    <div className="mx-auto max-w-[1600px]">
      <PageHeader title="Field report" sub="A strong model reads every product of this map and writes the intelligence that matters, each claim linked to its evidence">
        {rep && <button className="btn btn-sm" onClick={download}><Download className="size-3.5" /> Markdown</button>}
        {rep && <div className="seg text-[12px]">
          <button className={clsx(panel === "chat" && "on")} onClick={() => ask()} title={isMac ? "⌘L" : "Ctrl+L"}><MessageSquareText className="mr-1 inline size-3.5" />Chat</button>
          <button className={clsx(panel === "tools" && "on")} onClick={() => setPanel("tools")}><ListTree className="mr-1 inline size-3.5" />Reports</button>
        </div>}
      </PageHeader>
      <div className={clsx("grid gap-5", chatOpen ? "xl:grid-cols-[minmax(0,1fr)_440px]" : "xl:grid-cols-[minmax(0,1fr)_340px]")}>
        <div className="min-w-0">
          {!rid ? (
            <Card>
              <Empty title="No report yet" icon={<FileSearch className="size-6" />}>
                Choose a model and, if you like, a focus, then write the first report. It runs in the quick lane, so pipeline steps never hold it up.
              </Empty>
            </Card>
          ) : repLoading || !rep ? <PageSkeleton stats={false} /> : (
            <article ref={article} className="card px-6 py-6 md:px-10 md:py-8">
              <div className="mb-4 flex flex-wrap items-center gap-2 text-[12px]">
                <Pill tone="brand">{rep.model_label}</Pill>
                <span className="muted">{when(rep.created_at)} · data as of {rep.as_of}</span>
                {rep.focus && <span className="muted">· focus: “{rep.focus.length > 90 ? rep.focus.slice(0, 90) + "…" : rep.focus}”</span>}
                {!!rep.edits?.length && <Pill tone="amber" title="Changed from the chat">edited · {rep.edits.length}</Pill>}
              </div>
              <ReportMarkdown md={rep.markdown} className="report" />
              <div className="faint mt-8 border-t border-[var(--line)] pt-3 text-[11.5px]">
                {rep.model_slug} · {rep.usage?.prompt_tokens?.toLocaleString() ?? "?"} input / {rep.usage?.completion_tokens?.toLocaleString() ?? "?"} output tokens
                {rep.reasoning_tokens ? ` (${rep.reasoning_tokens.toLocaleString()} reasoning)` : ""}{rep.cost_usd != null ? ` · $${rep.cost_usd.toFixed(2)}` : ""}
                {rep.seconds ? ` · ${Math.round(rep.seconds / 60)} min` : ""} · evidence pack {Math.round(rep.evidence_chars / 1000)}k chars, sha {rep.evidence_sha256}.
                AI-written from this map only: open the linked papers and categories before acting on a claim.
                {!!rep.edits?.length && rep.markdown_original && rep.markdown !== rep.markdown_original && (
                  <button className="ml-2 inline-flex items-center gap-1 underline hover:text-[var(--text)]" onClick={restore}><RotateCcw className="size-3" />Restore the model's original</button>
                )}
              </div>
              <SelectionAsk root={article} onAsk={ask} />
            </article>
          )}
        </div>
        {chatOpen ? (
          <aside data-chat className="card flex h-[calc(100vh-7rem)] min-h-[480px] flex-col overflow-hidden xl:sticky xl:top-20">
            <ReportChat p={p} rid={rep!.id} report={rep!.markdown} quote={quote} setQuote={setQuote} focusTick={focusTick}
              onClose={() => setPanel("tools")} minContext={minContext} />
          </aside>
        ) : (
          <aside className="flex flex-col gap-5 xl:sticky xl:top-20 xl:self-start">
            <Generate p={p} models={models} job={job} onJob={(j) => setJobId(j.id)} />
            {toc.length > 1 && (
              <Card title="Contents">
                <nav className="flex flex-col gap-0.5 text-[13px]">
                  {toc.map((t) => (
                    <a key={t} href={`#${slug(t)}`} className="rounded-md px-2 py-1 hover:bg-[var(--surface-2)]"
                      onClick={(e) => { e.preventDefault(); document.getElementById(slug(t))?.scrollIntoView({ behavior: "smooth" }); }}>{t}</a>
                  ))}
                </nav>
              </Card>
            )}
            {list && list.length > 0 && (
              <Card title="Reports" subtitle={`${list.length} written for this project`}>
                <div className="scroll-thin flex max-h-80 flex-col gap-0.5 overflow-y-auto">
                  {list.map((r) => (
                    <button key={r.id} onClick={() => setSp((s) => { s.set("r", r.id); return s; })}
                      className={clsx("rounded-lg px-2.5 py-2 text-left hover:bg-[var(--surface-2)]", r.id === rid && "bg-[var(--nav-active)]")}>
                      <div className="line-clamp-2 text-[13px] font-medium">{r.title}</div>
                      <div className="faint mt-0.5 text-[11.5px]">{r.model_label} · {when(r.created_at)}{r.focus ? " · focused" : ""}</div>
                    </button>
                  ))}
                </div>
              </Card>
            )}
          </aside>
        )}
      </div>
    </div>
  );
}
