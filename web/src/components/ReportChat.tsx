import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowUp, Brain, Check, ChevronDown, ChevronRight, Copy, Database, History, Loader2, Pencil, Plus, Quote, RotateCcw, Square, Undo2, X } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, useApi } from "../api";
import { ModelPicker, modelLabel, useModels } from "./ModelPicker";
import { ReportMarkdown } from "./ReportMarkdown";
import { copyText, toast } from "./Toasts";

type Usage = { model: string; provider?: string; prompt_tokens?: number; completion_tokens?: number; cached_tokens?: number; cache_write_tokens?: number; reasoning_tokens?: number; cost?: number };
type EditState = "applied" | "rejected" | "undone";
export type Msg = { role: "user" | "assistant"; content: string; quote?: string; model?: string; usage?: Usage; reasoning?: string; edits?: Record<number, EditState>; error?: string; ms?: number };
type Chat = { id: string; title: string; model: string; created_at: string; updated_at?: string; messages: Msg[] };

const SUGGEST = [
  "What should I do first, and why?",
  "Which claims in this report are weakest against the evidence?",
  "Explain the top opportunity to a newcomer, with the papers to read",
  "What can this map not see?",
];
const newId = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
const k = (n?: number) => (n == null ? "?" : n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n));
const usd = (v?: number) => (v == null ? "" : v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(3)}`);

// ------------------------------------------------------------------ edit blocks

type Seg = { kind: "text"; text: string } | { kind: "edit"; find: string; replace: string; i: number; done: boolean };
const EDIT_RE = /```report-edit\s*\n([\s\S]*?)(```|$)/g;

/** Splits an answer into text and proposed edits; an edit still streaming is marked not done. */
function segments(md: string): Seg[] {
  const out: Seg[] = [];
  let last = 0, i = 0;
  for (const m of md.matchAll(EDIT_RE)) {
    if (m.index! > last) out.push({ kind: "text", text: md.slice(last, m.index) });
    const body = m[1];
    const f = body.match(/<<<<<<< FIND\n([\s\S]*?)\n=======\n([\s\S]*?)(\n>>>>>>> REPLACE|$)/);
    out.push({ kind: "edit", find: f?.[1] ?? "", replace: f?.[2] ?? "", i: i++, done: m[2] === "```" && !!f?.[3] });
    last = m.index! + m[0].length;
  }
  if (last < md.length) out.push({ kind: "text", text: md.slice(last) });
  return out;
}

function EditCard({ seg, state, report, onApply, onReject, onUndo }: {
  seg: Extract<Seg, { kind: "edit" }>; state?: EditState; report: string;
  onApply: () => Promise<void>; onReject: () => void; onUndo: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const found = seg.done && report.split(seg.find).length - 1;
  const run = async (f: () => Promise<void>) => { setBusy(true); try { await f(); } finally { setBusy(false); } };
  return (
    <div className="my-2 overflow-hidden rounded-lg border border-[var(--line)] text-[12.3px]">
      <div className="flex items-center gap-2 border-b border-[var(--line)] bg-[var(--surface-2)] px-2.5 py-1.5">
        <Pencil className="size-3.5 text-brand-600" />
        <span className="font-medium">Edit to the report</span>
        {!seg.done ? <span className="faint flex items-center gap-1"><Loader2 className="size-3 animate-spin" />drafting…</span>
          : state === "applied" ? <span className="flex items-center gap-1 text-emerald-600"><Check className="size-3.5" />applied</span>
          : state === "rejected" ? <span className="faint">rejected</span>
          : state === "undone" ? <span className="faint">undone</span>
          : found !== 1 ? <span className="text-amber-600">{found === 0 ? "passage not found in the report" : `passage occurs ${found}×`}</span> : null}
        <span className="ml-auto flex gap-1">
          {seg.done && !state && found === 1 && <>
            <button className="btn btn-sm h-6 px-2" disabled={busy} onClick={() => run(onApply)}>{busy ? <Loader2 className="size-3 animate-spin" /> : <Check className="size-3" />}Apply</button>
            <button className="btn btn-sm h-6 px-2" onClick={onReject}><X className="size-3" />Reject</button>
          </>}
          {state === "applied" && seg.replace && <button className="btn btn-sm h-6 px-2" disabled={busy} onClick={() => run(onUndo)}><Undo2 className="size-3" />Undo</button>}
        </span>
      </div>
      <div className={clsx("scroll-thin max-h-72 overflow-y-auto font-mono text-[11.5px] leading-relaxed", (state === "rejected" || state === "undone") && "opacity-50")}>
        {seg.find && <div className="whitespace-pre-wrap bg-red-500/8 px-2.5 py-1.5 text-red-800 line-through decoration-red-400/50 dark:text-red-300">{seg.find}</div>}
        {seg.replace && <div className="whitespace-pre-wrap bg-emerald-500/8 px-2.5 py-1.5 text-emerald-900 dark:text-emerald-200">{seg.replace}</div>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ one answer

function Thought({ text, tokens, live }: { text: string; tokens?: number; live: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-1.5">
      <button onClick={() => setOpen(!open)} className="faint flex items-center gap-1 text-[11.5px] hover:text-[var(--text)]">
        {open ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
        <Brain className={clsx("size-3", live && "animate-pulse")} />
        {live ? "Thinking…" : `Thought${tokens ? ` · ${k(tokens)} tokens` : ""}`}
      </button>
      {open && <div className="faint mt-1 max-h-48 overflow-y-auto border-l-2 border-[var(--line)] pl-2.5 text-[11.5px] leading-relaxed whitespace-pre-wrap">{text}</div>}
    </div>
  );
}

function Answer({ m, live, report, onEdit }: { m: Msg; live: boolean; report: string; onEdit: (i: number, s: EditState, seg: { find: string; replace: string }) => Promise<void> }) {
  const segs = useMemo(() => segments(m.content), [m.content]);
  const u = m.usage;
  return (
    <div className="group min-w-0">
      {m.reasoning && <Thought text={m.reasoning} tokens={u?.reasoning_tokens} live={live && !m.content} />}
      {live && !m.content && !m.reasoning && <div className="faint flex items-center gap-1.5 text-[12px]"><Loader2 className="size-3.5 animate-spin" />Reading the report…</div>}
      {segs.map((s, i) => s.kind === "text"
        ? <ReportMarkdown key={i} md={s.text} className="chat-md" />
        : <EditCard key={i} seg={s} state={m.edits?.[s.i]} report={report}
            onApply={() => onEdit(s.i, "applied", s)} onReject={() => onEdit(s.i, "rejected", s)} onUndo={() => onEdit(s.i, "undone", s)} />)}
      {live && m.content && <span className="chat-caret" />}
      {m.error && <div className="mt-1 rounded-lg bg-red-500/10 px-2.5 py-1.5 text-[12px] text-red-700 dark:text-red-300">{m.error}</div>}
      {u && !live && (
        <div className="faint mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10.5px] opacity-0 transition-opacity group-hover:opacity-100 group-last:opacity-100">
          <span>{u.provider ?? ""}</span>
          <span title="input tokens">{k(u.prompt_tokens)} in</span>
          {(u.cached_tokens ?? 0) > 0 && <span className="flex items-center gap-0.5 text-emerald-600 dark:text-emerald-400" title="read from the prompt cache"><Database className="size-2.5" />{k(u.cached_tokens)} cached</span>}
          {(u.cache_write_tokens ?? 0) > 0 && <span title="written to the prompt cache">{k(u.cache_write_tokens)} cache write</span>}
          <span>{k(u.completion_tokens)} out</span>
          {u.cost != null && <span>{usd(u.cost)}</span>}
          {m.ms != null && <span>{(m.ms / 1000).toFixed(1)}s</span>}
          <button className="ml-auto hover:text-[var(--text)]" title="Copy" onClick={() => copyText(m.content, "Answer")}><Copy className="size-3" /></button>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ the panel

export function ReportChat({ p, rid, report, quote, setQuote, focusTick, onClose, minContext }: {
  p: string; rid: string; report: string; quote: string | null; setQuote: (q: string | null) => void; focusTick: number;
  onClose: () => void; minContext: number;
}) {
  const qc = useQueryClient();
  const { data: defaults } = useApi<{ default_chat: string }>("/report-models");
  const { data: models } = useModels();
  const [model, setModelRaw] = useState<string | undefined>(() => { try { return localStorage.getItem("libera.chatModel") ?? undefined; } catch { return undefined; } });
  const setModel = (s: string) => { try { localStorage.setItem("libera.chatModel", s); } catch { /* private mode */ } setModelRaw(s); };
  const chatModel = model ?? defaults?.default_chat;
  const { data: saved } = useApi<Chat[]>(`/p/${p}/reports/${rid}/chats`, { staleTime: 0 });
  const [chat, setChat] = useState<Chat | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [showHist, setShowHist] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const ta = useRef<HTMLTextAreaElement>(null);

  // open the latest conversation of this report, or a fresh one
  useEffect(() => { setChat(null); stick.current = true; }, [rid]);
  useEffect(() => {
    if (chat === null && saved) setChat(saved[0] ?? null);
  }, [saved]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { ta.current?.focus(); }, [focusTick, quote]);
  useLayoutEffect(() => {
    const el = ta.current;
    if (el) { el.style.height = "auto"; el.style.height = Math.min(220, el.scrollHeight) + "px"; }
  }, [input]);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [chat]);

  const persist = (c: Chat) => api(`/p/${p}/reports/${rid}/chats`, { method: "PUT", json: c })
    .then(() => qc.invalidateQueries({ queryKey: [`/p/${p}/reports/${rid}/chats`] })).catch(() => {});

  async function send(text: string, base?: Chat) {
    if (!text.trim() || busy || !chatModel) return;
    const c1: Chat = base ?? chat ?? { id: newId(), title: "", model: chatModel, created_at: new Date().toISOString(), messages: [] };
    const c0 = c1.title ? c1 : { ...c1, title: text.trim().slice(0, 70) };
    const user: Msg = { role: "user", content: text.trim(), ...(quote ? { quote } : {}) };
    const draft: Msg = { role: "assistant", content: "", model: chatModel };
    let c: Chat = { ...c0, model: chatModel, messages: [...c0.messages, user, draft] };
    setChat(c); setInput(""); setQuote(null); setBusy(true); stick.current = true;
    const t0 = performance.now();
    const ctl = new AbortController();
    abort.current = ctl;
    const update = (f: (m: Msg) => Msg) => {
      const msgs = c.messages.slice();
      msgs[msgs.length - 1] = f(msgs[msgs.length - 1]);
      c = { ...c, messages: msgs };
      setChat(c);
    };
    try {
      const res = await fetch(`/api/p/${p}/reports/${rid}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: ctl.signal,
        body: JSON.stringify({ model: chatModel, session: c.id, messages: c.messages.slice(0, -1).filter((m) => !m.error).map(({ role, content, quote }) => ({ role, content, quote })) }),
      });
      if (!res.ok || !res.body) {
        let msg = res.statusText;
        try { msg = (await res.json()).detail; } catch { /* not JSON */ }
        throw new Error(msg);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "", text = "", reasoning = "", pending = false;
      // batch updates to one per animation frame: smooth at any token rate
      const flush = () => { pending = false; update((m) => ({ ...m, content: text, reasoning: reasoning || undefined })); };
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          const ev = JSON.parse(part.slice(6));
          if (ev.type === "delta") text += ev.text;
          else if (ev.type === "reasoning") reasoning += ev.text;
          else if (ev.type === "usage") { const { type: _t, ...u } = ev; update((m) => ({ ...m, content: text, reasoning: reasoning || undefined, usage: u })); }
          else if (ev.type === "error") update((m) => ({ ...m, content: text, error: ev.message }));
        }
        if (!pending) { pending = true; requestAnimationFrame(flush); }
      }
      flush();
    } catch (e: any) {
      update((m) => ({ ...m, error: e.name === "AbortError" ? undefined : String(e.message ?? e), content: m.content || (e.name === "AbortError" ? "*Stopped.*" : "") }));
    } finally {
      update((m) => ({ ...m, ms: performance.now() - t0 }));
      setBusy(false);
      abort.current = null;
      persist(c);
    }
  }

  const onEdit = async (msgIdx: number, i: number, s: EditState, seg: { find: string; replace: string }) => {
    try {
      if (s === "applied") await api(`/p/${p}/reports/${rid}/edit`, { method: "POST", json: seg });
      if (s === "undone") await api(`/p/${p}/reports/${rid}/edit`, { method: "POST", json: { find: seg.replace, replace: seg.find } });
    } catch (e: any) {
      toast({ title: "Could not change the report", body: e.message, tone: "error" });
      return;
    }
    if (s !== "rejected") qc.invalidateQueries({ queryKey: [`/p/${p}/reports/${rid}`] });
    const msgs = chat!.messages.slice();
    msgs[msgIdx] = { ...msgs[msgIdx], edits: { ...msgs[msgIdx].edits, [i]: s } };
    const c = { ...chat!, messages: msgs };
    setChat(c); persist(c);
    if (s === "applied") toast({ title: "Report updated", tone: "success", ttl: 1800 });
  };

  const regenerate = () => {
    if (!chat || busy) return;
    const msgs = chat.messages.slice();
    while (msgs.length && msgs[msgs.length - 1].role === "assistant") msgs.pop();
    const last = msgs.pop();
    if (!last) return;
    setQuote(last.quote ?? null);
    send(last.content, { ...chat, messages: msgs });
  };

  const total = (chat?.messages ?? []).reduce((a, m) => a + (m.usage?.cost ?? 0), 0);
  const cached = (chat?.messages ?? []).reduce((a, m) => a + (m.usage?.cached_tokens ?? 0), 0);
  const msgs = chat?.messages ?? [];
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-1.5 border-b border-[var(--line)] px-3 py-2">
        <div className="min-w-0 flex-1"><ModelPicker compact value={chatModel} onChange={setModel} minContext={minContext}
          suggested={["deepseek/deepseek-v4.1-flash", "google/gemini-3.6-flash", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4-pro", "anthropic/claude-sonnet-5.5"]} /></div>
        <div className="relative">
          <button className="btn btn-sm h-7 px-2" title="Earlier conversations" onClick={() => setShowHist(!showHist)}><History className="size-3.5" /></button>
          {showHist && (
            <div className="absolute right-0 z-40 mt-1 w-72 rounded-xl border border-[var(--line)] bg-[var(--surface)] p-1 shadow-xl" onMouseLeave={() => setShowHist(false)}>
              {!saved?.length && <div className="muted px-2 py-2 text-[12px]">No conversations yet.</div>}
              {saved?.map((s) => (
                <button key={s.id} onClick={() => { setChat(s); setShowHist(false); stick.current = true; }}
                  className={clsx("block w-full rounded-lg px-2 py-1.5 text-left hover:bg-[var(--surface-2)]", s.id === chat?.id && "bg-[var(--nav-active)]")}>
                  <div className="truncate text-[12.5px]">{s.title}</div>
                  <div className="faint text-[10.5px]">{s.messages.length} messages · {modelLabel(models, s.model)} · {(s.updated_at ?? s.created_at).slice(0, 16).replace("T", " ")}</div>
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="btn btn-sm h-7 px-2" title="New conversation" disabled={busy} onClick={() => { setChat({ id: newId(), title: "", model: chatModel ?? "", created_at: new Date().toISOString(), messages: [] }); ta.current?.focus(); }}><Plus className="size-3.5" /></button>
        <button className="btn btn-sm h-7 px-2" title="Close (⌘L)" onClick={onClose}><X className="size-3.5" /></button>
      </div>

      <div ref={scroller} className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-3"
        onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40; }}>
        {msgs.length === 0 ? (
          <div className="flex h-full flex-col justify-end gap-1.5 pb-2">
            <div className="faint mb-1 text-[12px]">Ask about this report. The assistant reads the report and the full evidence pack behind it; select a passage in the report and press <kbd className="key">⌘</kbd><kbd className="key">L</kbd> to ask about it, or ask it to rewrite a section.</div>
            {SUGGEST.map((s) => (
              <button key={s} onClick={() => send(s)} className="rounded-lg border border-[var(--line)] px-3 py-2 text-left text-[12.5px] hover:border-[var(--line-strong)] hover:bg-[var(--surface-2)]">{s}</button>
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {msgs.map((m, i) => m.role === "user" ? (
              <div key={i} className="self-end max-w-[92%] rounded-xl bg-[var(--surface-2)] px-3 py-2 text-[13px]">
                {m.quote && <div className="faint mb-1.5 line-clamp-3 border-l-2 border-brand-500 pl-2 text-[11.5px]">{m.quote}</div>}
                <div className="whitespace-pre-wrap">{m.content}</div>
              </div>
            ) : (
              <Answer key={i} m={m} live={busy && i === msgs.length - 1} report={report}
                onEdit={(ei, s, seg) => onEdit(i, ei, s, seg)} />
            ))}
            {!busy && msgs.length > 0 && msgs[msgs.length - 1].role === "assistant" && (
              <button className="faint -mt-2 flex items-center gap-1 self-start text-[11px] hover:text-[var(--text)]" onClick={regenerate}><RotateCcw className="size-3" />Regenerate</button>
            )}
          </div>
        )}
      </div>

      <div className="border-t border-[var(--line)] p-3">
        {quote && (
          <div className="mb-2 flex items-start gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[11.5px]">
            <Quote className="mt-0.5 size-3 shrink-0 text-brand-600" />
            <span className="line-clamp-3 min-w-0 flex-1">{quote}</span>
            <button className="faint hover:text-[var(--text)]" onClick={() => setQuote(null)}><X className="size-3" /></button>
          </div>
        )}
        <div className="flex items-end gap-2 rounded-xl border border-[var(--line)] bg-[var(--surface)] px-3 py-2 focus-within:border-brand-500">
          <textarea ref={ta} rows={1} value={input} onChange={(e) => setInput(e.target.value)}
            placeholder={quote ? "Ask about this passage, or say how to change it…" : "Ask anything about the report…"}
            className="max-h-[220px] min-h-[22px] flex-1 resize-none bg-transparent text-[13px] leading-relaxed outline-none"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(input); }
              if (e.key === "Escape" && busy) { e.preventDefault(); abort.current?.abort(); }
            }} />
          {busy
            ? <button className="grid size-7 shrink-0 place-items-center rounded-lg bg-[var(--text-strong)] text-[var(--surface)]" title="Stop (Esc)" onClick={() => abort.current?.abort()}><Square className="size-3" fill="currentColor" /></button>
            : <button className="grid size-7 shrink-0 place-items-center rounded-lg bg-brand-600 text-white disabled:opacity-40" title="Send (Enter)" disabled={!input.trim() || !chatModel} onClick={() => send(input)}><ArrowUp className="size-4" /></button>}
        </div>
        <div className="faint mt-1.5 flex items-center gap-2 text-[10.5px]">
          <span>Enter to send · Shift+Enter new line</span>
          {total > 0 && <span className="ml-auto">{usd(total)} this conversation{cached > 0 && <> · <Database className="inline size-2.5 text-emerald-600" /> {k(cached)} cached</>}</span>}
        </div>
      </div>
    </div>
  );
}
