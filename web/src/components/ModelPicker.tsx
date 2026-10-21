import clsx from "clsx";
import { Check, ChevronDown, Database, Loader2, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useApi } from "../api";
import { LS } from "../context";

export type OrModel = {
  id: string; name: string; created: number | null; context: number | null; max_output: number | null;
  in: number | null; out: number | null; cache_read: number | null; cache_write: number | null;
  reasoning: boolean; reasoning_mandatory: boolean; efforts: string[]; intelligence: number | null;
};

export function useModels() {
  return useApi<OrModel[]>("/openrouter/models", { staleTime: 3_600_000 });
}

/** A model choice kept per viewer (browser storage), with a default from the server. */
export function useStoredModel(key: string, fallback: string | undefined): [string | undefined, (s: string) => void] {
  const [v, setV] = useState<string | null>(() => LS.get(key));
  return [v ?? fallback, (s: string) => { LS.set(key, s); setV(s); }];
}

const short = (name: string) => name.replace(/^[^:]+:\s*/, "");
const price = (v: number | null) => (v == null ? "?" : v === 0 ? "free" : v < 0.1 ? `$${v.toFixed(3)}` : `$${v < 10 ? v.toFixed(2) : v.toFixed(0)}`);
const ctx = (n: number | null) => (n == null ? "" : n >= 1e6 ? `${(n / 1e6).toFixed(n % 1e6 ? 2 : 0)}M` : `${Math.round(n / 1000)}k`);

export function modelLabel(models: OrModel[] | undefined, id?: string) {
  const m = models?.find((x) => x.id === id);
  return m ? short(m.name) : id ?? "";
}

/**
 * Any OpenRouter model, searchable: suggested ones first, then the catalogue (newest first). Shows price per
 * million tokens (input / output), context and whether the model's provider discounts cached input.
 * `minContext` greys out models whose window cannot hold the prompt.
 */
export function ModelPicker({ value, onChange, suggested = [], minContext, compact, align = "left" }: {
  value?: string; onChange: (id: string) => void; suggested?: string[]; minContext?: number; compact?: boolean; align?: "left" | "right";
}) {
  const { data: models, isLoading } = useModels();
  const [open, setOpen] = useState(false);
  const [term, setTerm] = useState("");
  const [sel, setSel] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const cur = models?.find((m) => m.id === value);

  const rows = useMemo(() => {
    if (!models) return [];
    const t = term.trim().toLowerCase();
    const hit = (m: OrModel) => !t || t.split(/\s+/).every((w) => m.id.toLowerCase().includes(w) || m.name.toLowerCase().includes(w));
    const sug = suggested.map((s) => models.find((m) => m.id === s)).filter((m): m is OrModel => !!m && hit(m));
    const rest = models.filter((m) => !suggested.includes(m.id) && hit(m));
    return [...sug.map((m) => ({ m, sug: true })), ...rest.slice(0, 300).map((m) => ({ m, sug: false }))];
  }, [models, term, suggested]);

  useEffect(() => {
    if (!open) return;
    setTerm(""); setSel(0);
    setTimeout(() => input.current?.focus(), 0);
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  useEffect(() => { list.current?.querySelector<HTMLElement>(`[data-i="${sel}"]`)?.scrollIntoView({ block: "nearest" }); }, [sel]);

  const pick = (id: string) => { onChange(id); setOpen(false); };
  return (
    <div ref={box} className="relative min-w-0">
      <button type="button" onClick={() => setOpen(!open)} title={value}
        className={clsx("flex w-full min-w-0 items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--surface)] text-left hover:border-[var(--line-strong)]",
          compact ? "h-7 px-2 text-[12px]" : "h-9 px-2.5 text-[13px]")}>
        <span className="min-w-0 flex-1 truncate font-medium">{cur ? short(cur.name) : value ?? "Choose a model"}</span>
        {cur && !compact && <span className="faint shrink-0 text-[11px]">{price(cur.in)} / {price(cur.out)}</span>}
        <ChevronDown className="faint size-3.5 shrink-0" />
      </button>
      {open && (
        <div className={clsx("absolute z-50 mt-1 w-[min(440px,90vw)] overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface)] shadow-xl", align === "right" ? "right-0" : "left-0")}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(rows.length - 1, s + 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
            else if (e.key === "Enter" && rows[sel]) { e.preventDefault(); pick(rows[sel].m.id); }
            else if (e.key === "Escape") { e.stopPropagation(); setOpen(false); }
          }}>
          <div className="flex items-center gap-2 border-b border-[var(--line)] px-3">
            {isLoading ? <Loader2 className="faint size-4 animate-spin" /> : <Search className="faint size-4" />}
            <input ref={input} className="h-10 w-full bg-transparent text-[13px] outline-none" placeholder="Search OpenRouter models… (name or slug)"
              value={term} onChange={(e) => { setTerm(e.target.value); setSel(0); }} />
          </div>
          <div ref={list} className="scroll-thin max-h-[360px] overflow-y-auto py-1">
            {rows.length === 0 && !isLoading && (
              <div className="muted px-3 py-3 text-[12.5px]">
                No match.{/^[\w.~-]+\/[\w.:-]+$/.test(term.trim()) && <> <button className="cat-link underline" onClick={() => pick(term.trim())}>Use “{term.trim()}”</button></>}
              </div>
            )}
            {rows.map(({ m, sug }, i) => {
              const small = minContext != null && m.context != null && m.context < minContext;
              return (
                <div key={m.id}>
                  {(i === 0 && sug) && <div className="faint px-3 pt-1.5 pb-1 text-[10.5px] font-semibold tracking-wide uppercase">Suggested</div>}
                  {!sug && (i === 0 || rows[i - 1].sug) && <div className="faint px-3 pt-2 pb-1 text-[10.5px] font-semibold tracking-wide uppercase">All models · newest first</div>}
                  <button data-i={i} onMouseEnter={() => setSel(i)} onClick={() => pick(m.id)}
                    className={clsx("flex w-full items-start gap-2 px-3 py-1.5 text-left", i === sel && "bg-[var(--surface-2)]", small && "opacity-45")}>
                    <span className="mt-0.5 w-3.5 shrink-0">{m.id === value && <Check className="size-3.5 text-brand-600" />}</span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] font-medium">{short(m.name)}</span>
                      <span className="faint block truncate font-mono text-[10.5px]">{m.id}</span>
                    </span>
                    <span className="shrink-0 text-right text-[11px] leading-tight">
                      <span className="block">{price(m.in)} <span className="faint">/</span> {price(m.out)}</span>
                      <span className="faint flex items-center justify-end gap-1">
                        {m.cache_read != null && m.in ? <span title={`Cached input ${price(m.cache_read)}/M`} className="inline-flex items-center gap-0.5 text-emerald-600 dark:text-emerald-400"><Database className="size-2.5" />{Math.round((1 - m.cache_read / m.in) * 100)}%</span> : null}
                        {ctx(m.context)}{small && " · too small"}
                      </span>
                    </span>
                  </button>
                </div>
              );
            })}
          </div>
          <div className="faint border-t border-[var(--line)] px-3 py-1.5 text-[10.5px]">Price per million tokens, input / output · <Database className="inline size-2.5" /> discount on cached input · ↑↓ Enter</div>
        </div>
      )}
    </div>
  );
}
