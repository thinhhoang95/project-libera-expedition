import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import {
  ArrowDown, ArrowDownAZ, ArrowUp, CalendarDays, Gavel, Library, ListOrdered, Quote, TrendingUp, type LucideIcon,
} from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { LS } from "../context";
import { PaperLink } from "./PaperDrawer";

// ------------------------------------------------------------------ year and citations for any list of papers

export type PaperMeta = { year: number | null; cites: number | null; cites_here: number; venue: string | null };

/** Year, citations overall and citations from this map for the given papers (one request per list). */
export function usePaperMeta(ids: (string | null | undefined)[]) {
  const { project } = useParams();
  const key = useMemo(() => [...new Set(ids.filter((x): x is string => !!x))].sort(), [ids.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps
  const { data } = useQuery({
    queryKey: ["paper-meta", project, key.join(",")],
    queryFn: () => api<Record<string, PaperMeta>>(`/p/${project}/paper-meta`, { method: "POST", json: { ids: key } }),
    enabled: !!project && key.length > 0,
    staleTime: 600_000,
  });
  return data ?? {};
}

// ------------------------------------------------------------------ sort options

export type SortOpt<T> = {
  key: string; label: string; icon: LucideIcon;
  get: (r: T) => any;
  /** the first click's direction: -1 = largest first (citations, newest), 1 = smallest first (rank, A–Z) */
  dir?: 1 | -1;
  /** what each direction means, for the tooltip: [ascending, descending] */
  dirs?: [string, string];
};

export const by = {
  rank: <T,>(get: (r: T) => any, label = "Ranked order"): SortOpt<T> => ({ key: "rank", label, icon: ListOrdered, get, dir: 1, dirs: ["as ranked", "reversed"] }),
  year: <T,>(get: (r: T) => any): SortOpt<T> => ({ key: "year", label: "Year", icon: CalendarDays, get, dir: -1, dirs: ["oldest first", "newest first"] }),
  citedHere: <T,>(get: (r: T) => any, label = "Cited in this map"): SortOpt<T> => ({ key: "here", label, icon: Quote, get, dir: -1, dirs: ["fewest first", "most first"] }),
  cites: <T,>(get: (r: T) => any): SortOpt<T> => ({ key: "cites", label: "Citations overall", icon: TrendingUp, get, dir: -1, dirs: ["fewest first", "most first"] }),
  title: <T,>(get: (r: T) => any, label = "A–Z"): SortOpt<T> => ({ key: "az", label, icon: ArrowDownAZ, get: (r) => String(get(r) ?? "").toLowerCase(), dir: 1, dirs: ["A to Z", "Z to A"] }),
  judge: <T,>(get: (r: T) => any): SortOpt<T> => ({ key: "judge", label: "Judge score", icon: Gavel, get, dir: -1, dirs: ["lowest first", "highest first"] }),
  count: <T,>(key: string, label: string, get: (r: T) => any, icon: LucideIcon = Library): SortOpt<T> => ({ key, label, icon, get, dir: -1, dirs: ["fewest first", "most first"] }),
};

type State = { key: string; dir: 1 | -1 };

/**
 * Sorts a list by one of `opts`; the first option is the list's own order. `id` remembers the choice per list
 * (across visits). Returns the sorted rows and the bar of icon buttons to show above the list.
 */
export function useSortedList<T>(rows: T[], opts: SortOpt<T>[], id?: string) {
  const first = opts[0];
  const [s, setS] = useState<State>(() => {
    try {
      const saved = id ? JSON.parse(LS.get(`libera.sort.${id}`) ?? "null") : null;
      if (saved && opts.some((o) => o.key === saved.key)) return saved;
    } catch { /* ignore */ }
    return { key: first.key, dir: first.dir ?? 1 };
  });
  const set = (n: State) => {
    setS(n);
    if (id) LS.set(`libera.sort.${id}`, JSON.stringify(n));
  };
  const sorted = useMemo(() => {
    const o = opts.find((x) => x.key === s.key) ?? first;
    return rows.map((r, i) => [r, i] as const).sort(([a, i], [b, j]) => {
      const x = o.get(a), y = o.get(b);
      if (x == null && y == null) return i - j;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x > y ? 1 : x < y ? -1 : i - j) * s.dir;
    }).map(([r]) => r);
  }, [rows, s, opts]); // eslint-disable-line react-hooks/exhaustive-deps
  const bar = <SortBar opts={opts} value={s} onChange={set} />;
  return { sorted, bar, sort: s };
}

/** Icon buttons of a fixed size (a second click on the active one reverses it), then the current order in words. */
export function SortBar<T>({ opts, value, onChange, className }: { opts: SortOpt<T>[]; value: State; onChange: (s: State) => void; className?: string }) {
  const cur = opts.find((o) => o.key === value.key) ?? opts[0];
  const words = cur.dirs ? cur.dirs[value.dir === 1 ? 0 : 1] : "";
  const flip = () => onChange({ key: cur.key, dir: (value.dir * -1) as 1 | -1 });
  return (
    <div className={clsx("inline-flex max-w-full items-center gap-0.5 rounded-lg border border-[var(--line)] bg-[var(--surface-2)] p-0.5", className)} role="group" aria-label="Sort">
      <span className="faint px-1 text-[10.5px] font-semibold tracking-wide uppercase">Sort</span>
      {opts.map((o) => {
        const on = value.key === o.key;
        const I = o.icon;
        return (
          <button key={o.key} type="button" title={on ? `${o.label}: ${words}. Click to reverse` : `Sort by ${o.label.toLowerCase()}`}
            aria-pressed={on} aria-label={`Sort by ${o.label.toLowerCase()}`}
            onClick={() => (on ? flip() : onChange({ key: o.key, dir: o.dir ?? 1 }))}
            className={clsx("grid size-6 shrink-0 place-items-center rounded-md transition-colors",
              on ? "bg-[var(--surface)] text-[var(--chip-text)] shadow-sm" : "text-[var(--muted)] hover:text-[var(--text-strong)]")}>
            <I className="size-3.5" />
          </button>
        );
      })}
      <button type="button" onClick={flip} title="Reverse the order"
        className="ml-0.5 inline-flex min-w-0 items-center gap-1 rounded-md px-1.5 text-[11.5px] text-[var(--text)] hover:text-[var(--chip-text)]">
        <span className="truncate"><span className="font-medium">{cur.label}</span>{words && <span className="faint">, {words}</span>}</span>
        {value.dir === -1 ? <ArrowDown className="size-3 shrink-0" /> : <ArrowUp className="size-3 shrink-0" />}
      </button>
    </div>
  );
}

// ------------------------------------------------------------------ a sortable list of papers

type PRow = { paper_id: string; title: string | null; year?: number | null; [k: string]: any };

/**
 * Papers as a list (year, title, citations in this map) that the reader can re-sort: the list's own order first
 * (`orderLabel`), then year, citations in this map, citations overall and title. `extra` adds to each row.
 */
export function PaperList<T extends PRow>({ rows, id, orderLabel = "As listed", extra, numbered, className, max, dense, barClass }: {
  rows: T[]; id: string; orderLabel?: string; extra?: (r: T) => ReactNode; numbered?: boolean; className?: string; max?: number;
  dense?: boolean; barClass?: string;
}) {
  const meta = usePaperMeta(rows.map((r) => r.paper_id));
  const ranked = useMemo(() => rows.map((r, i) => ({ r, i: i + 1 })), [rows]);
  type R = { r: T; i: number };
  const { sorted, bar } = useSortedList<R>(ranked, [
    by.rank<R>((x) => x.i, orderLabel),
    by.year<R>((x) => x.r.year ?? meta[x.r.paper_id]?.year),
    by.citedHere<R>((x) => meta[x.r.paper_id]?.cites_here),
    by.cites<R>((x) => meta[x.r.paper_id]?.cites),
    by.title<R>((x) => x.r.title),
  ], id);
  if (!rows.length) return null;
  return (
    <div className={className}>
      {rows.length > 1 && <div className={clsx("mb-1 flex justify-end", barClass)}>{bar}</div>}
      <ol className="flex flex-col">
        {sorted.slice(0, max).map(({ r, i }) => {
          const m = meta[r.paper_id];
          return (
            <li key={r.paper_id} className={clsx("flex gap-2.5 border-b border-[var(--line)] last:border-0", dense ? "py-1 text-[12.5px]" : "py-1.5 text-[13px]")}>
              {numbered && <span className="faint w-5 shrink-0 text-right tabular-nums">{i}</span>}
              <span className="faint w-10 shrink-0 tabular-nums">{r.year ?? m?.year ?? "—"}</span>
              <span className="min-w-0 flex-1"><PaperLink pid={r.paper_id} title={r.title ?? r.paper_id} />{extra?.(r)}</span>
              {m && <span className="faint shrink-0 text-right text-[11px] tabular-nums leading-tight" title="Citations from this map · overall (OpenAlex)">
                {m.cites_here} here<br />{(m.cites ?? 0).toLocaleString()} all</span>}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
