import clsx from "clsx";
import { Pause, Play } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useCategorySheet } from "../context";
import { FACET_META, windowStats, type Category, type FacetBlock, type FacetKey } from "../facets";
import { DataTable } from "./DataTable";
import { Plot } from "./Plot";
import { Sparkline, pct, signedPct } from "./ui";

// ------------------------------------------------------------------ time seeker

/** A year window: one track with two thumbs, presets and a player that steps one year at a time. */
export function TimeSeeker({ y0, y1, value, onChange, asOf, compact, counts }: {
  y0: number; y1: number; value: [number, number]; onChange: (w: [number, number]) => void; asOf?: number; compact?: boolean;
  counts?: number[];
}) {
  const [t1, t2] = value;
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const h = setInterval(() => {
      // a moving one-year window; stops at the last year
      onChange(t2 >= y1 ? [y1, y1] : [t2 + 1, t2 + 1]);
      if (t2 >= y1) setPlaying(false);
    }, 900);
    return () => clearInterval(h);
  }, [playing, t2, y1, onChange]);
  const span = Math.max(y1 - y0, 1);
  const left = ((t1 - y0) / span) * 100, right = ((t2 - y0) / span) * 100;
  const T = asOf ?? y1;
  const presets: [string, [number, number]][] = [
    ["All", [y0, y1]], [`Last 3 yrs`, [Math.max(y0, T - 2), T]], [`Last 5`, [Math.max(y0, T - 4), T]], [`${T}`, [T, T]],
  ];
  const mx = counts ? Math.max(...counts, 1) : 0;
  return (
    <div className={clsx("flex flex-wrap items-center gap-x-3 gap-y-2", compact ? "text-[12px]" : "text-[12.5px]")}>
      <button className="btn btn-sm" title={playing ? "Stop" : "Play year by year"}
        onClick={() => { if (!playing && t2 >= y1) onChange([y0, y0]); setPlaying(!playing); }}>
        {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
      </button>
      <span className="h-title w-[84px] shrink-0 tabular-nums">{t1 === t2 ? t1 : `${t1}–${t2}`}</span>
      <div className="min-w-[180px] flex-1">
        {counts && (
          <div className="flex h-5 items-end gap-px px-[6px]">
            {counts.map((n, i) => (
              <div key={i} className={clsx("flex-1 rounded-t-sm", y0 + i >= t1 && y0 + i <= t2 ? "bg-brand-400/70" : "bg-[var(--line-strong)]")}
                style={{ height: `${Math.max(6, (n / mx) * 100)}%` }} title={`${y0 + i}: ${n}`} />
            ))}
          </div>
        )}
        <div className="dual-range">
          <div className="track" />
          <div className="fill" style={{ left: `${left}%`, width: `${Math.max(right - left, 0.8)}%` }} />
          <input type="range" min={y0} max={y1} value={t1} aria-label="Window start"
            onChange={(e) => onChange([Math.min(Number(e.target.value), t2), t2])} />
          <input type="range" min={y0} max={y1} value={t2} aria-label="Window end"
            onChange={(e) => onChange([t1, Math.max(Number(e.target.value), t1)])} />
        </div>
        <div className="faint flex justify-between text-[10.5px] tabular-nums"><span>{y0}</span><span>{y1}</span></div>
      </div>
      <div className="flex gap-1">
        {presets.map(([l, w]) => (
          <button key={l} className={clsx("btn btn-ghost btn-sm !px-2", w[0] === t1 && w[1] === t2 && "!bg-[var(--chip)] !text-[var(--chip-text)]")}
            onClick={() => { setPlaying(false); onChange(w); }}>{l}</button>
        ))}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ links and badges

/** A category label that opens its details sheet. */
export function CatLink({ cid, label, className, withCid, children }: {
  cid?: string | null; label?: ReactNode; className?: string; withCid?: string | null; children?: ReactNode;
}) {
  const { open } = useCategorySheet();
  if (!cid) return <span className={className}>{children ?? label}</span>;
  return (
    <button className={clsx("cat-link", className)} title="Open the details sheet"
      onClick={(e) => { e.stopPropagation(); open(cid, withCid); }}>
      {children ?? label}
    </button>
  );
}

export function FacetBadge({ facet, className }: { facet: FacetKey; className?: string }) {
  const M = FACET_META[facet];
  if (!M) return null;
  const I = M.icon;
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-md px-1.5 py-[1px] text-[11px] font-semibold", className)}
      style={{ color: M.color, background: `${M.color}18` }}>
      <I className="size-3" /> {M.label}
    </span>
  );
}

export const Dot = ({ color }: { color?: string | null }) => (
  <span className="inline-block size-2.5 shrink-0 rounded-full" style={{ background: color ?? "#9aa0a6" }} />
);

/** A community select with a link to the chosen community's details sheet. */
export function CommunityPicker({ value, onChange, options, className }: {
  value: string; onChange: (v: string) => void; options: [string, string][]; className?: string;
}) {
  return (
    <div className={clsx("mb-2 flex items-center gap-2", className)}>
      <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map(([id, l]) => <option key={id} value={id}>{l}</option>)}
      </select>
      <CatLink cid={value} className="btn btn-sm shrink-0 !no-underline">Details</CatLink>
    </div>
  );
}

// ------------------------------------------------------------------ category table (Start here, Explore)

export type CatRow = Category & { n: number; share: number; change: number | null; series: number[]; rank: number };

export function catRows(b: FacetBlock, y0: number, t1: number, t2: number, sparkFrom: number): CatRow[] {
  return b.categories
    .map((c) => ({ ...c, ...windowStats(c.counts, b.totals, y0, t1, t2), series: c.counts.slice(Math.max(0, sparkFrom - y0)), rank: 0 }))
    .sort((a, b) => b.share - a.share)
    .map((r, i) => ({ ...r, rank: i + 1 }));
}

export function CategoryTable({ block, y0, t1, t2, sparkFrom, maxHeight, filter, picked, onPick, limit }: {
  block: FacetBlock; y0: number; t1: number; t2: number; sparkFrom: number; maxHeight?: number; filter?: string;
  picked?: Set<string>; onPick?: (cid: string) => void; limit?: number;
}) {
  const { open } = useCategorySheet();
  const rows = useMemo(() => {
    let r = catRows(block, y0, t1, t2, sparkFrom);
    const f = filter?.trim().toLowerCase();
    if (f) r = r.filter((c) => (c.label + " " + (c.top_terms ?? []).join(" ") + " " + (c.description ?? "")).toLowerCase().includes(f));
    return limit ? r.slice(0, limit) : r;
  }, [block, y0, t1, t2, sparkFrom, filter, limit]);
  const win = t1 === t2 ? `${t1}` : `${t1}–${t2}`;
  return (
    <DataTable rows={rows} maxHeight={maxHeight} rowKey={(r) => r.cluster_id} onSelect={(r) => open(r.cluster_id)}
      empty="No categories match." cols={[
        ...(onPick ? [{ key: "pick", label: "", sortable: false, width: 28, render: (r: CatRow) => (
          <input type="checkbox" checked={picked?.has(r.cluster_id) ?? false} title="Compare trends"
            onClick={(e) => e.stopPropagation()} onChange={() => onPick(r.cluster_id)} className="accent-[var(--color-brand-600)]" />
        ) }] : []),
        { key: "label", label: FACET_META[block.facet].label, className: "min-w-[240px]", render: (r: CatRow) => (
          <span className="flex items-center gap-2 text-[var(--text-strong)]"><Dot color={r.color} />
            <span className="hover:text-brand-600">{r.label}</span>{r.kind === "hub" && <span className="faint" title="hub: a large, coherent body of work with a shifting boundary">◇</span>}</span>
        ) },
        { key: "size", label: "Papers", align: "right" as const },
        { key: "series", label: `${sparkFrom}–${y0 + block.totals.length - 1}`, sortable: false, render: (r: CatRow) => <Sparkline values={r.series} width={84} height={22} fill={false} color={r.color} /> },
        { key: "n", label: win, align: "right" as const, title: "Papers in the window" },
        { key: "share", label: "Share", align: "right" as const, render: (r: CatRow) => pct(r.share, 1), title: `Share of the ${FACET_META[block.facet].plural.toLowerCase()} in the window` },
        { key: "change", label: "Δ share", align: "right" as const, title: "Against the window of the same length just before",
          render: (r: CatRow) => r.change == null ? <span className="faint">—</span> :
            <span className={clsx(r.change > 0.005 ? "text-emerald-600" : r.change < -0.005 ? "text-red-600" : "muted")}>{signedPct(r.change)}</span> },
      ]} />
  );
}

// ------------------------------------------------------------------ compare trends

export function CompareTrends({ block, y0, ids, t1, t2, mode }: { block: FacetBlock; y0: number; ids: string[]; t1: number; t2: number; mode: "share" | "papers" }) {
  const years = block.totals.map((_, i) => y0 + i);
  const data = ids.map((id) => {
    const c = block.categories.find((x) => x.cluster_id === id)!;
    return { type: "scatter", mode: "lines+markers", name: c.label.slice(0, 40), x: years, line: { color: c.color, width: 2.2 }, marker: { size: 5 },
      y: mode === "share" ? c.counts.map((n, i) => (block.totals[i] ? n / block.totals[i] : 0)) : c.counts,
      hovertemplate: mode === "share" ? "%{x}: %{y:.1%}<extra>%{fullData.name}</extra>" : "%{x}: %{y} papers<extra>%{fullData.name}</extra>" };
  });
  return <Plot height={260} data={data} layout={{ margin: { l: 44, r: 10, t: 6, b: 30 }, yaxis: { tickformat: mode === "share" ? ".0%" : undefined, rangemode: "tozero" },
    legend: { orientation: "h", y: -0.2 }, shapes: [{ type: "rect", xref: "x", yref: "paper", x0: t1 - 0.5, x1: t2 + 0.5, y0: 0, y1: 1, line: { width: 0 }, fillcolor: "rgba(31,106,214,0.07)" }] } as any} />;
}


