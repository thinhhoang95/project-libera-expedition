import clsx from "clsx";
import {
  ArrowDownWideNarrow, BookOpen, CalendarDays, ChevronLeft, ChevronRight, Download, ExternalLink, Filter, Globe, Link2, Map as MapIcon, Search, Sparkles, Tags, Target, TrendingUp, Waypoints, X, Zap,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { useApi } from "../api";
import { useCategorySheet, usePaperDrawer, useRecordRecent, useSharedWindow } from "../context";
import { FACET_META, FACET_ORDER, facetOf, mapHref, useCatalogue, type FacetKey } from "../facets";
import { CatLink, Dot, FacetBadge, TimeSeeker } from "./Facets";
import { PaperLink } from "./PaperDrawer";
import { by, useSortedList } from "./Sort";
import { TimingPill } from "./Timing";
import { ACCENT, MUTED, Plot } from "./Plot";
import { isTopOverlay, useOverlay } from "./overlay";
import { copyText } from "./Toasts";
import { ErrorBox, Note, Pill, Seg, Spinner, pct, signedPct } from "./ui";

type PaperRow = {
  paper_id: string; year: number | null; title: string; cites: number | null; cites_here: number; sim: number | null;
  text: string | null; has_limitation: boolean; venue_id: string | null; venue: string | null;
} & Partial<Record<FacetKey, string | null>>;

type CatData = {
  meta: { facet: FacetKey; cluster_id: string; label: string; description: string | null; size: number; kind: string; color: string;
    top_terms: string[] | null; stability: number; alpha: number; resolution: string; on_map: boolean; is_category: boolean; merged_into: string | null };
  papers: PaperRow[];
  flows: { year: number; from_cluster: string; to_cluster: string; count: number }[];
  flow_labels: Record<string, string>;
  signals: { type: string; label: string; rank: number; role: string | null; title: string; paper_id: string | null; other: string | null; judge: number | null;
    year: number | null; timing: string | null }[];
};

type Filters = Partial<Record<FacetKey, string>>;

export function CategorySheet() {
  const { project } = useParams();
  const { cat, with: withCid, open } = useCategorySheet();
  const ref = useOverlay<HTMLElement>(!!cat && !!project);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || !isTopOverlay(ref.current)) return;
      if ((e.target as HTMLElement)?.closest?.("input, textarea") && (e.target as HTMLInputElement).value) return;
      open(null);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!cat || !project) return null;
  return (
    <div className="anim-fade fixed inset-0 z-40 flex justify-end bg-slate-900/25" onClick={() => open(null)}>
      <aside ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label="Category details"
        className="anim-slide scroll-thin h-full w-full max-w-[1080px] overflow-y-auto bg-[var(--bg)] shadow-2xl outline-none" onClick={(e) => e.stopPropagation()}>
        <SheetBody key={cat} p={project} cid={cat} withCid={withCid} onClose={() => open(null)} />
      </aside>
    </div>
  );
}

function Section({ title, icon, children, actions, sub }: { title: ReactNode; icon?: ReactNode; children: ReactNode; actions?: ReactNode; sub?: ReactNode }) {
  return (
    <section className="card px-5 py-4">
      <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="h-title flex items-center gap-2 text-[14.5px]">{icon}{title}</h3>
          {sub && <p className="muted mt-0.5 text-[12.3px]">{sub}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </header>
      {children}
    </section>
  );
}

function SheetBody({ p, cid, withCid, onClose }: { p: string; cid: string; withCid: string | null; onClose: () => void }) {
  const { data: d, error, isLoading } = useApi<CatData>(`/p/${p}/category/${encodeURIComponent(cid)}`, { staleTime: 600_000 });
  const cat = useCatalogue(p);
  const { open } = useCategorySheet();
  const facet = facetOf(cid);
  const block = cat.facet.get(facet);
  const y0 = cat.data?.y0 ?? 2000, y1 = cat.data?.y1 ?? new Date().getFullYear(), asOf = cat.data?.as_of ?? y1;
  // the project's shared window (Facets, the map, other sheets); by default the last three complete years
  const [win, setWin] = useSharedWindow(p);
  const [t1, t2] = win && win[0] >= y0 && win[1] <= y1 ? win : [Math.max(y0, asOf - 2), asOf];
  const lo = t1 <= y0 ? -Infinity : t1; // the first year also holds the earlier, sparse years
  const inWin = (y: number | null) => y != null && y >= lo && y <= t2;
  useRecordRecent(p, d ? { kind: "cat", id: cid, label: d.meta.label } : null);

  // siblings in the same facet, by size, for ‹ › stepping
  const siblings = block?.categories ?? [];
  const at = siblings.findIndex((c) => c.cluster_id === cid);
  const step = (k: number) => siblings.length && open(siblings[(at + k + siblings.length) % siblings.length].cluster_id);
  const [filters, setFilters] = useState<Filters>(() => (withCid ? { [facetOf(withCid)]: withCid } : {}));
  const [venue, setVenue] = useState<string | null>(null); // a venue_id ("" = unknown venue), filters the papers
  const toggleVenue = (v: string) => {
    setVenue((x) => (x === v ? null : v));
    document.getElementById("sheet-papers")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const toggleFilter = (f: FacetKey, id: string) => {
    setFilters((x) => (x[f] === id ? { ...x, [f]: undefined } : { ...x, [f]: id }));
    document.getElementById("sheet-papers")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const counts = useMemo(() => {
    const c = new Array(y1 - y0 + 1).fill(0);
    d?.papers.forEach((r) => r.year != null && r.year >= y0 && r.year <= y1 && c[r.year - y0]++);
    return c as number[];
  }, [d, y0, y1]);

  if (error) return <div className="p-6"><ErrorBox error={error} /><button className="btn btn-sm mt-3" onClick={onClose}>Close</button></div>;
  const M = FACET_META[facet];
  const m = d?.meta;
  return (
    <div className="flex min-h-full flex-col">
      <div className="sticky top-0 z-10 border-b border-[var(--line)] bg-[var(--surface)]/95 px-6 pt-4 pb-3 backdrop-blur">
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2 text-[12px]">
              <FacetBadge facet={facet} />
              {m?.kind === "hub" && <Pill tone="amber" title="A large, coherent body of work whose boundary shifts between clustering seeds">hub</Pill>}
              {m && !m.is_category && <Pill title="Not one of the facet's reference categories">α {m.alpha} · {m.resolution}{m.kind !== "stable" && m.kind !== "hub" ? ` · ${m.kind}` : ""}</Pill>}
              {block && at >= 0 && <span className="faint">{at + 1} of {siblings.length} {M.plural.toLowerCase()} by size</span>}
            </div>
            <h2 className="h-title mt-1 flex items-center gap-2 text-[21px] leading-tight">
              {m && <Dot color={m.color} />}<span className="min-w-0">{m?.label ?? "…"}</span>
            </h2>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {at >= 0 && <>
              <button className="btn btn-ghost btn-icon btn-sm" title="Previous (by size)" onClick={() => step(-1)}><ChevronLeft className="size-4" /></button>
              <button className="btn btn-ghost btn-icon btn-sm" title="Next (by size)" onClick={() => step(1)}><ChevronRight className="size-4" /></button>
            </>}
            <button className="btn btn-ghost btn-icon btn-sm" title="Close (Esc)" onClick={onClose}><X className="size-4" /></button>
          </div>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {m?.on_map && <Link to={mapHref(p, { cid })} className="btn btn-sm"><MapIcon className="size-3.5" /> Show on the map</Link>}
          <Link to={`/p/${p}/explore/${facet}`} className="btn btn-sm"><Tags className="size-3.5" /> All {M.plural.toLowerCase()}</Link>
          <Link to={`/p/${p}/clusters?facet=${facet}&alpha=${m?.alpha ?? ""}&res=${m?.resolution ?? "medium"}&sel=${encodeURIComponent(cid)}`} className="btn btn-sm">Review this cluster</Link>
          <button className="btn btn-ghost btn-sm" onClick={() => copyText(window.location.href)}>
            <Link2 className="size-3.5" /> Copy link
          </button>
        </div>
        <div className="mt-3"><TimeSeeker y0={y0} y1={y1} asOf={asOf} value={[t1, t2]} onChange={setWin} counts={counts} compact /></div>
      </div>

      {isLoading || !d ? <div className="px-6"><Spinner label="Loading the details…" /></div> : (
        <div className="flex flex-col gap-4 px-6 py-5">
          <Overview d={d} block={block} counts={counts} y0={y0} y1={y1} t1={t1} t2={t2} asOf={asOf} inWin={inWin} />
          <Relations d={d} p={p} facet={facet} y0={y0} y1={y1} t1={t1} t2={t2} inWin={inWin} withCid={withCid} filters={filters} onFilter={toggleFilter} />
          <Journals d={d} p={p} y0={y0} y1={y1} t1={t1} t2={t2} asOf={asOf} inWin={inWin} venue={venue} onVenue={toggleVenue} />
          <Flows d={d} t1={lo} t2={t2} />
          {d.signals.length > 0 && <Signals d={d} p={p} />}
          <Papers d={d} facet={facet} inWin={inWin} t1={t1} t2={t2} filters={filters} setFilters={setFilters} venue={venue} setVenue={setVenue} />
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ overview and trend

function Overview({ d, block, counts, y0, y1, t1, t2, asOf, inWin }: {
  d: CatData; block: any; counts: number[]; y0: number; y1: number; t1: number; t2: number; asOf: number; inWin: (y: number | null) => boolean;
}) {
  const m = d.meta;
  const years = counts.map((_, i) => y0 + i);
  const totals: number[] = block?.totals ?? [];
  const share = counts.map((n, i) => (totals[i] ? n / totals[i] : null));
  const len = t2 - t1 + 1;
  const sumY = (a: number[], s: number, e: number) => a.reduce((acc, v, i) => acc + (y0 + i >= s && y0 + i <= e ? v : 0), 0);
  const nWin = d.papers.filter((r) => inWin(r.year)).length;
  const sh = sumY(totals, t1, t2) ? sumY(counts, t1, t2) / sumY(totals, t1, t2) : null;
  const prev = t1 - len >= y0 && sumY(totals, t1 - len, t1 - 1) ? sumY(counts, t1 - len, t1 - 1) / sumY(totals, t1 - len, t1 - 1) : null;
  const yearsAll = d.papers.map((r) => r.year).filter((y): y is number => y != null).sort((a, b) => a - b);
  const peak = years[counts.indexOf(Math.max(...counts))];
  const rank = block && m.is_category
    ? [...block.categories].map((c: any) => ({ id: c.cluster_id, s: sumY(c.counts, t1, t2) })).sort((a, b) => b.s - a.s).findIndex((x) => x.id === m.cluster_id) + 1
    : null;
  const fig = useMemo(() => ({
    data: [
      { type: "bar", x: years, y: counts, name: "papers", hovertemplate: "%{x}: %{y} papers<extra></extra>",
        marker: { color: years.map((y) => (y > asOf ? "rgba(31,106,214,0.25)" : y >= t1 && y <= t2 ? m.color : MUTED)) } },
      { type: "scatter", mode: "lines+markers", x: years, y: share, yaxis: "y2", name: "share", line: { color: ACCENT, width: 2 }, marker: { size: 5 },
        hovertemplate: "%{x}: %{y:.1%} of the facet<extra></extra>" },
    ],
    layout: { margin: { l: 36, r: 44, t: 8, b: 26 }, showlegend: false, bargap: 0.2, xaxis: { dtick: years.length > 14 ? 2 : 1 },
      yaxis: { title: { text: "papers", font: { size: 10.5 } }, showgrid: true }, yaxis2: { overlaying: "y", side: "right", tickformat: ".0%", showgrid: false, rangemode: "tozero" },
      shapes: [{ type: "rect", xref: "x", yref: "paper", x0: t1 - 0.5, x1: t2 + 0.5, y0: 0, y1: 1, line: { color: ACCENT, width: 1.2 }, fillcolor: "rgba(0,0,0,0)" }] },
  }), [counts, share, t1, t2, asOf]); // eslint-disable-line react-hooks/exhaustive-deps
  const Stat = ({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) => (
    <div className="rounded-lg bg-[var(--surface-2)] px-3 py-2">
      <div className="muted text-[11.5px] font-medium">{label}</div>
      <div className="h-title text-[18px] leading-tight tabular-nums">{value}</div>
      {sub && <div className="faint text-[11px]">{sub}</div>}
    </div>
  );
  return (
    <Section title="Overview and trend" icon={<TrendingUp className="size-4 text-brand-600" />}
      sub={<>Bars: papers per year (window in colour{y1 > asOf ? `; ${asOf + 1} is a partial year` : ""}). Line: share of all {FACET_META[m.facet].plural.toLowerCase()}. Describes this map as sampled, not a forecast.</>}>
      {m.description && <p className="mb-2 text-[13.5px] leading-relaxed">{m.description}</p>}
      {m.top_terms && m.top_terms.length > 0 && <div className="mb-3 flex flex-wrap gap-1">{m.top_terms.slice(0, 12).map((t) => <Pill key={t}>{t}</Pill>)}</div>}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Papers" value={m.size.toLocaleString()} sub={`stability ${m.stability?.toFixed(2) ?? "—"}`} />
        <Stat label="In the window" value={nWin.toLocaleString()} sub={t1 === t2 ? `${t1}` : `${t1}–${t2}`} />
        <Stat label="Share in the window" value={pct(sh, 1)} sub={rank ? `#${rank} of ${block.categories.length}` : undefined} />
        <Stat label="Δ share" value={prev == null || sh == null ? "—" : <span className={clsx(sh - prev > 0.005 ? "text-emerald-600" : sh - prev < -0.005 ? "text-red-600" : "")}>{signedPct(sh - prev)}</span>}
          sub={`vs ${len} yr${len > 1 ? "s" : ""} before`} />
        <Stat label="First paper" value={yearsAll[0] ?? "—"} sub={`median year ${yearsAll[Math.floor(yearsAll.length / 2)] ?? "—"}`} />
        <Stat label="Peak year" value={peak ?? "—"} sub={`${Math.max(...counts)} papers`} />
      </div>
      <Plot className="mt-2" height={210} data={fig.data} layout={fig.layout as any} />
    </Section>
  );
}

// ------------------------------------------------------------------ relations to the other facets

function Relations({ d, p, facet, y0, y1, t1, t2, inWin, withCid, filters, onFilter }: {
  d: CatData; p: string; facet: FacetKey; y0: number; y1: number; t1: number; t2: number; inWin: (y: number | null) => boolean; withCid: string | null;
  filters: Filters; onFilter: (f: FacetKey, id: string) => void;
}) {
  const cat = useCatalogue(p);
  const others = FACET_ORDER.filter((f) => f !== facet);
  const [other, setOther] = useState<FacetKey>(withCid ? facetOf(withCid) : facet === "problem" ? "method" : "problem");
  const [mode, setMode] = useState<"window" | "time">("window");
  const [all, setAll] = useState(false);
  const { open } = useCategorySheet();
  const block = cat.facet.get(other);
  const rows = useMemo(() => {
    const n = new Map<string, number>();
    let assigned = 0, inW = 0;
    d.papers.forEach((r) => {
      if (!inWin(r.year)) return;
      inW++;
      const c = r[other];
      if (c) { assigned++; n.set(c, (n.get(c) ?? 0) + 1); }
    });
    // corpus baseline in the same window: each category's share of the facet's assigned papers
    const lo = t1 <= y0 ? -Infinity : t1;
    const base = new Map<string, number>();
    let btot = 0;
    block?.categories.forEach((c) => {
      const s = c.counts.reduce((a, v, i) => a + (y0 + i >= lo && y0 + i <= t2 ? v : 0), 0);
      base.set(c.cluster_id, s);
      btot += s;
    });
    const out = [...n.entries()].map(([id, k]) => {
      const share = k / Math.max(assigned, 1), b = btot ? (base.get(id) ?? 0) / btot : null;
      return { id, n: k, share, base: b, lift: b ? share / b : null, label: cat.byId.get(id)?.label ?? id, color: cat.byId.get(id)?.color };
    }).sort((a, b) => b.n - a.n);
    return { out, assigned, inW };
  }, [d, other, inWin, block, cat.byId, t1, t2, y0]);
  const maxShare = Math.max(...rows.out.map((r) => Math.max(r.share, r.base ?? 0)), 0.01);
  type Rel = (typeof rows.out)[number];
  const { sorted: relSorted, bar: relBar } = useSortedList(rows.out, [
    by.count<Rel>("n", "Papers", (r) => r.n),
    by.count<Rel>("lift", "Lift (over-represented here)", (r) => r.lift, Zap),
    by.count<Rel>("base", "Share on the whole map", (r) => r.base, Globe),
    by.title<Rel>((r) => r.label),
  ], "sheet-relations");
  const shown = all ? relSorted : relSorted.slice(0, 10);

  const timeFig = useMemo(() => {
    if (mode !== "time") return null;
    const years = Array.from({ length: y1 - y0 + 1 }, (_, i) => y0 + i);
    const top = rows.out.slice(0, 7).map((r) => r.id);
    const by = new Map<string, number[]>(), rest = new Array(years.length).fill(0), tot = new Array(years.length).fill(0);
    top.forEach((id) => by.set(id, new Array(years.length).fill(0)));
    d.papers.forEach((r) => {
      const c = r[other];
      if (!c || r.year == null) return;
      const i = Math.min(Math.max(r.year, y0), y1) - y0;
      tot[i]++;
      if (by.has(c)) by.get(c)![i]++;
      else rest[i]++;
    });
    const data: any[] = top.map((id) => ({ type: "bar", name: (cat.byId.get(id)?.label ?? id).slice(0, 38), x: years, y: by.get(id)!.map((v, i) => (tot[i] ? v / tot[i] : 0)),
      customdata: years.map(() => id), marker: { color: cat.byId.get(id)?.color }, hovertemplate: "%{x}: %{y:.0%}<extra>%{fullData.name}</extra>" }));
    data.push({ type: "bar", name: "other", x: years, y: rest.map((v, i) => (tot[i] ? v / tot[i] : 0)), marker: { color: "#d5dbe7" }, hovertemplate: "%{x}: %{y:.0%}<extra>other</extra>" });
    return { data, layout: { barmode: "stack", margin: { l: 40, r: 10, t: 6, b: 26 }, yaxis: { tickformat: ".0%", range: [0, 1] }, legend: { orientation: "h", y: -0.18, font: { size: 10.5 } },
      shapes: [{ type: "rect", xref: "x", yref: "paper", x0: Math.max(t1, y0) - 0.5, x1: t2 + 0.5, y0: 0, y1: 1, line: { color: ACCENT, width: 1.2 }, fillcolor: "rgba(0,0,0,0)" }] } };
  }, [mode, rows, d, other, y0, y1, t1, t2, cat.byId]);

  const OM = FACET_META[other];
  return (
    <Section title="How it relates to the other facets" icon={<Waypoints className="size-4 text-violet-600" />}
      sub={<>Where this category's papers fall in another facet. <b>Lift</b> compares with the whole map in the same window (2× = twice as common here). Select a row to filter the papers below; the arrow opens that category.</>}
      actions={<Seg size="sm" value={mode} onChange={setMode} options={[{ value: "window", label: "In the window" }, { value: "time", label: "Over time" }]} />}>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {others.map((f) => {
          const I = FACET_META[f].icon;
          return (
            <button key={f} onClick={() => { setOther(f); setAll(false); }}
              className={clsx("btn btn-sm", f === other && "!border-transparent !bg-[var(--chip)] !text-[var(--chip-text)]")}>
              <I className="size-3.5" style={{ color: FACET_META[f].color }} /> {FACET_META[f].plural}
            </button>
          );
        })}
      </div>
      {rows.assigned === 0 ? (
        <div className="muted text-[13px]">None of the {rows.inW} papers in the window has a {OM.label.toLowerCase()} category{other === "limitations" ? " (most abstracts state no limitation)" : ""}.</div>
      ) : mode === "window" ? (
        <>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <span className="muted text-[12px]">{rows.assigned} of {rows.inW} papers in the window have a {OM.label.toLowerCase()} category{other === "limitations" ? " (most abstracts state no limitation)" : ""}.</span>
            {relBar}
          </div>
          <div className="flex flex-col">
            {shown.map((r) => (
              <div key={r.id} className={clsx("group grid grid-cols-[minmax(0,1.3fr)_minmax(120px,1fr)_auto] items-center gap-3 rounded-md px-2 py-1.5 hover:bg-[var(--surface-2)]",
                filters[other] === r.id && "bg-[var(--nav-active)]")}>
                <button className="flex min-w-0 items-center gap-2 text-left text-[13px]" title="Filter the papers below"
                  onClick={() => onFilter(other, r.id)}>
                  <Dot color={r.color} /><span className="truncate text-[var(--text-strong)] group-hover:text-brand-600">{r.label}</span>
                </button>
                <div className="relative h-3.5 rounded bg-[var(--line)]" title={`${pct(r.share, 1)} here · ${pct(r.base, 1)} on the whole map`}>
                  <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${(r.share / maxShare) * 100}%`, background: r.color ?? OM.color, opacity: 0.85 }} />
                  {r.base != null && <div className="absolute -inset-y-0.5 w-0.5 bg-[var(--text-strong)]" style={{ left: `${(r.base / maxShare) * 100}%` }} />}
                </div>
                <div className="flex items-center gap-2 text-[12px] tabular-nums">
                  <span className="w-9 text-right">{r.n}</span>
                  <span className="w-11 text-right">{pct(r.share, 0)}</span>
                  <span className={clsx("w-12 text-right font-medium", r.lift && r.lift >= 1.5 ? "text-emerald-600" : r.lift && r.lift < 0.67 ? "text-red-600" : "muted")}>
                    {r.lift ? `${r.lift.toFixed(1)}×` : "—"}
                  </span>
                  <button className="btn btn-ghost btn-icon btn-sm !size-6 opacity-60 group-hover:opacity-100" title="Open this category" onClick={() => open(r.id)}>
                    <ChevronRight className="size-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-2 flex items-center justify-between">
            <span className="faint text-[11.5px]">Columns: papers · share of this category's papers · lift. The black tick is the share on the whole map.</span>
            {rows.out.length > 10 && <button className="btn btn-ghost btn-sm" onClick={() => setAll(!all)}>{all ? "Top 10" : `All ${rows.out.length}`}</button>}
          </div>
        </>
      ) : timeFig && (
        <>
          <Plot height={280} data={timeFig.data} layout={timeFig.layout as any}
            onClick={(pt: any) => pt.data?.customdata && onFilter(other, pt.data.customdata[0])} />
          <Note>Share of this category's papers per year, by {OM.label.toLowerCase()} category (top 7 in the window, the rest as other). The first year also holds earlier papers. Select a colour to filter the papers below.</Note>
        </>
      )}
    </Section>
  );
}

// ------------------------------------------------------------------ journals (venues) by year

type Venue = {
  id: string; name: string; n: number; nWin: number; counts: number[]; first: number | null; last: number | null;
  lastWin: number | null; age: number | null; share: number; base: number | null; lift: number | null;
};

/** How long ago the latest paper of the category appeared in a journal, against the map's as-of year. */
function AgePill({ age, last, asOf }: { age: number | null; last: number | null; asOf: number }) {
  if (age == null || last == null) return <span className="faint">—</span>;
  const tone = age <= 2 ? "green" : age >= 7 ? "amber" : "neutral";
  const words = age <= 0 ? "this year" : `${age} yr${age > 1 ? "s" : ""} ago`;
  return <Pill tone={tone} title={`The latest paper of this category in this journal is from ${last}, ${words} as of ${asOf}`}>{last} · {words}</Pill>;
}

function Journals({ d, p, y0: yTrend, y1, t1, t2, asOf, inWin, venue, onVenue }: {
  d: CatData; p: string; y0: number; y1: number; t1: number; t2: number; asOf: number; inWin: (y: number | null) => boolean;
  venue: string | null; onVenue: (v: string) => void;
}) {
  const { data: base } = useApi<{ venue_id: string[]; year: number[]; n: number[] }>(`/p/${p}/venues`, { staleTime: 600_000 });
  const [term, setTerm] = useState("");
  const [activeOnly, setActiveOnly] = useState(false);
  const [all, setAll] = useState(false);
  // the strip reaches back past the trend's start to the category's first paper (at most 30 columns): an old
  // venue is the point of this section
  const first = Math.min(...d.papers.map((r) => r.year ?? Infinity));
  const y0 = Math.min(yTrend, Math.max(Number.isFinite(first) ? first : yTrend, y1 - 29));
  const years = useMemo(() => Array.from({ length: y1 - y0 + 1 }, (_, i) => y0 + i), [y0, y1]);
  const lo = t1 <= yTrend ? -Infinity : t1;

  const rows = useMemo(() => {
    const m = new Map<string, Venue>();
    let nWinAll = 0;
    d.papers.forEach((r) => {
      const id = r.venue_id ?? "";
      let v = m.get(id);
      if (!v) {
        v = { id, name: r.venue ?? "Unknown venue", n: 0, nWin: 0, counts: new Array(years.length).fill(0), first: null, last: null,
          lastWin: null, age: null, share: 0, base: null, lift: null };
        m.set(id, v);
      }
      v.n++;
      if (r.year == null) return;
      v.counts[Math.min(Math.max(r.year, y0), y1) - y0]++;
      v.first = v.first == null ? r.year : Math.min(v.first, r.year);
      if (r.year <= asOf) v.last = v.last == null ? r.year : Math.max(v.last, r.year);
      if (inWin(r.year)) { v.nWin++; nWinAll++; v.lastWin = Math.max(v.lastWin ?? -Infinity, r.year); }
    });
    // whole-map baseline in the same window: each venue's share of all papers
    const bw = new Map<string, number>();
    let btot = 0;
    if (base) base.venue_id.forEach((id, i) => {
      const y = base.year[i];
      if (y < lo || y > t2) return;
      bw.set(id, (bw.get(id) ?? 0) + base.n[i]);
      btot += base.n[i];
    });
    const out = [...m.values()];
    out.forEach((v) => {
      v.age = v.last == null ? null : asOf - v.last;
      v.share = nWinAll ? v.nWin / nWinAll : 0;
      v.base = base && btot ? (bw.get(v.id) ?? 0) / btot : null;
      v.lift = v.base && v.nWin >= 2 ? v.share / v.base : null; // one paper says little
    });
    return { out: out.sort((a, b) => b.nWin - a.nWin || b.n - a.n), nWinAll };
  }, [d, years, y0, y1, asOf, inWin, base, lo, t2]);

  const filtered = useMemo(() => {
    const t = term.trim().toLowerCase();
    return rows.out.filter((v) => (!activeOnly || v.nWin > 0) && (!t || v.name.toLowerCase().includes(t)));
  }, [rows, term, activeOnly]);
  const { sorted, bar } = useSortedList(filtered, [
    by.count<Venue>("win", "Papers in the window", (v) => v.nWin),
    by.count<Venue>("all", "Papers, all years", (v) => v.n),
    { ...by.year<Venue>((v) => v.last), label: "Latest paper here" },
    { key: "first", label: "First paper here", icon: CalendarDays, get: (v: Venue) => v.first, dir: 1, dirs: ["oldest first", "newest first"] },
    by.count<Venue>("lift", "Lift (over-represented here)", (v) => v.lift, Zap),
    by.title<Venue>((v) => v.name),
  ], "sheet-journals");
  const shown = all || term ? sorted : sorted.slice(0, 12);
  const mx = Math.max(1, ...rows.out.map((v) => Math.max(...v.counts)));
  const known = rows.out.filter((v) => v.id);
  const top5 = [...known].sort((a, b) => b.nWin - a.nWin).slice(0, 5).reduce((a, v) => a + v.nWin, 0);
  const stale = known.filter((v) => v.age != null && v.age >= 7 && v.n >= 2).length;
  const color = d.meta.color;
  const winCol = (y: number) => (y >= t1 || (t1 <= yTrend && y <= yTrend)) && y <= t2;
  const folded = Number.isFinite(first) && first < y0;
  const tick = (y: number) => y === y0 || y === y1 || y % 5 === 0;
  const grid = "grid grid-cols-[minmax(0,1.15fr)_minmax(150px,1.6fr)_auto] items-center gap-3";

  return (
    <Section title="Journals" icon={<BookOpen className="size-4 text-sky-600" />}
      sub={<>Where this category's papers were published, year by year. A journal that carried the topic ten years ago is not the venue it is today: read the strip and the <b>latest paper</b> together before you aim a paper at it. Select a journal to filter the papers below.</>}
      actions={bar}>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px]">
        <span><b className="tabular-nums">{known.length}</b> <span className="muted">journals</span></span>
        <span><b className="tabular-nums">{pct(rows.nWinAll ? top5 / rows.nWinAll : null, 0)}</b> <span className="muted">of the window's papers in the top 5</span></span>
        <span><b className="tabular-nums">{stale}</b> <span className="muted">journals with 2+ papers but none here in the last 7 years</span></span>
      </div>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <div className="relative min-w-[200px] flex-1">
          <Search className="faint pointer-events-none absolute top-2.5 left-2.5 size-3.5" />
          <input className="input !pl-8" placeholder="Search journals" value={term} onChange={(e) => setTerm(e.target.value)} />
        </div>
        <label className="flex items-center gap-1.5 text-[12.5px]"><input type="checkbox" checked={activeOnly} onChange={(e) => setActiveOnly(e.target.checked)} className="accent-[var(--color-brand-600)]" /> Published here in the window</label>
      </div>
      <div className={clsx(grid, "faint px-2 pb-1 text-[10.5px]")}>
        <span>Journal</span>
        <div className="flex">{years.map((y) => <span key={y} className={clsx("flex-1 whitespace-nowrap tabular-nums", y === y0 ? "text-left" : "text-center", winCol(y) && "font-semibold text-[var(--chip-text)]")}>{y === y0 && folded ? `≤’${String(y).slice(2)}` : tick(y) ? `’${String(y).slice(2)}` : ""}</span>)}</div>
        <div className="flex items-center gap-2 text-right">
          <span className="w-9" title="Papers in the window">window</span><span className="w-9" title="Papers, all years">all</span>
          <span className="w-11" title="Lift over the whole map in the window">lift</span><span className="w-[118px] text-left">latest paper</span>
        </div>
      </div>
      <div className="flex flex-col">
        {shown.map((v) => (
          <div key={v.id || "-"} role="button" tabIndex={0} onClick={() => onVenue(v.id)} onKeyDown={(e) => e.key === "Enter" && onVenue(v.id)}
            title="Filter the papers below by this journal"
            className={clsx(grid, "group cursor-pointer rounded-md px-2 py-1.5 hover:bg-[var(--surface-2)]", venue === v.id && "bg-[var(--nav-active)]",
              v.nWin === 0 && "opacity-70")}>
            <div className="min-w-0">
              <div className={clsx("truncate text-[13px] text-[var(--text-strong)] group-hover:text-brand-600", !v.id && "italic")}>{v.name}</div>
              <div className="faint text-[11px] tabular-nums">{v.first == null ? "—" : v.first === v.last ? `${v.first}` : `${v.first}–${v.last ?? v.first}`}</div>
            </div>
            <div className="flex h-5 overflow-hidden rounded-sm">
              {v.counts.map((c, i) => {
                const y = years[i];
                return (
                  <div key={y} className={clsx("relative flex-1 border-r border-[var(--surface)] last:border-r-0", winCol(y) ? "bg-[var(--chip)]" : "bg-[var(--surface-2)]")}
                    title={`${v.name} · ${i === 0 && folded ? `${y0} and earlier` : y}: ${c} paper${c === 1 ? "" : "s"}`}>
                    {c > 0 && <div className="absolute inset-0" style={{ background: color, opacity: 0.25 + 0.75 * Math.sqrt(c / mx) }} />}
                    {y > asOf && <div className="absolute inset-0 bg-[repeating-linear-gradient(45deg,transparent_0_3px,rgba(255,255,255,0.5)_3px_5px)]" />}
                  </div>
                );
              })}
            </div>
            <div className="flex items-center gap-2 text-right text-[12px] tabular-nums">
              <span className="w-9 font-medium">{v.nWin || <span className="faint">0</span>}</span>
              <span className="muted w-9">{v.n}</span>
              <span className={clsx("w-11", v.lift && v.lift >= 1.5 ? "font-medium text-emerald-600" : v.lift && v.lift < 0.67 ? "text-red-600" : "muted")}>
                {v.lift ? `${v.lift.toFixed(1)}×` : "—"}
              </span>
              <span className="w-[118px] text-left"><AgePill age={v.age} last={v.last} asOf={asOf} /></span>
            </div>
          </div>
        ))}
        {!shown.length && <div className="muted py-4 text-center text-[13px]">No journal matches.</div>}
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <span className="faint text-[11.5px]">
          Cells: papers per year (darker = more; the window is shaded{folded ? `; the first column also holds the years before ${y0}` : ""}). <b>Lift</b> (2+ papers in the window) compares the journal's share of this category's papers with its share of the whole map in the window. Latest paper: green within 2 years of {asOf}, amber 7 or more years before.
        </span>
        {!term && filtered.length > 12 && <button className="btn btn-ghost btn-sm" onClick={() => setAll(!all)}>{all ? "Top 12" : `All ${filtered.length}`}</button>}
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ in-facet citation flows

function Flows({ d, t1, t2 }: { d: CatData; t1: number; t2: number }) {
  const cid = d.meta.cluster_id;
  const agg = (dir: "out" | "in") => {
    const m = new Map<string, number>();
    d.flows.forEach((f) => {
      if (f.year < t1 || f.year > t2) return;
      if (dir === "out" && f.from_cluster === cid) m.set(f.to_cluster, (m.get(f.to_cluster) ?? 0) + f.count);
      if (dir === "in" && f.to_cluster === cid) m.set(f.from_cluster, (m.get(f.from_cluster) ?? 0) + f.count);
    });
    return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  };
  const out = agg("out"), inn = agg("in");
  if (!out.length && !inn.length) return null;
  const mx = Math.max(1, ...out.map((x) => x[1]), ...inn.map((x) => x[1]));
  const List = ({ rows, title }: { rows: [string, number][]; title: string }) => (
    <div>
      <div className="muted mb-1.5 text-[12px] font-semibold">{title}</div>
      {rows.length ? rows.map(([id, n]) => (
        <div key={id} className="flex items-center gap-2 py-1 text-[12.8px]">
          <CatLink cid={id} label={d.flow_labels[id] ?? id} className="min-w-0 flex-1 truncate" />
          <div className="h-2 w-24 shrink-0 rounded bg-[var(--line)]"><div className="h-full rounded bg-brand-500/70" style={{ width: `${(n / mx) * 100}%` }} /></div>
          <span className="w-10 shrink-0 text-right tabular-nums">{n}</span>
        </div>
      )) : <div className="faint text-[12.5px]">None in the window.</div>}
    </div>
  );
  return (
    <Section title={`Citations with other ${FACET_META[d.meta.facet].plural.toLowerCase()}`} icon={<ArrowDownWideNarrow className="size-4 text-emerald-600" />}
      sub="Citations between this category and its siblings in the window (by the citing paper's year).">
      <div className="grid gap-5 md:grid-cols-2">
        <List rows={out} title="It cites" />
        <List rows={inn} title="Cited by" />
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ opportunities

type Sig = CatData["signals"][number];

function Signals({ d, p }: { d: CatData; p: string }) {
  const { sorted, bar } = useSortedList(d.signals, [
    by.rank<Sig>((s) => s.rank), by.judge<Sig>((s) => s.judge), by.year<Sig>((s) => s.year), by.title<Sig>((s) => s.title),
  ], "sheet-signals");
  const groups = useMemo(() => {
    const g = new Map<string, Sig[]>();
    sorted.forEach((s) => g.set(s.label, [...(g.get(s.label) ?? []), s]));
    return [...g.entries()];
  }, [sorted]);
  return (
    <Section title="Opportunities and signals naming it" icon={<Sparkles className="size-4 text-orange-600" />}
      sub="Ranked candidates from the signal layer where this category is a side. Evidence and judgments are in Map & grid › Bridges."
      actions={<>{bar}<Link to={`/p/${p}/map?facet=problem&alpha=0.5&res=medium&tab=Bridges`} className="btn btn-ghost btn-sm">Open Bridges <ExternalLink className="size-3" /></Link></>}>
      <div className="grid gap-4 md:grid-cols-2">
        {groups.map(([label, rows]) => (
          <div key={label}>
            <div className="muted mb-1 text-[12px] font-semibold">{label}</div>
            <ul className="flex flex-col">
              {rows.map((s, i) => (
                <li key={i} className="flex gap-2 border-b border-[var(--line)] py-1.5 text-[12.8px] last:border-0">
                  <span className="faint w-9 shrink-0 tabular-nums">#{s.rank}</span>
                  <span className="min-w-0 flex-1">
                    {s.paper_id ? <PaperLink pid={s.paper_id} title={s.title} /> : s.other ? <CatLink cid={s.other} label={s.title} /> : s.title}
                    {s.paper_id && (s.year || s.timing) && <span className="ml-1.5 inline-flex items-center gap-1"><span className="faint text-[11.5px]">{s.year}</span><TimingPill t={s.timing} /></span>}
                    {s.role === "from_cluster" && s.type.startsWith("bridge_paper") && <span className="faint"> (from here)</span>}
                  </span>
                  {s.judge != null && <span className="faint shrink-0" title="LLM judge 1–5">{s.judge}</span>}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ papers


function Papers({ d, facet, inWin, t1, t2, filters, setFilters, venue, setVenue }: {
  d: CatData; facet: FacetKey; inWin: (y: number | null) => boolean; t1: number; t2: number; filters: Filters; setFilters: (f: (x: Filters) => Filters) => void;
  venue: string | null; setVenue: (v: string | null) => void;
}) {
  const cat = useCatalogue(useParams().project ?? "");
  const { paper: openPaper } = usePaperDrawer();
  const [term, setTerm] = useState("");
  const [windowOnly, setWindowOnly] = useState(true);
  const [limOnly, setLimOnly] = useState(false);
  const [n, setN] = useState(40);
  const others = FACET_ORDER.filter((f) => f !== facet);
  const filtered = useMemo(() => {
    const t = term.trim().toLowerCase();
    let r = d.papers.filter((x) => (!windowOnly || inWin(x.year)) && (!limOnly || x.has_limitation)
      && others.every((f) => !filters[f] || x[f] === filters[f]) && (venue == null || (x.venue_id ?? "") === venue)
      && (!t || (x.title ?? "").toLowerCase().includes(t) || (x.text ?? "").toLowerCase().includes(t) || (x.venue ?? "").toLowerCase().includes(t)));
    return r;
  }, [d, term, windowOnly, limOnly, filters, venue, inWin]); // eslint-disable-line react-hooks/exhaustive-deps
  const { sorted: rows, bar, sort } = useSortedList(filtered, [
    { ...by.count<PaperRow>("sim", "Most typical", (r) => r.sim, Target), dirs: ["least typical first", "most typical first"] },
    by.year<PaperRow>((r) => r.year), by.citedHere<PaperRow>((r) => r.cites_here), by.cites<PaperRow>((r) => r.cites),
    by.title<PaperRow>((r) => r.title),
  ], "sheet-papers");
  useEffect(() => setN(40), [term, sort, windowOnly, limOnly, filters, venue, t1, t2]);
  // a journal picked for its older papers shows them: leave the window when none of its papers is in it
  useEffect(() => {
    if (venue != null && !d.papers.some((x) => (x.venue_id ?? "") === venue && inWin(x.year))) setWindowOnly(false);
  }, [venue]); // eslint-disable-line react-hooks/exhaustive-deps
  const venueName = venue == null ? null : venue === "" ? "Unknown venue" : d.papers.find((x) => x.venue_id === venue)?.venue ?? venue;
  const active = others.filter((f) => filters[f]);
  const csv = () => {
    const head = ["paper_id", "year", "venue", "title", "cites_here", "cites", ...others.map((f) => f)];
    const esc = (v: any) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const lines = [head.join(","), ...rows.map((r) => [r.paper_id, r.year, r.venue, r.title, r.cites_here, r.cites, ...others.map((f) => cat.byId.get(r[f] ?? "")?.label ?? "")].map(esc).join(","))];
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = `${d.meta.label.replace(/[^\w]+/g, "-").toLowerCase()}-papers.csv`;
    a.click();
  };
  return (
    <section id="sheet-papers" className="card scroll-mt-48 px-5 py-4">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="h-title flex items-center gap-2 text-[14.5px]"><Filter className="size-4 text-brand-600" />Papers <span className="muted text-[12.5px] font-normal">{rows.length.toLocaleString()} of {d.papers.length.toLocaleString()}</span></h3>
        <button className="btn btn-ghost btn-sm" onClick={csv} title="Download the filtered list as CSV"><Download className="size-3.5" /> CSV</button>
      </header>
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1">
          <Search className="faint pointer-events-none absolute top-2.5 left-2.5 size-3.5" />
          <input className="input !pl-8" placeholder={`Search titles, journals and ${FACET_META[facet].label.toLowerCase()} statements`} value={term} onChange={(e) => setTerm(e.target.value)} />
        </div>
        {bar}
        <label className="flex items-center gap-1.5 text-[12.5px]"><input type="checkbox" checked={windowOnly} onChange={(e) => setWindowOnly(e.target.checked)} className="accent-[var(--color-brand-600)]" /> In the window</label>
        <label className="flex items-center gap-1.5 text-[12.5px]"><input type="checkbox" checked={limOnly} onChange={(e) => setLimOnly(e.target.checked)} className="accent-[var(--color-brand-600)]" /> States a limitation</label>
      </div>
      {(active.length > 0 || venueName) && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {venueName && (
            <span className="chip !py-0.5">
              <span className="faint">Journal:</span> {venueName}
              <button className="ml-0.5 opacity-70 hover:opacity-100" onClick={() => setVenue(null)}><X className="size-3" /></button>
            </span>
          )}
          {active.map((f) => (
            <span key={f} className="chip !py-0.5">
              <span className="faint">{FACET_META[f].label}:</span> {cat.byId.get(filters[f]!)?.label ?? filters[f]}
              <button className="ml-0.5 opacity-70 hover:opacity-100" onClick={() => setFilters((x) => ({ ...x, [f]: undefined }))}><X className="size-3" /></button>
            </span>
          ))}
          <button className="btn btn-ghost btn-sm" onClick={() => { setFilters(() => ({})); setVenue(null); }}>Clear</button>
        </div>
      )}
      <ul className="mt-3 flex flex-col">
        {rows.slice(0, n).map((r) => (
          <li key={r.paper_id} className={clsx("border-b border-[var(--line)] py-2.5 last:border-0", openPaper === r.paper_id && "bg-[var(--nav-active)]")}>
            <div className="flex gap-3">
              <span className="faint w-10 shrink-0 pt-px text-[12px] tabular-nums">{r.year ?? "—"}</span>
              <div className="min-w-0 flex-1">
                <PaperLink pid={r.paper_id} title={r.title} className="text-[13.3px] font-medium text-[var(--text-strong)]" />
                {r.venue && <button className={clsx("faint block max-w-full truncate text-left text-[11.8px] italic hover:text-brand-600", venue === r.venue_id && "!text-brand-600")}
                  title="Filter by this journal" onClick={() => setVenue(r.venue_id ?? "")}>{r.venue}{r.year ? `, ${r.year}` : ""}</button>}
                {r.text && <p className="muted clamp-2 mt-0.5 text-[12.3px] leading-snug">{r.text}</p>}
                <div className="mt-1 flex flex-wrap gap-1">
                  {others.map((f) => r[f] && (
                    <button key={f} title={`${FACET_META[f].label}: filter by this`} onClick={() => setFilters((x) => ({ ...x, [f]: r[f]! }))}
                      className={clsx("inline-flex max-w-[260px] items-center gap-1 rounded border px-1.5 py-px text-[11px] hover:border-[var(--line-strong)]",
                        filters[f] === r[f] ? "border-brand-400 bg-[var(--chip)]" : "border-[var(--line)]")}>
                      <span className="font-semibold" style={{ color: FACET_META[f].color }}>{FACET_META[f].label[0]}</span>
                      <span className="truncate">{cat.byId.get(r[f]!)?.label ?? r[f]}</span>
                    </button>
                  ))}
                  {r.has_limitation && facet !== "limitations" && !r.limitations && <span className="faint text-[11px]">states a limitation</span>}
                </div>
              </div>
              <div className="w-[88px] shrink-0 text-right text-[11.5px] leading-tight">
                <div title="Citations from papers in this map"><span className="h-title tabular-nums">{r.cites_here}</span> <span className="faint">here</span></div>
                <div className="faint tabular-nums" title="OpenAlex citations">{(r.cites ?? 0).toLocaleString()} total</div>
              </div>
            </div>
          </li>
        ))}
      </ul>
      {!rows.length && <div className="muted py-6 text-center text-[13px]">No papers match. {windowOnly && "Widen the window, or "}clear a filter.</div>}
      {rows.length > n && <button className="btn btn-sm mt-3 w-full" onClick={() => setN(n + 60)}>Show more ({(rows.length - n).toLocaleString()} left)</button>}
    </section>
  );
}
