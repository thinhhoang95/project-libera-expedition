import clsx from "clsx";
import {
  ArrowDownWideNarrow, ChevronLeft, ChevronRight, Download, ExternalLink, Filter, Link2, Map as MapIcon, Search, Sparkles,
  Tags, TrendingUp, Waypoints, X,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { useApi } from "../api";
import { useCategorySheet, usePaperDrawer, useRecordRecent } from "../context";
import { FACET_META, FACET_ORDER, facetOf, mapHref, useCatalogue, type FacetKey } from "../facets";
import { CatLink, Dot, FacetBadge, TimeSeeker } from "./Facets";
import { PaperLink } from "./PaperDrawer";
import { ACCENT, Plot } from "./Plot";
import { ErrorBox, Note, Pill, Seg, Spinner, pct, signedPct } from "./ui";

type PaperRow = {
  paper_id: string; year: number | null; title: string; cites: number | null; cites_here: number; sim: number | null;
  text: string | null; has_limitation: boolean;
} & Partial<Record<FacetKey, string | null>>;

type CatData = {
  meta: { facet: FacetKey; cluster_id: string; label: string; description: string | null; size: number; kind: string; color: string;
    top_terms: string[] | null; stability: number; alpha: number; resolution: string; on_map: boolean; is_category: boolean; merged_into: string | null };
  papers: PaperRow[];
  flows: { year: number; from_cluster: string; to_cluster: string; count: number }[];
  flow_labels: Record<string, string>;
  signals: { type: string; label: string; rank: number; role: string | null; title: string; paper_id: string | null; other: string | null; judge: number | null }[];
};

type Filters = Partial<Record<FacetKey, string>>;

// the window survives hopping from one sheet to the next; a page with its own window hands it over
let lastWindow: [number, number] | null = null;
export const setSheetWindow = (w: [number, number] | null) => { lastWindow = w; };

export function CategorySheet() {
  const { project } = useParams();
  const { cat, with: withCid, open } = useCategorySheet();
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || new URLSearchParams(window.location.search).get("paper")) return;
      if ((e.target as HTMLElement)?.closest?.("input, textarea") && (e.target as HTMLInputElement).value) return;
      open(null);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open]);
  if (!cat || !project) return null;
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-slate-900/25" onClick={() => open(null)}>
      <aside className="scroll-thin h-full w-full max-w-[1080px] overflow-y-auto bg-[var(--bg)] shadow-2xl" onClick={(e) => e.stopPropagation()}>
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
  const [win, setWinRaw] = useState<[number, number] | null>(lastWindow);
  const setWin = (w: [number, number]) => { lastWindow = w; setWinRaw(w); };
  const [t1, t2] = win && win[0] >= y0 && win[1] <= y1 ? win : [y0, y1];
  const lo = t1 <= y0 ? -Infinity : t1; // the first year also holds the earlier, sparse years
  const inWin = (y: number | null) => y != null && y >= lo && y <= t2;
  useRecordRecent(p, d ? { kind: "cat", id: cid, label: d.meta.label } : null);

  // siblings in the same facet, by size, for ‹ › stepping
  const siblings = block?.categories ?? [];
  const at = siblings.findIndex((c) => c.cluster_id === cid);
  const step = (k: number) => siblings.length && open(siblings[(at + k + siblings.length) % siblings.length].cluster_id);
  const [copied, setCopied] = useState(false);
  const [filters, setFilters] = useState<Filters>(() => (withCid ? { [facetOf(withCid)]: withCid } : {}));
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
          <button className="btn btn-ghost btn-sm" onClick={() => { navigator.clipboard?.writeText(window.location.href); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>
            <Link2 className="size-3.5" /> {copied ? "Copied" : "Copy link"}
          </button>
        </div>
        <div className="mt-3"><TimeSeeker y0={y0} y1={y1} asOf={asOf} value={[t1, t2]} onChange={setWin} counts={counts} compact /></div>
      </div>

      {isLoading || !d ? <div className="px-6"><Spinner label="Loading the details…" /></div> : (
        <div className="flex flex-col gap-4 px-6 py-5">
          <Overview d={d} block={block} counts={counts} y0={y0} y1={y1} t1={t1} t2={t2} asOf={asOf} inWin={inWin} />
          <Relations d={d} p={p} facet={facet} y0={y0} y1={y1} t1={t1} t2={t2} inWin={inWin} withCid={withCid} filters={filters} onFilter={toggleFilter} />
          <Flows d={d} t1={lo} t2={t2} />
          {d.signals.length > 0 && <Signals d={d} p={p} />}
          <Papers d={d} facet={facet} inWin={inWin} t1={t1} t2={t2} filters={filters} setFilters={setFilters} />
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
        marker: { color: years.map((y) => (y > asOf ? "rgba(31,106,214,0.25)" : y >= t1 && y <= t2 ? m.color : "#c9cdd2")) } },
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
  const shown = all ? rows.out : rows.out.slice(0, 10);

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
          <div className="muted mb-2 text-[12px]">{rows.assigned} of {rows.inW} papers in the window have a {OM.label.toLowerCase()} category{other === "limitations" ? " (most abstracts state no limitation)" : ""}.</div>
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

function Signals({ d, p }: { d: CatData; p: string }) {
  const groups = useMemo(() => {
    const g = new Map<string, CatData["signals"]>();
    d.signals.forEach((s) => g.set(s.label, [...(g.get(s.label) ?? []), s]));
    return [...g.entries()];
  }, [d]);
  return (
    <Section title="Opportunities and signals naming it" icon={<Sparkles className="size-4 text-orange-600" />}
      sub="Ranked candidates from the signal layer where this category is a side. Evidence and judgments are in Map & grid › Bridges."
      actions={<Link to={`/p/${p}/map?facet=problem&alpha=0.5&res=medium&tab=Bridges`} className="btn btn-ghost btn-sm">Open Bridges <ExternalLink className="size-3" /></Link>}>
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

const SORTS = [
  { value: "sim", label: "Most typical" }, { value: "new", label: "Newest" }, { value: "old", label: "Oldest" },
  { value: "here", label: "Most cited here" }, { value: "cites", label: "Most cited overall" },
];

function Papers({ d, facet, inWin, t1, t2, filters, setFilters }: {
  d: CatData; facet: FacetKey; inWin: (y: number | null) => boolean; t1: number; t2: number; filters: Filters; setFilters: (f: (x: Filters) => Filters) => void;
}) {
  const cat = useCatalogue(useParams().project ?? "");
  const { paper: openPaper } = usePaperDrawer();
  const [term, setTerm] = useState("");
  const [sort, setSort] = useState("sim");
  const [windowOnly, setWindowOnly] = useState(true);
  const [limOnly, setLimOnly] = useState(false);
  const [n, setN] = useState(40);
  useEffect(() => setN(40), [term, sort, windowOnly, limOnly, filters, t1, t2]);
  const others = FACET_ORDER.filter((f) => f !== facet);
  const rows = useMemo(() => {
    const t = term.trim().toLowerCase();
    let r = d.papers.filter((x) => (!windowOnly || inWin(x.year)) && (!limOnly || x.has_limitation)
      && others.every((f) => !filters[f] || x[f] === filters[f])
      && (!t || (x.title ?? "").toLowerCase().includes(t) || (x.text ?? "").toLowerCase().includes(t)));
    const by: Record<string, (a: PaperRow, b: PaperRow) => number> = {
      sim: (a, b) => (b.sim ?? -1) - (a.sim ?? -1), new: (a, b) => (b.year ?? 0) - (a.year ?? 0), old: (a, b) => (a.year ?? 9999) - (b.year ?? 9999),
      here: (a, b) => b.cites_here - a.cites_here, cites: (a, b) => (b.cites ?? 0) - (a.cites ?? 0),
    };
    r = [...r].sort(by[sort]);
    return r;
  }, [d, term, sort, windowOnly, limOnly, filters, inWin]); // eslint-disable-line react-hooks/exhaustive-deps
  const active = others.filter((f) => filters[f]);
  const csv = () => {
    const head = ["paper_id", "year", "title", "cites_here", "cites", ...others.map((f) => f)];
    const esc = (v: any) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const lines = [head.join(","), ...rows.map((r) => [r.paper_id, r.year, r.title, r.cites_here, r.cites, ...others.map((f) => cat.byId.get(r[f] ?? "")?.label ?? "")].map(esc).join(","))];
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
          <input className="input !pl-8" placeholder={`Search titles and ${FACET_META[facet].label.toLowerCase()} statements`} value={term} onChange={(e) => setTerm(e.target.value)} />
        </div>
        <select className="select !w-auto" value={sort} onChange={(e) => setSort(e.target.value)}>{SORTS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}</select>
        <label className="flex items-center gap-1.5 text-[12.5px]"><input type="checkbox" checked={windowOnly} onChange={(e) => setWindowOnly(e.target.checked)} className="accent-[var(--color-brand-600)]" /> In the window</label>
        <label className="flex items-center gap-1.5 text-[12.5px]"><input type="checkbox" checked={limOnly} onChange={(e) => setLimOnly(e.target.checked)} className="accent-[var(--color-brand-600)]" /> States a limitation</label>
      </div>
      {active.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {active.map((f) => (
            <span key={f} className="chip !py-0.5">
              <span className="faint">{FACET_META[f].label}:</span> {cat.byId.get(filters[f]!)?.label ?? filters[f]}
              <button className="ml-0.5 opacity-70 hover:opacity-100" onClick={() => setFilters((x) => ({ ...x, [f]: undefined }))}><X className="size-3" /></button>
            </span>
          ))}
          <button className="btn btn-ghost btn-sm" onClick={() => setFilters(() => ({}))}>Clear</button>
        </div>
      )}
      <ul className="mt-3 flex flex-col">
        {rows.slice(0, n).map((r) => (
          <li key={r.paper_id} className={clsx("border-b border-[var(--line)] py-2.5 last:border-0", openPaper === r.paper_id && "bg-[var(--nav-active)]")}>
            <div className="flex gap-3">
              <span className="faint w-10 shrink-0 pt-px text-[12px] tabular-nums">{r.year ?? "—"}</span>
              <div className="min-w-0 flex-1">
                <PaperLink pid={r.paper_id} title={r.title} className="text-[13.3px] font-medium text-[var(--text-strong)]" />
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
