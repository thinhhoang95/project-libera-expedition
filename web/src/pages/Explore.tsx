import clsx from "clsx";
import { Grid3x3, LineChart, Search, Shapes } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { qs, useApi } from "../api";
import { setSheetWindow } from "../components/CategorySheet";
import { CategoryTable, CompareTrends, Dot, TimeSeeker } from "../components/Facets";
import { PageHeader } from "../components/Layout";
import { Plot } from "../components/Plot";
import { Card, Empty, Note, Seg, Spinner } from "../components/ui";
import { LS, useCategorySheet, useProjectId } from "../context";
import { FACET_META, FACET_ORDER, useCatalogue, type FacetKey } from "../facets";

export function Explore() {
  const p = useProjectId();
  const nav = useNavigate();
  const facet = (useParams().facet as FacetKey) || (LS.get("libera.explore.facet") as FacetKey) || "problem";
  const cat = useCatalogue(p);
  const d = cat.data;
  const [win, setWin] = useState<[number, number] | null>(null);
  const [filter, setFilter] = useState("");
  const [picked, setPicked] = useState<Record<string, string[]>>({});
  const [mode, setMode] = useState<"share" | "papers">("share");
  const w0 = win?.[0] ?? (d ? Math.max(d.y0, d.as_of - 2) : 0), w1 = win?.[1] ?? d?.as_of ?? 0;
  useEffect(() => { if (d?.built) setSheetWindow([w0, w1]); }, [w0, w1, d?.built]); // sheets opened here start on this window
  if (cat.isLoading) return <Spinner />;
  if (!d?.built)
    return (
      <div className="mx-auto max-w-3xl">
        <PageHeader title="Facets" />
        <Empty title="No categories yet" icon={<Shapes className="size-8 text-brand-500" />}>
          Categories come from the <code>core</code> step. <Link className="text-brand-600" to={`/p/${p}/pipeline`}>Open the pipeline →</Link>
        </Empty>
      </div>
    );
  const T = d.as_of;
  const [t1, t2] = win ?? [Math.max(d.y0, T - 2), T];
  const block = cat.facet.get(facet)!;
  const ids = picked[facet] ?? [];
  const pick = (cid: string) => setPicked((x) => {
    const cur = x[facet] ?? [];
    return { ...x, [facet]: cur.includes(cid) ? cur.filter((c) => c !== cid) : [...cur, cid].slice(-8) };
  });
  const go = (f: FacetKey) => {
    LS.set("libera.explore.facet", f);
    nav(`/p/${p}/explore/${f}`);
  };
  const M = FACET_META[facet];
  return (
    <div className="mx-auto max-w-[1500px]">
      <PageHeader title="Facets" sub="Every paper is read for its problem, method, innovation, result and limitations; similar statements form categories. Select a category for its details sheet." />
      <div className="mb-4 grid gap-2 sm:grid-cols-3 xl:grid-cols-5">
        {FACET_ORDER.map((f) => {
          const b = cat.facet.get(f);
          const I = FACET_META[f].icon;
          return (
            <button key={f} onClick={() => go(f)}
              className={clsx("card flex items-start gap-3 px-4 py-3 text-left transition-colors", f === facet ? "!border-brand-400 ring-2 ring-brand-500/15" : "hover:border-[var(--line-strong)]")}>
              <span className="grid size-9 shrink-0 place-items-center rounded-[10px]" style={{ background: `${FACET_META[f].color}18`, color: FACET_META[f].color }}><I className="size-4.5" /></span>
              <span className="min-w-0">
                <span className="h-title block text-[14px]">{FACET_META[f].plural}</span>
                <span className="muted block text-[12px]">{b?.categories.length ?? 0} categories · {(b?.assigned ?? 0).toLocaleString()} papers</span>
              </span>
            </button>
          );
        })}
      </div>

      <div className="card mb-4 px-4 py-3">
        <TimeSeeker y0={d.y0} y1={d.y1} asOf={T} value={[t1, t2]} onChange={setWin} counts={block.totals} />
        <Note>The window sets papers, share and Δ share (against the same number of years just before). The first year also holds earlier papers. {d.y1 > T && `${T + 1} is a partial year.`}</Note>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card title={`${M.plural}: categories`} icon={<M.icon className="size-4.5" style={{ color: M.color }} />} subtitle={M.blurb}
          actions={<div className="relative w-56"><Search className="faint pointer-events-none absolute top-2.5 left-2.5 size-3.5" />
            <input className="input !h-8 !pl-8" placeholder="Filter categories" value={filter} onChange={(e) => setFilter(e.target.value)} /></div>}>
          <CategoryTable block={block} y0={d.y0} t1={t1} t2={t2} sparkFrom={Math.max(d.y0, T - 9)} filter={filter} maxHeight={640} picked={new Set(ids)} onPick={pick} />
          <Note>Tick categories to compare their trends. ◇ = hub (a large body of work with a shifting boundary). Shares describe this map as sampled, not a forecast.</Note>
        </Card>
        <div className="flex min-w-0 flex-col gap-5">
          <Card title="Compare trends" icon={<LineChart className="size-4.5 text-brand-600" />}
            actions={<Seg size="sm" value={mode} onChange={setMode} options={[{ value: "share", label: "Share" }, { value: "papers", label: "Papers" }]} />}>
            {ids.length ? (
              <>
                <CompareTrends block={block} y0={d.y0} ids={ids} t1={t1} t2={t2} mode={mode} />
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {ids.map((id) => {
                    const c = cat.byId.get(id);
                    return <button key={id} className="chip !py-0.5" onClick={() => pick(id)} title="Remove"><Dot color={c?.color} /> {c?.label.slice(0, 34)} ×</button>;
                  })}
                </div>
              </>
            ) : <div className="muted py-8 text-center text-[13px]">Tick up to eight categories in the table to compare them here.</div>}
          </Card>
        </div>
      </div>
      <div className="mt-5"><CrossMatrix p={p} facet={facet} t1={t1} t2={t2} y0={d.y0} /></div>
    </div>
  );
}

// ------------------------------------------------------------------ cross-facet matrix

function CrossMatrix({ p, facet, t1, t2, y0 }: { p: string; facet: FacetKey; t1: number; t2: number; y0: number }) {
  const cat = useCatalogue(p);
  const { open } = useCategorySheet();
  const others = FACET_ORDER.filter((f) => f !== facet && f !== "limitations");
  const [b, setB] = useState<FacetKey>(facet === "problem" ? "method" : "problem");
  const other = b === facet ? others[0] : b;
  const { data, isLoading } = useApi<{ a: string[]; b: string[]; year: number[]; n: number[] }>(`/p/${p}/cooccurrence` + qs({ a: facet, b: other }));
  const [norm, setNorm] = useState<"n" | "row">("row");
  const fig = useMemo(() => {
    if (!data) return null;
    const lo = t1 <= y0 ? -Infinity : t1;
    const cell = new Map<string, number>(), rowT = new Map<string, number>(), colT = new Map<string, number>();
    data.a.forEach((a, i) => {
      if (data.year[i] < lo || data.year[i] > t2) return;
      const k = `${a}|${data.b[i]}`;
      cell.set(k, (cell.get(k) ?? 0) + data.n[i]);
      rowT.set(a, (rowT.get(a) ?? 0) + data.n[i]);
      colT.set(data.b[i], (colT.get(data.b[i]) ?? 0) + data.n[i]);
    });
    const rows = [...rowT.entries()].sort((x, y) => y[1] - x[1]).slice(0, 25).map((x) => x[0]);
    const cols = [...colT.entries()].sort((x, y) => y[1] - x[1]).slice(0, 25).map((x) => x[0]);
    const lab = (id: string) => { const l = cat.byId.get(id)?.label ?? id; return l.length > 44 ? l.slice(0, 42) + "…" : l; };
    const z = rows.map((r) => cols.map((c) => cell.get(`${r}|${c}`) ?? 0));
    const zz = norm === "row" ? z.map((r, i) => r.map((v) => v / Math.max(rowT.get(rows[i]) ?? 1, 1))) : z.map((r) => r.map((v) => Math.log10(1 + v)));
    return {
      rows, cols,
      data: [{ type: "heatmap", z: zz, customdata: z, x: cols.map(lab), y: rows.map(lab), xgap: 1, ygap: 1, showscale: false,
        colorscale: [[0, "#f3f6fb"], [0.15, "#cfe0f7"], [1, "#0359ad"]],
        hovertemplate: norm === "row" ? "%{y}<br>× %{x}<br>%{customdata} papers (%{z:.0%} of the row)<extra></extra>" : "%{y}<br>× %{x}<br>%{customdata} papers<extra></extra>" }],
      layout: { margin: { l: 260, r: 6, t: 4, b: 190 }, xaxis: { tickangle: -40, tickfont: { size: 10 } }, yaxis: { autorange: "reversed", tickfont: { size: 10 } } },
    };
  }, [data, t1, t2, y0, norm, cat.byId]);
  return (
    <Card title="Across facets" icon={<Grid3x3 className="size-4.5 text-violet-600" />}
      subtitle={`${FACET_META[facet].plural} × ${FACET_META[other].plural.toLowerCase()} in the window; select a cell for the papers in both`}
      actions={<Seg size="sm" value={norm} onChange={setNorm} options={[{ value: "row", label: "Row %" }, { value: "n", label: "Papers" }]} />}>
      <div className="mb-2 flex flex-wrap gap-1.5">
        {others.map((f) => (
          <button key={f} className={clsx("btn btn-sm", f === other && "!border-transparent !bg-[var(--chip)] !text-[var(--chip-text)]")} onClick={() => setB(f)}>
            × {FACET_META[f].plural}
          </button>
        ))}
      </div>
      {isLoading || !fig ? <Spinner /> : (
        <Plot height={Math.max(420, 190 + fig.rows.length * 24)} data={fig.data} layout={fig.layout as any}
          onClick={(pt: any) => {
            const r = fig.rows[pt.pointIndex?.[0] ?? -1], c = fig.cols[pt.pointIndex?.[1] ?? -1];
            if (r && c) open(r, c);
          }} />
      )}
      <Note>Top 25 categories on each side by papers in the window. Row % reads as: of this {FACET_META[facet].label.toLowerCase()} category's papers, the share in each column. The problem × method grid with method-transfer candidates is in Map & grid.</Note>
    </Card>
  );
}
