import { useMemo, useState } from "react";
import { qs, useApi } from "../api";
import { DataTable } from "../components/DataTable";
import { PaperCard, PaperLink, ReviewBar } from "../components/PaperDrawer";
import { ACCENT, Plot } from "../components/Plot";
import { HowToRead, Note, Pill, Seg, Spinner, pct, signedPct } from "../components/ui";
import { usePaperDrawer } from "../context";
import type { MapData } from "./MapGrid";

// ------------------------------------------------------------------ grid

export function GridPanel({ p, alpha, t1, t2, sel, haveSignals }: { p: string; alpha: number; t1: number; t2: number; sel: string | null; haveSignals: boolean }) {
  const { data: g, isLoading } = useApi<any>(`/p/${p}/grid` + qs({ alpha }));
  const fig = useMemo(() => {
    if (!g?.built) return null;
    const rows: string[] = g.rows.map((r: any) => r.cluster_id), cols: string[] = g.cols.map((r: any) => r.cluster_id);
    const lab = new Map<string, string>([...g.rows, ...g.cols].map((r: any) => [r.cluster_id, (r.label + (r.kind === "hub" ? " ◇" : "")).slice(0, 28)]));
    const C = g.cells;
    const cell = new Map<string, { n: number; first: number }>();
    C.year.forEach((y: number, i: number) => {
      if (y < t1 || y > t2) return;
      const k = `${C.problem_cluster[i]}|${C.method_cluster[i]}`;
      const c = cell.get(k) ?? { n: 0, first: Infinity };
      c.n += C.count[i];
      c.first = Math.min(c.first, C.first_year[i]);
      cell.set(k, c);
    });
    const z = rows.map((r) => cols.map((c) => cell.get(`${r}|${c}`)?.n ?? 0));
    const data: any[] = [{ type: "heatmap", z: z.map((r) => r.map((v) => Math.log10(1 + v))), customdata: z, x: cols.map((c) => lab.get(c)), y: rows.map((r) => lab.get(r)),
      colorscale: [[0, "#f3f6fb"], [0.15, "#cfe0f7"], [1, "#0359ad"]], showscale: false, hovertemplate: "%{y} × %{x}<br>%{customdata:.0f} papers in window<extra></extra>", xgap: 1, ygap: 1 }];
    const newC = [...cell.entries()].filter(([k, v]) => { const [r, c] = k.split("|"); return v.first >= Math.max(t1, t2 - 1) && v.first <= t2 && rows.includes(r) && cols.includes(c); });
    if (newC.length) data.push({ type: "scatter", mode: "markers", x: newC.map(([k]) => lab.get(k.split("|")[1])), y: newC.map(([k]) => lab.get(k.split("|")[0])),
      marker: { symbol: "square-open", size: 13, color: ACCENT, line: { width: 2 } }, text: newC.map(([, v]) => String(v.first)), hovertemplate: "new combination (first year %{text})<extra></extra>" });
    const tr = (g.transfer as any[]).filter((t) => cols.includes(t.method) && rows.includes(t.problem));
    if (haveSignals && tr.length) data.push({ type: "scatter", mode: "markers", x: tr.map((t) => lab.get(t.method)), y: tr.map((t) => lab.get(t.problem)),
      marker: { symbol: "diamond-open", size: 14, color: ACCENT, line: { width: 2 } }, text: tr.map((t) => `method transfer candidate #${t.rank}`), hovertemplate: "%{text}<extra></extra>" });
    const shapes: any[] = [];
    if (sel && rows.includes(sel)) shapes.push({ type: "rect", xref: "paper", x0: 0, x1: 1, y0: rows.indexOf(sel) - 0.5, y1: rows.indexOf(sel) + 0.5, line: { color: ACCENT, width: 2 } });
    if (sel && cols.includes(sel)) shapes.push({ type: "rect", yref: "paper", y0: 0, y1: 1, x0: cols.indexOf(sel) - 0.5, x1: cols.indexOf(sel) + 0.5, line: { color: ACCENT, width: 2 } });
    return { data, layout: { margin: { l: 170, r: 4, t: 4, b: 150 }, xaxis: { tickangle: -50, tickfont: { size: 9 } }, yaxis: { autorange: "reversed", tickfont: { size: 9 } }, showlegend: false, shapes },
      nNew: newC.length, nTr: tr.length };
  }, [g, t1, t2, sel, haveSignals]);
  if (isLoading) return <Spinner />;
  if (!g?.built || !fig) return <div className="muted text-[13px]">No matrix for this α.</div>;
  return (
    <>
      <Plot data={fig.data} layout={fig.layout as any} height={620} />
      <Note>Medium clusters at α = {alpha} (◇ = hub). Colour = papers in {t1}–{t2} (log). □ new combinations first seen in {Math.max(t1, t2 - 1)}–{t2} ({fig.nNew}). {alpha === 0.5 ? `◇ top method-transfer candidates (${fig.nTr}).` : "Method-transfer candidates are outlined at α = 0.5."}</Note>
    </>
  );
}

// ------------------------------------------------------------------ detail

export function DetailPanel({ p, m, sel, setSel, res, alpha, t1, t2, nWin, status, scores, disp, selPaper, setSelPaper }: {
  p: string; m: MapData; sel: string | null; setSel: (c: string | null) => void; res: string; alpha: number; t1: number; t2: number;
  nWin: Map<string, number>; status: Map<string, string | null>; scores: Map<string, Record<string, any>>; disp: (c: string) => string;
  selPaper: string | null; setSelPaper: (s: string | null) => void;
}) {
  const c = m.clusters.find((x) => x.cluster_id === sel);
  const { data: central } = useApi<any[]>(sel ? `/p/${p}/cluster/${encodeURIComponent(sel)}/papers` + qs({ res, alpha }) : null);
  const s = sel ? scores.get(sel) : undefined;
  return (
    <div className="flex flex-col gap-3">
      <select className="select" value={sel ?? ""} onChange={(e) => setSel(e.target.value || null)}>
        <option value="">— select a cluster (or click the map) —</option>
        {[...m.clusters].sort((a, b) => b.size - a.size).map((x) => <option key={x.cluster_id} value={x.cluster_id}>{disp(x.cluster_id)} ({x.cluster_id.split(":").pop()})</option>)}
      </select>
      {c && (
        <div>
          <div className="h-title flex items-center gap-2 text-[16px]"><span style={{ color: c.color }}>●</span> {c.label}</div>
          {c.kind === "hub" && <div className="mt-2 rounded-lg bg-amber-500/10 px-3 py-2 text-[12.5px]">Hub cluster: a coherent body of work whose boundary shifts between clustering seeds (stability {c.stability.toFixed(2)}, centrality z {c.centrality?.toFixed(2)}). Its signals are ranked separately from stable clusters.</div>}
          {c.description && <p className="muted mt-1.5 text-[13px]">{c.description}</p>}
          <div className="mt-2 text-[13px]">{c.size} papers · stability {c.stability.toFixed(2)} · {nWin.get(c.cluster_id) ?? 0} in {t1}–{t2}</div>
          <div className="mt-1.5 flex flex-wrap gap-1">{c.top_terms.slice(0, 10).map((t) => <Pill key={t}>{t}</Pill>)}</div>
          {s && (
            <div className="mt-2 rounded-lg bg-[var(--surface-2)] px-3 py-2 text-[12.5px]">
              <b>As of {t2}:</b> status <b>{status.get(c.cluster_id) ?? "—"}</b> · growth g {s.g ?? "—"} · Δg {s.dg ?? "—"} · emergence E {s.E ?? "—"} · niche N {s.N ?? "—"} · net author migration {s.mig_net ?? "—"}
            </div>
          )}
          <div className="h-title mt-3 mb-1.5 text-[13px]">Central papers (evidence)</div>
          <DataTable rows={central ?? []} dense rowKey={(r) => r.paper_id} onSelect={(r) => setSelPaper(r.paper_id)} selected={(r) => r.paper_id === selPaper}
            cols={[{ key: "year", label: "Year", width: 50 }, { key: "title", label: "Title" }, { key: "sim", label: "Sim", align: "right" }]} />
        </div>
      )}
      {selPaper && (
        <div className="border-t border-[var(--line)] pt-3">
          <PaperCard pid={selPaper} compact />
        </div>
      )}
      {!c && !selPaper && <Note>Click a paper or a cluster centroid on the map.</Note>}
    </div>
  );
}

// ------------------------------------------------------------------ ranking

export function RankingPanel({ p, m, derived, t1, t2, onSelect }: { p: string; m: MapData; derived: any; t1: number; t2: number; onSelect: (c: string) => void }) {
  const [variant, setVariant] = useState("all");
  const { data: ns } = useApi<Record<string, any>>(variant === "no_semantic" ? `/p/${p}/ranking/no-semantic` : null);
  const rows = m.clusters.map((c) => {
    const sc = variant === "all" ? derived.scores.get(c.cluster_id) : ns?.[c.cluster_id];
    const share = (derived.nWin.get(c.cluster_id) ?? 0) / Math.max(derived.totWin, 1);
    const prev = (derived.nPrev.get(c.cluster_id) ?? 0) / Math.max(derived.totPrev, 1);
    return { cluster_id: c.cluster_id, label: c.label, kind: c.kind, size: c.size, stability: c.stability, in_window: derived.nWin.get(c.cluster_id) ?? 0,
      share, dshare: share - prev, status: variant === "all" ? derived.status.get(c.cluster_id) : sc?.status, g: sc?.g, E: sc?.E, N: sc?.N, mig_net: sc?.mig_net };
  }).sort((a, b) => b.share - a.share);
  const cols = [
    { key: "label", label: "Cluster", render: (r: any) => <span className="text-[var(--text-strong)]">{r.label}</span> },
    { key: "size", label: "Size", align: "right" as const },
    { key: "in_window", label: "In window", align: "right" as const },
    { key: "share", label: "Share", align: "right" as const, render: (r: any) => pct(r.share, 1) },
    { key: "dshare", label: "Δ share", align: "right" as const, render: (r: any) => signedPct(r.dshare) },
    { key: "status", label: "Status" },
    { key: "g", label: "g", align: "right" as const, title: "slope of log share, last 3 years" },
    { key: "E", label: "E′", align: "right" as const, title: "z(momentum) + z(acceleration) − z(new-author share)" },
    { key: "N", label: "N", align: "right" as const },
    { key: "mig_net", label: "Net mig.", align: "right" as const },
  ];
  return (
    <div className="flex flex-col gap-3">
      <Seg size="sm" value={variant} onChange={setVariant} options={[{ value: "all", label: "Whole corpus" }, { value: "no_semantic", label: "Without semantic route" }]} />
      {variant === "no_semantic" && t2 !== m.y1 && <Note>The no-semantic variant is computed at {m.y1} only.</Note>}
      {(["stable", "hub"] as const).map((k) => {
        const part = rows.filter((r) => r.kind === k);
        return part.length ? (
          <div key={k}>
            <div className="h-title mb-1.5 text-[13px]">{k === "stable" ? "Stable clusters" : "Hub clusters (unstable boundary)"}</div>
            <DataTable rows={part} cols={cols} dense rowKey={(r) => r.cluster_id} onSelect={(r) => onSelect(r.cluster_id)} maxHeight={420} />
          </div>
        ) : null;
      })}
      <Note>Δ share compares {t1}–{t2} with the {t2 - t1 + 1} years before it. Scores are as of {t2}. Shares and statuses describe this corpus as sampled; direction claims need a stable yearly sample (see Experiments).</Note>
    </div>
  );
}

// ------------------------------------------------------------------ bridges

function ClusterPairs({ p }: { p: string }) {
  const { data } = useApi<{ types: Record<string, string>; lists: Record<string, any[]> }>(`/p/${p}/bridges`);
  const [typ, setTyp] = useState("bridge_lim_inn");
  const [rank, setRank] = useState<number | null>(null);
  const { data: det } = useApi<any>(rank != null ? `/p/${p}/bridges/${typ}/${rank}` : null);
  if (!data) return <Spinner />;
  const list = data.lists[typ] ?? [];
  return (
    <div className="flex flex-col gap-3">
      <Seg size="sm" value={typ} onChange={(t) => { setTyp(t); setRank(null); }} options={Object.entries(data.types).map(([k, v]) => ({ value: k, label: `${v} (${data.lists[k]?.length ?? 0})` }))} />
      <DataTable rows={list} dense maxHeight={280} rowKey={(r) => r.rank} onSelect={(r) => setRank(r.rank)} selected={(r) => r.rank === rank}
        empty="No candidates of this type for this run." cols={[
          { key: "rank", label: "#", width: 36 }, { key: "candidate", label: "Candidate" },
          { key: "judge", label: "Judge 1–5", align: "right" }, { key: "score", label: "Score", align: "right" },
        ]} />
      {rank == null && <Note>Ranked by the LLM judge (top 50 judged), then by score. Select a candidate for its evidence.</Note>}
      {det && (
        <div className="rounded-lg border border-[var(--line)] p-3 text-[13px]">
          <div className="h-title">#{det.rank} {det.title}</div>
          {det.rationale && <p className="mt-1">Judge {det.judge}/5: {det.rationale}</p>}
          {det.obvious_reason && <p className="muted mt-1 text-[12.5px]">Obvious reason it hasn't been done: {det.obvious_reason}</p>}
          {det.pairs ? (
            <ul className="mt-2 flex flex-col gap-2">
              {det.pairs.map((pr: any, i: number) => (
                <li key={i} className="rounded-lg bg-[var(--surface-2)] px-3 py-2">
                  <div><b>Limitation</b> ({pr.limitation.year}, <PaperLink pid={pr.limitation.paper_id} title={(pr.limitation.title ?? "").slice(0, 80)} className="italic" />): {pr.limitation.text}</div>
                  <div className="mt-1"><b>Innovation</b> ({pr.innovation.year}, <PaperLink pid={pr.innovation.paper_id} title={(pr.innovation.title ?? "").slice(0, 80)} className="italic" />): {pr.innovation.text}</div>
                  <div className="faint mt-1 text-[11.5px]">cosine {pr.sim}</div>
                </li>
              ))}
            </ul>
          ) : (
            <ul className="mt-2 flex flex-col gap-1">{det.evidence.map((e: any) => <li key={e.paper_id}><span className="faint mr-2">{e.year}</span><PaperLink pid={e.paper_id} title={e.title ?? e.paper_id} /></li>)}</ul>
          )}
          {typ === "bridge_lim_inn" && <Note>Citations A→B: {det.features.flow_xy ?? "—"} (max from A to any cluster: {det.features.flow_max ?? "—"}).</Note>}
          <ReviewBar view="bridges" subject={{ signal_type: typ, subject: det.subject, rank: det.rank }} actions={["worth pursuing", "obvious", "wrong"]} alpha={{ problem: 0.5, method: 0.5 }} />
        </div>
      )}
    </div>
  );
}

function PaperBridges({ p, sel }: { p: string; sel: string | null }) {
  const { data } = useApi<{ clusters: any[]; lists: Record<string, Record<string, any[]>> }>(`/p/${p}/paper-bridges`);
  const [pick, setPick] = useState<string | null>(null);
  const [chosen, setChosen] = useState<any>(null);
  const cluster = pick ?? (data?.clusters.some((c) => c.cluster_id === sel) ? sel : data?.clusters[0]?.cluster_id) ?? null;
  const { data: inn } = useApi<any>(chosen ? `/p/${p}/facet-text` + qs({ ids: chosen.paper_id, facet: "innovation" }) : null);
  const { data: lim } = useApi<any>(chosen?.best_matching_limitation ? `/p/${p}/facet-text` + qs({ ids: chosen.best_matching_limitation, facet: "limitations" }) : null);
  if (!data) return <Spinner />;
  if (!data.clusters.length) return <div className="muted text-[13px]">No paper bridges. Run signals.</div>;
  const lists = (cluster && data.lists[cluster]) || { bridge_paper: [], bridge_paper_gem: [] };
  const label = data.clusters.find((c) => c.cluster_id === cluster)?.label;
  return (
    <div className="flex flex-col gap-3">
      <HowToRead>{"Pick a problem cluster. Each row is an **innovation paper from another cluster that no paper here cites yet**, ranked by how likely this cluster is to start citing it in the next three years.\n\n- **Likely to be brought in**: the top 10. About 1 in 4 is picked up within three years in the backtest (popularity alone: about 1 in 5).\n- **Hidden gems**: papers cited at most once so far. Fewer are picked up, but popularity can't find them at all.\n- **from** is the paper's own cluster, **cites** its corpus citations so far, and **score** the model's log-odds. Compare scores within one list only.\n- **Select a row** to see the innovation next to this cluster's closest stated limitation. Mark *wrong* if they don't connect, and *obvious* if everyone in the field already knows it.\n\nThe ranking mostly reflects shared references with the cluster. The limitation match explains a pick; it is not what drives the ranking."}</HowToRead>
      <select className="select" value={cluster ?? ""} onChange={(e) => { setPick(e.target.value); setChosen(null); }}>
        {data.clusters.map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.label}{c.kind === "hub" ? " ◇hub" : ""} ({c.size})</option>)}
      </select>
      {([["bridge_paper", "Likely to be brought in", "backtest p@10 0.23 vs popularity 0.19"], ["bridge_paper_gem", "Hidden gems (cited ≤ 1×)", "long-tail AUC 0.84 vs popularity 0.55"]] as const).map(([k, title, help]) =>
        lists[k]?.length ? (
          <div key={k}>
            <div className="h-title mb-1.5 text-[13px]" title={help}>{title} <span className="faint font-normal">· {help}</span></div>
            <DataTable rows={lists[k]} dense rowKey={(r) => r.paper_id} onSelect={setChosen} selected={(r) => r === chosen} cols={[
              { key: "rank", label: "#", width: 32 }, { key: "title", label: "Paper" }, { key: "from", label: "From" },
              { key: "cites", label: "Cites", align: "right" }, { key: "score", label: "Score", align: "right" },
            ]} />
          </div>
        ) : null,
      )}
      {chosen && (
        <div className="rounded-lg border border-[var(--line)] p-3 text-[13px]">
          <p><b>Innovation</b> (<PaperLink pid={chosen.paper_id} title={chosen.title} className="italic" />): {inn?.[chosen.paper_id]?.text ?? "…"}</p>
          {chosen.best_matching_limitation && lim && (
            <p className="mt-2"><b>Closest limitation in {label}</b> ({lim[chosen.best_matching_limitation]?.year}, <PaperLink pid={chosen.best_matching_limitation} title={(lim[chosen.best_matching_limitation]?.title ?? "").slice(0, 90)} className="italic" />): {lim[chosen.best_matching_limitation]?.text ?? "—"}</p>
          )}
          <Note>coupling {chosen.coupling ?? "—"} · problem similarity {chosen.problem_sim ?? "—"} · limitation similarity {chosen.limitation_sim ?? "—"} · {chosen.cites ?? 0} citations so far</Note>
          <ReviewBar view="bridges" subject={{ signal_type: lists.bridge_paper.includes(chosen) ? "bridge_paper" : "bridge_paper_gem", subject: { cluster_id: cluster, paper_id: chosen.paper_id, title: chosen.title }, rank: chosen.rank }}
            actions={["worth pursuing", "obvious", "wrong"]} alpha={{ problem: 0.5 }} />
        </div>
      )}
    </div>
  );
}

function ExternalImports({ p }: { p: string }) {
  const { data } = useApi<{ built: boolean; as_of: number; rows: any[] }>(`/p/${p}/external-imports`);
  const [scope, setScope] = useState("outside");
  const [recent, setRecent] = useState(true);
  const [cluster, setCluster] = useState<string>("");
  const [chosen, setChosen] = useState<string | null>(null);
  const { data: det } = useApi<any>(chosen ? `/p/${p}/external-imports/${chosen}` : null);
  if (!data) return <Spinner />;
  if (!data.built) return <div className="muted text-[13px]">No external-import list for this project: run <code>external</code> and <code>portfolio</code>.</div>;
  let d = data.rows.filter((r) => !recent || r.year >= data.as_of - 10);
  d = d.filter((r) => scope === "all" || (scope === "outside" ? r.outside_field : !r.outside_field));
  const clusters = [...new Map(d.filter((r) => r.cluster_id).map((r) => [r.cluster_id, r.cluster_label])).entries()];
  if (cluster) d = d.filter((r) => r.cluster_id === cluster);
  d = d.slice(0, 40);
  return (
    <div className="flex flex-col gap-3">
      <HowToRead>{`Works **outside the corpus** that corpus papers cite, as of ${data.as_of}, ranked by how likely the field is to cite them more and more over the next three years. Only works with at least 3 recent citing papers in the corpus are listed.\n\n- **What drives the ranking** (experiment 6): the work's fit with the problems the field is moving *towards*, and its citation momentum across all of science (**global trend**). Both add to the in-field citation rate, modestly: about +0.025 AUC.\n- **Outside the field** keeps the third of the list whose abstracts are farthest from the corpus. **In-field, missing from the corpus** shows the rest: often core papers the corpus never fetched, a coverage gap rather than a borrowing.\n- **Select a row** for the work's innovation and the corpus papers now citing it.`}</HowToRead>
      <div className="flex flex-wrap items-center gap-2">
        <Seg size="sm" value={scope} onChange={setScope} options={[{ value: "outside", label: "Outside the field" }, { value: "infield", label: "In-field, missing" }, { value: "all", label: "All" }]} />
        <label className="flex items-center gap-1.5 text-[12.5px]"><input type="checkbox" checked={recent} onChange={(e) => setRecent(e.target.checked)} className="accent-[var(--color-brand-600)]" /> Last 10 years</label>
      </div>
      <select className="select" value={cluster} onChange={(e) => setCluster(e.target.value)}>
        <option value="">All closest problem clusters</option>
        {clusters.map(([c, l]) => <option key={c} value={c}>{l}</option>)}
      </select>
      <DataTable rows={d} dense maxHeight={340} rowKey={(r) => r.paper_id} onSelect={(r) => setChosen(r.paper_id)} selected={(r) => r.paper_id === chosen} cols={[
        { key: "title", label: "Work" }, { key: "year", label: "Year" }, { key: "subfield", label: "Subfield" },
        { key: "global_trend", label: "Global trend", align: "right" }, { key: "recent_citers", label: "Citers, 3 yrs", align: "right" },
        { key: "score", label: "Score", align: "right" },
      ]} />
      <Note>{d.length} shown. Global trend 1.5 means 50% more citations this year than the average of the two before, across all of science.</Note>
      {det && (
        <div className="rounded-lg border border-[var(--line)] p-3 text-[13px]">
          <div className="h-title">{det.work.title}</div>
          {Object.entries(det.facets ?? {}).map(([k, v]) => <p key={k} className="mt-1"><b className="capitalize">{k}:</b> {String(v)}</p>)}
          <Note>{det.work.venue ?? ""} · {det.work.year} · {det.work.cited_by_count ?? 0} citations overall · {det.work.corpus_citers_total} citing papers in the corpus · <a className="text-brand-600" href={`https://openalex.org/${det.work.paper_id}`} target="_blank" rel="noreferrer">OpenAlex</a></Note>
          {det.citing_recent?.length > 0 && (
            <>
              <div className="h-title mt-2 mb-1 text-[12.5px]">Corpus papers citing it since {det.work.as_of - 2} ({det.citing_recent.length})</div>
              <ul className="scroll-thin max-h-40 overflow-y-auto">{det.citing_recent.map((e: any) => <li key={e.paper_id} className="py-0.5"><span className="faint mr-2">{e.year}</span><PaperLink pid={e.paper_id} title={e.title} /></li>)}</ul>
            </>
          )}
          {det.work.closest_corpus_paper && <p className="mt-2 text-[12.5px]"><span className="muted">Closest paper in the corpus:</span> <PaperLink pid={det.work.closest_corpus_paper} title="open its card" /></p>}
          <ReviewBar view="bridges" subject={{ signal_type: "external_import", paper_id: det.work.paper_id, title: det.work.title }} actions={["worth pursuing", "obvious", "wrong"]} />
        </div>
      )}
    </div>
  );
}

export function BridgesPanel({ p, sel }: { p: string; sel: string | null; onPaper: (pid: string) => void }) {
  const [mode, setMode] = useState("pairs");
  return (
    <div className="flex flex-col gap-3">
      <Seg size="sm" value={mode} onChange={setMode} options={[{ value: "pairs", label: "Cluster pairs" }, { value: "papers", label: "Papers to bring in" }, { value: "external", label: "From outside the field" }]} />
      {mode === "pairs" && <ClusterPairs p={p} />}
      {mode === "papers" && <PaperBridges p={p} sel={sel} />}
      {mode === "external" && <ExternalImports p={p} />}
    </div>
  );
}

// ------------------------------------------------------------------ people

export function PeoplePanel({ p }: { p: string }) {
  const { data } = useApi<any>(`/p/${p}/people`);
  const { open } = usePaperDrawer();
  if (!data) return <Spinner />;
  return (
    <div className="flex flex-col gap-4">
      <Note>Internal decision support only: these lists are not a public ranking of people. Descriptive roles, not predictions: in the Phase 8 backtest they did not beat plain popularity at predicting future citations.</Note>
      <div className="grid gap-3 md:grid-cols-2">
        {([["seeder_author", "Seeders", "early papers in growing clusters"], ["broker_author", "Brokers", "participation across clusters"],
          ["institution_seeder", "Institutions (seeders)", "weighted score"], ["institution_broker", "Institutions (brokers)", "weighted score"]] as const).map(([k, t, col]) =>
          data[k]?.length ? (
            <div key={k}>
              <div className="h-title mb-1.5 text-[13px]">{t}</div>
              <DataTable rows={data[k]} dense maxHeight={260} cols={[{ key: "rank", label: "#", width: 32 }, { key: "name", label: "Name" }, { key: "score", label: col, align: "right" }]} />
            </div>
          ) : null,
        )}
      </div>
      {data.early_work?.length > 0 && (
        <div>
          <div className="h-title mb-1.5 text-[13px]">Early work</div>
          <DataTable rows={data.early_work} dense maxHeight={360} rowKey={(r) => r.paper_id} onSelect={(r) => open(r.paper_id)} cols={[
            { key: "rank", label: "#", width: 32 }, { key: "year", label: "Year" }, { key: "title", label: "Paper" },
            { key: "cites", label: "Cites", align: "right" }, { key: "score", label: "Score", align: "right" },
          ]} />
          <HowToRead>{"Papers from the last two years that the model expects to be in the **top quarter of their year for citations from this field** over the next three years (backtest AUC 0.82).\n\n- **What drives it:** shared references with the field, early citations, and closeness to its cluster's core. Novelty counts *against* a paper.\n- **It predicts in-field uptake, not global impact.** Read it as *what this community will read next*, not *the best papers*.\n- **score** is log-odds, so compare ranks rather than values. Select a row to open the paper."}</HowToRead>
        </div>
      )}
    </div>
  );
}
