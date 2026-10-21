import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { ChevronLeft, ChevronRight, DollarSign, FileWarning, Layers3, Library } from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CatLink } from "../components/Facets";
import { api, qs, useApi } from "../api";
import { DataTable } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { PaperCard, ReviewBar } from "../components/PaperDrawer";
import { Plot } from "../components/Plot";
import { Bar, Card, ErrorBox, Note, Pill, Seg, Spinner, StatCard, pct } from "../components/ui";
import { useProjectId, useReviewer } from "../context";
import { ALPHAS, GLOSS } from "./MapGrid";

const ROUTE_COLORS: Record<string, string> = { seed: "#e8590c", ai_report: "#ae3ec9", back: "#1c7ed6", forward: "#37b24d", author: "#f59f00", semantic: "#868e96" };

// ------------------------------------------------------------------ corpus

export function Corpus() {
  const p = useProjectId();
  const { data: d, isLoading } = useApi<any>(`/p/${p}/corpus`);
  const [route, setRoute] = useState("seed");
  const [pid, setPid] = useState<string | null>(null);
  const { data: papers } = useApi<any[]>(`/p/${p}/corpus/papers` + qs({ route }));
  if (isLoading || !d) return <Spinner />;
  const years = [...new Set<string>(d.by_year.map((r: any) => r.year))].sort((a, b) => (a === "≤2009" ? -1 : b === "≤2009" ? 1 : a.localeCompare(b)));
  const llm = (d.spend.llm_extraction ?? 0) + (d.spend.llm_labels ?? 0) + (d.spend.llm_judge ?? 0) + (d.spend.llm_external_facets ?? 0);
  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader title="Corpus view" sub="Composition by entry route, hop and year. Use it to spot seed bias, e.g. a sub-field visible only through the semantic route." />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={<Library className="size-4.5" />} label="Papers" value={d.papers.toLocaleString()} />
        <StatCard icon={<FileWarning className="size-4.5" />} tone="orange" label="Missing abstract" value={pct(d.missing_abstract, 1)} hint="title-only after S2 and arXiv fill" />
        <StatCard icon={<DollarSign className="size-4.5" />} tone="green" label="Metadata spend" value={`$${(d.spend.openalex ?? 0).toFixed(3)}`} />
        <StatCard icon={<DollarSign className="size-4.5" />} tone="violet" label="LLM spend" value={`$${llm.toFixed(2)}`} />
      </div>
      <Card className="mt-5" title="Papers by year and route" icon={<Layers3 className="size-4.5 text-brand-600" />}>
        <Plot height={320} data={Object.keys(ROUTE_COLORS).filter((r) => d.by_year.some((x: any) => x.route === r)).map((r) => {
          const s = d.by_year.filter((x: any) => x.route === r);
          return { type: "bar", name: r, x: s.map((x: any) => x.year), y: s.map((x: any) => x.n), marker: { color: ROUTE_COLORS[r] } };
        })} layout={{ barmode: "stack", xaxis: { type: "category", categoryorder: "array", categoryarray: years } }} />
      </Card>
      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Card title="Routes and hops"><DataTable rows={d.routes} cols={[
          { key: "route", label: "Route" }, { key: "hop", label: "Hop" }, { key: "papers", label: "Papers", align: "right" },
          { key: "missing_abstract", label: "Missing abstract", align: "right", render: (r) => pct(r.missing_abstract, 1) },
          { key: "s2_filled", label: "S2 filled", align: "right" }, { key: "arxiv_filled", label: "arXiv filled", align: "right" },
          { key: "first_year", label: "First" }, { key: "last_year", label: "Last" },
        ]} /></Card>
        <Card title="Primary topics: citation routes vs semantic route"><DataTable rows={d.topics} cols={[
          { key: "topic", label: "Topic" }, { key: "citation_routes", label: "Citation routes", align: "right" }, { key: "semantic", label: "Semantic", align: "right" },
        ]} /></Card>
      </div>
      <div className="mt-5 grid gap-5 xl:grid-cols-[1.3fr_1fr]">
        <Card title="Browse papers by route" actions={<Seg size="sm" value={route} onChange={(r) => { setRoute(r); setPid(null); }} options={[...new Set<string>(d.routes.map((r: any) => r.route))]} />}>
          <DataTable rows={papers ?? []} maxHeight={460} dense rowKey={(r) => r.paper_id} onSelect={(r) => setPid(r.paper_id)} selected={(r) => r.paper_id === pid} cols={[
            { key: "year", label: "Year", width: 50 }, { key: "title", label: "Title" }, { key: "hop", label: "Hop" },
            { key: "score", label: "Score", align: "right" }, { key: "abstract_source", label: "Abstract" },
          ]} />
        </Card>
        <Card title="Paper">{pid ? <PaperCard pid={pid} /> : <Note>Select a paper.</Note>}</Card>
      </div>
      <ReviewBar view="corpus" subject={{ run_id: p }} actions={["composition checked"]} label="Composition notes (e.g. 'geometric analysis papers via the semantic route look off-field')" />
    </div>
  );
}

// ------------------------------------------------------------------ facet review

const FAITH = ["supported", "partly supported", "unsupported"];
const GL = ["correct variant", "variant unclear (ok)", "wrong variant"];

export function FacetReview() {
  const p = useProjectId();
  const qc = useQueryClient();
  const { reviewer } = useReviewer();
  const [mode, setMode] = useState("gold");
  const { data: pool } = useApi<{ pool: string[]; done: string[] }>(`/p/${p}/facets/pool` + qs({ mode, reviewer: reviewer || "anonymous" }), { staleTime: 0 });
  const [i, setI] = useState<number | null>(null);
  useEffect(() => setI(null), [mode]);
  const idx = i ?? (pool ? Math.max(0, pool.pool.findIndex((x) => !pool.done.includes(x))) : 0);
  const pid = pool?.pool[idx];
  const { data: f } = useApi<any>(pid ? `/p/${p}/facets/paper/${pid}` : null);
  const { data: acc } = useApi<any>(`/p/${p}/facets/acceptance`, { staleTime: 0 });
  const [ratings, setRatings] = useState<Record<string, string>>({});
  const [note, setNote] = useState("");
  useEffect(() => { setRatings({}); setNote(""); }, [pid]);
  const save = useMutation({
    mutationFn: async () => {
      for (const [k, v] of Object.entries(ratings))
        await api(`/p/${p}/reviews`, { method: "POST", json: { view: "facet_review", reviewer: reviewer || "anonymous", action: v, note,
          subject: { paper_id: pid, facet: k, model_id: f.model, prompt_version: f.prompt_version, gold_set: f.gold } } });
    },
    onSuccess: () => {
      qc.invalidateQueries({ predicate: (q) => String(q.queryKey[0]).includes("/facets/") });
      setI(Math.min((pool?.pool.length ?? 1) - 1, idx + 1));
    },
  });
  const fields = f ? [...f.faithful.map((x: any) => ({ ...x, opts: FAITH })), ...f.glosses.map((x: any) => ({ ...x, opts: GL, gloss: true }))] : [];
  const missing = fields.filter((x: any) => !ratings[x.facet]);
  const done = pool ? pool.done.length : 0;
  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader title="Facet review" sub="Rate whether each extracted facet is supported by the abstract. No default rating: a preselected 'supported' would bias the gold set upwards." />
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Seg value={mode} onChange={setMode} options={[{ value: "gold", label: "Gold set (200, fixed sample)" }, { value: "flagged", label: "QA-flagged" }, { value: "limitations", label: "With limitations" }]} />
        {pool && <div className="flex min-w-[240px] flex-1 items-center gap-2 text-[12.5px]"><Bar value={done / Math.max(pool.pool.length, 1)} className="max-w-xs" /> {done} of {pool.pool.length} reviewed by {reviewer || "anonymous"}</div>}
      </div>
      {acc && (
        <div className="mb-4 grid gap-4 sm:grid-cols-3">
          <StatCard icon={<Library className="size-4.5" />} tone="green" label="Faithful facets supported (target ≥ 90%)" value={pct(acc.supported)} />
          <StatCard icon={<Library className="size-4.5" />} tone="orange" label="Limitations unsupported (target < 5%)" value={pct(acc.limitations_unsupported)} />
          <StatCard icon={<Library className="size-4.5" />} tone="violet" label="Gloss correct variant (target ≥ 85%)" value={pct(acc.gloss_ok)} />
        </div>
      )}
      {!pool ? <Spinner /> : !pool.pool.length ? <Note>Nothing in this pool.</Note> : (
        <>
          <div className="mb-3 flex items-center gap-2">
            <button className="btn btn-sm" onClick={() => setI(Math.max(0, idx - 1))}><ChevronLeft className="size-4" /> Previous</button>
            <button className="btn btn-sm" onClick={() => setI(Math.min(pool.pool.length - 1, idx + 1))}>Next <ChevronRight className="size-4" /></button>
            <span className="muted text-[12.5px]">Paper {idx + 1} of {pool.pool.length} · <code>{pid}</code>{pid && pool.done.includes(pid) && " · already reviewed"}</span>
          </div>
          {f && (
            <div className="grid gap-5 xl:grid-cols-2">
              <Card title={`${f.title} (${f.year})`}>
                <p className="text-[13.5px] leading-relaxed">{f.abstract_raw}</p>
                <Note>abstract source: {f.abstract_source} · tags: <code>{f.tags.problem_class}</code> / <code>{f.tags.method_family}</code> · {f.tags.contribution_type} · QA flags: {Object.keys(f.qa_flags).length ? JSON.stringify(f.qa_flags) : "none"}</Note>
              </Card>
              <Card title="Facets">
                <div className="flex flex-col gap-3">
                  {fields.map((x: any) => (
                    <div key={x.facet} className="border-b border-[var(--line)] pb-3 last:border-0">
                      <div className="text-[13px]"><b>{x.facet}</b>{x.gloss && <span className="faint"> (model knowledge)</span>}</div>
                      <div className="mt-0.5 text-[13px]">{x.text ?? <i className="faint">null</i>}</div>
                      <div className="mt-1.5 flex flex-wrap gap-1.5">
                        {x.opts.map((o: string) => (
                          <button key={o} onClick={() => setRatings({ ...ratings, [x.facet]: o })}
                            className={clsx("btn btn-sm", ratings[x.facet] === o && "!border-brand-500 !bg-[var(--chip)] !text-[var(--chip-text)]")}>{o}</button>
                        ))}
                      </div>
                    </div>
                  ))}
                  <input className="input" placeholder="Note" value={note} onChange={(e) => setNote(e.target.value)} />
                  <button className="btn btn-primary" disabled={!!missing.length || save.isPending} onClick={() => save.mutate()}>
                    {missing.length ? `Rate every field (${missing.length} left)` : "Save ratings"}
                  </button>
                  {save.error && <ErrorBox error={save.error} />}
                </div>
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ cluster review

export function ClusterReview() {
  const p = useProjectId();
  const { reviewer } = useReviewer();
  const qc = useQueryClient();
  const [sp] = useSearchParams(); // opened from a details sheet: ?facet=&alpha=&res=&sel=
  const [facet, setFacet] = useState(sp.get("facet") ?? "problem");
  const [alphaRaw, setAlpha] = useState(sp.get("alpha") ? Number(sp.get("alpha")) : 0.3);
  const [res, setRes] = useState(sp.get("res") ?? "medium");
  const alpha = GLOSS.has(facet) ? alphaRaw : 0;
  const { data: cl } = useApi<any[]>(`/p/${p}/clusters` + qs({ facet, alpha, res }));
  const [sel, setSel] = useState<string | null>(sp.get("sel"));
  const r = cl?.find((c) => c.cluster_id === sel);
  const { data: det } = useApi<any>(sel ? `/p/${p}/clusters/${encodeURIComponent(sel)}` + qs({ facet, alpha, res }) : null, { staleTime: 0 });
  const [action, setAction] = useState("approve");
  const [newLabel, setNewLabel] = useState("");
  const [mergeInto, setMergeInto] = useState("");
  const [note, setNote] = useState("");
  const save = useMutation({
    mutationFn: () => api(`/p/${p}/reviews`, { method: "POST", json: {
      view: "cluster_review", reviewer: reviewer || "anonymous", action, note, alpha: { [facet]: alpha },
      subject: { cluster_id: sel, facet, resolution: res, ...(action === "rename" ? { new_label: newLabel } : {}), ...(action === "merge" ? { merge_into: mergeInto } : {}) } } }),
    onSuccess: () => { setNote(""); qc.invalidateQueries({ predicate: (q) => String(q.queryKey[0]).includes("/clusters/") }); },
  });
  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader title="Cluster review" />
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <label><span className="label">Facet</span><select className="select" value={facet} onChange={(e) => { setFacet(e.target.value); setSel(null); }}>{["problem", "method", "innovation", "result", "limitations"].map((f) => <option key={f}>{f}</option>)}</select></label>
        <div><span className="label">Gloss weight α</span><Seg size="sm" value={alpha} onChange={(a) => { setAlpha(a); setSel(null); }} options={(GLOSS.has(facet) ? ALPHAS : [0]).map((a) => ({ value: a, label: String(a) }))} /></div>
        <div><span className="label">Resolution</span><Seg size="sm" value={res} onChange={(v) => { setRes(v); setSel(null); }} options={["coarse", "medium", "fine"]} /></div>
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Clusters">
          {!cl ? <Spinner /> : <DataTable rows={cl} maxHeight={640} dense rowKey={(c) => c.cluster_id} onSelect={(c) => setSel(c.cluster_id)} selected={(c) => c.cluster_id === sel} cols={[
            { key: "label", label: "Label", render: (c) => <span><span style={{ color: c.color }}>● </span>{c.label}</span> },
            { key: "size", label: "Size", align: "right" }, { key: "stability", label: "Stability", align: "right", render: (c) => c.stability?.toFixed(2) },
            { key: "kind", label: "Kind", render: (c) => <Pill tone={c.kind === "stable" ? "green" : c.kind === "hub" ? "amber" : "neutral"}>{c.kind}</Pill> },
          ]} />}
          <Note>stable: Jaccard ≥ 0.6 across 5 Leiden seeds. hub: unstable boundary but large and coherent; kept and reported separately. unstable: papers unassigned downstream.</Note>
        </Card>
        {r ? (
          <Card title={<span><span style={{ color: r.color }}>● </span>{r.label}</span>} subtitle={`${r.cluster_id} · labelled by ${r.label_model ?? "—"} · parent ${r.parent_cluster_id ?? "—"}`}
            actions={<CatLink cid={r.cluster_id} className="btn btn-sm !no-underline">Details sheet</CatLink>}>
            {r.description && <p className="text-[13px]">{r.description}</p>}
            <div className="mt-2 text-[13px]"><b>{r.size} papers · stability {r.stability?.toFixed(2)}</b></div>
            <div className="mt-1.5 flex flex-wrap gap-1">{(r.top_terms ?? []).map((t: string) => <Pill key={t}>{t}</Pill>)}</div>
            {det?.series?.length > 0 && <Plot height={140} data={[{ type: "scatter", mode: "lines+markers", x: det.series.map((s: any) => s.year), y: det.series.map((s: any) => s.share), line: { color: r.color } }]} layout={{ yaxis: { tickformat: ".0%" }, margin: { l: 40, r: 10, t: 6, b: 24 } }} />}
            <div className="h-title mt-2 mb-1.5 text-[13px]">Central papers (facet text)</div>
            <DataTable rows={det?.central ?? []} dense maxHeight={320} cols={[{ key: "year", label: "Year", width: 48 }, { key: "title", label: "Title" }, { key: "text", label: "Facet text" }, { key: "sim", label: "Sim", align: "right" }]} />
            <div className="mt-3 rounded-lg border border-[var(--line)] bg-[var(--surface-2)] p-3">
              <div className="muted mb-2 text-[12px] font-semibold">Your judgment</div>
              <Seg size="sm" value={action} onChange={setAction} options={["approve", "rename", "merge", "split", "flag"]} />
              {action === "rename" && <input className="input mt-2" placeholder="New label" value={newLabel} onChange={(e) => setNewLabel(e.target.value)} />}
              {action === "merge" && (
                <select className="select mt-2" value={mergeInto} onChange={(e) => setMergeInto(e.target.value)}>
                  <option value="">Merge into…</option>
                  {cl!.filter((c) => c.cluster_id !== sel).map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.label}</option>)}
                </select>
              )}
              <input className="input mt-2" placeholder="Note (why; for split, how)" value={note} onChange={(e) => setNote(e.target.value)} />
              <button className="btn btn-primary mt-2" disabled={save.isPending || (action === "merge" && !mergeInto)} onClick={() => save.mutate()}>Save</button>
              {save.isSuccess && <span className="ml-2 text-[12px] text-emerald-600">Saved.</span>}
            </div>
            {det?.reviews?.length > 0 && <div className="mt-3"><DataTable rows={det.reviews} dense cols={[{ key: "reviewer", label: "Reviewer" }, { key: "action", label: "Action" }, { key: "note", label: "Note" }, { key: "ts", label: "When" }]} /></div>}
          </Card>
        ) : <Card><Note>Select a cluster.</Note></Card>}
      </div>
    </div>
  );
}
