import { Microscope } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, qs, useApi } from "../api";
import { DataTable } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { ReviewBar } from "../components/PaperDrawer";
import { PaperList } from "../components/Sort";
import { Plot } from "../components/Plot";
import { Card, Empty, Note, Seg, Spinner, StatCard } from "../components/ui";
import { useProjectId } from "../context";
import { useQuery } from "@tanstack/react-query";

const PALETTE = ["#1f6ad6", "#e8590c", "#2f9e44", "#ae3ec9", "#f59f00", "#868e96", "#0ca678", "#d6336c"];

function Evidence({ ids }: { ids: string[] }) {
  const p = useProjectId();
  const { data } = useQuery({
    queryKey: ["titles", p, ids.join(",")],
    queryFn: () => api<any[]>(`/p/${p}/titles`, { method: "POST", json: ids }),
    enabled: ids.length > 0,
  });
  if (!ids.length) return null;
  return (
    <PaperList rows={data ?? []} id="lab-evidence" className="mt-2" dense />
  );
}

function Dynamics({ d }: { d: any }) {
  const p = useProjectId();
  const [sparse, setSparse] = useState(false);
  const [sel, setSel] = useState<string | null>(null);
  const year = d.audit.complete_through;
  const routes = [...new Set<string>(d.routes.map((r: any) => r.route))];
  const trends = (d.trends as any[]).filter((t) => sparse || (t.before_n >= 20 && t.after_n >= 20));
  const { data: tr } = useApi<any>(sel ? `/p/${p}/lab/trend/${encodeURIComponent(sel)}` : null);
  const series = useMemo(() => {
    if (!tr) return [];
    const rs = [...new Set<string>(tr.series.map((r: any) => r.route))];
    return rs.map((r, i) => {
      const s = tr.series.filter((x: any) => x.route === r).sort((a: any, b: any) => a.year - b.year);
      return { type: "scatter", mode: "lines+markers", name: r, x: s.map((x: any) => x.year), y: s.map((x: any) => x.share), line: { color: PALETTE[i % 8] },
        customdata: s.map((x: any) => [x.n, x.denominator]), hovertemplate: "%{x}: %{y:.1%} · %{customdata[0]} / %{customdata[1]} works<extra>" + r + "</extra>" };
    });
  }, [tr]);
  return (
    <div className="flex flex-col gap-4">
      <Plot height={280} data={routes.map((r, i) => {
        const s = d.routes.filter((x: any) => x.route === r);
        return { type: "bar", name: r, x: s.map((x: any) => x.year), y: s.map((x: any) => x.share), customdata: s.map((x: any) => x.n), marker: { color: PALETTE[i % 8] }, hovertemplate: "%{x}: %{y:.1%} · %{customdata} records<extra>" + r + "</extra>" };
      })} layout={{ barmode: "stack", yaxis: { tickformat: ".0%", title: { text: "Sampling route share" } } }} />
      <Note>{year + 1} is partial and is excluded from the comparison below. Provenance records the first route that found each paper; route shares are not inclusion probabilities.</Note>
      <div className="flex items-center justify-between gap-3">
        <p className="text-[13px]">Compare {year - 5}–{year - 3} with {year - 2}–{year}. The observed share change splits exactly into change within routes and change from the route mix. A sign reversal means the direction is sensitive to sampling.</p>
        <label className="flex shrink-0 items-center gap-1.5 text-[12.5px]"><input type="checkbox" checked={sparse} onChange={(e) => setSparse(e.target.checked)} className="accent-[var(--color-brand-600)]" /> Include small communities</label>
      </div>
      <DataTable rows={trends} rowKey={(r) => r.cluster_id} onSelect={(r) => setSel(r.cluster_id)} selected={(r) => r.cluster_id === sel} cols={[
        { key: "label", label: "Community" }, { key: "kind", label: "Kind" },
        { key: "raw_change", label: "Observed", align: "right", render: (r) => r.raw_change.toFixed(4) },
        { key: "within_route_change", label: "Within routes", align: "right", render: (r) => r.within_route_change.toFixed(4) },
        { key: "composition_change", label: "Route mix", align: "right", render: (r) => r.composition_change.toFixed(4) },
        { key: "sign_flip", label: "Sign flip", render: (r) => (r.sign_flip ? <span className="text-red-600">yes</span> : "no") },
        { key: "before_n", label: "Before n", align: "right" }, { key: "after_n", label: "After n", align: "right" },
      ]} />
      {sel && tr && (
        <Card title={trends.find((t) => t.cluster_id === sel)?.label}>
          <Plot height={300} data={series} layout={{ yaxis: { tickformat: ".1%" } }} />
          <Note>Very small route denominators make individual trajectories noisy. This sensitivity analysis cannot recover an unbiased field trend.</Note>
          <details className="mt-2"><summary className="muted cursor-pointer text-[12.5px] font-medium">Recent evidence papers</summary>
            <PaperList rows={tr.evidence} id="lab-trend-evidence" className="mt-2" dense />
          </details>
        </Card>
      )}
    </div>
  );
}

function LabBridges({ d }: { d: any }) {
  const p = useProjectId();
  const [cluster, setCluster] = useState<string>(d.bridge_clusters[0]?.cluster_id ?? "");
  const [mode, setMode] = useState("broader");
  const [i, setI] = useState(0);
  const { data } = useApi<any[]>(cluster ? `/p/${p}/lab/bridges` + qs({ cluster, mode }) : null);
  const row = data?.[i];
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2">
        <select className="select max-w-md" value={cluster} onChange={(e) => { setCluster(e.target.value); setI(0); }}>
          {d.bridge_clusters.map((c: any) => <option key={c.cluster_id} value={c.cluster_id}>{c.label} · {c.kind} · {c.cluster_id.split(":").pop()}</option>)}
        </select>
        <Seg size="sm" value={mode} onChange={(m) => { setMode(m); setI(0); }} options={[{ value: "broader", label: "Broader methods" }, { value: "likely", label: "Likely adoption" }, { value: "contrastive", label: "Contrastive experiment" }]} />
      </div>
      <Note>Scores rank likely citation adoption, not demonstrated usefulness or scientific opportunity. Broader methods limits each list to three papers per method community. Aliases already cited by this community are excluded.</Note>
      {!data ? <Spinner /> : (
        <DataTable rows={data.map((r, k) => ({ ...r, _k: k }))} rowKey={(r) => r._k} onSelect={(r) => setI(r._k)} selected={(r) => r._k === i} cols={[
          { key: "title", label: "Paper" }, { key: "pop", label: "Pop.", align: "right" }, { key: "age", label: "Age", align: "right" },
          { key: "method_label", label: "Method community" }, { key: "n_lim", label: "Limitations", align: "right" },
          { key: "baseline_model", label: "Baseline", align: "right" }, { key: "contrastive_model", label: "Contrastive", align: "right" },
        ]} />
      )}
      {row && (
        <Card title="Candidate">
          {row.limitation_text ? (
            <div className="text-[13px]">
              <div className="h-title text-[12.5px]">Stated limitation in the target community</div><p>{row.limitation_text}</p>
              <div className="h-title mt-2 text-[12.5px]">Candidate innovation</div><p>{row.innovation_text}</p>
              <Note>This match is a hypothesis. Read the papers to check assumptions, chronology and compatibility.</Note>
            </div>
          ) : <Note>This community has no retained explicit limitation for this candidate. The score uses problem similarity as fallback.</Note>}
          <Evidence ids={[row.i, ...(row.best_limitation ? [row.best_limitation] : [])]} />
          <ReviewBar view="research_lab_bridge" label="Expert judgment" subject={{ cluster_id: cluster, paper_id: row.i, limitation_id: row.best_limitation, list: mode, snapshot: d.audit.as_of }}
            actions={["useful and non-obvious", "obvious", "wrong", "uncertain"]} alpha={{ problem: 0.5, method: 0.5 }} />
        </Card>
      )}
    </div>
  );
}

function Underconnected({ d }: { d: any }) {
  const p = useProjectId();
  const [group, setGroup] = useState("stable");
  const rows = (d.opportunities as any[]).filter((o) => (group === "hub") === (o.source_kind === "hub" || o.target_kind === "hub"))
    .map((o) => ({ ...o, ratio: o.observed / o.expected, pair: `${o.source_label} → ${o.target_label}` }));
  const [sel, setSel] = useState<any>(null);
  const row = sel ?? rows[0];
  const { data: ev } = useApi<any[]>(row ? `/p/${p}/lab/opportunity` + qs({ source: row.source, target: row.target }) : null);
  return (
    <div className="flex flex-col gap-3">
      <p className="text-[13px]">These communities are semantically near but cite each other less than a degree-based null model expects. The null accounts for each community's outgoing and incoming citation volume. Counts use the last three completed publication years.</p>
      <Note>A deficit is descriptive. Topic boundaries, homophily and incompatible assumptions can explain it; these pairs have not been validated as research opportunities.</Note>
      <Seg size="sm" value={group} onChange={(g) => { setGroup(g); setSel(null); }} options={[{ value: "stable", label: "Stable communities" }, { value: "hub", label: "Includes a hub" }]} />
      {rows.length ? (
        <>
          <Plot height={320} data={[{ type: "scatter", mode: "markers", x: rows.map((r) => r.semantic_similarity), y: rows.map((r) => r.ratio), text: rows.map((r) => r.pair),
            marker: { size: rows.map((r) => 6 + Math.sqrt(r.expected) * 2), color: rows.map((r) => r.score), colorscale: "Blues", showscale: true, line: { width: 1, color: "var(--ring)" } },
            hovertemplate: "%{text}<br>similarity %{x:.3f} · observed/expected %{y:.2f}<extra></extra>" }]}
            layout={{ xaxis: { title: { text: "semantic similarity" } }, yaxis: { title: { text: "observed / expected" } }, margin: { l: 50, r: 10, t: 10, b: 40 } }} />
          <DataTable rows={rows} rowKey={(r) => r.pair} onSelect={setSel} selected={(r) => r.pair === row?.pair} cols={[
            { key: "source_label", label: "From" }, { key: "target_label", label: "To" },
            { key: "observed", label: "Observed", align: "right" }, { key: "expected", label: "Expected", align: "right" },
            { key: "semantic_similarity", label: "Similarity", align: "right" }, { key: "score", label: "Score", align: "right" },
          ]} />
          {row && (
            <Card title={row.pair} subtitle={`Observed ${row.observed} citations; null expectation ${row.expected.toFixed(1)}. Evidence is the nearest pair of problem facets, not a confirmed solution transfer.`}>
              <PaperList rows={ev ?? []} id="lab-pair-evidence" dense />
              <ReviewBar view="research_lab_opportunity" label="Opportunity judgment" subject={{ source: row.source, target: row.target, snapshot: d.audit.as_of }}
                actions={["useful and non-obvious", "obvious", "wrong", "uncertain"]} alpha={{ problem: 0.3 }} />
            </Card>
          )}
        </>
      ) : <Note>No pairs satisfy the minimum exposure and semantic-nearness gates.</Note>}
    </div>
  );
}

function PeopleByTopic({ d }: { d: any }) {
  const p = useProjectId();
  const [topic, setTopic] = useState<string>(d.topics[0]?.cluster_id ?? "");
  const [role, setRole] = useState("external_adoption");
  const [sel, setSel] = useState<any>(null);
  const { data } = useApi<any[]>(topic ? `/p/${p}/lab/people` + qs({ topic, role }) : null);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2">
        <select className="select max-w-md" value={topic} onChange={(e) => { setTopic(e.target.value); setSel(null); }}>
          {d.topics.map((t: any) => <option key={t.cluster_id} value={t.cluster_id}>{t.label} · {t.cluster_id.split(":").pop()}</option>)}
        </select>
        <Seg size="sm" value={role} onChange={setRole} options={[{ value: "external_adoption", label: "Recent external adoption" }, { value: "recent_fractional_works", label: "Recent contribution" }, { value: "participation", label: "Cross-community participation" }]} />
      </div>
      <Note>Internal topic-specific roles. Works are deduplicated, author credit is fractional, and external adoption excludes citations from teams sharing any author. Counts reflect this sampled corpus. Participation describes breadth, not influence.</Note>
      {!data ? <Spinner /> : (
        <DataTable rows={data} rowKey={(r) => r.author_id} onSelect={setSel} selected={(r) => r.author_id === sel?.author_id} cols={[
          { key: "name", label: "Name" }, { key: "recent_fractional_works", label: "Recent works", align: "right" },
          { key: "external_adoption", label: "External adoption", align: "right" }, { key: "participation", label: "Participation", align: "right" },
          { key: "topics", label: "Topics", align: "right" }, { key: "first_topic_year", label: "First year", align: "right" },
        ]} />
      )}
      {sel && <Card title={`Evidence for ${sel.name}`}><Evidence ids={sel.evidence} /></Card>}
    </div>
  );
}

function LabExperiments({ d }: { d: any }) {
  const protocols = [...new Set<string>(d.metrics.map((m: any) => m.protocol))];
  const [protocol, setProtocol] = useState(protocols[0]);
  const scores = d.metrics.filter((m: any) => m.protocol === protocol);
  const summary = useMemo(() => {
    const by = new Map<string, any[]>();
    scores.forEach((s: any) => by.set(s.ranking, [...(by.get(s.ranking) ?? []), s]));
    const mean = (xs: any[], k: string) => xs.reduce((a, x) => a + (x[k] ?? 0), 0) / xs.length;
    return [...by.entries()].map(([ranking, xs]) => ({ ranking, "p@10": mean(xs, "p@10"), "methods@10": mean(xs, "methods@10"), auc: mean(xs, "auc"), tail_auc: mean(xs, "tail_auc") }));
  }, [scores]);
  const chosen = summary.filter((s) => ["lpop", "baseline_model", "contrastive_model", "diverse_baseline"].includes(s.ranking));
  return (
    <div className="flex flex-col gap-3">
      <select className="select max-w-md" value={protocol} onChange={(e) => setProtocol(e.target.value)}>{protocols.map((x) => <option key={x}>{x}</option>)}</select>
      <Plot height={280} data={[{ type: "bar", x: chosen.map((c) => c.ranking), y: chosen.map((c) => c["p@10"]), marker: { color: "#1f6ad6" } }]} layout={{ yaxis: { tickformat: ".0%", title: { text: "Future adoption precision@10" } } }} />
      <DataTable rows={summary} />
      <Note>{d.audit.protocols?.[protocol]}</Note>
      <p className="text-[13px]">All rankings share the same alias-aware candidate pool. Centering and CSLS remove common semantic directions and generic matches. The contrastive model adds these features to the existing five-feature model. No experimental model replaced production weights.</p>
      <details><summary className="muted cursor-pointer text-[12.5px] font-medium">Per-cutoff results and geometry audit</summary>
        <div className="mt-2 flex flex-col gap-3"><DataTable rows={scores} maxHeight={360} /><DataTable rows={d.geometry} /></div>
      </details>
      {(d.audit.limitations ?? []).map((l: string, k: number) => <Note key={k}>{l}</Note>)}
    </div>
  );
}

export function ResearchLab() {
  const p = useProjectId();
  const { data: d, isLoading } = useApi<any>(`/p/${p}/lab`);
  const [tab, setTab] = useState("Dynamics");
  if (isLoading) return <Spinner />;
  if (!d?.built) return (
    <div className="mx-auto max-w-3xl"><PageHeader title="Research lab" />
      <Empty title="The research lab is not built" icon={<Microscope className="size-8 text-brand-500" />}>
        It needs the backtest cutoffs, the 2019 cutoff and the research-lab step. <Link to={`/p/${p}/pipeline`} className="text-brand-600">Queue them on the Pipeline page →</Link>
      </Empty></div>
  );
  const a = d.audit;
  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader title="Research lab" sub={`Offline experiments · snapshot ${a.as_of} · completed publication years through ${a.complete_through} · corpus selected retrospectively`} />
      <p className="mb-4 max-w-4xl text-[13.5px]">Inspect how sampling changes the story, compare bridge lists, and explore underconnected communities. These are candidates for expert review; similarity alone does not show that a problem is solved.</p>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={<Microscope className="size-4.5" />} label="Unique works" value={a.unique_works?.toLocaleString()} />
        <StatCard icon={<Microscope className="size-4.5" />} tone="orange" label="Explicit limitations" value={a.explicit_limitations} />
        <StatCard icon={<Microscope className="size-4.5" />} tone="violet" label="Trend sign reversals" value={`${a.supported_direction_sign_flips} / ${a.supported_direction_clusters}`} />
        <StatCard icon={<Microscope className="size-4.5" />} tone="green" label="Underconnected pairs" value={a.flow_opportunities} />
      </div>
      <div className="card mt-5">
        <div className="scroll-thin flex overflow-x-auto border-b border-[var(--line)] px-5">
          {["Dynamics", "Paper bridges", "Underconnected communities", "People by topic", "Experiments"].map((t) => <button key={t} className={"tab" + (t === tab ? " active" : "")} onClick={() => setTab(t)}>{t}</button>)}
        </div>
        <div className="p-5">
          {tab === "Dynamics" && <Dynamics d={d} />}
          {tab === "Paper bridges" && <LabBridges d={d} />}
          {tab === "Underconnected communities" && <Underconnected d={d} />}
          {tab === "People by topic" && <PeopleByTopic d={d} />}
          {tab === "Experiments" && <LabExperiments d={d} />}
        </div>
      </div>
    </div>
  );
}
