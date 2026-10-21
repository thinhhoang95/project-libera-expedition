import { Sparkles } from "lucide-react";
import { useMemo, useState } from "react";
import { DataTable } from "./DataTable";
import { CatLink } from "./Facets";
import { PaperLink } from "./PaperDrawer";
import { FadePill, FlowState, LineTrend, StatusPill, TIMING_HOW, TimingPill, yearSpan, type FlowChange, type Line, type PairWhen, type PaperWhen } from "./Timing";
import { Card, HowToRead, Note, Seg } from "./ui";

type Opps = {
  as_of: number;
  gems: { paper_id: string; title: string; community: string; cluster_id: string; from_cluster: string; when: PaperWhen | null; to_line: Line | null;
    status: string | null; age: number | null; cites: number | null }[];
  lim_inn: { from: string; to: string; from_cluster: string; to_cluster: string; judge: number | null; why: string; when: PairWhen;
    from_line: Line | null; to_line: Line | null; flow: FlowChange; status: string | null }[];
  method_transfer: { method: string; problem: string; method_cluster: string; problem_cluster: string; judge: number | null;
    method_line: Line | null; problem_line: Line | null; elsewhere_since: number | null; elsewhere_recent: number; elsewhere_before: number }[];
  underconnected: { source_label: string; target_label: string; source: string; target: string; observed: number; expected: number;
    source_line: Line | null; target_line: Line | null; flow: FlowChange }[] | null;
};

const GEM_FILTERS = [
  { value: "all", label: "Any time" }, { value: "fresh", label: "Fresh" },
  { value: "active line", label: "Active line" }, { value: "revive", label: "Fading or revival" },
];

/** Start here › Opportunities to inspect, each placed in time. */
export function OpportunitiesCard({ o }: { o: Opps }) {
  const [opp, setOpp] = useState("gems");
  const [when, setWhen] = useState("all");
  const T = o.as_of;
  const gems = useMemo(() => o.gems.filter((g) => when === "all" || (when === "revive"
    ? ["fading line", "quiet line"].includes(g.when?.timing ?? "") : g.when?.timing === when)), [o, when]);
  return (
    <Card className="mt-5" title="Opportunities to inspect" icon={<Sparkles className="size-4.5 text-brand-600" />}
      subtitle={`Each placed in time as of ${T}: is it fresh, part of a line of work that is active now, or a revival of one that has gone quiet?`}
      actions={<Seg size="sm" value={opp} onChange={setOpp} options={[
        { value: "gems", label: "Hidden gems" }, { value: "lim", label: "Limitation → innovation" },
        { value: "mt", label: "Method transfer" }, { value: "uc", label: "Underconnected" },
      ]} />}>
      {opp === "gems" && (<>
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <Seg size="sm" value={when} onChange={setWhen} options={GEM_FILTERS} />
          <span className="faint text-[12px]">{gems.length} of {o.gems.length}</span>
        </div>
        <DataTable rows={gems} rowKey={(r) => r.paper_id} empty="None with this timing." cols={[
          { key: "title", label: "Paper", render: (r) => <PaperLink pid={r.paper_id} title={r.title} /> },
          { key: "year", label: "Year", align: "right", value: (r) => r.when?.year, render: (r) => r.when?.year ?? "—" },
          { key: "status", label: "Status", render: (r) => (
            <span className="flex flex-col items-start gap-0.5"><StatusPill s={r.status} /><FadePill age={r.age} cites={r.cites} /></span>
          ) },
          { key: "timing", label: "Timing", value: (r) => r.when?.timing, render: (r) => (
            <span className="flex flex-col items-start gap-0.5">
              <TimingPill t={r.when?.timing} />
              <span className="faint whitespace-nowrap text-[11px]">{r.when?.last_cited ? `last cited ${r.when.last_cited}` : "never cited here"}</span>
            </span>
          ) },
          { key: "line", label: "Its line of work", sortable: false, render: (r) => <LineTrend line={r.when?.line} compact /> },
          { key: "community", label: "For community", render: (r) => (
            <span className="flex flex-col gap-0.5"><CatLink cid={r.cluster_id} label={r.community} /><LineTrend line={r.to_line} compact /></span>
          ) },
        ]} />
        <Note>Papers cited at most once that a community is likely to take up, where popularity is at chance (experiment 1). The ranking adds whether the community is already converging on the paper's idea: “being taken” gems were cited about 7× as often (experiment 30). The timing columns are context, not part of the model.</Note>
      </>)}
      {opp === "lim" && (<>
        <DataTable rows={o.lim_inn} rowKey={(_, i) => i} cols={[
          { key: "from", label: "Limitations in", render: (r) => <span className="flex flex-col gap-0.5"><CatLink cid={r.from_cluster} label={r.from} /><LineTrend line={r.from_line} compact /></span> },
          { key: "to", label: "Innovations in", render: (r) => <span className="flex flex-col gap-0.5"><CatLink cid={r.to_cluster} label={r.to} /><LineTrend line={r.to_line} compact /></span> },
          { key: "when", label: "Evidence years", sortable: false, render: (r) => (
            <span className="flex flex-col items-start gap-0.5 text-[12px]">
              <TimingPill t={r.when.timing} />
              <span className="faint whitespace-nowrap">limitation {yearSpan(r.when.limitation_years)}</span>
              <span className="faint whitespace-nowrap">innovation {yearSpan(r.when.innovation_years)}</span>
            </span>
          ) },
          { key: "flow", label: "Citing already", value: (r) => r.flow.recent, render: (r) => <FlowState f={r.flow} /> },
          { key: "status", label: "Status", render: (r) => <StatusPill s={r.status} /> },
          { key: "judge", label: "Judge", align: "right" },
          { key: "why", label: "Why", render: (r) => <span className="muted clamp-2 text-[12px]" title={r.why}>{r.why}</span> },
        ]} />
        <Note>Communities whose stated limitations match innovations elsewhere, with few citations between them (Phase 8: about 2.4× random). Innovations the community already cites are left out: closed gaps made up most of this list's backtest hits, and without them about 1 in 9 of the top 20 is taken up within three years (experiment 30). Only a minority of abstracts state limitations: check the evidence in Map & grid › Bridges.</Note>
      </>)}
      {opp === "mt" && (<>
        <DataTable rows={o.method_transfer} rowKey={(_, i) => i} cols={[
          { key: "method", label: "Method", render: (r) => <span className="flex flex-col gap-0.5"><CatLink cid={r.method_cluster} label={r.method} /><LineTrend line={r.method_line} compact /></span> },
          { key: "problem", label: "Problem", render: (r) => <span className="flex flex-col gap-0.5"><CatLink cid={r.problem_cluster} label={r.problem} /><LineTrend line={r.problem_line} compact /></span> },
          { key: "elsewhere_recent", label: "Similar problems using it", align: "right", render: (r) => (
            <span className="text-[12px] tabular-nums" title={`Papers in similar problems using this method: ${T - 2}–${T} / ${T - 5}–${T - 3}`}>
              {r.elsewhere_recent} / {r.elsewhere_before}{" "}
              <span className={r.elsewhere_recent > r.elsewhere_before * 1.15 ? "font-medium text-emerald-600" : r.elsewhere_recent < r.elsewhere_before * 0.85 ? "font-medium text-[#e8590c]" : "muted"}>
                {r.elsewhere_recent > r.elsewhere_before * 1.15 ? "spreading" : r.elsewhere_recent < r.elsewhere_before * 0.85 ? "cooling" : "steady"}
              </span>
            </span>
          ) },
          { key: "judge", label: "Judge", align: "right" },
        ]} />
        <Note>Methods not yet used for a problem, though they are used for similar problems (experiment 4: AUC 0.70–0.80). Shown as diamonds in the grid. "Similar problems using it" counts papers in the last 3 years / the 3 before.</Note>
      </>)}
      {opp === "uc" && (o.underconnected ? (<>
        <DataTable rows={o.underconnected} rowKey={(_, i) => i} cols={[
          { key: "source_label", label: "From", render: (r) => <span className="flex flex-col gap-0.5"><CatLink cid={r.source} label={r.source_label} /><LineTrend line={r.source_line} compact /></span> },
          { key: "target_label", label: "To", render: (r) => <span className="flex flex-col gap-0.5"><CatLink cid={r.target} label={r.target_label} /><LineTrend line={r.target_line} compact /></span> },
          { key: "observed", label: "Citations seen", align: "right" }, { key: "expected", label: "Expected", align: "right" },
          { key: "flow", label: "Recent / before", value: (r) => r.flow.recent, render: (r) => <FlowState f={r.flow} /> },
        ]} />
        <Note>Semantically close communities that cite each other less than their size predicts (Phase 8c). Hypotheses to review, not confirmed gaps.</Note>
      </>) : <div className="muted text-[13px]">Run the research lab to add these.</div>)}
      <HowToRead title="How to read the timing">{TIMING_HOW}</HowToRead>
    </Card>
  );
}
