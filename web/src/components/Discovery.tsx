import { useState } from "react";
import { CommunityPicker } from "./Facets";
import { PaperLink } from "./PaperDrawer";
import { HowToRead, Note, Pill } from "./ui";

type Ref = { paper_id: string; title: string; year?: number | null };

/** Papers that build on a quiet foundation without citing it, behind a toggle. */
export function Builders({ rows }: { rows?: Ref[] }) {
  const [open, setOpen] = useState(false);
  if (!rows?.length) return null;
  return (
    <span className="block">
      <button className="text-[11.5px] text-brand-600 hover:underline" onClick={() => setOpen(!open)}>
        {open ? "Hide" : `${rows.length} recent paper${rows.length > 1 ? "s" : ""} build on it without citing it`}
      </button>
      {open && (
        <ul className="mt-1 flex flex-col gap-0.5 border-l-2 border-[var(--line)] pl-2">
          {rows.map((b) => <li key={b.paper_id} className="text-[12px]"><span className="faint mr-1.5">{b.year}</span><PaperLink pid={b.paper_id} title={b.title} /></li>)}
        </ul>
      )}
    </span>
  );
}

// ------------------------------------------------------------------ quiet foundations

export function QuietFoundations({ data }: { data: { rows: any[]; evidence?: string | null } }) {
  return (
    <>
      <ol className="flex flex-col">
        {data.rows.map((r, i) => (
          <li key={r.paper_id} className="flex gap-3 border-b border-[var(--line)] py-2 text-[13px] last:border-0">
            <span className="faint w-5 shrink-0 text-right tabular-nums">{i + 1}</span>
            <span className="min-w-0 flex-1">
              <PaperLink pid={r.paper_id} title={r.title} /> <span className="faint text-[11.5px]">{r.year}</span>
              <span className="muted block text-[11.5px]">
                earliest source of {r.attributed_quiet} later papers that do not cite it · cited by {r.cites_by_others} others
                {r.outside_top50_cited && <Pill tone="amber" className="ml-1.5">not among the 50 most cited</Pill>}
              </span>
              <Builders rows={r.builders} />
            </span>
          </li>
        ))}
      </ol>
      {!data.rows.length && <div className="muted text-[13px]">None for this community.</div>}
      {data.evidence && <HowToRead>{"Quiet foundations are papers that later work builds on (closest earliest source of its ideas) far more often than it cites them. " + data.evidence}</HowToRead>}
    </>
  );
}

// ------------------------------------------------------------------ tools to learn

const STATUS: Record<string, { tone: "green" | "amber" | "neutral"; head: string }> = {
  settled: { tone: "green", head: "Learn these" },
  "in between": { tone: "neutral", head: "Also used" },
  "in flux": { tone: "amber", head: "Watch these" },
};

export function ToolsList({ rows }: { rows: any[] }) {
  return (
    <div className="flex flex-col gap-3">
      {["settled", "in between", "in flux"].map((s) => {
        const part = rows.filter((r) => r.status === s);
        if (!part.length) return null;
        return (
          <div key={s}>
            <div className="mb-1 flex items-center gap-2 text-[12.5px] font-semibold"><Pill tone={STATUS[s].tone}>{s}</Pill> {STATUS[s].head}</div>
            <ul className="flex flex-col">
              {part.map((r) => (
                <li key={r.method_cluster} className="border-b border-[var(--line)] py-1.5 text-[13px] last:border-0">
                  <span className="text-[var(--text-strong)]">{r.method_label}</span>
                  <span className="faint ml-2 text-[11.5px]">{r.papers_here_recent} recent papers here</span>
                  {r.examples?.length > 0 && (
                    <span className="muted block text-[12px]">e.g. {r.examples.slice(0, 2).map((e: Ref, i: number) => (
                      <span key={e.paper_id}>{i > 0 && "; "}<PaperLink pid={e.paper_id} title={e.title} /></span>))}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </div>
  );
}

export function ToolsToLearn({ data, picker = true }: { data: { rows: any[]; evidence?: string | null }; picker?: boolean }) {
  const clusters = [...new Map(data.rows.map((r) => [r.cluster_id, r.label])).entries()];
  const [c, setC] = useState(clusters[0]?.[0] ?? "");
  const rows = data.rows.filter((r) => r.cluster_id === c);
  return (
    <div>
      {picker && clusters.length > 1 && (
        <CommunityPicker value={c} onChange={setC} options={clusters} />
      )}
      <ToolsList rows={rows} />
      <Note>Method communities this community used in {rows[0]?.recent_years}. <b>Settled</b> tools are used broadly, with few competing variants and little innovation on the tool itself; <b>in flux</b> ones are still changing.</Note>
      {data.evidence && <HowToRead>{data.evidence}</HowToRead>}
    </div>
  );
}

// ------------------------------------------------------------------ untried applications

export function UntriedApplications({ data }: { data: { rows: any[]; evidence?: string | null } }) {
  const personal = data.rows.some((r) => r.paper_fit != null);
  return (
    <div>
      <Pill tone="amber">early evidence · small sample</Pill>
      {!data.rows.length && <div className="muted mt-2 text-[13px]">Your home community already uses every application domain on this map, or is too small to rank.</div>}
      <ol className="mt-2 flex flex-col">
        {data.rows.map((r, i) => (
          <li key={r.domain_id} className="border-b border-[var(--line)] py-2 text-[13px] last:border-0">
            <div className="flex items-baseline gap-2">
              <span className="faint w-5 shrink-0 text-right tabular-nums">{i + 1}</span>
              <span className="h-title text-[13px]">Domain {r.domain_id}</span>
              <span className="faint text-[11.5px]">{r.domain_papers} papers on the map</span>
              <span className="ml-auto text-[12px] tabular-nums">
                {personal && <span className="mr-2" title="Your problem statement vs the domain">yours {r.paper_fit?.toFixed(3)}</span>}
                <span className="faint" title="The community's problems vs the domain">community {r.fit.toFixed(3)}</span>
              </span>
            </div>
            <ul className="muted mt-1 ml-7 list-disc pl-4 text-[12.5px]">{r.domain_examples.map((t: string, k: number) => <li key={k}>{t}</li>)}</ul>
          </li>
        ))}
      </ol>
      <Note>Application domains your home community has not used yet, closest to {personal ? "your problem" : "its problems"} first. Domains are clusters of the data and settings papers describe, so the example statements say what each one is.</Note>
      {data.evidence && <HowToRead>{data.evidence}</HowToRead>}
    </div>
  );
}

// ------------------------------------------------------------------ crowding gauge

export function CrowdingGauge({ c }: { c: any }) {
  if (!c?.available) return <div className="muted text-[13px]">{c?.reason ?? "Not built."}</div>;
  const pct = c.percentile ?? 0;
  return (
    <div>
      <div className="grid grid-cols-3 gap-2">
        {[["same_question", "Same question", "papers in the window ask it"], ["near_duplicates", "Close twins", "same question and method"], ["limitations_answered", "Limitations answered", "stated by other papers"]].map(([k, l, s]) => (
          <div key={k} className="rounded-lg bg-[var(--surface-2)] px-3 py-2">
            <div className="muted text-[11.5px]">{l}</div>
            <div className="h-title text-[22px] leading-tight">{c[k]}</div>
            <div className="faint text-[11px]">{s}</div>
          </div>
        ))}
      </div>
      <div className="mt-3">
        <div className="muted mb-1 flex justify-between text-[11.5px]"><span>quiet question</span><span>busy question</span></div>
        <div className="relative h-2 rounded-full bg-gradient-to-r from-[var(--line)] via-brand-200 to-brand-500">
          <span className="absolute -top-1 size-4 -translate-x-1/2 rounded-full border-2 border-white bg-brand-600 shadow" style={{ left: `${pct * 100}%` }} />
        </div>
        <div className="faint mt-1 text-[11.5px]">Busier than {(pct * 100).toFixed(0)}% of the papers on this map ({c.window[0]}–{c.window[1]}).</div>
      </div>
      {c.same_question_examples.length > 0 && (
        <div className="mt-2 text-[12.5px]"><span className="muted">Asking the same question:</span>
          <ul className="mt-0.5">{c.same_question_examples.map((e: Ref) => <li key={e.paper_id}><span className="faint mr-1.5">{e.year}</span><PaperLink pid={e.paper_id} title={e.title} /></li>)}</ul></div>
      )}
      {c.limitation_examples.length > 0 && (
        <div className="mt-2 text-[12.5px]"><span className="muted">Limitations it may answer:</span>
          <ul className="mt-0.5">{c.limitation_examples.map((e: Ref) => <li key={e.paper_id}><span className="faint mr-1.5">{e.year}</span><PaperLink pid={e.paper_id} title={e.title} /></li>)}</ul></div>
      )}
      <Note>Crowded questions draw attention: on this map, papers entering busy questions, and even those with close twins, were taken up more over three years. Answering a stated limitation drew the community's later work towards the paper.{c.computed_here && " Computed for this paper against the map; your own earlier papers are not left out."}</Note>
      {c.evidence && <HowToRead>{c.evidence}</HowToRead>}
    </div>
  );
}

export const quietRoleLabel = (role: string) => (role === "quiet foundation" ? "built on, rarely cited" : role);
export const roleTone = (role: string): "brand" | "violet" | "amber" => (role === "classic" ? "brand" : role === "quiet foundation" ? "amber" : "violet");
