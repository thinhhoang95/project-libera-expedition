import clsx from "clsx";
import { BookOpen, CalendarRange, Compass, DoorOpen, Gem, Globe2, Layers, Library, Network, Sparkles, Telescope, Waves } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { useApi } from "../api";
import { DataTable } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { PaperLink } from "../components/PaperDrawer";
import { Plot } from "../components/Plot";
import { Card, Empty, HowToRead, Md, Note, Pill, Seg, Sparkline, Spinner, StatCard, pct, signedPct } from "../components/ui";
import { useProjectId } from "../context";
import { EntryRoutes, Lasting, ReadThisFirst, SurveyGaps } from "../components/Orientation";

export const SECOND_WAVES_HOW =
  "Established ideas reaching **new audiences**: communities that began citing (or, for methods, using) them in the " +
  "last three complete years. **Citing papers** compares the last three years with the three before; only ideas " +
  "gaining use are listed (research works, and outside works from the last 15 years). **Quiet uptake** counts " +
  "communities whose recent work moved towards the idea without citing it (experiment 9: most uptake is uncited). " +
  "**Likely next** comes from the backtested models: community warmth for papers (experiment 10), method transfer " +
  "for methods (experiment 4), the closest community for outside works. This is a watchlist of diffusion already " +
  "under way, not advance warning: on the pilot, the import model improved ordering modestly (+0.025 AUC) without " +
  "a sharper top of the list. Select an idea for its evidence.";

const short = (xs: string[]) => xs.slice(0, 2).join(", ") + (xs.length > 2 ? ` +${xs.length - 2}` : "");

export function SecondWavesTable({ rows }: { rows: any[] }) {
  const [sel, setSel] = useState<number | null>(null);
  const r = sel != null ? rows[sel] : null;
  return (
    <div>
      <DataTable rows={rows.map((x, i) => ({ ...x, _i: i }))} rowKey={(x) => x._i} onSelect={(x) => setSel(x._i === sel ? null : x._i)} selected={(x) => x._i === sel} cols={[
        { key: "idea", label: "Idea", render: (x) => <span className="text-[var(--text-strong)]">{x.idea}</span> },
        { key: "origin", label: "From" },
        { key: "year", label: "Year", width: 52 },
        { key: "n_new", label: "New audiences", render: (x) => `${x.n_new}: ${short(x.new_audiences)}` },
        { key: "recent", label: "Citing (3 yrs / 3 before)", render: (x) => `${x.recent} / ${x.earlier}` },
        { key: "n_quiet", label: "Quiet uptake", align: "right" },
        { key: "likely_next", label: "Likely next", sortable: false, render: (x) => short(x.likely_next) },
      ]} />
      {r && (
        <div className="mt-3 rounded-lg border border-[var(--line)] bg-[var(--surface-2)] p-3 text-[13px]">
          <div className="h-title">{r.idea}</div>
          <div className="mt-1"><span className="muted">New audiences:</span> {r.new_audiences.join(", ")}</div>
          {r.quiet_uptake.length > 0 && <div><span className="muted">Moving towards it without citing it:</span> {r.quiet_uptake.join(", ")}</div>}
          {r.likely_next.length > 0 && <div><span className="muted">Likely next:</span> {r.likely_next.join(", ")}</div>}
          <div><span className="muted">Backed by:</span> {r.backed_by}</div>
          {r.evidence.length > 0 && (
            <ul className="mt-2 flex flex-col gap-1">
              {r.evidence.map((e: any) => <li key={e.paper_id} className="flex gap-2"><span className="faint w-10 shrink-0">{e.year}</span><PaperLink pid={e.paper_id} title={e.title} /></li>)}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

const TRUST = `| View | Verdict on the pilot | Backed by |
| --- | --- | --- |
| My paper › warm communities | **Use** | experiment 10: a top-3 community cites the paper in 69% of cases (25% at random) |
| Bridges › Papers to bring in, hidden gems | **Use** | experiment 1 and the 8c chronological test (19.5% vs 13.6% for popularity in the top 10) |
| People › Early work | **Use** as a reading list | experiment 2: AUC 0.82 |
| Bridges › From outside the field | **Use** as a review list | experiment 6: modest gains (+0.025 AUC) |
| Second waves (here and on My paper) | **Use** as a watchlist | experiments 4, 6, 9, 10: diffusion already under way, not advance warning |
| Bridges › Cluster pairs, grid method transfer | **Use with care** | Phase 8, experiment 4 |
| Status badges, Ranking, Rising pockets | **Describe only** | direction is sensitive to sampling |
| People, institutions | **Describe only** | not better than popularity at predicting citations |

Every result table, including the experiments that did not work, is in **Experiments**.`;

export function StartHere() {
  const p = useProjectId();
  const { data: d, isLoading } = useApi<any>(`/p/${p}/overview`);
  const [opp, setOpp] = useState("gems");
  const [wave, setWave] = useState("paper");
  if (isLoading) return <Spinner />;
  if (!d?.built)
    return (
      <div className="mx-auto max-w-3xl">
        <PageHeader title="Start here" />
        <Empty title="The portfolio is not built yet" icon={<Compass className="size-8 text-brand-500" />}>
          Queue the core map for this project; Start here appears once <code>portfolio</code> has run.
          <div className="mt-3"><Link to={`/p/${p}/pipeline`} className="btn btn-primary">Open the pipeline</Link></div>
        </Empty>
      </div>
    );
  const m = d.metrics, T = d.as_of;
  const waves = (d.second_waves as any[]).filter((w) => w.kind === wave);
  return (
    <div className="mx-auto max-w-[1500px]">
      <PageHeader title="Start here" sub={<>As of {T} (the latest complete publication year) · built {d.built_at.slice(0, 16).replace("T", " ")}</>} />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={<BookOpen className="size-4.5" />} label="Papers in the map" value={m.papers.toLocaleString()} spark={m.papers_by_year.map((r: any) => r.n)} hint="Papers per year (last 15 years)" />
        <StatCard icon={<CalendarRange className="size-4.5" />} tone="green" label="Publication years" value={`${m.year_lo}–${m.year_hi}`} sub={<span className="muted">95% of papers</span>} hint={`The oldest reference is from ${m.year_min}; ${m.year_hi} is a partial year.`} />
        <StatCard icon={<Network className="size-4.5" />} tone="violet" label="Problem communities" value={m.stable} sub={<span className="muted">+ {m.hubs} hubs</span>} />
        <StatCard icon={<Globe2 className="size-4.5" />} tone="orange" label="References outside the map" value={pct(m.external_reference_share)} sub={<span className="muted">{m.external_works?.toLocaleString()} works tracked</span>} />
      </div>
      <details className="card mt-4 px-5 py-3">
        <summary className="h-title cursor-pointer text-[14px]">What this map covers, and what it cannot tell you</summary>
        <Md className="mt-2 text-[13px]">{
          `The map grows from one or a few seed papers through their citation graph: references, papers citing them, their authors' newest work, and semantically close papers in the same topics (${Object.entries(d.routes).map(([k, v]) => `${k} ${(v as number).toLocaleString()}`).join(", ")} papers by route). It is a **map of one field as reached from its seed**, not a sample of all of science, and parts of it were fetched in citation order.\n\n` +
          "- **Reliable here:** what to read, which papers and ideas a community is likely to take up next, and which communities are likely to be warm to a paper (all backtested; see *How far to trust each view* below).\n" +
          "- **Read with care:** whether a topic is growing or shrinking. Several topics reverse direction once changes in how the map was sampled are separated out.\n" +
          `- **Not covered:** most references point outside the map; the ${(m.external_works ?? 0).toLocaleString()} external works cited at least 5 times are tracked separately (below). Nothing here predicts citations from all of science, or where a brand-new community will form.`}</Md>
      </details>

      <div className="mt-5 grid gap-5 xl:grid-cols-[1.4fr_1fr]">
        <Card title="The communities" icon={<Layers className="size-4.5 text-brand-600" />} actions={<Link to={`/p/${p}/map`} className="btn btn-sm">Open the map</Link>}>
          <DataTable rows={d.communities} maxHeight={380} rowKey={(r) => r.cluster_id} cols={[
            { key: "label", label: "Community", render: (r) => <span className="text-[var(--text-strong)]">{r.label}{r.kind === "hub" && <span className="faint"> ◇</span>}</span> },
            { key: "size", label: "Papers", align: "right" },
            { key: "series", label: `${T - 9}–${T}`, sortable: false, render: (r) => <Sparkline values={r.series} width={80} height={22} fill={false} /> },
            { key: "recent", label: `${T - 2}–${T}`, align: "right" },
            { key: "share_now", label: "Share now", align: "right", render: (r) => pct(r.share_now, 1) },
            { key: "change", label: "vs 3 yrs before", align: "right", render: (r) => <span className={clsx(r.change > 0.005 ? "text-emerald-600" : r.change < -0.005 ? "text-red-600" : "muted")}>{signedPct(r.change)}</span> },
          ]} />
          <Note>Problem communities (◇ = hub: a large, coherent body of work with a shifting boundary). The share change describes this map as sampled; it is not a forecast.</Note>
        </Card>
        {d.import_map && (
          <Card title="Where this field borrows from" icon={<Telescope className="size-4.5 text-brand-600" />}>
            <Plot height={380} data={[
              { type: "bar", orientation: "h", y: [...d.import_map.rows].reverse().map((r: any) => r.subfield), x: [...d.import_map.rows].reverse().map((r: any) => r.share_early), name: d.import_map.early_years, marker: { color: "#c9cdd2" } },
              { type: "bar", orientation: "h", y: [...d.import_map.rows].reverse().map((r: any) => r.subfield), x: [...d.import_map.rows].reverse().map((r: any) => r.share_recent), name: d.import_map.recent_years, marker: { color: "#1f6ad6" } },
            ]} layout={{ barmode: "group", margin: { l: 10, r: 10, t: 4, b: 40 }, yaxis: { automargin: true }, xaxis: { tickformat: ".0%" }, legend: { orientation: "h", y: -0.12 } }} />
            <Note>Share of the citations to works outside the map, by the cited work's OpenAlex subfield. Experiment 6: the field imports more of what fits where its questions are moving and what trends across science.</Note>
          </Card>
        )}
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Card title="Being imported now" icon={<Globe2 className="size-4.5 text-orange-600" />}>
          <DataTable rows={d.imported_now ?? []} rowKey={(r) => r.paper_id} cols={[
            { key: "title", label: "Work", render: (r) => <a href={`https://openalex.org/${r.paper_id}`} target="_blank" rel="noreferrer" className="hover:text-brand-600">{r.title}</a> },
            { key: "subfield", label: "Subfield" },
            { key: "recent_citers", label: "Citing here", align: "right" },
            { key: "cluster_label", label: "Closest community" },
          ]} empty="Not built: run `external` and `portfolio`." />
          <Note>Recent works from outside the field that this field cites more and more ({T - 2}–{T}), ranked by the experiment-6 model. More in Map & grid › Bridges › From outside the field.</Note>
        </Card>
        <Card title="Read now" icon={<BookOpen className="size-4.5 text-brand-600" />}>
          <ol className="flex flex-col">
            {d.read_now.map((r: any) => (
              <li key={r.paper_id} className="flex gap-3 border-b border-[var(--line)] py-2 text-[13px] last:border-0">
                <span className="faint w-5 shrink-0 text-right tabular-nums">{r.rank}</span>
                <span><PaperLink pid={r.paper_id} title={r.title} /><Lasting v={r.lasting} /></span>
              </li>
            ))}
          </ol>
          <Note>Recent papers this field is most likely to read and cite next (experiment 2: AUC 0.82). A reading list, not a quality ranking.</Note>
        </Card>
      </div>

      {(d.reading_paths || d.entry_routes) && (
        <div className="mt-5 grid gap-5 xl:grid-cols-2">
          {d.reading_paths && (
            <Card title="Read this first" icon={<Library className="size-4.5 text-brand-600" />} subtitle="Per community: its classics and the sources of its current methods">
              <ReadThisFirst rows={d.reading_paths} />
            </Card>
          )}
          {d.entry_routes && (
            <Card title="How people got in" icon={<DoorOpen className="size-4.5 text-violet-600" />} subtitle="How newcomers entered each community, and what happened next">
              <EntryRoutes rows={d.entry_routes} effects={d.entry_effects} />
            </Card>
          )}
        </div>
      )}

      {d.survey_gaps && d.survey_gaps.some((r: any) => r.status === "survey gap" || r.status === "thin") && (
        <Card className="mt-5" title="Where a survey is missing" icon={<BookOpen className="size-4.5 text-amber-600" />}
          subtitle={`Communities whose share of surveys (${d.survey_gaps[0]?.window}) is well below the field's`}>
          <SurveyGaps rows={d.survey_gaps} />
        </Card>
      )}

      <Card className="mt-5" title="Opportunities to inspect" icon={<Sparkles className="size-4.5 text-brand-600" />}
        actions={<Seg size="sm" value={opp} onChange={setOpp} options={[
          { value: "gems", label: "Hidden gems" }, { value: "lim", label: "Limitation → innovation" },
          { value: "mt", label: "Method transfer" }, { value: "uc", label: "Underconnected" },
        ]} />}>
        {opp === "gems" && (<>
          <DataTable rows={d.opportunities.gems} cols={[
            { key: "title", label: "Paper", render: (r) => <PaperLink pid={r.paper_id} title={r.title} /> },
            { key: "community", label: "For community" },
          ]} />
          <Note>Papers cited at most once that a community is likely to take up (experiment 1: long-tail AUC 0.84, where popularity is at chance).</Note>
        </>)}
        {opp === "lim" && (<>
          <DataTable rows={d.opportunities.lim_inn} cols={[
            { key: "from", label: "Limitations in" }, { key: "to", label: "Innovations in" },
            { key: "judge", label: "Judge", align: "right" }, { key: "why", label: "Why", render: (r) => <span className="muted text-[12px]">{r.why}</span> },
          ]} />
          <Note>Communities whose stated limitations match innovations elsewhere, with few citations between them (Phase 8: about 2.4× random). Only a minority of abstracts state limitations: check the evidence in Map & grid › Bridges.</Note>
        </>)}
        {opp === "mt" && (<>
          <DataTable rows={d.opportunities.method_transfer} cols={[{ key: "method", label: "Method" }, { key: "problem", label: "Problem" }, { key: "judge", label: "Judge", align: "right" }]} />
          <Note>Methods not yet used for a problem, though they are used for similar problems (experiment 4: AUC 0.70–0.80). Shown as diamonds in the grid.</Note>
        </>)}
        {opp === "uc" && (d.opportunities.underconnected ? (<>
          <DataTable rows={d.opportunities.underconnected} cols={[
            { key: "source_label", label: "From" }, { key: "target_label", label: "To" },
            { key: "observed", label: "Citations seen", align: "right" }, { key: "expected", label: "Expected", align: "right" },
          ]} />
          <Note>Semantically close communities that cite each other less than their size predicts (Phase 8c). Hypotheses to review, not confirmed gaps.</Note>
        </>) : <div className="muted text-[13px]">Run the research lab to add these.</div>)}
      </Card>

      {d.second_waves.length > 0 && (
        <Card className="mt-5" title="Second waves: ideas reaching new audiences" icon={<Waves className="size-4.5 text-brand-600" />}
          actions={<Seg size="sm" value={wave} onChange={setWave} options={[{ value: "paper", label: "Papers in the map" }, { value: "external", label: "From outside" }, { value: "method", label: "Methods" }]} />}>
          {waves.length ? <SecondWavesTable rows={waves} /> : <div className="muted text-[13px]">None this year.</div>}
          <HowToRead>{SECOND_WAVES_HOW}</HowToRead>
        </Card>
      )}

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        {d.coverage_gaps && (
          <Card title="Coverage gaps" icon={<Gem className="size-4.5 text-violet-600" />}>
            <DataTable rows={d.coverage_gaps} cols={[
              { key: "title", label: "Work", render: (r) => <a href={`https://openalex.org/${r.paper_id}`} target="_blank" rel="noreferrer" className="hover:text-brand-600">{r.title}</a> },
              { key: "year", label: "Year" }, { key: "recent_citers", label: "Citing here", align: "right" }, { key: "cluster_label", label: "Community" },
            ]} />
            <Note>Works this field cites heavily that are not in the map: often core papers the seed's citation graph never reached. Adding them (a new seed or a corpus expansion) makes every view more complete.</Note>
          </Card>
        )}
        <Card title="How far to trust each view" icon={<Pill tone="green">evidence</Pill>}>
          <Md className="text-[12.8px]">{TRUST}</Md>
        </Card>
      </div>
    </div>
  );
}
