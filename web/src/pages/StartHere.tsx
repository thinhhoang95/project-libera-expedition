import clsx from "clsx";
import { BookOpen, CalendarRange, Wrench, Compass, DoorOpen, Gem, Globe2, Layers, Library, Network, Telescope, Waves } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useApi } from "../api";
import { DataTable } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { PaperList } from "../components/Sort";
import { MUTED, Plot } from "../components/Plot";
import { Card, Empty, HowToRead, Md, Note, PageSkeleton, Pill, Seg, Spinner, StatCard, pct } from "../components/ui";
import { useProjectId } from "../context";
import { CatLink, CategoryTable } from "../components/Facets";
import { OpportunitiesCard } from "../components/Opportunities";
import { FACET_META, FACET_ORDER, useCatalogue, type FacetKey } from "../facets";
import { EntryRoutes, Lasting, ReadThisFirst, SurveyGaps } from "../components/Orientation";
import { ToolsToLearn } from "../components/Discovery";

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
            <PaperList rows={r.evidence} id="wave-evidence" orderLabel="Newest first" className="mt-2" dense />
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

/** The field by facet: problems (the communities), methods, innovations, results and limitations. */
function FacetsCard({ p, T }: { p: string; T: number }) {
  const cat = useCatalogue(p);
  const [facet, setFacet] = useState<FacetKey>("problem");
  const b = cat.facet.get(facet);
  const M = FACET_META[facet];
  return (
    <Card title="The field by facet" icon={<Layers className="size-4.5 text-brand-600" />}
      subtitle={M.blurb}
      actions={<Link to={`/p/${p}/explore/${facet}`} className="btn btn-sm">Explore {M.plural.toLowerCase()}</Link>}>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {FACET_ORDER.map((f) => {
          const I = FACET_META[f].icon;
          return (
            <button key={f} onClick={() => setFacet(f)} className={clsx("btn btn-sm", f === facet && "!border-transparent !bg-[var(--chip)] !text-[var(--chip-text)]")}>
              <I className="size-3.5" style={{ color: FACET_META[f].color }} /> {f === "problem" ? "Communities" : FACET_META[f].plural}
              <span className="faint text-[11px]">{cat.facet.get(f)?.categories.length ?? ""}</span>
            </button>
          );
        })}
      </div>
      {!cat.data ? <Spinner /> : b ? (
        <CategoryTable block={b} y0={cat.data.y0} t1={Math.max(cat.data.y0, T - 2)} t2={T} sparkFrom={Math.max(cat.data.y0, T - 9)} maxHeight={380} />
      ) : <div className="muted text-[13px]">No categories for this facet.</div>}
      <Note>{facet === "problem" ? "Problem communities" : `${M.label} categories`} (◇ = hub: a large, coherent body of work with a shifting boundary), {T - 2}–{T} against the three years before. Select one for its details sheet: papers, trend, and how it relates to the other facets. Share changes describe this map as sampled; they are not a forecast.</Note>
    </Card>
  );
}

const SECTIONS: [string, string][] = [["facets", "Facets"], ["borrow", "Borrowing"], ["read", "Read now"], ["paths", "Read this first"],
  ["tools", "Tools"], ["surveys", "Surveys"], ["opps", "Opportunities"], ["waves", "Second waves"], ["trust", "Trust"]];

/** Where you are on the page: the section whose top last passed under the sticky bar. */
function useScrollSpy(ids: string[], offset = 150) {
  const [active, setActive] = useState(ids[0]);
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      let cur = ids[0];
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top <= offset) cur = id;
      }
      // at the very bottom the last sections may never reach the bar: pick the last one
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) cur = ids.filter((id) => document.getElementById(id)).slice(-1)[0] ?? cur;
      setActive(cur);
    };
    const on = () => { if (!raf) raf = requestAnimationFrame(update); };
    update();
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("resize", on);
    return () => {
      window.removeEventListener("scroll", on);
      window.removeEventListener("resize", on);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [ids.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps
  return active;
}

function SectionNav() {
  const active = useScrollSpy(SECTIONS.map(([id]) => id));
  const bar = useRef<HTMLElement>(null);
  useEffect(() => {
    // keep the active chip visible in the bar (only the bar scrolls, never the page)
    const b = bar.current, c = b?.querySelector<HTMLElement>(`[data-sec="${active}"]`);
    if (b && c && (c.offsetLeft < b.scrollLeft || c.offsetLeft + c.offsetWidth > b.scrollLeft + b.clientWidth))
      b.scrollTo({ left: c.offsetLeft - 8, behavior: "smooth" });
  }, [active]);
  return (
    <nav ref={bar} aria-label="Sections" className="scroll-thin sticky top-[60px] z-20 -mx-1 mb-4 flex gap-1 overflow-x-auto bg-[var(--bg)]/90 px-1 py-2 backdrop-blur">
      {SECTIONS.map(([id, l]) => (
        <a key={id} href={`#${id}`} data-sec={id} aria-current={id === active ? "location" : undefined}
          className={clsx("chip shrink-0 ring-1 transition-colors", id === active
            ? "!bg-[var(--chip)] !text-[var(--chip-text)] ring-brand-300/60"
            : "!bg-[var(--surface)] !text-[var(--text)] ring-[var(--line)] hover:!text-brand-600")}
          onClick={(e) => { const el = document.getElementById(id); if (el) { e.preventDefault(); el.scrollIntoView({ behavior: "smooth", block: "start" }); history.replaceState(null, "", `#${id}`); } }}>{l}</a>
      ))}
    </nav>
  );
}

export function StartHere() {
  const p = useProjectId();
  const { data: d, isLoading } = useApi<any>(`/p/${p}/overview`);
  const [wave, setWave] = useState("paper");
  if (isLoading) return <PageSkeleton />;
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
      <SectionNav />
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

      <div id="facets" className="mt-5 grid scroll-mt-28 gap-5 xl:grid-cols-[1.4fr_1fr]">
        <FacetsCard p={p} T={T} />
        {d.import_map && (
          <Card title="Where this field borrows from" icon={<Telescope className="size-4.5 text-brand-600" />} className="scroll-mt-28" >
            <Plot height={380} data={[
              { type: "bar", orientation: "h", y: [...d.import_map.rows].reverse().map((r: any) => r.subfield), x: [...d.import_map.rows].reverse().map((r: any) => r.share_early), name: d.import_map.early_years, marker: { color: MUTED } },
              { type: "bar", orientation: "h", y: [...d.import_map.rows].reverse().map((r: any) => r.subfield), x: [...d.import_map.rows].reverse().map((r: any) => r.share_recent), name: d.import_map.recent_years, marker: { color: "#1f6ad6" } },
            ]} layout={{ barmode: "group", margin: { l: 10, r: 10, t: 4, b: 40 }, yaxis: { automargin: true }, xaxis: { tickformat: ".0%" }, legend: { orientation: "h", y: -0.12 } }} />
            <Note>Share of the citations to works outside the map, by the cited work's OpenAlex subfield. Experiment 6: the field imports more of what fits where its questions are moving and what trends across science.</Note>
          </Card>
        )}
      </div>

      <div id="borrow" className="mt-5 grid scroll-mt-28 gap-5 xl:grid-cols-2">
        <Card title="Being imported now" icon={<Globe2 className="size-4.5 text-orange-600" />}>
          <DataTable rows={d.imported_now ?? []} rowKey={(r) => r.paper_id} cols={[
            { key: "title", label: "Work", render: (r) => <a href={`https://openalex.org/${r.paper_id}`} target="_blank" rel="noreferrer" className="hover:text-brand-600">{r.title}</a> },
            { key: "year", label: "Year", align: "right" },
            { key: "subfield", label: "Subfield" },
            { key: "recent_citers", label: "Citing here", align: "right" },
            { key: "cluster_label", label: "Closest community", render: (r) => <CatLink cid={r.cluster_id} label={r.cluster_label} /> },
          ]} empty="Not built: run `external` and `portfolio`." />
          <Note>Recent works from outside the field that this field cites more and more ({T - 2}–{T}), ranked by the experiment-6 model. More in Map & grid › Bridges › From outside the field.</Note>
        </Card>
        <Card title="Read now" icon={<BookOpen className="size-4.5 text-brand-600" />} className="scroll-mt-28">
          <span id="read" className="block scroll-mt-28" />
          <PaperList rows={d.read_now} id="read-now" orderLabel="Most likely read next" numbered extra={(r: any) => <Lasting v={r.lasting} />} />
          <Note>Recent papers this field is most likely to read and cite next (experiment 2: AUC 0.82). A reading list, not a quality ranking.</Note>
        </Card>
      </div>

      {(d.reading_paths || d.entry_routes) && (
        <div id="paths" className="mt-5 grid scroll-mt-28 gap-5 xl:grid-cols-2">
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

      <span id="tools" className="block scroll-mt-28" />
      {d.tools?.rows?.length > 0 && (
        <Card className="mt-5" title="Tools to learn" icon={<Wrench className="size-4.5 text-emerald-600" />} subtitle="Per community: settled methods first, then the ones still changing">
          <ToolsToLearn data={d.tools} />
        </Card>
      )}

      <span id="surveys" className="block scroll-mt-28" />
      {d.survey_gaps && d.survey_gaps.some((r: any) => r.status === "survey gap" || r.status === "thin") && (
        <Card className="mt-5" title="Where a survey is missing" icon={<BookOpen className="size-4.5 text-amber-600" />}
          subtitle={`Communities whose share of surveys (${d.survey_gaps[0]?.window}) is well below the field's`}>
          <SurveyGaps rows={d.survey_gaps} />
        </Card>
      )}

      <span id="opps" className="block scroll-mt-28" />
      <OpportunitiesCard o={d.opportunities} />

      <span id="waves" className="block scroll-mt-28" />
      {d.second_waves.length > 0 && (
        <Card className="mt-5" title="Second waves: ideas reaching new audiences" icon={<Waves className="size-4.5 text-brand-600" />}
          actions={<Seg size="sm" value={wave} onChange={setWave} options={[{ value: "paper", label: "Papers in the map" }, { value: "external", label: "From outside" }, { value: "method", label: "Methods" }]} />}>
          {waves.length ? <SecondWavesTable rows={waves} /> : <div className="muted text-[13px]">None this year.</div>}
          <HowToRead>{SECOND_WAVES_HOW}</HowToRead>
        </Card>
      )}

      <div id="trust" className="mt-5 grid scroll-mt-28 gap-5 xl:grid-cols-2">
        {d.coverage_gaps && (
          <Card title="Coverage gaps" icon={<Gem className="size-4.5 text-violet-600" />}>
            <DataTable rows={d.coverage_gaps} cols={[
              { key: "title", label: "Work", render: (r) => <a href={`https://openalex.org/${r.paper_id}`} target="_blank" rel="noreferrer" className="hover:text-brand-600">{r.title}</a> },
              { key: "year", label: "Year" }, { key: "recent_citers", label: "Citing here", align: "right" }, { key: "cluster_label", label: "Community", render: (r) => <CatLink cid={r.cluster_id} label={r.cluster_label} /> },
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
