import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import {
  ArrowRight, BookOpen, Calendar, DoorOpen, Library, ChevronLeft, ChevronRight, Download, ExternalLink, FileText, Flame, HelpCircle,
  Lightbulb, Link2, Loader2, Network, Quote, Search, Tag, Target, Users, Waves,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, qs, useApi, type Job, type Project } from "../api";
import { DataTable } from "../components/DataTable";
import { PaperLink, ReviewBar } from "../components/PaperDrawer";
import { Card, ErrorBox, HowToRead, Note, Pill, Spinner, StatCard } from "../components/ui";
import { SecondWavesTable, SECOND_WAVES_HOW } from "./StartHere";
import { EntryRoutes, Lasting, ReadingPath, SurveyGaps } from "../components/Orientation";

type Warm = { rank: number; cluster_id: string; community: string; kind: string; warmth: number; content_fit: number; why: string[] };
type MP = {
  paper: { paper_id: string; title: string; year: number | null; doi: string | null; in_corpus: boolean; abstract: string | null;
    venue: string | null; type: string | null; cited_by_count: number | null; analysed_at?: string; llm_cost_usd?: number };
  source: string;
  facets: Record<string, string>;
  home: { cluster_id: string; label: string; kind: string } | null;
  already_citing: string[];
  warm: Warm[];
  stats: { cited_by_count: number | null; map_citations: number; map_citations_by_year: { year: number; n: number }[];
    references: number; refs_in_map: number; warm_candidates: number; already_citing: number };
  limitations: { paper_id: string; limitation: string; title: string; year: number; first_author: string | null; closeness: number }[];
  outside: { paper_id: string; title: string; subfield: string; fit: number; recent_citers: number; year: number }[];
  bring_in: { paper_id: string; title: string; year: number | null; list: string; fit: number | null }[];
  methods: { method: string; judge: number | null }[];
  early_work: { paper_id: string; title: string; year: number | null; closeness: number }[];
  second_waves: any[];
  people: { name: string; community: string; papers: number; earlier: number; track_record: string }[];
  people_since?: number;
  top3: string[];
  reading_path: any[] | null;
  entry_routes: any[];
  survey_gaps: any[] | null;
  entry_effects: any;
};

const TYPE_LABEL: Record<string, string> = { article: "Journal article", preprint: "Preprint", "book-chapter": "Book chapter", review: "Review", dissertation: "Dissertation" };
const TABS = ["Overview", "Related papers", "Communities", "Ideas & insights", "Papers to bring in"] as const;

function downloadCsv(name: string, rows: Record<string, any>[]) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc = (v: any) => `"${String(Array.isArray(v) ? v.join("; ") : v ?? "").replace(/"/g, '""')}"`;
  const blob = new Blob([[cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ------------------------------------------------------------------ picking a paper

function AnalyseByDoi({ compact }: { compact?: boolean }) {
  const { project } = useParams();
  const qc = useQueryClient();
  const nav = useNavigate();
  const [doi, setDoi] = useState("");
  const [jobId, setJobId] = useState<number | null>(null);
  const m = useMutation({
    mutationFn: () => api<Job>(`/p/${project}/my-paper/analyse`, { method: "POST", json: { doi } }),
    onSuccess: (j) => setJobId(j.id),
  });
  const { data: job } = useApi<Job>(jobId ? `/jobs/${jobId}` : null, { refetchInterval: 1500, staleTime: 0 });
  useEffect(() => {
    if (job?.status === "done") {
      qc.invalidateQueries({ queryKey: [`/p/${project}/my-paper/analysed`] });
      api<{ paper_id: string; doi: string }[]>(`/p/${project}/my-paper/analysed`).then((list) => {
        const hit = list.find((r) => (r.doi ?? "").toLowerCase().endsWith(doi.toLowerCase().trim()));
        if (hit) nav(`/p/${project}/paper/${hit.paper_id}?source=doi`);
      });
    }
  }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div>
      <div className={clsx("flex gap-2", compact && "flex-col")}>
        <input className="input font-mono" placeholder="10.1109/TAC.2020.3006812" value={doi} onChange={(e) => setDoi(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && doi && m.mutate()} />
        <button className="btn btn-primary" disabled={!doi || m.isPending || (job && !["done", "failed", "cancelled"].includes(job.status))} onClick={() => m.mutate()}>
          Analyse
        </button>
      </div>
      {m.error && <div className="mt-2"><ErrorBox error={m.error} /></div>}
      {job && (
        <div className="muted mt-2 flex items-center gap-2 text-[12.5px]">
          {!["done", "failed", "cancelled"].includes(job.status) && <Loader2 className="size-3.5 animate-spin" />}
          {job.status === "queued" ? "Queued (quick lane)…" : job.status === "running" ? job.steps[0]?.message?.slice(0, 80) || "Fetching, extracting facets and scoring…" : job.status === "failed" ? <span className="text-red-600">{job.error?.split("\n").slice(-2).join(" ")}</span> : job.status}
        </div>
      )}
      {!compact && <Note>Fetches the paper from OpenAlex, extracts its facets with one LLM call (less than a cent) unless it is already in the map, and scores every community. Runs in the quick lane, so it never waits for a pipeline step.</Note>}
    </div>
  );
}

function FindInMap({ onPick }: { onPick: (pid: string) => void }) {
  const { project } = useParams();
  const [term, setTerm] = useState("");
  const { data, isFetching } = useApi<{ paper_id: string; year: number; title: string }[]>(
    term.length >= 2 ? `/p/${project}/search` + qs({ term, limit: 40 }) : null,
  );
  return (
    <div>
      <div className="relative">
        <Search className="faint absolute top-2.5 left-3 size-4" />
        <input className="input pl-9" placeholder="Title contains… e.g. Wasserstein" value={term} onChange={(e) => setTerm(e.target.value)} />
      </div>
      {isFetching && <div className="muted mt-2 text-[12px]">Searching…</div>}
      {data && (
        <div className="scroll-thin mt-2 max-h-72 overflow-y-auto rounded-lg border border-[var(--line)]">
          {data.length === 0 && <div className="muted px-3 py-3 text-[12.5px]">No title matches.</div>}
          {data.map((h) => (
            <button key={h.paper_id} onClick={() => onPick(h.paper_id)} className="flex w-full gap-3 border-b border-[var(--line)] px-3 py-2 text-left text-[13px] last:border-0 hover:bg-[var(--surface-2)]">
              <span className="faint w-10 shrink-0">{h.year}</span>
              <span className="min-w-0 flex-1">{h.title}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function AnalysedList({ onPick, current }: { onPick: (pid: string) => void; current?: string }) {
  const { project } = useParams();
  const { data } = useApi<{ paper_id: string; title: string; year: number; analysed_at: string; in_corpus: boolean }[]>(`/p/${project}/my-paper/analysed`);
  if (!data?.length) return <div className="muted text-[12.5px]">No papers analysed by DOI yet.</div>;
  return (
    <div className="flex flex-col gap-1">
      {data.map((r) => (
        <button key={r.paper_id} onClick={() => onPick(r.paper_id)}
          className={clsx("flex gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] hover:bg-[var(--surface-2)]", r.paper_id === current && "bg-[var(--nav-active)]")}>
          <FileText className="faint mt-0.5 size-3.5 shrink-0" />
          <span className="min-w-0 flex-1 truncate">{r.title}</span>
          <span className="faint shrink-0 text-[11.5px]">{r.year}</span>
        </button>
      ))}
    </div>
  );
}

function PaperStatusCard({ source, current }: { source: string; current: string }) {
  const { project } = useParams();
  const nav = useNavigate();
  const [mode, setMode] = useState(source === "doi" ? "doi" : "map");
  return (
    <Card title="Paper status" icon={<FileText className="size-4 text-brand-600" />}>
      <div className="flex flex-col gap-2">
        {[["map", "Find a paper in the map"], ["doi", "A paper analysed by DOI"]].map(([v, l]) => (
          <label key={v} className="flex cursor-pointer items-center gap-2.5 text-[13.5px]">
            <span className={clsx("grid size-[18px] place-items-center rounded-full border-2", mode === v ? "border-brand-600" : "border-[var(--line-strong)]")}>
              {mode === v && <span className="size-2 rounded-full bg-brand-600" />}
            </span>
            <input type="radio" className="hidden" checked={mode === v} onChange={() => setMode(v)} />
            {l}
          </label>
        ))}
      </div>
      <div className="mt-3">
        {mode === "map" ? (
          <FindInMap onPick={(pid) => nav(`/p/${project}/paper/${pid}`)} />
        ) : (
          <div className="flex flex-col gap-3">
            <AnalysedList current={current} onPick={(pid) => nav(`/p/${project}/paper/${pid}?source=doi`)} />
            <AnalyseByDoi compact />
          </div>
        )}
      </div>
    </Card>
  );
}

function Picker() {
  const { project } = useParams();
  const nav = useNavigate();
  const { data: proj } = useApi<Project>(`/projects/${project}`);
  return (
    <div className="mx-auto max-w-[1100px]">
      <h1 className="h-title text-[28px]">My paper</h1>
      <p className="muted mt-1 max-w-3xl text-[13.5px]">
        Which communities are most likely to be warm to a paper, whose stated problems it may answer, ideas from outside the field that fit,
        and papers to bring in. Pick a paper in {proj?.name ?? "the map"}, or analyse any paper by its DOI.
      </p>
      {proj && !proj.artefacts.portfolio && (
        <div className="mt-4"><ErrorBox error={`The portfolio is not built yet for ${project}. Queue the core map on the Pipeline page.`} /></div>
      )}
      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <Card title="Find a paper in the map" icon={<Search className="size-4 text-brand-600" />}>
          <FindInMap onPick={(pid) => nav(`/p/${project}/paper/${pid}`)} />
          <Note>Papers need an extracted innovation statement to be scored.</Note>
        </Card>
        <Card title="Analyse a paper by DOI" icon={<Link2 className="size-4 text-brand-600" />}>
          <AnalyseByDoi />
          <div className="muted mt-4 mb-1.5 text-[12px] font-semibold">Analysed papers</div>
          <AnalysedList onPick={(pid) => nav(`/p/${project}/paper/${pid}?source=doi`)} />
        </Card>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ the paper

function Abstract({ text }: { text: string | null }) {
  const [full, setFull] = useState(false);
  return (
    <Card
      title="Abstract"
      icon={<span className="grid size-8 place-items-center rounded-lg bg-[var(--chip)]"><FileText className="size-4 text-brand-600" /></span>}
      actions={text && text.length > 420 ? (
        <button className="btn btn-sm border-transparent bg-[var(--chip)] text-[var(--chip-text)]" onClick={() => setFull(!full)}>
          {full ? "Show less" : "View full abstract"} <ArrowRight className="size-3.5" />
        </button>
      ) : undefined}
    >
      <p className={clsx("text-[14px] leading-[1.65]", !full && "line-clamp-4")}>{text || "No abstract available."}</p>
    </Card>
  );
}

function Tags({ d }: { d: MP }) {
  const tags = [d.facets.problem_class, d.facets.method_family, d.facets.contribution_type, d.home?.label].filter(Boolean) as string[];
  return (
    <Card title="Tags" icon={<Tag className="size-4 text-[var(--muted)]" />}>
      <div className="flex flex-wrap gap-2">
        {tags.map((t, i) => <span key={i} className="chip" title={i === 3 ? "Home community" : undefined}>{t}</span>)}
        {!tags.length && <span className="muted text-[12.5px]">No tags extracted.</span>}
      </div>
    </Card>
  );
}

function WarmTable({ d, selected, onSelect, limit }: { d: MP; selected: string | null; onSelect: (c: string) => void; limit?: number }) {
  return (
    <DataTable
      rows={limit ? d.warm.slice(0, limit) : d.warm}
      rowKey={(r) => r.cluster_id}
      maxHeight={limit ? undefined : 520}
      onSelect={(r) => onSelect(r.cluster_id)}
      selected={(r) => r.cluster_id === selected}
      cols={[
        { key: "sel", label: "", sortable: false, width: 28, render: (r) => (
          <span className={clsx("grid size-4 place-items-center rounded border", r.cluster_id === selected ? "border-brand-600 bg-brand-600 text-white" : "border-[var(--line-strong)]")}>
            {r.cluster_id === selected && <svg viewBox="0 0 12 12" className="size-2.5"><path d="M2 6l3 3 5-6" fill="none" stroke="currentColor" strokeWidth="2" /></svg>}
          </span>) },
        { key: "rank", label: "#", width: 36 },
        { key: "community", label: "Community", render: (r) => <span className="text-[var(--text-strong)]">{r.community}{r.kind === "hub" && <span className="faint" title="hub: a large, coherent body of work with a shifting boundary"> ◇</span>}</span> },
        { key: "warmth", label: "Warmth", width: 150, render: (r) => (
          <span className="flex items-center gap-2">
            <span className="h-[7px] w-[72px] overflow-hidden rounded-full bg-[var(--line)]"><span className="block h-full rounded-full bg-brand-500" style={{ width: `${r.warmth * 100}%` }} /></span>
            <span className="text-[12px] tabular-nums">{(r.warmth * 100).toFixed(1)}%</span>
          </span>) },
        { key: "content_fit", label: "Content fit", align: "right", render: (r) => <span className="tabular-nums">{r.content_fit.toFixed(3)}</span> },
        { key: "why", label: "Why", sortable: false, render: (r) => <span className="muted text-[12px]">{r.why.length ? r.why.join("; ").replace(/^./, (c) => c.toUpperCase()) + "." : "No strong signal."}</span> },
      ]}
    />
  );
}

function Closest({ pid, source, cluster, label }: { pid: string; source: string; cluster: string | null; label?: string }) {
  const { project } = useParams();
  const { data, isLoading } = useApi<{ paper_id: string; year: number; title: string; closeness: number }[]>(
    cluster ? `/p/${project}/my-paper/${pid}/community/${encodeURIComponent(cluster)}` + qs({ source }) : null,
  );
  return (
    <Card title="Its recent papers closest to yours" icon={<FileText className="size-4 text-brand-600" />} subtitle={label}>
      {!cluster && <div className="muted text-[12.5px]">Select a community.</div>}
      {isLoading && <Spinner />}
      {data && (
        <DataTable rows={data} dense rowKey={(r) => r.paper_id} cols={[
          { key: "year", label: "Year", width: 52 },
          { key: "title", label: "Title", render: (r) => <PaperLink pid={r.paper_id} title={r.title} className="line-clamp-2" /> },
        ]} empty="No recent papers with an innovation statement." />
      )}
      <Note>Who to read, cite or contact in that community: its last four years, closest to your innovation.</Note>
    </Card>
  );
}

function Limitations({ d }: { d: MP }) {
  return (
    <Card title="Whose stated problems it may answer" icon={<Target className="size-4.5 text-orange-600" />}
      actions={<HelpCircle className="faint size-4" />}>
      <DataTable rows={d.limitations} rowKey={(r) => r.paper_id} dense cols={[
        { key: "first_author", label: "Researcher", width: 150, render: (r) => <span className="text-[var(--text-strong)]">{r.first_author ?? "—"}</span> },
        { key: "limitation", label: "Stated problem", render: (r) => <span title={r.title}>{r.limitation}</span> },
        { key: "go", label: "", width: 24, sortable: false, render: (r) => <PaperLink pid={r.paper_id} title="›" className="faint text-[16px]" /> },
      ]} empty="No innovation statement to compare." />
      <Note>Limitations stated in other papers' abstracts that are closest to your innovation. Few abstracts state limitations, and closeness of wording does not establish that the assumptions are compatible: candidates to read.</Note>
    </Card>
  );
}

function Outside({ d }: { d: MP }) {
  return (
    <Card title="Ideas from outside the field that fit" icon={<Lightbulb className="size-4.5 text-amber-500" />}>
      <DataTable rows={d.outside} rowKey={(r) => r.paper_id} dense cols={[
        { key: "title", label: "Idea", render: (r) => <a className="hover:text-brand-600" href={`https://openalex.org/${r.paper_id}`} target="_blank" rel="noreferrer">{r.title}</a> },
        { key: "subfield", label: "Source", width: 150 },
        { key: "fit", label: "Fit", align: "right", width: 56 },
      ]} empty="Not built: outside works need `landscape external` and the portfolio." />
      <Note>Works from outside the field that this field is importing (experiment 6), ranked by closeness to your problem; works you already cite are left out.</Note>
    </Card>
  );
}

function PaperView({ pid }: { pid: string }) {
  const { project } = useParams();
  const [sp] = useSearchParams();
  const source = sp.get("source") ?? "corpus";
  const [tab, setTab] = useState<(typeof TABS)[number]>("Overview");
  const { data: d, error, isLoading } = useApi<MP>(`/p/${project}/my-paper/${pid}` + qs({ source }));
  const [sel, setSel] = useState<string | null>(null);
  useEffect(() => setSel(null), [pid]);
  const selected = sel ?? d?.warm[0]?.cluster_id ?? null;
  const spark = useMemo(() => {
    const by = d?.stats.map_citations_by_year ?? [];
    if (!by.length) return [];
    const ys = by.map((r) => r.year);
    const out: number[] = [];
    for (let y = Math.min(...ys); y <= Math.max(...ys); y++) out.push(by.find((r) => r.year === y)?.n ?? 0);
    return out;
  }, [d]);
  const cum = useMemo(() => spark.reduce<number[]>((a, v) => [...a, (a[a.length - 1] ?? 0) + v], []), [spark]);
  if (isLoading) return <Spinner label="Scoring communities…" />;
  if (error) return (
    <div className="mx-auto max-w-3xl">
      <Link to={`/p/${project}/paper`} className="muted mb-3 inline-flex items-center gap-1 text-[13px]"><ChevronLeft className="size-4" /> My paper</Link>
      <ErrorBox error={error} />
    </div>
  );
  if (!d) return null;
  const p = d.paper;
  const link = p.doi ? (p.doi.startsWith("http") ? p.doi : `https://doi.org/${p.doi}`) : `https://openalex.org/${p.paper_id}`;
  const selLabel = d.warm.find((w) => w.cluster_id === selected)?.community;
  return (
    <div className="mx-auto max-w-[1500px]">
      <div className="muted mb-2 flex items-center gap-1.5 text-[13px]">
        <Link to={`/p/${project}/paper`} className="flex items-center hover:text-brand-600"><ChevronLeft className="size-4" /> My paper</Link>
        <ChevronRight className="size-3.5" /> <span className="font-medium text-[var(--text-strong)]">Paper details</span>
      </div>
      <div className="grid gap-5 xl:grid-cols-[1fr_380px]">
        <div className="min-w-0">
          <h1 className="h-title text-[26px] leading-[1.2] md:text-[30px]">{p.title}</h1>
          <div className="mt-3 flex flex-wrap gap-2">
            {p.year && <span className="chip !bg-[var(--surface-2)] !text-[var(--text)]"><Calendar className="size-3.5" /> {p.year}</span>}
            {p.venue && <span className="chip !bg-[var(--surface-2)] !text-[var(--text)]"><BookOpen className="size-3.5" /> {p.venue}</span>}
            {p.type && <span className="chip !bg-[var(--surface-2)] !text-[var(--text)]">{TYPE_LABEL[p.type] ?? p.type}</span>}
            <span className="chip">{p.in_corpus ? "In the map" : "Analysed by DOI"}</span>
          </div>
          <div className="scroll-thin mt-4 flex overflow-x-auto border-b border-[var(--line)]">
            {TABS.map((t) => <button key={t} className={clsx("tab", t === tab && "active")} onClick={() => setTab(t)}>{t}</button>)}
          </div>
        </div>
        <div className="flex items-start justify-end gap-2 xl:pt-1">
          <a className="btn btn-primary" href={link} target="_blank" rel="noreferrer">Open the paper <ExternalLink className="size-4" /></a>
        </div>
      </div>

      {tab === "Overview" && (
        <>
          <div className="mt-5 grid gap-5 xl:grid-cols-[1fr_380px]">
            <Abstract text={p.abstract} />
            <div className="flex flex-col gap-5">
              <PaperStatusCard source={source} current={pid} />
              <Tags d={d} />
            </div>
              <div className="grid gap-4 sm:grid-cols-2 xl:col-span-2 xl:grid-cols-4">
              <StatCard icon={<Quote className="size-4.5" />} tone="green" label="Citations" value={p.cited_by_count?.toLocaleString() ?? "—"} spark={cum} sparkColor="#1e9e5a" hint="All of science, from OpenAlex. The sparkline is the cumulative citations from papers in the map." />
              <StatCard icon={<Network className="size-4.5" />} label="Cited in the map" value={d.stats.map_citations.toLocaleString()} sub={<span className="muted">papers</span>} spark={spark} hint="Papers in this project's map citing it, by year" />
              <StatCard icon={<Users className="size-4.5" />} tone="violet" label="Warm communities" value={d.stats.warm_candidates} sub={<span className="muted">{d.stats.already_citing} already cite it</span>} hint="Communities that do not cite it yet, ranked by warmth" />
              <StatCard icon={<Link2 className="size-4.5" />} tone="orange" label="References" value={d.stats.references} sub={<span className="muted">{d.stats.refs_in_map} in the map</span>} />
            </div>
          </div>
          <div className="mt-5 grid items-start gap-5 xl:grid-cols-[1fr_380px]">
            <Card
              title="Communities most likely to be warm to it"
              icon={<Flame className="size-4.5 text-red-500" />}
              actions={<button className="btn btn-sm" onClick={() => downloadCsv(`warm-communities-${pid}.csv`, d.warm)}><Download className="size-3.5" /> Export</button>}
            >
              <WarmTable d={d} selected={selected} onSelect={setSel} limit={10} />
              <Note>
                {d.home && <>Home community: <b>{d.home.label}</b>. </>}
                {d.already_citing.length > 0 && <>Already citing it: {d.already_citing.join(", ")}. </>}
                Ranked by the community-warmth model (experiment 10): among communities that do not cite a paper yet, a top-3 community went on to cite it within three years for 69% of papers (25% at random). <b>Content fit</b> is the similarity of your innovation to the community's problems.
              </Note>
              <HowToRead>{"**Warmth** is relative within this paper (100% = the warmest). **Why** names the strongest contributions: shared references with the community's papers, how close its recent methods and innovations are to your innovation, how related its problems are, and whether its stated limitations match. Select a community to see its recent papers closest to yours."}</HowToRead>
            </Card>
            <Closest pid={pid} source={source} cluster={selected} label={selLabel} />
          </div>
          <div className="mt-5 grid gap-5 xl:grid-cols-[1fr_1fr]">
            <Limitations d={d} />
            <Outside d={d} />
          </div>
          <ReviewBar view="my_paper" subject={{ paper_id: pid, warm_top3: d.top3 }} actions={["useful", "obvious", "wrong"]} alpha={{ problem: 0.5 }} label="Were these suggestions useful?" />
        </>
      )}

      {tab === "Communities" && (
        <div className="mt-5 grid gap-5 xl:grid-cols-[1fr_380px]">
          <div className="flex min-w-0 flex-col gap-5">
            <Card title="All candidate communities" icon={<Flame className="size-4.5 text-red-500" />}
              actions={<button className="btn btn-sm" onClick={() => downloadCsv(`warm-communities-${pid}.csv`, d.warm)}><Download className="size-3.5" /> Export</button>}>
              <WarmTable d={d} selected={selected} onSelect={setSel} />
            </Card>
            {d.entry_routes.length > 0 && (
              <Card title="How people got in" icon={<DoorOpen className="size-4.5 text-violet-600" />} subtitle="Your home community, then your warmest ones">
                <EntryRoutes key={pid} rows={d.entry_routes} effects={d.entry_effects} />
              </Card>
            )}
            {d.survey_gaps && d.survey_gaps.length > 0 && (
              <Card title="Is a survey missing?" icon={<BookOpen className="size-4.5 text-amber-600" />} subtitle="Your home community, then your warmest ones">
                <SurveyGaps rows={d.survey_gaps} compact />
              </Card>
            )}
            <Card title="Active in your warmest communities" icon={<Users className="size-4.5 text-violet-600" />}>
              <DataTable rows={d.people} cols={[
                { key: "name", label: "Name" }, { key: "community", label: "Community" },
                { key: "papers", label: `Papers since ${d.people_since ?? ""}`, align: "right" },
                { key: "earlier", label: "Earlier papers in the map", align: "right" },
                { key: "track_record", label: "Track record", render: (r) => <Pill tone={r.track_record === "insider" ? "brand" : "neutral"}>{r.track_record}</Pill> },
              ]} />
              <Note>Authors with the most recent papers in your three warmest communities. <b>Insider</b> = at least 2 earlier papers in the map: insiders' papers draw more citations from others (experiments 5 and 11), while big names stepping somewhere new do not start a follow-up wave (experiment 11). Descriptive decision support, not a ranking of people.</Note>
            </Card>
          </div>
          <Closest pid={pid} source={source} cluster={selected} label={selLabel} />
        </div>
      )}

      {tab === "Related papers" && (
        <div className="mt-5 grid gap-5 xl:grid-cols-2">
          <Card title="Early work nearby" icon={<BookOpen className="size-4.5 text-brand-600" />}>
            <DataTable rows={d.early_work} rowKey={(r) => r.paper_id} cols={[
              { key: "year", label: "Year", width: 52 },
              { key: "title", label: "Paper", render: (r: any) => <span><PaperLink pid={r.paper_id} title={r.title} /><Lasting v={r.lasting} /></span> },
              { key: "closeness", label: "Closeness", align: "right" },
            ]} />
            <Note>Recent papers this field is likely to read next (experiment 2), closest to yours.</Note>
          </Card>
          <Limitations d={d} />
          {d.reading_path && d.home && (
            <Card title="Read this first in your home community" icon={<Library className="size-4.5 text-brand-600" />} subtitle={`${d.home.label} · papers you already cite are left out`}>
              <ReadingPath rows={d.reading_path} />
            </Card>
          )}
        </div>
      )}

      {tab === "Ideas & insights" && (
        <div className="mt-5 grid gap-5 xl:grid-cols-2">
          <Outside d={d} />
          <Card title="Methods new to your problem" icon={<Lightbulb className="size-4.5 text-brand-600" />}>
            <DataTable rows={d.methods} cols={[{ key: "method", label: "Method community" }, { key: "judge", label: "Judge 1–5", align: "right" }]}
              empty="No home community, or no transfer candidates for it." />
            <Note>Method communities not yet used for your home community's problems, though similar problems use them (experiment 4).</Note>
          </Card>
          <div className="xl:col-span-2">
            <Card title="Second waves near you" icon={<Waves className="size-4.5 text-brand-600" />}>
              {d.second_waves.length ? <SecondWavesTable rows={d.second_waves} /> : <div className="muted text-[13px]">No ideas reaching your home or warmest communities right now.</div>}
              <Note>Ideas reaching your home or warmest communities as new audiences, quiet uptake or likely next.</Note>
              <HowToRead>{SECOND_WAVES_HOW}</HowToRead>
            </Card>
          </div>
        </div>
      )}

      {tab === "Papers to bring in" && (
        <div className="mt-5">
          <Card title="Papers to bring in" icon={<ArrowRight className="size-4.5 text-brand-600" />} subtitle={d.home ? `For your home community: ${d.home.label}` : undefined}>
            <DataTable rows={d.bring_in} rowKey={(r) => r.paper_id} cols={[
              { key: "year", label: "Year", width: 52 },
              { key: "title", label: "Paper", render: (r: any) => <span><PaperLink pid={r.paper_id} title={r.title} /><Lasting v={r.lasting} /></span> },
              { key: "list", label: "List", render: (r) => <Pill tone={r.list === "hidden gem" ? "violet" : "brand"}>{r.list}</Pill> },
              { key: "fit", label: "Fit", align: "right" },
            ]} empty="No home community to draw the list from." />
            <Note>Innovations your home community is likely to take up but does not cite yet (experiment 1), closest to your problem first; papers you already cite are left out.</Note>
          </Card>
        </div>
      )}
    </div>
  );
}

export function MyPaper() {
  const { pid } = useParams();
  if (!pid) return <Picker />;
  return <PaperView pid={pid} />;
}

