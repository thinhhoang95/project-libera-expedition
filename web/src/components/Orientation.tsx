import { Hourglass } from "lucide-react";
import { useState } from "react";
import { DataTable } from "./DataTable";
import { Builders, quietRoleLabel, roleTone } from "./Discovery";
import { CatLink, CommunityPicker } from "./Facets";
import { PaperLink } from "./PaperDrawer";
import { by, usePaperMeta, useSortedList } from "./Sort";
import { HowToRead, Note, Pill, pct } from "./ui";

/** Experiment 14: lasting = top third of breadth among equally generic ideas. Papers before 2018 are not assessed. */
export function Lasting({ v }: { v: boolean | null | undefined }) {
  if (v === undefined) return null;
  if (v === null) return <span className="faint ml-1.5 text-[11px]" title="Published before 2018: not assessed">not assessed</span>;
  return v ? (
    <Pill tone="green" className="ml-1.5" title="Broad at publication among equally generic ideas: more likely to still be used 5+ years later (experiment 14, AUC 0.58)">
      <Hourglass className="size-3" /> lasting
    </Pill>
  ) : null;
}

const LASTING_HOW =
  "**Lasting** marks papers whose idea was already close to several communities' earlier work when it was published, compared " +
  "among equally generic ideas (experiment 14: AUC 0.58 for still being used five or more years later; novel ideas fade " +
  "faster, and early citations say nothing). Papers before 2018 are *not assessed*: there is no window to check them.";

export type ReadingRow = { cluster_id: string; label: string; rank: number; paper_id: string; title: string; year: number | null;
  role: string; cited_by_community: number; source_of_methods: number; lasting: boolean; lasting_assessed: boolean;
  builders?: { paper_id: string; title: string; year?: number | null }[] };

export function ReadingPath({ rows, limit }: { rows: ReadingRow[]; limit?: number }) {
  const meta = usePaperMeta(rows.map((r) => r.paper_id));
  const { sorted, bar } = useSortedList(rows, [
    by.rank<ReadingRow>((r) => r.rank, "Reading order"),
    by.year<ReadingRow>((r) => r.year),
    by.citedHere<ReadingRow>((r) => r.cited_by_community, "Cited by this community"),
    by.count<ReadingRow>("methods", "Source of recent methods", (r) => r.source_of_methods),
    by.cites<ReadingRow>((r) => meta[r.paper_id]?.cites),
    by.title<ReadingRow>((r) => r.title),
  ], "reading-path");
  return (
    <>
      <div className="mb-1.5 flex justify-end">{bar}</div>
      <ol className="flex flex-col">
        {sorted.slice(0, limit).map((r) => (
          <li key={r.paper_id} className="flex gap-3 border-b border-[var(--line)] py-2 text-[13px] last:border-0">
            <span className="faint w-5 shrink-0 text-right tabular-nums">{r.rank}</span>
            <span className="min-w-0 flex-1">
              <PaperLink pid={r.paper_id} title={r.title} />
              <span className="faint ml-1.5 text-[11.5px]">{r.year}{meta[r.paper_id]?.cites != null && ` · ${meta[r.paper_id].cites!.toLocaleString()} citations overall`}</span>
              <Lasting v={r.lasting_assessed ? r.lasting : null} />
              <span className="muted block text-[11.5px]">
                <Pill tone={roleTone(r.role)}>{quietRoleLabel(r.role)}</Pill>
                <span className="ml-2">cited by {r.cited_by_community} here{r.source_of_methods ? ` · source of ${r.source_of_methods} recent methods` : ""}</span>
              </span>
              {r.role === "quiet foundation" && <Builders rows={r.builders} />}
            </span>
          </li>
        ))}
      </ol>
      <HowToRead>{"Classics (most cited by the community) alternate with the **sources of its current methods** (the earliest close sources of what it used in the last three years). Experiment 13: newcomers mostly cite the classics but build on the method sources; alternating covers most of both.\n\n**Built on, rarely cited** marks quiet foundations: papers later work here builds on far more often than it cites (experiment 17: this persists, AUC 0.83, where citations are at chance).\n\n" + LASTING_HOW}</HowToRead>
    </>
  );
}

export function ReadThisFirst({ rows }: { rows: ReadingRow[] }) {
  const clusters = [...new Map(rows.map((r) => [r.cluster_id, r.label])).entries()];
  const [c, setC] = useState(clusters[0]?.[0] ?? "");
  return (
    <div>
      <CommunityPicker value={c} onChange={setC} options={clusters} />
      <ReadingPath rows={rows.filter((r) => r.cluster_id === c)} />
    </div>
  );
}

export type EntryRow = { cluster_id: string; label: string; strategy: string; entries: number; stayed_rate: number | null;
  taken_up_rate: number | null; insider_coauthor_rate: number | null; examples_taken_up: { paper_id: string; title: string }[];
  recent_entries: number; outcome_year: number; recent_year: number };

export function EntryRoutes({ rows, effects, picker = true }: { rows: EntryRow[]; effects?: any; picker?: boolean }) {
  const clusters = [...new Map(rows.map((r) => [r.cluster_id, r.label])).entries()];
  const [c, setC] = useState(clusters[0]?.[0] ?? "");
  const part = rows.filter((r) => r.cluster_id === c);
  const y = part[0];
  const examples = part.flatMap((r) => r.examples_taken_up.map((e) => ({ ...e, strategy: r.strategy })));
  return (
    <div>
      {picker && clusters.length > 1 && (
        <CommunityPicker value={c} onChange={setC} options={clusters} />
      )}
      <DataTable rows={part} rowKey={(r) => r.strategy} dense cols={[
        { key: "strategy", label: "How they came in", render: (r) => <span className="text-[var(--text-strong)]">{r.strategy}</span> },
        { key: "entries", label: y ? `Entries ${y.outcome_year}` : "Entries", align: "right" },
        { key: "stayed_rate", label: "Stayed", align: "right", render: (r) => pct(r.stayed_rate) },
        { key: "taken_up_rate", label: "Taken up", align: "right", render: (r) => pct(r.taken_up_rate) },
        { key: "insider_coauthor_rate", label: "With an insider", align: "right", render: (r) => pct(r.insider_coauthor_rate) },
        { key: "recent_entries", label: y ? `Entries ${y.recent_year}` : "Recent", align: "right" },
      ]} />
      {examples.length > 0 && (
        <div className="mt-2 text-[12.5px]">
          <span className="muted">Entries that were taken up:</span>
          <ul className="mt-1 flex flex-col gap-0.5">
            {examples.slice(0, 6).map((e) => <li key={e.paper_id}><Pill>{e.strategy}</Pill> <PaperLink pid={e.paper_id} title={e.title} /></li>)}
          </ul>
        </div>
      )}
      <Note>What happened on this map over three years, not advice: first papers in the community in {y?.outcome_year}, whether their authors published there again (stayed) and whether the community cited the paper (taken up). The last column is the mix of entries in {y?.recent_year}, with no outcomes yet.</Note>
      {effects?.available && (
        <HowToRead title="Like-for-like differences on the pilot">
          {`${effects.note}\n\n| Outcome | Strategy / trait | Adjusted difference | 90% interval |\n| --- | --- | --- | --- |\n` +
            effects.effects.map((e: any) => `| ${e.outcome.replace("_", " ")} | ${e["strategy / trait"]} | ${e["adjusted difference"] >= 0 ? "+" : ""}${(e["adjusted difference"] * 100).toFixed(1)} pts | ${e["90% CI"]} |`).join("\n")}
        </HowToRead>
      )}
    </div>
  );
}

export type SurveyGapRow = { cluster_id: string; label: string; recent_papers: number; survey_share: number | null;
  field_survey_share: number | null; ratio_to_field: number | null; status: string; recent_surveys: number;
  latest_survey_id: string | null; latest_survey_title: string | null; latest_survey_year: number | null;
  most_cited_survey_id: string | null; most_cited_survey_title: string | null; most_cited_survey_year: number | null;
  most_cited_survey_cites_here: number; window: string; evidence: string };

const SURVEY_TONE: Record<string, "amber" | "neutral" | "green"> = { "survey gap": "amber", thin: "neutral", covered: "green", "not judged": "neutral" };

export function SurveyStatus({ s }: { s: string }) {
  return <Pill tone={SURVEY_TONE[s] ?? "neutral"}>{s}</Pill>;
}

const SURVEY_HOW =
  "A community's **survey share** is the share of its papers from the last three complete years that are surveys, " +
  "compared with the whole field's. **Survey gap** = under half the field's share, **thin** = under three quarters. " +
  "Experiment 15: where surveys were scarce, a new survey was taken up by others far more often than surveys elsewhere " +
  "(47% vs 10% in the top quarter of uptake, +38 points, 90% interval +16 to +54), and the effect grows the scarcer surveys are. " +
  "Surveys are cited rather than built on: the community's methods do not move towards them. Communities with fewer than 10 " +
  "recent papers are not judged.";

export function SurveyGaps({ rows, compact = false }: { rows: SurveyGapRow[]; compact?: boolean }) {
  const shown = compact ? rows : rows.filter((r) => r.status !== "not judged");
  return (
    <>
      <DataTable rows={shown} rowKey={(r) => r.cluster_id} dense cols={[
        { key: "label", label: "Community", render: (r) => <CatLink cid={r.cluster_id} label={r.label} /> },
        { key: "status", label: "Surveys", render: (r) => <SurveyStatus s={r.status} /> },
        { key: "survey_share", label: "Survey share", align: "right",
          render: (r) => <span title={`field: ${pct(r.field_survey_share)}`}>{pct(r.survey_share)} <span className="faint">({r.recent_surveys} of {r.recent_papers})</span></span> },
        { key: "latest_survey_title", label: "Latest survey here",
          render: (r) => r.latest_survey_id ? <span><PaperLink pid={r.latest_survey_id} title={r.latest_survey_title ?? ""} /> <span className="faint">{r.latest_survey_year}</span></span> : <span className="faint">none</span> },
        ...(compact ? [] : [{ key: "most_cited_survey_title", label: "Survey it cites most",
          render: (r: SurveyGapRow) => r.most_cited_survey_id ? <span><PaperLink pid={r.most_cited_survey_id} title={r.most_cited_survey_title ?? ""} /> <span className="faint">{r.most_cited_survey_year} · {r.most_cited_survey_cites_here}×</span></span> : <span className="faint">none</span> }]),
      ]} />
      <HowToRead>{SURVEY_HOW}</HowToRead>
    </>
  );
}
