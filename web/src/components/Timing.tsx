import clsx from "clsx";
import { CatLink } from "./Facets";
import { Pill, Sparkline } from "./ui";

/** A cluster's recent papers against the three years before (timing.py `line`). */
export type Line = {
  cluster_id: string; series: number[]; from: number; recent: number; before: number;
  share_recent: number; share_before: number | null; ratio: number | null; trend: "growing" | "steady" | "shrinking" | "quiet";
};
export type PaperWhen = { year: number | null; age: number | null; timing: string; last_cited: number | null; cites_recent: number; line: Line | null };
export type PairWhen = { limitation_years: number[]; innovation_years: number[]; limitation_last: number | null; innovation_last: number | null; timing: string };
export type FlowChange = { recent: number; before: number; state: "closing" | "open" | "steady" };

const TREND_COLOR: Record<string, string> = { growing: "#2f9e44", steady: "#1f6ad6", shrinking: "#e8590c", quiet: "#c92a2a" };

export const TIMING: Record<string, { tone: "green" | "brand" | "amber" | "red" | "neutral"; label: string; help: string }> = {
  fresh: { tone: "green", label: "fresh", help: "Published in the last three complete years" },
  "active line": { tone: "brand", label: "active line", help: "Older paper; the line of work its idea belongs to is holding or growing its share" },
  "fading line": { tone: "amber", label: "fading line", help: "Older paper; its line of work is losing share: taking it up means going against the current" },
  "quiet line": { tone: "red", label: "revival", help: "Older paper; its line of work has gone quiet. Taking it up would revive it" },
  older: { tone: "neutral", label: "older", help: "Older paper with no idea category to judge its line by" },
  "fresh answer": { tone: "green", label: "fresh answer", help: "The matching innovations include one from the last three years" },
  "still stated": { tone: "amber", label: "still stated", help: "The limitation was stated in the last three years, but the matching innovations are older" },
  "older pairing": { tone: "neutral", label: "older pairing", help: "Both the limitation and the matching innovations predate the last three years" },
};

/** Backtested status (experiment 30): is the community already converging on the idea, cited or not? */
export const STATUS: Record<string, { tone: "green" | "neutral"; label: string; help: string }> = {
  "being taken": { tone: "green", label: "being taken", help: "This community's recent papers already say what the idea says, without citing it yet. In the backtest such items were cited 4–7× as often within three years: act soon (experiment 30)" },
  open: { tone: "neutral", label: "open", help: "No sign yet that this community is moving towards the idea. In the backtest open items were rarely cited within three years (limitation → innovation: 1.7%): untouched, which can mean non-obvious or not viable (experiment 30)" },
};

export function StatusPill({ s }: { s?: string | null }) {
  if (!s || !STATUS[s]) return null;
  const m = STATUS[s];
  return <Pill tone={m.tone} title={m.help}>{m.label}</Pill>;
}

/** Uncited papers lose their pull with age (experiment 29): flag gems still uncited 3 or more years on. */
export function FadePill({ age, cites }: { age?: number | null; cites?: number | null }) {
  if (age == null || age < 3 || (cites ?? 0) > 1) return null;
  return <Pill tone="amber" title="Papers cited at most once 3–5 years after publication were picked up about half as often as their score suggests (experiment 29)">{cites ? `1 cite in ${age} y` : `uncited ${age} y`}</Pill>;
}

export function TimingPill({ t }: { t?: string | null }) {
  if (!t) return null;
  const m = TIMING[t] ?? { tone: "neutral" as const, label: t, help: "" };
  return <Pill tone={m.tone} title={m.help}>{m.label}</Pill>;
}

/** A sparkline of a line of work with its trend word; the label (optional) opens the category sheet. */
export function LineTrend({ line, label, compact }: { line: Line | null | undefined; label?: string | null; compact?: boolean }) {
  if (!line) return <span className="faint">—</span>;
  const c = TREND_COLOR[line.trend];
  const yrs = `${line.from}–${line.from + line.series.length - 1}`;
  return (
    <span className="inline-flex min-w-0 items-center gap-2" title={`${yrs}: ${line.series.join(", ")} papers per year. ` +
      `Share of the facet, last 3 years vs the 3 before: ${(line.share_recent * 100).toFixed(1)}% vs ${line.share_before == null ? "—" : (line.share_before * 100).toFixed(1) + "%"}`}>
      <Sparkline values={line.series} width={compact ? 56 : 70} height={20} fill={false} color={c} />
      <span className="shrink-0 text-[11.5px] font-medium" style={{ color: c }}>{line.trend}</span>
      {label && <CatLink cid={line.cluster_id} label={label} className="min-w-0 truncate text-[12.5px]" />}
    </span>
  );
}

export function FlowState({ f }: { f?: FlowChange | null }) {
  if (!f) return <span className="faint">—</span>;
  return (
    <span className="whitespace-nowrap text-[12px] tabular-nums" title="Citations from the first side to the second: last 3 years / the 3 before">
      {f.recent} / {f.before}{" "}
      <span className={clsx("font-medium", f.state === "closing" ? "text-emerald-600" : f.state === "open" ? "text-brand-600" : "muted")}>
        {f.state === "closing" ? "closing" : f.state === "open" ? "still open" : "steady"}
      </span>
    </span>
  );
}

export const yearSpan = (ys: number[]) => (!ys.length ? "—" : ys[0] === ys[ys.length - 1] ? `${ys[0]}` : `${ys[0]}–${ys[ys.length - 1]}`);

export const TIMING_HOW =
  "**Status** and the **uncited** flag are backtested (experiments 29 and 30):\n\n" +
  "- **being taken**: the community's papers of the last three years already say what the idea says, closer than three to " +
  "five years before, without citing it. Such hidden gems were cited about 7× as often within three years, and the gem " +
  "ranking uses it. **open**: no such sign; open limitation → innovation items were rarely cited within three years.\n" +
  "- **Already cited** innovations are left out of limitation → innovation: a gap the community has closed is not an " +
  "opportunity, and such items made up most of the list's backtest hits.\n" +
  "- **uncited N y** (or **1 cite in N y**): a paper cited at most once 3–5 years on is picked up about half as often as its score suggests.\n\n" +
  "**Timing** places each opportunity in time, as of the latest complete year. It describes this map as sampled; it was " +
  "**not backtested** as a predictor, so read it as context for a decision, not as a score.\n\n" +
  "- **Papers** (hidden gems, papers to bring in): **fresh** = published in the last three years. Older papers are judged by " +
  "their **line of work** (their innovation category, else their problem category): **active line** if its share of the field " +
  "held or grew over the last three years against the three before, **fading line** if it lost share, **revival** if it has " +
  "gone quiet (under half its earlier share, or fewer than 3 recent papers). *Last cited* is the latest citation from this field.\n" +
  "- **Limitation → innovation**: the years of the evidence pairs. **Fresh answer** = a matching innovation from the last three " +
  "years; **still stated** = the limitation is recent but the answers are older. **Citing already** counts citations from the " +
  "limitation side to the innovation side (last 3 years / the 3 before): **closing** means the gap is being bridged already.\n" +
  "- **Method transfer**: the method's and the problem's lines, and how much the similar problems used the method recently " +
  "(last 3 years / the 3 before): a method spreading now is a different bet from one similar problems used long ago.\n" +
  "- **Trends** compare shares, so the corpus thinning in recent years (sampling) does not read as decline.";
