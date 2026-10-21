import { AlertTriangle, BarChart3, Lightbulb, Target, Wrench, type LucideIcon } from "lucide-react";
import { useMemo } from "react";
import { useApi } from "./api";

export type FacetKey = "problem" | "method" | "innovation" | "result" | "limitations";

export const FACET_ORDER: FacetKey[] = ["problem", "method", "innovation", "result", "limitations"];

export const FACET_META: Record<FacetKey, { label: string; plural: string; icon: LucideIcon; color: string; blurb: string }> = {
  problem: { label: "Problem", plural: "Problems", icon: Target, color: "#1f6ad6",
    blurb: "What the papers set out to solve. Problem categories are the field's communities." },
  method: { label: "Method", plural: "Methods", icon: Wrench, color: "#0f9d76",
    blurb: "How they solve it: the techniques and tools papers use." },
  innovation: { label: "Innovation", plural: "Innovations", icon: Lightbulb, color: "#7c3aed",
    blurb: "What each paper claims is new." },
  result: { label: "Result", plural: "Results", icon: BarChart3, color: "#d97706",
    blurb: "What the papers report finding or achieving." },
  limitations: { label: "Limitation", plural: "Limitations", icon: AlertTriangle, color: "#dc2626",
    blurb: "Limitations the papers state. Only a minority of abstracts state one, so these categories are small." },
};

export const facetOf = (cid: string) => cid.split(":")[0] as FacetKey;
export const alphaOf = (cid: string) => Number(cid.split(":")[1]);
export const resOf = (cid: string) => cid.split(":")[2] ?? "medium";

export type Category = {
  cluster_id: string; label: string; description: string | null; size: number; kind: string; color: string;
  top_terms: string[] | null; stability: number; on_map: boolean; counts: number[];
};
export type FacetBlock = { facet: FacetKey; alpha: number; papers: number; assigned: number; totals: number[]; categories: Category[] };
export type Catalogue = { built: boolean; y0: number; y1: number; as_of: number; resolution: string; facets: FacetBlock[] };

/** The facet catalogue with lookups: category by id, facet block by key. */
export function useCatalogue(p: string) {
  const q = useApi<Catalogue>(p ? `/p/${p}/catalogue` : null, { staleTime: 600_000 });
  const extra = useMemo(() => {
    const d = q.data;
    const byId = new Map<string, Category & { facet: FacetKey }>();
    const facet = new Map<FacetKey, FacetBlock>();
    d?.facets?.forEach((b) => {
      facet.set(b.facet, b);
      b.categories.forEach((c) => byId.set(c.cluster_id, { ...c, facet: b.facet }));
    });
    return { byId, facet };
  }, [q.data]);
  return { ...q, ...extra };
}

/** Sum of a yearly series (indexed from y0) over [t1, t2]. */
export function sumWindow(counts: number[], y0: number, t1: number, t2: number) {
  let s = 0;
  for (let y = Math.max(t1, y0); y <= t2; y++) s += counts[y - y0] ?? 0;
  return s;
}

/** Papers, share of the facet, and the share change against the window of the same length just before. */
export function windowStats(counts: number[], totals: number[], y0: number, t1: number, t2: number) {
  const len = t2 - t1 + 1;
  const n = sumWindow(counts, y0, t1, t2), tot = sumWindow(totals, y0, t1, t2);
  const pn = sumWindow(counts, y0, t1 - len, t1 - 1), ptot = sumWindow(totals, y0, t1 - len, t1 - 1);
  const share = tot ? n / tot : 0;
  const prev = ptot ? pn / ptot : null;
  return { n, share, prev, change: prev == null ? null : share - prev };
}

/** Where a category or a paper lives on the map. */
export function mapHref(p: string, o: { cid?: string; paper?: string }) {
  if (o.cid) {
    const f = facetOf(o.cid);
    return `/p/${p}/map?facet=${f}&alpha=${alphaOf(o.cid)}&res=${resOf(o.cid)}&sel=${encodeURIComponent(o.cid)}&tab=Detail`;
  }
  return `/p/${p}/map?facet=problem&alpha=0.5&res=medium&focus=${o.paper}&tab=Detail`;
}
