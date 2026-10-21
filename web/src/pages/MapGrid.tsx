import clsx from "clsx";
import { SlidersHorizontal } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigationType, useSearchParams } from "react-router-dom";
import { qs, useApi } from "../api";
import { PageHeader } from "../components/Layout";
import { CatLink, TimeSeeker } from "../components/Facets";
import { usePaperDrawer, useProjectId, useSharedWindow } from "../context";
import { useCatalogue } from "../facets";
import { ACCENT, MUTED, Plot } from "../components/Plot";
import { Card, Empty, Note, PageSkeleton, Seg, Spinner } from "../components/ui";
import { BridgesPanel, DetailPanel, GridPanel, PeoplePanel, RankingPanel } from "./MapPanels";

export const ALPHAS = [0, 0.3, 0.5, 0.8];
export const GLOSS = new Set(["problem", "method"]);
const FACETS = ["problem", "method", "innovation", "result", "limitations"];
const RES = ["coarse", "medium", "fine"];
const BADGE: Record<string, string> = { emerging: "▲", niche: "◆", dormant: "○" };
const FLOW_UP = "#2f9e44", FLOW_DOWN = "#c92a2a";

export type MapData = {
  built: boolean; reason?: string; y0: number; y1: number; years: { year: number; n: number }[]; have_signals: boolean;
  points: { paper_id: string[]; x: number[]; y: number[]; year: number[]; title: string[]; cluster_id: (string | null)[] };
  clusters: { cluster_id: string; label: string; description: string | null; size: number; stability: number; kind: string;
    centrality: number | null; color: string; top_terms: string[]; parent_cluster_id: string | null; x: number | null; y: number | null }[];
  cluster_year: { cluster_id: string[]; year: number[]; n: number[]; share: number[] };
  status: { cluster_id: string; year: number; status: string | null; scores: Record<string, any> }[];
};

function arc(x0: number, y0: number, x1: number, y1: number, bend = 0.15, n = 20) {
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2, dx = x1 - x0, dy = y1 - y0;
  const cx = mx - bend * dy, cy = my + bend * dx;
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    xs.push((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1);
    ys.push((1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1);
  }
  return [xs, ys];
}

function hist2d(xs: number[], ys: number[], xr: [number, number], yr: [number, number], bins = 40) {
  const z = Array.from({ length: bins }, () => new Array(bins).fill(0));
  const bx = (xr[1] - xr[0]) / bins, by = (yr[1] - yr[0]) / bins;
  xs.forEach((x, i) => {
    const a = Math.min(bins - 1, Math.max(0, Math.floor((x - xr[0]) / bx)));
    const b = Math.min(bins - 1, Math.max(0, Math.floor((ys[i] - yr[0]) / by)));
    z[b][a] += 1;
  });
  const tot = Math.max(xs.length, 1) * bx * by;
  return { z: z.map((r) => r.map((v) => v / tot)), xc: Array.from({ length: bins }, (_, i) => xr[0] + (i + 0.5) * bx), yc: Array.from({ length: bins }, (_, i) => yr[0] + (i + 0.5) * by) };
}

const quantile = (vals: number[], q: number) => {
  if (!vals.length) return Infinity;
  const s = [...vals].sort((a, b) => a - b);
  const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
};

const MAP_H = 600;

type Placed = { id: string; x: number; y: number; text: string; size: number; pos: string };

/**
 * Cluster labels that do not collide: the selected cluster first, then flagged ones, then by share of the window.
 * A label is kept when its box (estimated from its length and font size, in screen pixels) overlaps none already
 * placed; near an edge it turns inward so it is not cut off. Long labels are shortened (the hover has them whole).
 */
function placeLabels(cs: { cluster_id: string; x: number | null; y: number | null; share: number }[], o: {
  sel: string | null; badged?: (c: string) => boolean; label: (c: string) => string; xs: number[]; ys: number[]; width: number; height: number;
}): Placed[] {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  o.xs.forEach((v, i) => { x0 = Math.min(x0, v); x1 = Math.max(x1, v); y0 = Math.min(y0, o.ys[i]); y1 = Math.max(y1, o.ys[i]); });
  const k = Math.min(o.width / Math.max(x1 - x0, 1e-9), o.height / Math.max(y1 - y0, 1e-9)); // px per unit (aspect locked)
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const px = (x: number) => o.width / 2 + (x - cx) * k, py = (y: number) => o.height / 2 - (y - cy) * k;
  const rank = (c: string) => (c === o.sel ? 0 : o.badged?.(c) ? 1 : 2);
  const order = [...cs].filter((c) => c.x != null).sort((a, b) => rank(a.cluster_id) - rank(b.cluster_id) || b.share - a.share);
  const boxes: [number, number, number, number][] = [];
  const out: Placed[] = [];
  for (const c of order) {
    if (out.length >= 22) break;
    const full = o.label(c.cluster_id);
    const text = full.length > 40 ? full.slice(0, 38).trimEnd() + "…" : full;
    const size = Math.min(15, Math.max(10, 9 + 25 * Math.sqrt(c.share)));
    const w = text.length * size * 0.55 + 6, h = size * 1.35;
    const x = px(c.x!), y = py(c.y!) - 6; // "top": the text sits just above the marker
    let left = x - w / 2, pos = "top center";
    if (left < 2) { left = x - 4; pos = "top right"; }
    else if (left + w > o.width - 2) { left = x - w + 4; pos = "top left"; }
    const b: [number, number, number, number] = [left, y - h, left + w, y];
    const hit = boxes.some((q) => b[0] < q[2] && b[2] > q[0] && b[1] < q[3] && b[3] > q[1]);
    if (hit && c.cluster_id !== o.sel) continue;
    boxes.push(b);
    out.push({ id: c.cluster_id, x: c.x!, y: c.y!, text, size, pos });
  }
  return out;
}

type Layers = { papers: boolean; clusters: boolean; density: boolean; compare: boolean; badges: boolean; flows: boolean;
  migration: boolean; novel: boolean; pockets: boolean; bridges: boolean; people: boolean };

const TABS = ["Grid", "Detail", "Ranking", "Bridges", "People"];

/** The map's view lives in the URL (shareable, survives reloads); a link that changes it remounts the view. */
export function MapGrid() {
  const [sp] = useSearchParams();
  const navType = useNavigationType();
  const sig = ["facet", "alpha", "res", "sel", "focus", "tab"].map((k) => sp.get(k) ?? "").join("|");
  const last = useRef(sig);
  const [key, setKey] = useState(0);
  useEffect(() => {
    if (sig === last.current) return;
    if (navType === "PUSH") setKey((k) => k + 1);
    last.current = sig;
  }, [sig, navType]);
  return <MapView key={key} />;
}

function MapView() {
  const p = useProjectId();
  const { open } = usePaperDrawer();
  const [sp, setSp] = useSearchParams();
  const init = useRef({
    facet: FACETS.includes(sp.get("facet") ?? "") ? sp.get("facet")! : "problem",
    alpha: sp.get("alpha") != null ? Number(sp.get("alpha")) : 0.3,
    res: RES.includes(sp.get("res") ?? "") ? sp.get("res")! : "medium",
    sel: sp.get("sel"), focus: sp.get("focus"), tab: TABS.includes(sp.get("tab") ?? "") ? sp.get("tab")! : "Grid",
    win: /^\d{4}-\d{4}$/.test(sp.get("w") ?? "") ? (sp.get("w")!.split("-").map(Number) as [number, number]) : null,
  }).current;
  const [facet, setFacet] = useState(init.facet);
  const [alphaRaw, setAlpha] = useState(ALPHAS.includes(init.alpha) ? init.alpha : 0.3);
  const [res, setRes] = useState(init.res);
  const focus = init.focus;
  const alpha = GLOSS.has(facet) ? alphaRaw : 0;
  const [L, setL] = useState<Layers>({ papers: true, clusters: true, density: false, compare: false, badges: false, flows: false,
    migration: false, novel: false, pockets: false, bridges: false, people: false });
  const [topN, setTopN] = useState(30);
  // the map's width in pixels (rounded), so labels can be placed without overlapping
  const [mapW, setMapW] = useState(900);
  const ro = useRef<ResizeObserver | null>(null);
  const mapBox = useCallback((el: HTMLDivElement | null) => {
    ro.current?.disconnect();
    if (!el) return;
    ro.current = new ResizeObserver(([e]) => setMapW(Math.round(e.contentRect.width / 40) * 40 || 900));
    ro.current.observe(el);
  }, []);
  const [author, setAuthor] = useState("");
  const [tab, setTab] = useState(init.tab);
  const [sel, setSel] = useState<string | null>(init.sel);
  const [selPaper, setSelPaper] = useState<string | null>(focus);
  const { data: m, isLoading, isPlaceholderData: switching } = useApi<MapData>(`/p/${p}/map` + qs({ facet, alpha, res }), { staleTime: 600_000, keep: true });
  // the window is the project's shared one (Facets and details sheets read it too); a link's `w` sets it
  const [win, setWin] = useSharedWindow(p);
  useEffect(() => { if (init.win) setWin(init.win); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const asOf = useCatalogue(p).data?.as_of;
  // a focused paper selects its cluster once the map is in
  useEffect(() => {
    if (!m?.built || !focus || init.sel) return;
    const i = m.points.paper_id.indexOf(focus);
    if (i >= 0 && m.points.cluster_id[i]) setSel(m.points.cluster_id[i]);
  }, [m]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    setSp((prev) => {
      const n = new URLSearchParams(prev);
      const put = (k: string, v: string | null | undefined) => (v ? n.set(k, v) : n.delete(k));
      put("facet", facet); put("alpha", String(alpha)); put("res", res); put("sel", sel); put("tab", tab);
      put("w", win && m && (win[0] !== m.y0 || win[1] !== m.y1) ? `${win[0]}-${win[1]}` : null);
      return n;
    }, { replace: true });
  }, [facet, alpha, res, sel, tab, win, m]); // eslint-disable-line react-hooks/exhaustive-deps

  const nf = ["problem", "method", "innovation"].includes(facet) ? facet : "abstract";
  const { data: novelty } = useApi<{ paper_id: string[]; value: number[] }>(L.novel ? `/p/${p}/map/scores` + qs({ metric: "novelty", facet: nf }) : null);
  const { data: pockets } = useApi<{ paper_id: string[]; value: number[] }>(L.pockets ? `/p/${p}/map/scores?metric=local_momentum` : null);
  const { data: flowsC } = useApi<any>(L.flows && res === "medium" ? `/p/${p}/map/flows` + qs({ kind: "citation", facet, alpha }) : null);
  const { data: flowsM } = useApi<any>(L.migration && res === "medium" ? `/p/${p}/map/flows` + qs({ kind: "migration", facet, alpha }) : null);
  const onBridgeMap = facet === "problem" && alpha === 0.5 && res === "medium";
  const { data: arcs } = useApi<any[]>(L.bridges && onBridgeMap ? `/p/${p}/map/bridge-arcs` : null);
  const { data: authors } = useApi<{ author_id: string; label: string }[]>(L.people ? `/p/${p}/map/authors` : null);
  const { data: path } = useApi<{ year: number; x: number; y: number }[]>(L.people && author ? `/p/${p}/map/author-path` + qs({ author, facet, alpha }) : null);

  const t1 = m ? Math.min(Math.max(win?.[0] ?? m.y0, m.y0), m.y1) : 0, t2 = m ? Math.max(Math.min(win?.[1] ?? m.y1, m.y1), t1) : 0;
  const derived = useMemo(() => {
    if (!m?.built) return null;
    const P = m.points, n = P.x.length;
    const inWin: number[] = [], before: number[] = [], prevIdx: number[] = [];
    const prevLen = t2 - t1 + 1;
    for (let i = 0; i < n; i++) {
      const y = P.year[i];
      if (y >= t1 && y <= t2) inWin.push(i);
      else if (y < t1) before.push(i);
      if (y >= t1 - prevLen && y < t1) prevIdx.push(i);
    }
    const CY = m.cluster_year;
    const nWin = new Map<string, number>(), nPrev = new Map<string, number>();
    CY.cluster_id.forEach((c, i) => {
      const y = CY.year[i];
      if (y >= t1 && y <= t2) nWin.set(c, (nWin.get(c) ?? 0) + CY.n[i]);
      if (y >= t1 - prevLen && y < t1) nPrev.set(c, (nPrev.get(c) ?? 0) + CY.n[i]);
    });
    const status = new Map<string, string | null>(), scores = new Map<string, Record<string, any>>();
    m.status.forEach((s) => {
      if (s.year === t2) {
        status.set(s.cluster_id, s.status);
        scores.set(s.cluster_id, s.scores);
      }
    });
    const byId = new Map(m.clusters.map((c) => [c.cluster_id, c]));
    return { inWin, before, prevIdx, prevLen, nWin, nPrev, totWin: inWin.length, totPrev: prevIdx.length, status, scores, byId };
  }, [m, t1, t2]);

  const disp = (c: string) => {
    const cl = derived?.byId.get(c);
    const b = L.badges ? BADGE[derived?.status.get(c) ?? ""] ?? "" : "";
    return `${b} ${cl?.label ?? c}${cl?.kind === "hub" ? " ◇hub" : ""}`.trim();
  };

  const mapFig = useMemo(() => {
    if (!m?.built || !derived) return null;
    const P = m.points;
    const data: any[] = [];
    const annotations: any[] = [];
    const pick = (idx: number[], arr: any[]) => idx.map((i) => arr[i]);
    const pos = new Map(m.clusters.filter((c) => c.x != null).map((c) => [c.cluster_id, [c.x!, c.y!]]));
    const colors = new Map(m.clusters.map((c) => [c.cluster_id, c.color]));
    if (L.density && derived.inWin.length > 20) {
      const xr: [number, number] = [Math.min(...P.x), Math.max(...P.x)], yr: [number, number] = [Math.min(...P.y), Math.max(...P.y)];
      const now = hist2d(pick(derived.inWin, P.x), pick(derived.inWin, P.y), xr, yr);
      let z = now.z, cs: any = "Greys", zmid: number | undefined;
      if (L.compare && derived.prevIdx.length > 20) {
        const prev = hist2d(pick(derived.prevIdx, P.x), pick(derived.prevIdx, P.y), xr, yr);
        z = now.z.map((r, i) => r.map((v, j) => v - prev.z[i][j]));
        cs = "RdBu";
        zmid = 0;
      }
      data.push({ type: "contour", z, x: now.xc, y: now.yc, colorscale: cs, reversescale: cs === "Greys", zmid, opacity: 0.45, showscale: L.compare,
        contours: { coloring: "heatmap" }, line: { width: 0 }, hoverinfo: "skip" });
    }
    if (L.papers) {
      data.push({ type: "scattergl", mode: "markers", x: pick(derived.before, P.x), y: pick(derived.before, P.y), hoverinfo: "skip",
        marker: { size: 3, color: MUTED, opacity: 0.35 } });
      data.push({
        type: "scattergl", mode: "markers", x: pick(derived.inWin, P.x), y: pick(derived.inWin, P.y), customdata: pick(derived.inWin, P.paper_id),
        text: derived.inWin.map((i) => `${(P.title[i] ?? "").slice(0, 90)}<br>${P.year[i]} · ${derived.byId.get(P.cluster_id[i] ?? "")?.label ?? "unclustered"}`),
        hovertemplate: "%{text}<extra></extra>",
        marker: { size: 5, opacity: 0.85, color: derived.inWin.map((i) => {
          const c = P.cluster_id[i];
          return !sel || c === sel ? colors.get(c ?? "") ?? "#9aa0a6" : MUTED;
        }) },
      });
    }
    const winIds = new Set(derived.inWin.map((i) => P.paper_id[i]));
    const xyOf = new Map(derived.inWin.map((i) => [P.paper_id[i], i]));
    if (L.novel && novelty) {
      const vals = novelty.paper_id.map((id, k) => [id, novelty.value[k]] as const).filter(([id]) => winIds.has(id));
      const q = quantile(vals.map((v) => v[1]), 0.95);
      const top = vals.filter((v) => v[1] >= q);
      data.push({ type: "scatter", mode: "markers", x: top.map(([id]) => P.x[xyOf.get(id)!]), y: top.map(([id]) => P.y[xyOf.get(id)!]), customdata: top.map((v) => v[0]),
        text: top.map(([id, v]) => `novel (${nf}): ${v.toFixed(3)} (descriptive; novelty does not predict 3-year citations)<br>${(P.title[xyOf.get(id)!] ?? "").slice(0, 80)}`),
        hovertemplate: "%{text}<extra></extra>", marker: { size: 11, symbol: "circle-open", color: "#1f2328", line: { width: 1.5 } } });
    }
    if (L.pockets && pockets) {
      const top = pockets.paper_id.map((id, k) => [id, pockets.value[k]] as const).filter(([id]) => winIds.has(id));
      data.push({ type: "scatter", mode: "markers", x: top.map(([id]) => P.x[xyOf.get(id)!]), y: top.map(([id]) => P.y[xyOf.get(id)!]), customdata: top.map((v) => v[0]),
        text: top.map(([id, v]) => `rising pocket: local momentum ${v.toFixed(2)}<br>${(P.title[xyOf.get(id)!] ?? "").slice(0, 80)}`),
        hovertemplate: "%{text}<extra></extra>", marker: { size: 13, symbol: "circle-open", color: FLOW_UP, line: { width: 2 } } });
    }
    const windowFlows = (f: any) => {
      const now = new Map<string, number>(), prev = new Map<string, number>();
      f.year.forEach((y: number, i: number) => {
        const k = `${f.from_cluster[i]}|${f.to_cluster[i]}`;
        if (y >= t1 && y <= t2) now.set(k, (now.get(k) ?? 0) + f.count[i]);
        if (y >= t1 - derived.prevLen && y < t1) prev.set(k, (prev.get(k) ?? 0) + f.count[i]);
      });
      const top = [...now.entries()].filter(([k]) => { const [a, b] = k.split("|"); return pos.has(a) && pos.has(b); })
        .sort((a, b) => b[1] - a[1]).slice(0, topN);
      return { top, prev };
    };
    if (L.flows && flowsC && res === "medium") {
      const { top, prev } = windowFlows(flowsC);
      const mx = Math.max(1, ...top.map((t) => t[1]));
      top.forEach(([k, n]) => {
        const [a, b] = k.split("|");
        const [xs, ys] = arc(pos.get(a)![0], pos.get(a)![1], pos.get(b)![0], pos.get(b)![1]);
        const d = n / derived.prevLen - (prev.get(k) ?? 0) / derived.prevLen;
        data.push({ type: "scatter", mode: "lines", x: xs, y: ys, opacity: 0.55, hoverinfo: "text",
          line: { width: 1 + (7 * n) / mx, color: d >= 0 ? FLOW_UP : FLOW_DOWN },
          hovertext: `${disp(a)} → ${disp(b)}<br>${n} citations in ${t1}–${t2} (${d >= 0 ? "+" : ""}${d.toFixed(1)}/yr vs previous window)` });
      });
    }
    if (L.migration && flowsM && res === "medium") {
      const { top } = windowFlows(flowsM);
      const mx = Math.max(1, ...top.map((t) => t[1]));
      top.forEach(([k, n]) => {
        const [a, b] = k.split("|");
        annotations.push({ x: pos.get(b)![0], y: pos.get(b)![1], ax: pos.get(a)![0], ay: pos.get(a)![1], xref: "x", yref: "y", axref: "x", ayref: "y",
          showarrow: true, arrowhead: 3, arrowsize: 1, arrowwidth: 1 + (4 * n) / mx, arrowcolor: "#495057", opacity: 0.7,
          hovertext: `${n} authors moved ${disp(a)} → ${disp(b)}`, text: "" });
      });
    }
    if (L.bridges && arcs && onBridgeMap) {
      arcs.filter((r) => r.rank <= Math.max(5, Math.floor(topN / 2)) && pos.has(r.a) && pos.has(r.b)).forEach((r) => {
        const [xs, ys] = arc(pos.get(r.a)![0], pos.get(r.a)![1], pos.get(r.b)![0], pos.get(r.b)![1], -0.2);
        data.push({ type: "scatter", mode: "lines", x: xs, y: ys, line: { color: ACCENT, width: 2, dash: "dash" }, hoverinfo: "text",
          hovertext: `${r.label} #${r.rank}<br>${disp(r.a)} → ${disp(r.b)}<br>judge: ${r.judge ?? "—"}` });
      });
    }
    if (L.people && path?.length) {
      const pth = path.filter((r) => r.year <= t2);
      data.push({ type: "scatter", mode: "lines+markers+text", x: pth.map((r) => r.x), y: pth.map((r) => r.y), text: pth.map((r) => String(r.year)),
        textposition: "top right", textfont: { size: 9 }, line: { color: "#1f2328", width: 2 }, marker: { size: 6, color: "#1f2328" }, hoverinfo: "skip" });
      if (pth.length) data.push({ type: "scatter", mode: "markers", x: [pth[pth.length - 1].x], y: [pth[pth.length - 1].y], marker: { size: 14, color: ACCENT, symbol: "star" }, hoverinfo: "skip" });
    }
    if (L.clusters) {
      const cs = m.clusters.filter((c) => c.x != null).map((c) => ({ ...c, share: (derived.nWin.get(c.cluster_id) ?? 0) / Math.max(derived.totWin, 1) }))
        .sort((a, b) => b.share - a.share);
      data.push({ type: "scatter", mode: "markers", x: cs.map((c) => c.x), y: cs.map((c) => c.y), customdata: cs.map((c) => "cluster:" + c.cluster_id), hoverinfo: "text",
        marker: { size: cs.map((c) => (c.kind === "hub" ? 11 : 7)), color: cs.map((c) => c.color), symbol: cs.map((c) => (c.kind === "hub" ? "diamond" : "circle")), line: { color: "var(--ring)", width: 1 } },
        hovertext: cs.map((c) => `${disp(c.cluster_id)}<br>${(c.share * 100).toFixed(1)}% of window` + (derived.status.get(c.cluster_id) ? `<br>status ${t2}: ${derived.status.get(c.cluster_id)}` : "")) });
      const txt = placeLabels(cs, { sel, badged: L.badges ? (c) => !!BADGE[derived.status.get(c) ?? ""] : undefined, label: disp,
        xs: P.x, ys: P.y, width: mapW, height: MAP_H });
      // annotations, not a text trace: they draw above the WebGL points, on a translucent backing
      txt.forEach((t) => annotations.push({ x: t.x, y: t.y, text: t.text, showarrow: false, yshift: 5, yanchor: "bottom",
        xanchor: t.pos === "top right" ? "left" : t.pos === "top left" ? "right" : "center", borderpad: 1.5, bgcolor: "var(--label-bg)",
        font: { size: t.size, color: t.id === sel ? ACCENT : "var(--text-strong)" }, captureevents: false }));
    }
    // the focused paper (from "Show on the map"): a star, and the view zoomed around it
    const fi = focus ? P.paper_id.indexOf(focus) : -1;
    const xaxis: any = { visible: false }, yaxis: any = { visible: false, scaleanchor: "x" };
    if (fi >= 0) {
      data.push({ type: "scatter", mode: "markers+text", x: [P.x[fi]], y: [P.y[fi]], customdata: [focus], text: ["this paper"], textposition: "bottom center",
        textfont: { size: 11, color: ACCENT }, hovertext: (P.title[fi] ?? "").slice(0, 90), hoverinfo: "text",
        marker: { size: 18, symbol: "star", color: ACCENT, line: { color: "var(--ring)", width: 1.5 } } });
      const dx = (Math.max(...P.x) - Math.min(...P.x)) / 7, dy = (Math.max(...P.y) - Math.min(...P.y)) / 7;
      xaxis.range = [P.x[fi] - dx, P.x[fi] + dx];
      yaxis.range = [P.y[fi] - dy, P.y[fi] + dy];
    }
    return { data, layout: { margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false, xaxis, yaxis, dragmode: "pan", annotations,
      uirevision: `${facet}|${alpha}|${res}|${focus ?? ""}` } };
  }, [m, derived, L, sel, novelty, pockets, flowsC, flowsM, arcs, path, topN, t1, t2, res, onBridgeMap, nf, focus, mapW]); // eslint-disable-line react-hooks/exhaustive-deps

  const axisFig = useMemo(() => {
    if (!m?.built) return null;
    const yr = m.years.filter((r) => r.year >= m.y0);
    const data: any[] = [{ type: "bar", x: yr.map((r) => r.year), y: yr.map((r) => r.n), marker: { color: yr.map((r) => (r.year >= t1 && r.year <= t2 ? "#7d8590" : MUTED)) }, hovertemplate: "%{x}: %{y} papers<extra></extra>" }];
    if (sel) {
      const CY = m.cluster_year;
      const pts = CY.cluster_id.map((c, i) => [c, CY.year[i], CY.share[i]] as const).filter(([c, y]) => c === sel && y >= m.y0).sort((a, b) => a[1] - b[1]);
      data.push({ type: "scatter", x: pts.map((p) => p[1]), y: pts.map((p) => p[2]), yaxis: "y2", mode: "lines+markers", line: { color: ACCENT }, hovertemplate: "%{x}: %{y:.1%}<extra>selected</extra>" });
    }
    return { data, layout: { margin: { l: 30, r: 40, t: 4, b: 22 }, showlegend: false, bargap: 0.15, xaxis: { dtick: 2 },
      yaxis: { showgrid: false }, yaxis2: { overlaying: "y", side: "right", tickformat: ".0%", showgrid: false },
      shapes: [{ type: "rect", xref: "x", yref: "paper", x0: t1 - 0.5, x1: t2 + 0.5, y0: 0, y1: 1, line: { color: ACCENT, width: 1.5 }, fillcolor: "rgba(0,0,0,0)" }] } };
  }, [m, t1, t2, sel]);

  const toggle = (k: keyof Layers) => setL((l) => ({ ...l, [k]: !l[k] }));
  const nosig = !m?.have_signals;
  const active = [L.density, L.flows, L.migration, L.novel, L.pockets, L.bridges, L.people].filter(Boolean).length;

  if (isLoading && !m) return <PageSkeleton stats={false} />;
  if (m && !m.built)
    return (
      <div className="mx-auto max-w-3xl">
        <PageHeader title="Map & grid" />
        <Empty title={m.reason === "core" ? "Communities are not built yet" : "No layout for this facet yet"}>
          Run <code>{m.reason ?? "viz"}</code> for this project. <Link className="text-brand-600" to={`/p/${p}/pipeline`}>Open the pipeline →</Link>
        </Empty>
      </div>
    );
  return (
    <div className="mx-auto max-w-[1700px]">
      <PageHeader title="Map & grid" sub="One map, one grid, one time axis. Every layer reads the same window. The view is kept in the address, so a link reopens it.">
        {sel && <CatLink cid={sel} className="btn btn-sm !no-underline">Open the selected cluster's details</CatLink>}
      </PageHeader>
      <div className="grid gap-4 xl:grid-cols-[240px_minmax(0,1fr)] min-[1800px]:grid-cols-[240px_minmax(0,1.4fr)_minmax(0,1fr)]">
        <Card title="Controls" icon={<SlidersHorizontal className="size-4 text-brand-600" />} className="self-start">
          <div className="flex flex-col gap-3 text-[13px]">
            <div>
              <label className="label">Facet</label>
              <select className="select" value={facet} onChange={(e) => setFacet(e.target.value)}>{FACETS.map((f) => <option key={f}>{f}</option>)}</select>
            </div>
            <div>
              <label className="label" title="0 = the abstract's wording; higher groups by concept">Gloss weight α</label>
              <Seg size="sm" value={alpha} onChange={setAlpha} options={(GLOSS.has(facet) ? ALPHAS : [0]).map((a) => ({ value: a, label: String(a) }))} />
            </div>
            <div>
              <label className="label">Resolution</label>
              <Seg size="sm" value={res} onChange={setRes} options={RES} />
            </div>
            <div>
              <label className="label">Layers</label>
              {([["papers", "Papers"], ["clusters", "Clusters"], ["density", "Activity shift"], ["compare", "Compare with previous window"],
                ["badges", "Status badges"], ["flows", "Citation flows"], ["migration", "Author migration"], ["novel", "Novel papers"],
                ["pockets", "Rising pockets"], ["bridges", "Bridges"], ["people", "People"]] as [keyof Layers, string][]).map(([k, label]) => {
                const dis = (k === "compare" && !L.density) || (nosig && ["badges", "flows", "migration", "novel", "pockets", "bridges", "people"].includes(k));
                return (
                  <label key={k} className={clsx("flex items-center gap-2 py-0.5", k === "compare" && "pl-5", dis && "opacity-45")}
                    title={k === "badges" ? "Direction labels (emerging / niche / dormant). Weak in the backtest and sensitive to how the corpus was sampled: read as prompts, not trends." : k === "pockets" ? "Recent papers whose neighbourhood is filling up fastest (top 5% local momentum)" : undefined}>
                    <input type="checkbox" disabled={dis} checked={L[k]} onChange={() => toggle(k)} className="accent-[var(--color-brand-600)]" /> {label}
                  </label>
                );
              })}
              {active > 3 && <Note>Tip: more than three active layers get hard to read.</Note>}
            </div>
            {(L.flows || L.migration || L.bridges) && (
              <div>
                <label className="label">Arcs / arrows shown: {topN}</label>
                <input type="range" min={5} max={60} value={topN} onChange={(e) => setTopN(Number(e.target.value))} className="w-full accent-[var(--color-brand-600)]" />
              </div>
            )}
            {L.people && (
              <div>
                <label className="label">Author path</label>
                <select className="select" value={author} onChange={(e) => setAuthor(e.target.value)}>
                  <option value="">—</option>
                  {(authors ?? []).map((a) => <option key={a.author_id} value={a.author_id}>{a.label}</option>)}
                </select>
              </div>
            )}
          </div>
        </Card>

        <div className="flex min-w-0 flex-col gap-3">
          <div ref={mapBox} className="card relative overflow-hidden">
            {switching && (
              <div className="anim-fade absolute inset-0 z-10 grid place-items-center bg-[var(--surface)]/60 backdrop-blur-[1px]">
                <Spinner label={`Loading the ${facet} map…`} />
              </div>
            )}
            {mapFig && <Plot data={mapFig.data} layout={mapFig.layout as any} height={MAP_H} config={{ scrollZoom: true, displayModeBar: "hover", modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"] }}
              onClick={(pt: any) => {
                const cd = pt.customdata as string | undefined;
                if (!cd) return;
                if (cd.startsWith("cluster:")) {
                  setSel(cd.slice(8));
                  setTab("Detail");
                  return;
                }
                setSelPaper(cd);
                const i = m!.points.paper_id.indexOf(cd);
                const c = m!.points.cluster_id[i];
                if (c && derived?.byId.has(c)) setSel(c);
                setTab("Detail");
              }} />}
          </div>
          <div className="card px-4 py-3">
            {m && <TimeSeeker y0={m.y0} y1={m.y1} asOf={asOf && asOf <= m.y1 ? asOf : undefined} value={[t1, t2]} onChange={setWin} />}
            {axisFig && <Plot data={axisFig.data} layout={axisFig.layout as any} height={130}
              onClick={(pt: any) => setWin([Number(pt.x), Number(pt.x)])} />}
            {L.pockets && <Note>Rising pockets (green rings): recent papers whose closest earlier neighbours are filling up faster than last year's (top 5%). "This small neighbourhood is getting busier": a place to look, not a forecast (about twice random in the backtest). A paper recorded twice in OpenAlex can make a false pocket.</Note>}
            {L.badges && <Note>Status at {t2}: ▲ emerging · ◆ niche · ○ dormant · ◇ hub. <b>Read with care:</b> emerging was barely better than chance in the backtest, and several topics reverse direction once changes in the sampling mix are separated out.</Note>}
            {(L.flows || L.migration) && res !== "medium" && <Note>Citation flows and migration are computed for medium clusters.</Note>}
            {L.bridges && !onBridgeMap && <Note>Bridges link problem clusters at α = 0.5, medium resolution: switch the map there to draw them. The Bridges tab lists them from any view.</Note>}
          </div>
        </div>

        <div className="min-w-0 xl:col-span-2 min-[1800px]:col-span-1">
          <div className="card">
            <div className="scroll-thin flex overflow-x-auto border-b border-[var(--line)] px-4">
              {TABS.map((t) => <button key={t} className={clsx("tab", t === tab && "active")} onClick={() => setTab(t)}>{t}</button>)}
            </div>
            <div className="p-4">
              {m && derived && tab === "Grid" && <GridPanel p={p} alpha={GLOSS.has(facet) ? alpha : 0.3} t1={t1} t2={t2} sel={sel} haveSignals={m.have_signals} />}
              {m && derived && tab === "Detail" && <DetailPanel p={p} m={m} sel={sel} setSel={setSel} res={res} alpha={alpha} t1={t1} t2={t2} nWin={derived.nWin}
                status={derived.status} scores={derived.scores} disp={disp} selPaper={selPaper} setSelPaper={setSelPaper} />}
              {m && derived && tab === "Ranking" && <RankingPanel p={p} m={m} derived={derived} t1={t1} t2={t2} onSelect={(c) => { setSel(c); setTab("Detail"); }} />}
              {tab === "Bridges" && <BridgesPanel p={p} sel={sel} onPaper={(pid) => open(pid)} />}
              {tab === "People" && <PeoplePanel p={p} />}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
