import type Plotly from "plotly.js-dist-min";
import { useEffect, useRef, useState } from "react";

// Plotly is large: it loads with the first chart, not with the app
let plotly: Promise<typeof Plotly> | null = null;
const loadPlotly = () => (plotly ??= import("plotly.js-dist-min").then((m: any) => m.default ?? m));

export const ACCENT = "#e8590c"; // reserved for what a reviewer should look at: selection, playhead, new cells
export const MUTED = "var(--plot-muted)"; // context: what lies outside the window or the selection
/** Light-to-brand scale for heatmaps; its stops follow the theme (resolved by `Plot`). */
export const HEAT: [number, string][] = [[0, "var(--heat-0)"], [0.15, "var(--heat-1)"], [1, "var(--heat-2)"]];

/** Changes whenever the light/dark theme does (the theme switch or the system setting). */
export function useThemeKey() {
  const [k, setK] = useState(0);
  useEffect(() => {
    const bump = () => setK((x) => x + 1);
    const mo = new MutationObserver(bump);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", bump);
    return () => {
      mo.disconnect();
      mq.removeEventListener("change", bump);
    };
  }, []);
  return k;
}

/** Plotly does not read CSS variables: every "var(--x)" string in a figure becomes the current theme's value. */
function resolveVars<T>(v: T, css: CSSStyleDeclaration): T {
  if (typeof v === "string")
    return (v.includes("var(--") ? v.replace(/var\((--[\w-]+)\)/g, (m, name) => css.getPropertyValue(name).trim() || m) : v) as T;
  // numeric arrays (coordinates, counts) are passed through untouched; [stop, colour] pairs are not numeric
  if (Array.isArray(v)) return (typeof v[0] === "number" && typeof v[v.length - 1] === "number" ? v : v.map((x) => resolveVars(x, css))) as T;
  if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, resolveVars(x, css)])) as T;
  return v;
}

function themeVars() {
  const s = getComputedStyle(document.documentElement);
  return {
    text: s.getPropertyValue("--muted").trim() || "#64708a",
    strong: s.getPropertyValue("--text-strong").trim() || "#0f1b3d",
    grid: s.getPropertyValue("--plot-grid").trim() || "#edf0f6",
    surface: s.getPropertyValue("--surface").trim() || "#fff",
  };
}

export function baseLayout(extra: Partial<Plotly.Layout> = {}): Partial<Plotly.Layout> {
  const t = themeVars();
  return {
    margin: { l: 40, r: 10, t: 10, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter Variable, system-ui, sans-serif", size: 11.5, color: t.text },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.grid, font: { color: t.strong, size: 12 } },
    xaxis: { gridcolor: t.grid, zerolinecolor: t.grid, linecolor: t.grid },
    yaxis: { gridcolor: t.grid, zerolinecolor: t.grid, linecolor: t.grid },
    legend: { orientation: "h", y: -0.18 },
    ...extra,
  };
}

/** Thin Plotly wrapper: re-renders with Plotly.react, forwards clicks. */
export function Plot({ data, layout, height = 300, onClick, config, className }: {
  data: any[]; layout?: Partial<Plotly.Layout>; height?: number;
  onClick?: (pt: Plotly.PlotDatum) => void; config?: Partial<Plotly.Config>; className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;
  const theme = useThemeKey();
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let alive = true;
    loadPlotly().then((P) => {
      const el = ref.current as any;
      if (!alive || !el) return;
      const css = getComputedStyle(document.documentElement);
      const base = baseLayout();
      const full: Record<string, any> = { ...base, height, ...layout };
      // a chart's own axis settings add to the themed axis style instead of replacing it
      Object.keys(full).forEach((k) => { if (/^[xy]axis\d*$/.test(k)) full[k] = { ...base.xaxis, ...full[k] }; });
      P.react(el, resolveVars(data, css) as Plotly.Data[], resolveVars(full, css) as Plotly.Layout, {
        displaylogo: false, responsive: true, displayModeBar: false, ...config,
      });
      if (!el.__clicks) {
        el.__clicks = true;
        el.on?.("plotly_click", (e: Plotly.PlotMouseEvent) => e.points?.[0] && clickRef.current?.(e.points[0]));
      }
      setReady(true);
    });
    return () => {
      alive = false;
    };
  }, [data, layout, height, config, theme]);
  useEffect(() => {
    const el = ref.current as any;
    const ro = new ResizeObserver(() => el.isConnected && el._fullLayout && plotly?.then((P) => P.Plots.resize(el)));
    ro.observe(el);
    return () => {
      ro.disconnect();
      el.removeAllListeners?.("plotly_click");
      if (el._fullLayout) plotly?.then((P) => P.purge(el));
    };
  }, []);
  return (
    <div className={className} style={{ position: "relative", width: "100%", height }}>
      {!ready && <div className="skel absolute inset-0 rounded-lg opacity-60" />}
      <div ref={ref} style={{ width: "100%", height }} />
    </div>
  );
}
