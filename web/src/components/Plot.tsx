import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "react";

export const ACCENT = "#e8590c"; // reserved for what a reviewer should look at: selection, playhead, new cells
export const MUTED = "#c9cdd2";

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
  useEffect(() => {
    const el = ref.current!;
    Plotly.react(el, data as Plotly.Data[], { ...baseLayout(), height, ...layout } as Plotly.Layout, {
      displaylogo: false, responsive: true, displayModeBar: false, ...config,
    });
  }, [data, layout, height, config]);
  useEffect(() => {
    const el = ref.current as any;
    const h = (e: Plotly.PlotMouseEvent) => e.points?.[0] && clickRef.current?.(e.points[0]);
    el.on?.("plotly_click", h);
    const ro = new ResizeObserver(() => el.isConnected && el._fullLayout && Plotly.Plots.resize(el));
    ro.observe(el);
    return () => {
      ro.disconnect();
      el.removeAllListeners?.("plotly_click");
      Plotly.purge(el);
    };
  }, []);
  return <div ref={ref} className={className} style={{ width: "100%", height }} />;
}
