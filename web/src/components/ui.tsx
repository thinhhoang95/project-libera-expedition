import clsx from "clsx";
import { ChevronDown, ChevronUp, Info, Loader2 } from "lucide-react";
import { Component, useMemo, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function Card({ title, icon, actions, children, className, pad = true, subtitle }: {
  title?: ReactNode; icon?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; pad?: boolean;
  subtitle?: ReactNode;
}) {
  return (
    <section className={clsx("card min-w-0", className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-5 pt-4 pb-2">
          <div className="min-w-0">
            <h2 className="h-title flex items-center gap-2 text-[15px]">
              {icon}
              <span className="truncate">{title}</span>
            </h2>
            {subtitle && <p className="muted mt-0.5 text-[12.5px]">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx(pad && (title ? "px-5 pb-4" : "card-pad"))}>{children}</div>
    </section>
  );
}

export function IconBadge({ children, tone = "brand" }: { children: ReactNode; tone?: "brand" | "green" | "orange" | "violet" | "red" }) {
  const tones = {
    brand: "bg-[var(--chip)] text-brand-600",
    green: "bg-emerald-500/12 text-emerald-600",
    orange: "bg-orange-500/12 text-orange-600",
    violet: "bg-violet-500/12 text-violet-600",
    red: "bg-red-500/12 text-red-600",
  };
  return <span className={clsx("grid size-9 shrink-0 place-items-center rounded-[10px]", tones[tone])}>{children}</span>;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="muted flex items-center gap-2 py-6 text-[13px]">
      <Loader2 className="size-4 animate-spin" /> {label ?? "Loading…"}
    </div>
  );
}

export function Empty({ title, children, icon }: { title: ReactNode; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-[var(--line-strong)] px-6 py-10 text-center">
      {icon}
      <div className="h-title text-[14px]">{title}</div>
      {children && <div className="muted max-w-lg text-[13px]">{children}</div>}
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  return (
    <div className="rounded-xl border border-red-300/60 bg-red-500/5 px-4 py-3 text-[13px] text-red-700 dark:text-red-300">
      {error instanceof Error ? error.message : String(error)}
    </div>
  );
}

export function Md({ children, className }: { children: string; className?: string }) {
  return (
    <div className={clsx("prose-sm", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}

export function HowToRead({ children, title = "How to read this" }: { children: string; title?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2 rounded-lg border border-[var(--line)] bg-[var(--surface-2)]">
      <button onClick={() => setOpen(!open)} className="muted flex w-full items-center gap-1.5 px-3 py-2 text-[12.5px] font-medium">
        <Info className="size-3.5" /> {title}
        {open ? <ChevronUp className="ml-auto size-3.5" /> : <ChevronDown className="ml-auto size-3.5" />}
      </button>
      {open && <Md className="px-3 pb-3 text-[12.8px]">{children}</Md>}
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="muted mt-2.5 text-[12.3px] leading-relaxed">{children}</p>;
}

export function Seg<T extends string | number>({ value, options, onChange, size }: {
  value: T; options: { value: T; label: ReactNode }[] | readonly T[]; onChange: (v: T) => void; size?: "sm";
}) {
  const opts = (options as any[]).map((o) => (typeof o === "object" ? o : { value: o, label: String(o) }));
  return (
    <div className={clsx("seg", size === "sm" && "text-[12px]")}>
      {opts.map((o) => (
        <button key={String(o.value)} className={clsx(o.value === value && "on")} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Bar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={clsx("h-[7px] w-full overflow-hidden rounded-full bg-[var(--line)]", className)}>
      <div className="h-full rounded-full bg-brand-500" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

export function Sparkline({ values, color = "var(--color-brand-500)", width = 96, height = 30, fill = true }: {
  values: number[]; color?: string; width?: number; height?: number; fill?: boolean;
}) {
  const id = useMemo(() => `g${Math.random().toString(36).slice(2)}`, []);
  if (!values.length) return null;
  const max = Math.max(...values, 1e-9), min = Math.min(...values, 0);
  const pts = values.map((v, i) => [
    (i / Math.max(values.length - 1, 1)) * (width - 2) + 1,
    height - 2 - ((v - min) / (max - min || 1)) * (height - 4),
  ]);
  // smooth path (Catmull-Rom → Bezier)
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] ?? p2;
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${p2[0]},${p2[1]}`;
  }
  return (
    <svg width={width} height={height} className="overflow-visible">
      {fill && (
        <>
          <defs>
            <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor={color} stopOpacity="0.22" />
              <stop offset="1" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={`${d} L${pts[pts.length - 1][0]},${height} L${pts[0][0]},${height} Z`} fill={`url(#${id})`} />
        </>
      )}
      <path d={d} fill="none" stroke={color} strokeWidth={1.8} strokeLinecap="round" />
    </svg>
  );
}

export function StatCard({ icon, tone, label, value, sub, spark, sparkColor, hint }: {
  icon: ReactNode; tone?: "brand" | "green" | "orange" | "violet"; label: string; value: ReactNode; sub?: ReactNode;
  spark?: number[]; sparkColor?: string; hint?: string;
}) {
  return (
    <div className="card flex items-center gap-3.5 px-4 py-4" title={hint}>
      <IconBadge tone={tone}>{icon}</IconBadge>
      <div className="min-w-0 flex-1">
        <div className="muted truncate text-[12.5px] font-medium">{label}</div>
        <div className="flex items-baseline gap-2">
          <span className="h-title text-[24px] leading-tight">{value}</span>
          {sub && <span className="text-[12px] font-medium">{sub}</span>}
        </div>
      </div>
      {spark && spark.length > 1 && (
        <div className="hidden shrink-0 sm:block">
          <Sparkline values={spark} color={sparkColor} />
        </div>
      )}
    </div>
  );
}

export function Pill({ children, tone = "neutral", className, title }: {
  children: ReactNode; tone?: "neutral" | "brand" | "green" | "red" | "amber" | "violet"; className?: string; title?: string;
}) {
  const t = {
    neutral: "bg-[var(--surface-2)] text-[var(--muted)] border-[var(--line)]",
    brand: "bg-[var(--chip)] text-[var(--chip-text)] border-transparent",
    green: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/20",
    red: "bg-red-500/10 text-red-700 dark:text-red-300 border-red-500/20",
    amber: "bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/25",
    violet: "bg-violet-500/10 text-violet-700 dark:text-violet-300 border-violet-500/20",
  }[tone];
  return <span title={title} className={clsx("inline-flex items-center gap-1 rounded-md border px-1.5 py-[1px] text-[11.5px] font-medium", t, className)}>{children}</span>;
}

export const pct = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 0) =>
  v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
export const signedPct = (v: number | null | undefined, d = 1) =>
  v == null ? "—" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%`;

/** Keeps one failing view from blanking the app; resets when `resetKey` changes (e.g. the route). */
export class Boundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }
  render() {
    if (this.state.error)
      return (
        <div className="mx-auto max-w-2xl">
          <ErrorBox error={`This view failed: ${this.state.error.message}`} />
          <button className="btn btn-sm mt-3" onClick={() => this.setState({ error: null })}>Try again</button>
        </div>
      );
    return this.props.children;
  }
}
