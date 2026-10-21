import clsx from "clsx";
import { CheckCircle2, Info, X, XCircle } from "lucide-react";
import { useEffect, useSyncExternalStore, type ReactNode } from "react";
import { Link } from "react-router-dom";

export type Toast = { id: number; title: ReactNode; body?: ReactNode; tone?: "success" | "error" | "info"; href?: string; linkLabel?: string; ttl?: number };

let toasts: Toast[] = [];
let next = 1;
const subs = new Set<() => void>();
const emit = () => subs.forEach((f) => f());

/** Shows a short notice in the corner: confirmations, finished or failed jobs. Errors stay until dismissed. */
export function toast(t: Omit<Toast, "id">) {
  toasts = [...toasts.slice(-3), { ...t, id: next++ }];
  emit();
}

const dismiss = (id: number) => {
  toasts = toasts.filter((t) => t.id !== id);
  emit();
};

/** Copies text and confirms it with a toast. */
export function copyText(text: string, what = "Link") {
  navigator.clipboard?.writeText(text).then(
    () => toast({ title: `${what} copied`, tone: "success", ttl: 1800 }),
    () => toast({ title: `Could not copy the ${what.toLowerCase()}`, tone: "error", ttl: 3000 }),
  );
}

function ToastItem({ t }: { t: Toast }) {
  const ttl = t.ttl ?? (t.tone === "error" ? 0 : 5000);
  useEffect(() => {
    if (!ttl) return;
    const h = setTimeout(() => dismiss(t.id), ttl);
    return () => clearTimeout(h);
  }, [t.id, ttl]);
  const Icon = t.tone === "success" ? CheckCircle2 : t.tone === "error" ? XCircle : Info;
  return (
    <div role={t.tone === "error" ? "alert" : "status"}
      className="anim-toast card pointer-events-auto flex w-[340px] max-w-[calc(100vw-32px)] items-start gap-2.5 px-3.5 py-3 shadow-xl">
      <Icon className={clsx("mt-0.5 size-4 shrink-0", t.tone === "success" ? "text-emerald-600" : t.tone === "error" ? "text-red-600" : "text-brand-500")} />
      <div className="min-w-0 flex-1 text-[13px]">
        <div className="font-medium text-[var(--text-strong)]">{t.title}</div>
        {t.body && <div className="muted mt-0.5 text-[12.5px] leading-snug">{t.body}</div>}
        {t.href && <Link to={t.href} onClick={() => dismiss(t.id)} className="mt-1 inline-block text-[12.5px] font-medium text-brand-600 hover:underline">{t.linkLabel ?? "Open"} →</Link>}
      </div>
      <button className="faint -mt-0.5 -mr-1 rounded p-0.5 hover:text-[var(--text-strong)]" onClick={() => dismiss(t.id)} aria-label="Dismiss"><X className="size-3.5" /></button>
    </div>
  );
}

export function Toaster() {
  const list = useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, () => toasts);
  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-[70] flex flex-col items-end gap-2" aria-live="polite">
      {list.map((t) => <ToastItem key={t.id} t={t} />)}
    </div>
  );
}
