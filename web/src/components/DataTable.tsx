import clsx from "clsx";
import { ArrowDown, ArrowUp } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

export type Col<T> = {
  key: string;
  label: ReactNode;
  render?: (row: T, i: number) => ReactNode;
  value?: (row: T) => any; // sort value
  align?: "left" | "right" | "center";
  width?: number | string;
  className?: string;
  sortable?: boolean;
  title?: string;
};

export function fmtCell(v: any): ReactNode {
  if (v == null || (typeof v === "number" && Number.isNaN(v))) return <span className="faint">—</span>;
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(Math.abs(v) < 1 ? 3 : 2);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Sortable table with optional row selection; columns default to the row's keys. */
export function DataTable<T extends Record<string, any>>({ rows, cols, onSelect, selected, maxHeight, rowKey, empty,
  dense, initialSort, className }: {
  rows: T[]; cols?: Col<T>[]; onSelect?: (row: T, i: number) => void; selected?: (row: T) => boolean;
  maxHeight?: number; rowKey?: (row: T, i: number) => string | number; empty?: ReactNode; dense?: boolean;
  initialSort?: { key: string; dir: 1 | -1 }; className?: string;
}) {
  const columns: Col<T>[] = useMemo(
    () => cols ?? (rows[0] ? Object.keys(rows[0]).map((k) => ({ key: k, label: k.replace(/_/g, " ") })) : []),
    [cols, rows],
  );
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(initialSort ?? null);
  const sorted = useMemo(() => {
    if (!sort) return rows.map((r, i) => [r, i] as const);
    const c = columns.find((c) => c.key === sort.key);
    const get = c?.value ?? ((r: T) => r[sort.key]);
    return rows
      .map((r, i) => [r, i] as const)
      .sort(([a], [b]) => {
        const x = get(a), y = get(b);
        if (x == null) return 1;
        if (y == null) return -1;
        return (x > y ? 1 : x < y ? -1 : 0) * sort.dir;
      });
  }, [rows, sort, columns]);
  if (!rows.length) return <div className="muted py-4 text-[13px]">{empty ?? "Nothing to show."}</div>;
  return (
    <div className={clsx("scroll-thin overflow-auto rounded-lg border border-[var(--line)]", className)} style={{ maxHeight }}>
      <table className={clsx("table", dense && "[&_td]:py-1.5")}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                style={{ width: c.width, textAlign: c.align }}
                title={c.title}
                className={clsx(c.sortable !== false && "cursor-pointer hover:text-[var(--text-strong)]")}
                onClick={() =>
                  c.sortable !== false &&
                  setSort((s) => (s?.key === c.key ? (s.dir === -1 ? { key: c.key, dir: 1 } : null) : { key: c.key, dir: -1 }))
                }
              >
                <span className="inline-flex items-center gap-1">
                  {c.label}
                  {sort?.key === c.key && (sort.dir === -1 ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />)}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(([r, i]) => (
            <tr
              key={rowKey ? rowKey(r, i) : i}
              className={clsx(onSelect && "clickable", selected?.(r) && "selected")}
              onClick={() => onSelect?.(r, i)}
            >
              {columns.map((c) => (
                <td key={c.key} style={{ textAlign: c.align }} className={c.className}>
                  {c.render ? c.render(r, i) : fmtCell(r[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
