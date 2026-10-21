import clsx from "clsx";
import { ArrowDown, ArrowUp, ArrowUpDown, Download } from "lucide-react";
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
  /** Hidden when the table's box is narrower than about 640px (a sparkline, a secondary count). */
  narrow?: boolean;
  /** The value written to CSV, when neither `value` nor the row's field is it. */
  csv?: (row: T) => any;
};

/** Saves rows as a CSV file. */
export function downloadCsv(name: string, rows: Record<string, any>[]) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc = (v: any) => `"${String(Array.isArray(v) ? v.join("; ") : v != null && typeof v === "object" ? JSON.stringify(v) : v ?? "").replace(/"/g, '""')}"`;
  const blob = new Blob([[cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

const plain = (n: ReactNode, fallback: string) => (typeof n === "string" || typeof n === "number" ? String(n) : fallback);

/** The table as shown (its columns, in its current order) for a CSV export. */
function tableRows<T extends Record<string, any>>(rows: T[], cols: Col<T>[]) {
  const keep = cols.filter((c) => !["pick", "sel"].includes(c.key) && c.key !== "series");
  return rows.map((r) => Object.fromEntries(keep.map((c) => [plain(c.label, c.key) || c.key, c.csv ? c.csv(r) : c.value ? c.value(r) : r[c.key]])));
}

const NARROW = "@max-[640px]:hidden";

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
  dense, initialSort, className, exportName }: {
  rows: T[]; cols?: Col<T>[]; onSelect?: (row: T, i: number) => void; selected?: (row: T) => boolean;
  maxHeight?: number; rowKey?: (row: T, i: number) => string | number; empty?: ReactNode; dense?: boolean;
  initialSort?: { key: string; dir: 1 | -1 }; className?: string; exportName?: string;
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
  const save = () => {
    const page = window.location.pathname.split("/").filter(Boolean).slice(-1)[0] ?? "table";
    downloadCsv(exportName ?? `libera-${page}-${new Date().toISOString().slice(0, 10)}.csv`, tableRows(sorted.map(([r]) => r), columns));
  };
  return (
    <div className={clsx("group/table @container relative", className)}>
    <button onClick={save} title={`Export these ${rows.length} rows as CSV`} aria-label="Export as CSV"
      className="btn btn-sm btn-icon absolute top-1 right-1 z-[2] !size-6 !rounded-md bg-[var(--surface)] opacity-0 shadow-sm group-hover/table:opacity-100 focus-visible:opacity-100">
      <Download className="size-3.5" />
    </button>
    <div className="scroll-thin overflow-auto rounded-lg border border-[var(--line)]" style={{ maxHeight }}>
      <table className={clsx("table", dense && "[&_td]:py-1.5")}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                style={{ width: c.width, textAlign: c.align }}
                title={c.sortable !== false ? `${c.title ? c.title + ". " : ""}Click to sort; again to reverse; a third time for the original order` : c.title}
                className={clsx(c.sortable !== false && "group/th cursor-pointer hover:text-[var(--text-strong)]", c.narrow && NARROW)}
                aria-sort={sort?.key === c.key ? (sort.dir === -1 ? "descending" : "ascending") : undefined}
                onClick={() =>
                  c.sortable !== false &&
                  setSort((s) => (s?.key === c.key ? (s.dir === -1 ? { key: c.key, dir: 1 } : null) : { key: c.key, dir: -1 }))
                }
              >
                <span className="inline-flex items-center gap-1">
                  {c.label}
                  {sort?.key === c.key ? (sort.dir === -1 ? <ArrowDown className="size-3 text-[var(--chip-text)]" /> : <ArrowUp className="size-3 text-[var(--chip-text)]" />)
                    : c.sortable !== false && <ArrowUpDown className="size-3 opacity-30 group-hover/th:opacity-80" aria-hidden />}
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
                <td key={c.key} style={{ textAlign: c.align }} className={clsx(c.className, c.narrow && NARROW)}>
                  {c.render ? c.render(r, i) : fmtCell(r[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    </div>
  );
}
