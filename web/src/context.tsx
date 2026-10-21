import { createContext, useCallback, useContext, useEffect, useState, useSyncExternalStore, type ReactNode } from "react";
import { useParams, useSearchParams } from "react-router-dom";

const LS = {
  get(k: string) {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set(k: string, v: string) {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* private mode */
    }
  },
};
export { LS };

// ------------------------------------------------------------------ reviewer (stored with every judgment)

const ReviewerCtx = createContext<{ reviewer: string; setReviewer: (s: string) => void }>({ reviewer: "", setReviewer: () => {} });

export function ReviewerProvider({ children }: { children: ReactNode }) {
  const [reviewer, set] = useState(() => LS.get("libera.reviewer") ?? "");
  const setReviewer = useCallback((s: string) => {
    set(s);
    LS.set("libera.reviewer", s);
  }, []);
  return <ReviewerCtx.Provider value={{ reviewer, setReviewer }}>{children}</ReviewerCtx.Provider>;
}
export const useReviewer = () => useContext(ReviewerCtx);

// ------------------------------------------------------------------ paper drawer and category sheet (in the URL)

/** One search parameter as open/close state: opening pushes a history entry (Back closes it), closing replaces. */
function useParamState(key: string, clear: string[] = []) {
  const [sp, setSp] = useSearchParams();
  const value = sp.get(key);
  const open = useCallback(
    (v: string | null, extra?: Record<string, string | null>) =>
      setSp(
        (prev) => {
          const n = new URLSearchParams(prev);
          if (v) n.set(key, v);
          else n.delete(key);
          if (v) clear.forEach((k) => n.delete(k));
          Object.entries(extra ?? {}).forEach(([k, x]) => (x ? n.set(k, x) : n.delete(k)));
          return n;
        },
        { replace: !v },
      ),
    [setSp, key], // eslint-disable-line react-hooks/exhaustive-deps
  );
  return [value, open] as const;
}

export function usePaperDrawer() {
  const [paper, open] = useParamState("paper");
  return { paper, open: (pid: string | null) => open(pid) };
}

/** The category details sheet: `?cat=<cluster id>`, optionally `&with=<cluster id>` to filter its papers. */
export function useCategorySheet() {
  const [cat, open] = useParamState("cat", ["paper", "with"]);
  const [sp] = useSearchParams();
  return { cat, with: sp.get("with"), open: (cid: string | null, withCid?: string | null) => open(cid, { with: withCid ?? null }) };
}

// ------------------------------------------------------------------ recently opened papers and categories

export type Recent = { kind: "paper" | "cat"; id: string; label: string; at: number };

export function getRecents(project: string): Recent[] {
  try {
    return JSON.parse(LS.get(`libera.recent.${project}`) ?? "[]");
  } catch {
    return [];
  }
}

export function useRecordRecent(project: string | undefined, r: Omit<Recent, "at"> | null) {
  const key = r ? `${r.kind}:${r.id}:${r.label}` : "";
  useEffect(() => {
    if (!project || !r) return;
    const rest = getRecents(project).filter((x) => !(x.kind === r.kind && x.id === r.id));
    LS.set(`libera.recent.${project}`, JSON.stringify([{ ...r, at: Date.now() }, ...rest].slice(0, 12)));
  }, [project, key]); // eslint-disable-line react-hooks/exhaustive-deps
}

// ------------------------------------------------------------------ the time window, one per project

export type YearWindow = [number, number];
const winSubs = new Set<() => void>();
const winKey = (project: string) => `libera.window.${project}`;
const winCache = new Map<string, YearWindow | null>();

function readWindow(project: string): YearWindow | null {
  if (!winCache.has(project)) {
    const m = /^(\d{4})-(\d{4})$/.exec(LS.get(winKey(project)) ?? "");
    winCache.set(project, m ? [Number(m[1]), Number(m[2])] : null);
  }
  return winCache.get(project)!;
}

export function setSharedWindow(project: string, w: YearWindow | null) {
  const cur = readWindow(project);
  if (cur && w && cur[0] === w[0] && cur[1] === w[1]) return;
  winCache.set(project, w);
  LS.set(winKey(project), w ? `${w[0]}-${w[1]}` : "");
  winSubs.forEach((f) => f());
}

/**
 * The year window the user last chose in this project, shared by Facets, the map and every details sheet, and kept
 * across reloads. `null` until they choose one: each view then shows its own default.
 */
export function useSharedWindow(project: string): [YearWindow | null, (w: YearWindow | null) => void] {
  const w = useSyncExternalStore(
    (f) => { winSubs.add(f); return () => winSubs.delete(f); },
    () => readWindow(project),
  );
  const set = useCallback((x: YearWindow | null) => setSharedWindow(project, x), [project]);
  return [w, set];
}

// ------------------------------------------------------------------ project from the URL

export function useProjectId(): string {
  return useParams().project ?? "";
}

// ------------------------------------------------------------------ theme

let theme = LS.get("libera.theme") ?? "system";
const themeSubs = new Set<() => void>();
const applyTheme = () => {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
};
applyTheme();

/** light, dark or system; one value for the whole app (the sidebar switch and the command palette share it). */
export function useTheme(): [string, (t: string) => void] {
  const t = useSyncExternalStore((f) => { themeSubs.add(f); return () => themeSubs.delete(f); }, () => theme);
  const set = useCallback((x: string) => {
    theme = x;
    LS.set("libera.theme", x);
    applyTheme();
    themeSubs.forEach((f) => f());
  }, []);
  return [t, set];
}

export function initials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
}
