import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { useParams } from "react-router-dom";

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

// ------------------------------------------------------------------ paper drawer

const DrawerCtx = createContext<{ paper: string | null; open: (pid: string | null) => void }>({ paper: null, open: () => {} });

export function DrawerProvider({ children }: { children: ReactNode }) {
  const [paper, open] = useState<string | null>(null);
  return <DrawerCtx.Provider value={{ paper, open }}>{children}</DrawerCtx.Provider>;
}
export const usePaperDrawer = () => useContext(DrawerCtx);

// ------------------------------------------------------------------ project from the URL

export function useProjectId(): string {
  return useParams().project ?? "";
}

// ------------------------------------------------------------------ theme

export function useTheme(): [string, (t: string) => void] {
  const [theme, setTheme] = useState(() => LS.get("libera.theme") ?? "system");
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    LS.set("libera.theme", theme);
  }, [theme]);
  return [theme, setTheme];
}

export function initials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
}
