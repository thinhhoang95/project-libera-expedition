import { useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Bell, Boxes, CheckCircle2, Compass, FileText, FlaskConical, FolderKanban, Grid2x2, Home, Layers3,
  ListChecks, Loader2, Menu, Microscope, Monitor, Moon, Plus, Search, Sun, Tags, Workflow, X, XCircle,
} from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { api, qs, useApi, type Job, type Project } from "../api";
import { initials, useReviewer, useTheme } from "../context";
import { PaperDrawer } from "./PaperDrawer";
import { Boundary } from "./ui";

export const NAV = [
  { to: "start", label: "Start", icon: Home },
  { to: "paper", label: "My paper", icon: FileText },
  { to: "map", label: "Map & grid", icon: Grid2x2 },
  { to: "lab", label: "Research lab", icon: Microscope },
  { to: "experiments", label: "Experiments", icon: FlaskConical },
  { to: "corpus", label: "Corpus view", icon: Layers3 },
  { to: "facets", label: "Facet review", icon: Tags },
  { to: "clusters", label: "Cluster review", icon: Boxes },
];

function Brand() {
  return (
    <Link to="/" className="flex items-center gap-2.5 px-2">
      <img src="/logo.png" alt="" className="size-10 dark:hidden" />
      <img src="/logo-white.png" alt="" className="hidden size-10 dark:block" />
      <div className="leading-tight">
        <div className="h-title text-[16.5px]">Libera Expedition</div>
        <div className="muted text-[11.5px]">Research landscape intelligence</div>
      </div>
    </Link>
  );
}

function NavItem({ to, icon: Icon, label, end, badge }: { to: string; icon: any; label: string; end?: boolean; badge?: ReactNode }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        clsx(
          "flex items-center gap-3 rounded-[10px] px-3 py-2 text-[14px] font-medium transition-colors",
          isActive ? "bg-[var(--nav-active)] text-brand-600 dark:text-brand-300" : "text-[var(--text)] hover:bg-[var(--surface-2)]",
        )
      }
    >
      <Icon className="size-[18px] shrink-0 opacity-90" strokeWidth={1.8} />
      <span className="truncate">{label}</span>
      {badge && <span className="ml-auto">{badge}</span>}
    </NavLink>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { project } = useParams();
  const { data: projects } = useApi<Project[]>("/projects", { refetchInterval: 10_000 });
  const { data: jobs } = useApi<Job[]>("/jobs" + qs({ project, limit: 5 }), { enabled: !!project, refetchInterval: 4000 });
  const running = jobs?.find((j) => ["running", "pausing", "cancelling"].includes(j.status));
  const queued = jobs?.filter((j) => j.status === "queued").length ?? 0;
  const { reviewer, setReviewer } = useReviewer();
  const [theme, setTheme] = useTheme();
  const [editing, setEditing] = useState(false);
  const loc = useLocation();
  const page = project ? loc.pathname.split("/")[3] ?? "start" : "start";
  return (
    <nav className="flex h-full flex-col gap-1 px-4 pt-5 pb-4" onClick={(e) => (e.target as HTMLElement).closest("a") && onNavigate?.()}>
      <Brand />
      <div className="mt-6 flex flex-col gap-0.5">
        {project ? (
          <>
            {NAV.map((n) => (
              <NavItem key={n.to} to={`/p/${project}/${n.to}`} icon={n.icon} label={n.label} />
            ))}
            <div className="my-2 border-t border-[var(--line)]" />
            <NavItem
              to={`/p/${project}/pipeline`}
              icon={Workflow}
              label="Pipeline"
              badge={
                running ? (
                  <span className="flex items-center gap-1 text-[11.5px] text-brand-600"><Loader2 className="size-3 animate-spin" />{Math.round(running.progress * 100)}%</span>
                ) : queued ? (
                  <span className="rounded-full bg-[var(--chip)] px-1.5 text-[11px] font-semibold text-[var(--chip-text)]">{queued}</span>
                ) : null
              }
            />
          </>
        ) : (
          <>
            <NavItem to="/" end icon={FolderKanban} label="Projects" />
            <NavItem to="/new" icon={Plus} label="New project" />
          </>
        )}
        <NavItem to="/queue" icon={ListChecks} label="Job queue" />
      </div>

      <div className="mt-5 flex items-center justify-between px-3">
        <span className="muted text-[12.5px] font-medium">Projects</span>
        <Link to="/new" className="faint hover:text-brand-600" title="New project"><Plus className="size-4" /></Link>
      </div>
      <div className="scroll-thin mt-1 flex min-h-0 flex-col gap-0.5 overflow-y-auto">
        {(projects ?? []).map((p) => (
          <Link
            key={p.id}
            to={`/p/${p.id}/${page}`}
            className={clsx(
              "flex items-center gap-2.5 rounded-[9px] px-3 py-1.5 text-[13px]",
              p.id === project ? "bg-[var(--nav-active)] font-medium text-brand-600 dark:text-brand-300" : "hover:bg-[var(--surface-2)]",
            )}
          >
            <Compass className="size-3.5 shrink-0 opacity-70" />
            <span className="truncate">{p.name}</span>
            {p.job && ["running", "queued"].includes(p.job.status) && (
              <Loader2 className={clsx("ml-auto size-3 shrink-0", p.job.status === "running" ? "animate-spin text-brand-500" : "faint")} />
            )}
          </Link>
        ))}
        <Link to="/" className="faint px-3 py-1.5 text-[12.5px] hover:text-brand-600">All projects →</Link>
      </div>

      <div className="mt-auto pt-3">
        <div className="flex items-center gap-3 rounded-xl px-2 py-2">
          <span className="grid size-10 shrink-0 place-items-center rounded-full bg-[var(--chip)] text-[14px] font-semibold text-[var(--chip-text)]">
            {initials(reviewer || "?")}
          </span>
          <div className="min-w-0 flex-1">
            {editing ? (
              <input
                autoFocus
                className="input h-7 text-[13px]"
                defaultValue={reviewer}
                placeholder="Your name"
                onBlur={(e) => {
                  setReviewer(e.target.value.trim());
                  setEditing(false);
                }}
                onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
              />
            ) : (
              <button onClick={() => setEditing(true)} className="block w-full text-left">
                <div className="h-title truncate text-[13.5px]">{reviewer || "Set your name"}</div>
                <div className="muted truncate text-[11.5px]">Reviewer · stored with judgments</div>
              </button>
            )}
          </div>
          <button
            className="btn btn-ghost btn-icon btn-sm"
            title={`Theme: ${theme}`}
            onClick={() => setTheme(theme === "system" ? "light" : theme === "light" ? "dark" : "system")}
          >
            {theme === "dark" ? <Moon className="size-4" /> : theme === "light" ? <Sun className="size-4" /> : <Monitor className="size-4" />}
          </button>
        </div>
      </div>
    </nav>
  );
}

// ------------------------------------------------------------------ command palette (⌘K)

function SearchPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { project } = useParams();
  const nav = useNavigate();
  const [term, setTerm] = useState("");
  const [sel, setSel] = useState(0);
  const { data: hits } = useApi<{ paper_id: string; title: string; year: number }[]>(
    project && term.length >= 2 ? `/p/${project}/search` + qs({ term, limit: 12 }) : null,
  );
  const { data: projects } = useApi<Project[]>("/projects");
  useEffect(() => {
    if (open) {
      setTerm("");
      setSel(0);
    }
  }, [open]);
  if (!open) return null;
  const t = term.toLowerCase();
  const items: { key: string; label: ReactNode; hint: string; go: () => void }[] = [
    ...(project
      ? NAV.filter((n) => n.label.toLowerCase().includes(t)).map((n) => ({
          key: "nav" + n.to, label: n.label, hint: "Page", go: () => nav(`/p/${project}/${n.to}`),
        }))
      : []),
    ...(projects ?? [])
      .filter((p) => !t || p.name.toLowerCase().includes(t) || p.id.includes(t))
      .slice(0, 4)
      .map((p) => ({ key: "p" + p.id, label: p.name, hint: "Project", go: () => nav(`/p/${p.id}/start`) })),
    ...(hits ?? []).map((h) => ({
      key: h.paper_id,
      label: (<span><span className="faint mr-2">{h.year}</span>{h.title}</span>),
      hint: "Open in My paper",
      go: () => nav(`/p/${project}/paper/${h.paper_id}`),
    })),
  ];
  const go = (i: number) => {
    items[i]?.go();
    onClose();
  };
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-slate-900/30 px-4 pt-[12vh] backdrop-blur-[2px]" onClick={onClose}>
      <div className="card w-full max-w-2xl overflow-hidden shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b border-[var(--line)] px-4">
          <Search className="faint size-4" />
          <input
            autoFocus
            value={term}
            onChange={(e) => {
              setTerm(e.target.value);
              setSel(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") onClose();
              if (e.key === "ArrowDown") setSel((s) => Math.min(s + 1, items.length - 1));
              if (e.key === "ArrowUp") setSel((s) => Math.max(s - 1, 0));
              if (e.key === "Enter") go(sel);
            }}
            placeholder={project ? "Search papers by title or DOI, pages, projects…" : "Search projects…"}
            className="h-12 flex-1 bg-transparent text-[14.5px] text-[var(--text-strong)] outline-none"
          />
          <kbd className="faint text-[11px]">esc</kbd>
        </div>
        <div className="scroll-thin max-h-[55vh] overflow-y-auto p-1.5">
          {items.length === 0 && <div className="muted px-3 py-6 text-center text-[13px]">{term.length < 2 ? "Type to search" : "No matches"}</div>}
          {items.map((it, i) => (
            <button
              key={it.key}
              onMouseEnter={() => setSel(i)}
              onClick={() => go(i)}
              className={clsx("flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[13.5px]", i === sel && "bg-[var(--nav-active)]")}
            >
              <span className="min-w-0 flex-1 truncate">{it.label}</span>
              <span className="faint shrink-0 text-[11.5px]">{it.hint}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ notifications

type Ev = { id: number; ts: string; project: string | null; job_id: number | null; level: string; message: string; seen: number };

function Notifications() {
  const [open, setOpen] = useState(false);
  const qc = useQueryClient();
  const { data } = useQuery({ queryKey: ["events"], queryFn: () => api<{ events: Ev[]; unseen: number }>("/events"), refetchInterval: 5000 });
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const h = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  const icon = (l: string) =>
    l === "success" ? <CheckCircle2 className="size-4 text-emerald-600" /> : l === "error" ? <XCircle className="size-4 text-red-600" /> : <Workflow className="faint size-4" />;
  return (
    <div className="relative" ref={ref}>
      <button
        className="btn btn-ghost btn-icon relative"
        onClick={async () => {
          setOpen(!open);
          if (!open && data?.unseen) {
            await api("/events/seen", { method: "POST" });
            qc.invalidateQueries({ queryKey: ["events"] });
          }
        }}
        title="Pipeline notifications"
      >
        <Bell className="size-[19px]" strokeWidth={1.8} />
        {!!data?.unseen && <span className="absolute top-1.5 right-2 size-2 rounded-full bg-red-500 ring-2 ring-[var(--surface)]" />}
      </button>
      {open && (
        <div className="card absolute right-0 z-40 mt-2 w-96 overflow-hidden shadow-xl">
          <div className="h-title border-b border-[var(--line)] px-4 py-2.5 text-[13.5px]">Pipeline activity</div>
          <div className="scroll-thin max-h-96 overflow-y-auto">
            {(data?.events ?? []).length === 0 && <div className="muted px-4 py-6 text-center text-[13px]">Nothing yet.</div>}
            {(data?.events ?? []).map((e) => (
              <Link
                key={e.id}
                to={e.project ? `/p/${e.project}/pipeline` : "/queue"}
                onClick={() => setOpen(false)}
                className="flex gap-3 border-b border-[var(--line)] px-4 py-2.5 last:border-0 hover:bg-[var(--surface-2)]"
              >
                <span className="mt-0.5">{icon(e.level)}</span>
                <span className="min-w-0 flex-1">
                  <span className="block text-[13px] text-[var(--text-strong)]">{e.message}</span>
                  <span className="faint text-[11.5px]">{e.project} · {e.ts.replace("T", " ").slice(0, 16)}</span>
                </span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Topbar({ onSearch, onMenu }: { onSearch: () => void; onMenu: () => void }) {
  const { reviewer } = useReviewer();
  const { project } = useParams();
  const { data: p } = useApi<Project>(project ? `/projects/${project}` : null);
  return (
    <header className="sticky top-0 z-30 flex h-[60px] items-center gap-3 border-b border-[var(--line)] bg-[var(--surface)]/85 px-4 backdrop-blur md:px-6">
      <button className="btn btn-ghost btn-icon lg:hidden" onClick={onMenu}><Menu className="size-5" /></button>
      <button
        onClick={onSearch}
        className="flex h-10 min-w-0 max-w-xl flex-1 items-center gap-2.5 rounded-[10px] border border-[var(--line)] bg-[var(--surface-2)] px-3 text-left text-[13.5px] text-[var(--faint)] hover:border-[var(--line-strong)]"
      >
        <Search className="size-4" />
        <span className="flex-1 truncate">{project ? `Search papers in ${p?.name ?? project}, pages, projects…` : "Search projects…"}</span>
        <kbd className="hidden rounded border border-[var(--line)] px-1.5 text-[11px] sm:block">⌘ K</kbd>
      </button>
      <div className="ml-auto flex items-center gap-1.5">
        <Notifications />
        <div className="mx-1 hidden h-6 border-l border-[var(--line)] sm:block" />
        <span className="grid size-9 place-items-center rounded-full bg-[var(--chip)] text-[13px] font-semibold text-[var(--chip-text)]" title={reviewer || "Set your name in the sidebar"}>
          {initials(reviewer || "?")}
        </span>
      </div>
    </header>
  );
}

export function Layout() {
  const loc = useLocation();
  const [search, setSearch] = useState(false);
  const [menu, setMenu] = useState(false);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearch((s) => !s);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);
  return (
    <div className="flex min-h-full">
      <aside className="hidden w-[272px] shrink-0 border-r border-[var(--line)] bg-[var(--surface)] lg:block">
        <div className="sticky top-0 h-screen"><Sidebar /></div>
      </aside>
      {menu && (
        <div className="fixed inset-0 z-50 bg-slate-900/30 lg:hidden" onClick={() => setMenu(false)}>
          <aside className="h-full w-[280px] bg-[var(--surface)] shadow-xl" onClick={(e) => e.stopPropagation()}>
            <button className="btn btn-ghost btn-icon absolute top-3 left-[236px]" onClick={() => setMenu(false)}><X className="size-5" /></button>
            <Sidebar onNavigate={() => setMenu(false)} />
          </aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar onSearch={() => setSearch(true)} onMenu={() => setMenu(true)} />
        <main className="min-w-0 flex-1 px-4 py-5 md:px-7 md:py-6">
          <Boundary resetKey={loc.pathname}>
            <Outlet />
          </Boundary>
        </main>
      </div>
      <SearchPalette open={search} onClose={() => setSearch(false)} />
      <PaperDrawer />
    </div>
  );
}

export function PageHeader({ title, crumbs, children, sub }: { title: ReactNode; crumbs?: ReactNode; children?: ReactNode; sub?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        {crumbs && <div className="muted mb-1 flex items-center gap-1.5 text-[13px]">{crumbs}</div>}
        <h1 className="h-title text-[26px] leading-tight md:text-[28px]">{title}</h1>
        {sub && <div className="muted mt-1 text-[13px]">{sub}</div>}
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

