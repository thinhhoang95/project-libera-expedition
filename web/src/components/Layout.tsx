import { useIsFetching, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Bell, Boxes, CheckCircle2, Clock, Compass, FileText, FlaskConical, FolderKanban, Grid2x2, Home, Keyboard, Layers3,
  ListChecks, Loader2, Menu, Microscope, Monitor, Moon, PanelLeftClose, PanelLeftOpen, Plus, Search, Shapes, Sun, Tags, Workflow, X, XCircle,
} from "lucide-react";
import { Suspense, useEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { api, qs, useApi, type Job, type Project } from "../api";
import { LS, getRecents, initials, useCategorySheet, usePaperDrawer, useReviewer, useTheme } from "../context";
import { FACET_META, FACET_ORDER, type FacetKey } from "../facets";
import { CategorySheet } from "./CategorySheet";
import { Dot } from "./Facets";
import { PaperDrawer } from "./PaperDrawer";
import { isTopOverlay, useOverlay } from "./overlay";
import { Toaster, copyText, toast } from "./Toasts";
import { Boundary, PageSkeleton } from "./ui";

export const NAV = [
  { to: "start", label: "Start", icon: Home, group: "Explore", key: "s" },
  { to: "explore", label: "Facets", icon: Shapes, group: "Explore", key: "f" },
  { to: "map", label: "Map & grid", icon: Grid2x2, group: "Explore", key: "m" },
  { to: "paper", label: "My paper", icon: FileText, group: "Explore", key: "p" },
  { to: "lab", label: "Research lab", icon: Microscope, group: "Evidence", key: "l" },
  { to: "experiments", label: "Experiments", icon: FlaskConical, group: "Evidence", key: "x" },
  { to: "corpus", label: "Corpus view", icon: Layers3, group: "Review", key: "c" },
  { to: "facets", label: "Facet review", icon: Tags, group: "Review", key: "r" },
  { to: "clusters", label: "Cluster review", icon: Boxes, group: "Review", key: "k" },
];
const GROUPS = ["Explore", "Evidence", "Review"];

function Brand({ collapsed }: { collapsed?: boolean }) {
  return (
    <Link to="/" className={clsx("flex items-center gap-2.5", collapsed ? "justify-center" : "px-2")} title="Libera Expedition">
      <img src="/logo.png" alt="" className="size-10 shrink-0 dark:hidden" />
      <img src="/logo-white.png" alt="" className="hidden size-10 shrink-0 dark:block" />
      <div className={clsx("leading-tight", collapsed && "hidden")}>
        <div className="h-title text-[16.5px]">Libera Expedition</div>
        <div className="muted text-[11.5px]">Research intelligence</div>
      </div>
    </Link>
  );
}

function NavItem({ to, icon: Icon, label, end, badge, collapsed }: {
  to: string; icon: any; label: string; end?: boolean; badge?: ReactNode; collapsed?: boolean;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      title={collapsed ? label : undefined}
      className={({ isActive }) =>
        clsx(
          "relative flex items-center gap-3 rounded-[10px] py-2 text-[14px] font-medium transition-colors",
          collapsed ? "justify-center px-0" : "px-3",
          isActive ? "bg-[var(--nav-active)] text-brand-600 dark:text-brand-300" : "text-[var(--text)] hover:bg-[var(--surface-2)]",
        )
      }
    >
      <Icon className="size-[18px] shrink-0 opacity-90" strokeWidth={1.8} />
      {!collapsed && <span className="truncate">{label}</span>}
      {badge && !collapsed && <span className="ml-auto">{badge}</span>}
      {badge && collapsed && <span className="absolute top-1.5 right-2 size-2 rounded-full bg-brand-500 ring-2 ring-[var(--surface)]" />}
    </NavLink>
  );
}

function Sidebar({ onNavigate, collapsed, onToggle }: { onNavigate?: () => void; collapsed?: boolean; onToggle?: () => void }) {
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
    <nav className={clsx("flex h-full flex-col gap-1 pt-5 pb-4", collapsed ? "px-3" : "px-4")}
      onClick={(e) => (e.target as HTMLElement).closest("a") && onNavigate?.()}>
      <div className={clsx("flex items-center", collapsed ? "flex-col gap-3" : "justify-between")}>
        <Brand collapsed={collapsed} />
        {onToggle && (
          <button className="btn btn-ghost btn-icon btn-sm faint shrink-0" onClick={onToggle}
            title={collapsed ? "Expand the sidebar (⌘B)" : "Collapse the sidebar (⌘B)"} aria-label={collapsed ? "Expand the sidebar" : "Collapse the sidebar"}>
            {collapsed ? <PanelLeftOpen className="size-4" /> : <PanelLeftClose className="size-4" />}
          </button>
        )}
      </div>
      <div className="mt-5 flex flex-col gap-0.5">
        {project ? (
          <>
            {GROUPS.map((g, i) => (
              <div key={g} className={clsx("flex flex-col gap-0.5", i > 0 && "mt-2")}>
                {collapsed ? (i > 0 && <div className="mx-2 mb-1 border-t border-[var(--line)]" />) : (
                  <div className="faint px-3 pb-0.5 text-[11px] font-semibold tracking-wide uppercase">{g}</div>
                )}
                {NAV.filter((n) => n.group === g).map((n) => (
                  <NavItem key={n.to} to={`/p/${project}/${n.to}`} icon={n.icon} label={n.label} collapsed={collapsed} />
                ))}
              </div>
            ))}
            <div className="my-2 border-t border-[var(--line)]" />
            <NavItem
              to={`/p/${project}/pipeline`}
              icon={Workflow}
              label="Pipeline"
              collapsed={collapsed}
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
            <NavItem to="/" end icon={FolderKanban} label="Projects" collapsed={collapsed} />
            <NavItem to="/new" icon={Plus} label="New project" collapsed={collapsed} />
          </>
        )}
        <NavItem to="/queue" icon={ListChecks} label="Job queue" collapsed={collapsed} />
      </div>

      <div className={clsx("mt-5 flex items-center justify-between px-3", collapsed && "hidden")}>
        <span className="muted text-[12.5px] font-medium">Projects</span>
        <Link to="/new" className="faint hover:text-brand-600" title="New project"><Plus className="size-4" /></Link>
      </div>
      <div className="scroll-thin mt-1 flex min-h-0 flex-col gap-0.5 overflow-y-auto">
        {(projects ?? []).map((p) => (
          <Link
            key={p.id}
            to={`/p/${p.id}/${page}`}
            title={collapsed ? p.name : undefined}
            className={clsx(
              "flex items-center gap-2.5 rounded-[9px] py-1.5 text-[13px]",
              collapsed ? "justify-center px-0" : "px-3",
              p.id === project ? "bg-[var(--nav-active)] font-medium text-brand-600 dark:text-brand-300" : "hover:bg-[var(--surface-2)]",
            )}
          >
            <Compass className="size-3.5 shrink-0 opacity-70" />
            {!collapsed && <span className="truncate">{p.name}</span>}
            {!collapsed && p.job && ["running", "queued"].includes(p.job.status) && (
              <Loader2 className={clsx("ml-auto size-3 shrink-0", p.job.status === "running" ? "animate-spin text-brand-500" : "faint")} />
            )}
          </Link>
        ))}
        {!collapsed && <Link to="/" className="faint px-3 py-1.5 text-[12.5px] hover:text-brand-600">All projects →</Link>}
      </div>

      <div className="mt-auto pt-3">
        <div className={clsx("flex items-center rounded-xl py-2", collapsed ? "flex-col gap-2 px-0" : "gap-3 px-2")}>
          <button
            className="grid size-10 shrink-0 place-items-center rounded-full bg-[var(--chip)] text-[14px] font-semibold text-[var(--chip-text)]"
            title={collapsed ? `${reviewer || "Set your name"}: expand the sidebar to edit` : "Edit your name"}
            onClick={() => (collapsed ? onToggle?.() : setEditing(true))}
          >
            {initials(reviewer || "?")}
          </button>
          <div className={clsx("min-w-0 flex-1", collapsed && "hidden")}>
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

// ------------------------------------------------------------------ loading bar

/** A thin bar across the top while something new loads (not while polls refresh what is already shown). */
function TopProgress() {
  const n = useIsFetching({ predicate: (q) => q.state.data === undefined });
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (!n) return setShow(false);
    const t = setTimeout(() => setShow(true), 220); // quick answers never flash the bar
    return () => clearTimeout(t);
  }, [n > 0]); // eslint-disable-line react-hooks/exhaustive-deps
  return show ? <div className="top-progress" role="progressbar" aria-label="Loading" /> : null;
}

// ------------------------------------------------------------------ command palette (⌘K)

type Item = { key: string; label: ReactNode; hint: ReactNode; go: () => void; group: string };

/** The term in `text`, emphasised. */
function Hl({ text, term }: { text: string; term: string }) {
  const i = term ? text.toLowerCase().indexOf(term.toLowerCase()) : -1;
  if (i < 0) return <>{text}</>;
  return <>{text.slice(0, i)}<mark className="rounded-sm bg-brand-500/15 px-px text-inherit">{text.slice(i, i + term.length)}</mark>{text.slice(i + term.length)}</>;
}

function useDebounced<T>(v: T, ms: number) {
  const [d, setD] = useState(v);
  useEffect(() => {
    const h = setTimeout(() => setD(v), ms);
    return () => clearTimeout(h);
  }, [v, ms]);
  return d;
}

function SearchPalette({ open, onClose, onToggleSidebar, onHelp }: { open: boolean; onClose: () => void; onToggleSidebar: () => void; onHelp: () => void }) {
  const { project } = useParams();
  const nav = useNavigate();
  const sheet = useCategorySheet();
  const drawer = usePaperDrawer();
  const [theme, setTheme] = useTheme();
  const [term, setTerm] = useState("");
  const [sel, setSel] = useState(0);
  const q = useDebounced(term.trim(), 160); // one request per pause in typing, not per key
  const { data: hits, isFetching: f1 } = useApi<{ paper_id: string; title: string; year: number }[]>(
    project && q.length >= 2 ? `/p/${project}/search` + qs({ term: q, limit: 10 }) : null, { keep: true },
  );
  const { data: cats, isFetching: f2 } = useApi<{ facet: FacetKey; cluster_id: string; label: string; size: number; color: string }[]>(
    project && q.length >= 2 ? `/p/${project}/categories/search` + qs({ term: q, limit: 8 }) : null, { keep: true },
  );
  const { data: projects } = useApi<Project[]>("/projects");
  const ref = useOverlay<HTMLDivElement>(open, { autoFocus: false });
  useEffect(() => {
    if (open) {
      setTerm("");
      setSel(0);
    }
  }, [open]);
  if (!open) return null;
  const t = term.trim().toLowerCase();
  const live = t.length >= 2;
  const recents = project && !t ? getRecents(project) : [];
  const actions: { label: string; hint: string; go: () => void; when?: boolean }[] = [
    { label: `Theme: ${theme === "dark" ? "light" : "dark"}`, hint: "Appearance", go: () => setTheme(theme === "dark" ? "light" : "dark") },
    { label: "Theme: follow the system", hint: "Appearance", go: () => setTheme("system"), when: theme !== "system" },
    { label: "Collapse or expand the sidebar", hint: "⌘B", go: onToggleSidebar },
    { label: "Copy a link to this view", hint: "Link", go: () => copyText(window.location.href) },
    { label: "Keyboard shortcuts", hint: "?", go: onHelp },
    { label: "Open the pipeline", hint: "Run steps", go: () => nav(`/p/${project}/pipeline`), when: !!project },
    { label: "Job queue", hint: "All projects", go: () => nav("/queue") },
    { label: "New project", hint: "From seed DOIs", go: () => nav("/new") },
  ];
  const actionItems: Item[] = actions.filter((a) => a.when !== false && (!t || a.label.toLowerCase().includes(t)))
    .map((a) => ({ key: "a" + a.label, group: "Actions", label: <Hl text={a.label} term={t} />, hint: a.hint, go: a.go }));
  const items: Item[] = [
    ...recents.map((r) => ({
      key: "r" + r.kind + r.id, group: "Recent",
      label: <span className="flex items-center gap-2"><Clock className="faint size-3.5 shrink-0" /><span className="truncate">{r.label}</span></span>,
      hint: r.kind === "cat" ? "Category" : "Paper",
      go: () => (r.kind === "cat" ? sheet.open(r.id) : drawer.open(r.id)),
    })),
    ...(project
      ? NAV.filter((n) => n.label.toLowerCase().includes(t)).map((n) => ({
          key: "nav" + n.to, group: "Pages", label: <Hl text={n.label} term={t} />, hint: <span>Page <kbd className="key ml-1">g</kbd> <kbd className="key">{n.key}</kbd></span>,
          go: () => nav(`/p/${project}/${n.to}`),
        }))
      : []),
    ...(project && live
      ? FACET_ORDER.filter((f) => FACET_META[f].plural.toLowerCase().includes(t)).map((f) => ({
          key: "fx" + f, group: "Pages", label: `Explore ${FACET_META[f].plural.toLowerCase()}`, hint: "Facet", go: () => nav(`/p/${project}/explore/${f}`),
        }))
      : []),
    ...(t ? actionItems : []),
    ...(live ? cats ?? [] : []).map((c) => ({
      key: c.cluster_id, group: "Categories",
      label: <span className="flex items-center gap-2"><Dot color={c.color} /><span className="truncate"><Hl text={c.label} term={t} /></span></span>,
      hint: <span><span style={{ color: FACET_META[c.facet]?.color }} className="font-semibold">{FACET_META[c.facet]?.label}</span> · {c.size} papers</span>,
      go: () => sheet.open(c.cluster_id),
    })),
    ...(live ? hits ?? [] : []).map((h) => ({
      key: h.paper_id, group: "Papers",
      label: (<span><span className="faint mr-2">{h.year}</span><Hl text={h.title} term={t} /></span>),
      hint: "Open paper",
      go: () => drawer.open(h.paper_id),
    })),
    ...(projects ?? [])
      .filter((p) => (t ? p.name.toLowerCase().includes(t) || p.id.includes(t) : p.id !== project))
      .slice(0, 4)
      .map((p) => ({ key: "p" + p.id, group: "Projects", label: <Hl text={p.name} term={t} />, hint: "Project", go: () => nav(`/p/${p.id}/start`) })),
    ...(t ? [] : actionItems), // typed: matching actions come before categories and papers
  ];
  const go = (i: number) => {
    onClose();
    items[i]?.go();
  };
  const loading = live && (f1 || f2 || q !== term.trim());
  return (
    <div className="anim-fade fixed inset-0 z-50 flex items-start justify-center bg-slate-900/30 px-4 pt-[12vh] backdrop-blur-[2px]" onClick={onClose}>
      <div ref={ref} role="dialog" aria-modal="true" aria-label="Search" className="anim-pop card w-full max-w-2xl overflow-hidden shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b border-[var(--line)] px-4">
          {loading ? <Loader2 className="faint size-4 animate-spin" /> : <Search className="faint size-4" />}
          <input
            autoFocus
            value={term}
            onChange={(e) => {
              setTerm(e.target.value);
              setSel(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") onClose();
              if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, items.length - 1)); }
              if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
              if (e.key === "Enter") go(sel);
            }}
            placeholder={project ? "Search papers, categories, pages, actions…" : "Search projects and actions…"}
            aria-label="Search"
            className="h-12 flex-1 bg-transparent text-[14.5px] text-[var(--text-strong)] outline-none"
          />
          <kbd className="faint text-[11px]">esc</kbd>
        </div>
        <div className="scroll-thin max-h-[55vh] overflow-y-auto p-1.5" role="listbox">
          {items.length === 0 && <div className="muted px-3 py-6 text-center text-[13px]">{loading ? "Searching…" : t.length < 2 ? "Type to search" : `Nothing matches “${term.trim()}”`}</div>}
          {items.map((it, i) => (
            <div key={it.key}>
              {(i === 0 || items[i - 1].group !== it.group) && <div className="faint px-3 pt-2 pb-1 text-[11px] font-semibold tracking-wide uppercase">{it.group}</div>}
              <button
                ref={(el) => { if (el && i === sel) el.scrollIntoView({ block: "nearest" }); }}
                role="option"
                aria-selected={i === sel}
                onMouseMove={() => sel !== i && setSel(i)}
                onClick={() => go(i)}
                className={clsx("flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[13.5px]", i === sel && "bg-[var(--nav-active)]")}
              >
                <span className="min-w-0 flex-1 truncate">{it.label}</span>
                <span className="faint shrink-0 text-[11.5px]">{it.hint}</span>
              </button>
            </div>
          ))}
        </div>
        <div className="faint flex items-center gap-3 border-t border-[var(--line)] px-4 py-2 text-[11.5px]">
          <span><kbd className="key">↑</kbd> <kbd className="key">↓</kbd> move</span><span><kbd className="key">↵</kbd> open</span>
          <span className="ml-auto"><kbd className="key">?</kbd> all shortcuts</span>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ keyboard shortcuts

function Shortcuts({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useOverlay<HTMLDivElement>(open);
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && isTopOverlay(ref.current) && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!open) return null;
  const rows: [ReactNode, string][] = [
    [<><kbd className="key">⌘</kbd> <kbd className="key">K</kbd> or <kbd className="key">/</kbd></>, "Search papers, categories, pages"],
    [<><kbd className="key">⌘</kbd> <kbd className="key">B</kbd></>, "Collapse or expand the sidebar"],
    [<kbd className="key">Esc</kbd>, "Close the paper, then the category sheet"],
    [<kbd className="key">?</kbd>, "This list"],
    ...NAV.map((n) => [<><kbd className="key">g</kbd> <kbd className="key">{n.key}</kbd></>, `Go to ${n.label}`] as [ReactNode, string]),
  ];
  return (
    <div className="anim-fade fixed inset-0 z-50 flex items-start justify-center bg-slate-900/30 px-4 pt-[12vh]" onClick={onClose}>
      <div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" className="anim-pop card w-full max-w-md p-5 shadow-2xl outline-none" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="h-title flex items-center gap-2 text-[15px]"><Keyboard className="size-4" /> Keyboard shortcuts</h2>
          <button className="btn btn-ghost btn-icon btn-sm" onClick={onClose} aria-label="Close"><X className="size-4" /></button>
        </div>
        <ul className="flex flex-col">
          {rows.map(([k, l], i) => (
            <li key={i} className="flex items-center justify-between border-b border-[var(--line)] py-1.5 text-[13px] last:border-0"><span>{l}</span><span className="shrink-0">{k}</span></li>
          ))}
        </ul>
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
  // events that arrive while the app is open pop up as toasts; a finished step also refreshes that project's views
  const lastId = useRef<number | null>(null);
  useEffect(() => {
    const evs = data?.events ?? [];
    if (!data) return;
    const top = Math.max(0, ...evs.map((e) => e.id));
    if (lastId.current == null) {
      lastId.current = top;
      return;
    }
    const fresh = evs.filter((e) => e.id > lastId.current!).sort((a, b) => a.id - b.id);
    lastId.current = Math.max(lastId.current, top);
    fresh.forEach((e) => {
      if (e.level === "success" && e.project)
        qc.invalidateQueries({ predicate: (q) => String(q.queryKey[0]).startsWith(`/p/${e.project}/`) || String(q.queryKey[0]).startsWith("/projects") });
      if (e.level === "success" || e.level === "error")
        toast({ title: e.message, body: e.project ?? undefined, tone: e.level, href: e.project ? `/p/${e.project}/pipeline` : "/queue", linkLabel: "Open the pipeline" });
    });
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps
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

function Topbar({ onSearch, onMenu, onHelp }: { onSearch: () => void; onMenu: () => void; onHelp: () => void }) {
  const { reviewer } = useReviewer();
  const { project } = useParams();
  const { data: p } = useApi<Project>(project ? `/projects/${project}` : null);
  return (
    <header className="sticky top-0 z-30 flex h-[60px] items-center gap-3 border-b border-[var(--line)] bg-[var(--surface)]/85 px-4 backdrop-blur md:px-6">
      <button className="btn btn-ghost btn-icon lg:hidden" onClick={onMenu} aria-label="Open the menu"><Menu className="size-5" /></button>
      <button
        onClick={onSearch}
        className="flex h-10 min-w-0 max-w-xl flex-1 items-center gap-2.5 rounded-[10px] border border-[var(--line)] bg-[var(--surface-2)] px-3 text-left text-[13.5px] text-[var(--faint)] hover:border-[var(--line-strong)]"
      >
        <Search className="size-4" />
        <span className="flex-1 truncate">{project ? `Search papers and categories in ${p?.name ?? project}…` : "Search projects…"}</span>
        <kbd className="hidden rounded border border-[var(--line)] px-1.5 text-[11px] sm:block">⌘ K</kbd>
      </button>
      <div className="ml-auto flex items-center gap-1.5">
        <button className="btn btn-ghost btn-icon hidden sm:inline-flex" title="Keyboard shortcuts (?)" onClick={onHelp}><Keyboard className="size-[18px]" strokeWidth={1.8} /></button>
        <Notifications />
        <div className="mx-1 hidden h-6 border-l border-[var(--line)] sm:block" />
        <span className="grid size-9 place-items-center rounded-full bg-[var(--chip)] text-[13px] font-semibold text-[var(--chip-text)]" title={reviewer || "Set your name in the sidebar"}>
          {initials(reviewer || "?")}
        </span>
      </div>
    </header>
  );
}

const OTHER_TITLES: Record<string, string> = { pipeline: "Pipeline", new: "New project", queue: "Job queue" };

/** The browser tab says where you are: "Facets · dro-pilot", so tabs and history entries can be told apart. */
function useDocumentTitle() {
  const loc = useLocation();
  const { project } = useParams();
  const { data: p } = useApi<Project>(project ? `/projects/${project}` : null);
  useEffect(() => {
    const seg = project ? loc.pathname.split("/")[3] ?? "start" : loc.pathname.split("/")[1] ?? "";
    const page = NAV.find((n) => n.to === seg)?.label ?? OTHER_TITLES[seg] ?? (project ? "" : "Projects");
    document.title = [page, project ? p?.name ?? project : null, "Libera Expedition"].filter(Boolean).join(" · ");
  }, [loc.pathname, project, p?.name]);
}

export function Layout() {
  const loc = useLocation();
  const nav = useNavigate();
  const { project } = useParams();
  useDocumentTitle();
  const [search, setSearch] = useState(false);
  const [help, setHelp] = useState(false);
  const chord = useRef<number>(0);
  const [menu, setMenu] = useState(false);
  const [collapsed, setCollapsed] = useState(() => LS.get("libera.sidebar") === "collapsed");
  const toggle = () =>
    setCollapsed((c) => {
      LS.set("libera.sidebar", c ? "expanded" : "collapsed");
      return !c;
    });
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearch((s) => !s);
      }
      const typing = !!(e.target as HTMLElement)?.closest?.("input, textarea, select, [contenteditable]");
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "b" && !typing) {
        e.preventDefault();
        toggle();
      }
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "/") { e.preventDefault(); setSearch(true); }
      else if (e.key === "?") setHelp((x) => !x);
      else if (e.key === "g") chord.current = Date.now();
      else if (Date.now() - chord.current < 1200) {
        const n = NAV.find((x) => x.key === e.key);
        chord.current = 0;
        if (n && project) nav(`/p/${project}/${n.to}`);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [project]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="flex min-h-full">
      <TopProgress />
      <aside className={clsx("hidden shrink-0 border-r border-[var(--line)] bg-[var(--surface)] whitespace-nowrap lg:block",
        collapsed ? "w-[72px]" : "w-[272px]")}>
        <div className="sticky top-0 h-screen overflow-hidden"><Sidebar collapsed={collapsed} onToggle={toggle} /></div>
      </aside>
      {menu && (
        <div className="anim-fade fixed inset-0 z-50 bg-slate-900/30 lg:hidden" onClick={() => setMenu(false)}>
          <aside className="anim-slide-left h-full w-[280px] bg-[var(--surface)] shadow-xl" onClick={(e) => e.stopPropagation()}>
            <button className="btn btn-ghost btn-icon absolute top-3 left-[236px]" onClick={() => setMenu(false)}><X className="size-5" /></button>
            <Sidebar onNavigate={() => setMenu(false)} />
          </aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar onSearch={() => setSearch(true)} onMenu={() => setMenu(true)} onHelp={() => setHelp(true)} />
        <main className="min-w-0 flex-1 px-4 py-5 md:px-7 md:py-6">
          <Boundary resetKey={loc.pathname}>
            <Suspense fallback={<PageSkeleton />}>
              <Outlet />
            </Suspense>
          </Boundary>
        </main>
      </div>
      <SearchPalette open={search} onClose={() => setSearch(false)} onToggleSidebar={toggle} onHelp={() => setHelp(true)} />
      <Shortcuts open={help} onClose={() => setHelp(false)} />
      <CategorySheet />
      <PaperDrawer />
      <Toaster />
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

