import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T = any>(path: string, init?: RequestInit & { json?: unknown }): Promise<T> {
  const { json, ...rest } = init ?? {};
  const res = await fetch(`/api${path}`, {
    ...rest,
    headers: json !== undefined ? { "Content-Type": "application/json", ...(rest.headers ?? {}) } : rest.headers,
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const b = await res.json();
      msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return res.json();
}

export function useApi<T = any>(path: string | null, opts?: Partial<UseQueryOptions<T>>) {
  return useQuery<T>({
    queryKey: [path],
    queryFn: () => api<T>(path!),
    enabled: !!path,
    staleTime: 60_000,
    retry: (n, e) => !(e instanceof ApiError && e.status < 500) && n < 1,
    ...opts,
  });
}

export const qs = (o: Record<string, string | number | boolean | undefined | null>) =>
  "?" +
  Object.entries(o)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join("&");

// ------------------------------------------------------------------ types

export type Step = {
  job_id: number;
  idx: number;
  name: string;
  label: string;
  argv: string[];
  status: "pending" | "running" | "done" | "failed" | "interrupted" | "skipped";
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  log: string | null;
  progress: number | null;
  message: string | null;
  attempts: number;
};

export type Job = {
  id: number;
  project: string;
  kind: string;
  title: string;
  lane: "pipeline" | "quick";
  status: "queued" | "running" | "pausing" | "paused" | "cancelling" | "cancelled" | "done" | "failed";
  params: Record<string, any>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  steps: Step[];
  progress: number;
  current: string | null;
  queue_position?: number;
};

export type Project = {
  id: string;
  name: string;
  description?: string;
  seed_dois?: string[];
  seed_titles?: { title: string; year: number | null }[];
  created_at?: string;
  in_database: boolean;
  artefacts: Record<string, number | boolean>;
  spend?: Record<string, number>;
  paper_budget?: number;
  as_of?: number;
  job?: Job | null;
  params?: any;
};

export type PaperRef = { paper_id: string; title: string; year: number | null };
