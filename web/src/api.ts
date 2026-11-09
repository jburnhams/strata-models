// Typed client for the service's API (src/stratamodels/server/app.py). Same origin; works at the root or behind a proxy at a path.
import { fetchJson } from 'strata-react'

export interface Health { ok: boolean; version: string; gpu: boolean }
export interface GpuProcess { pid: number; used_mb: number; command: string; ours: boolean }
export interface Gpu { name: string; total_mb: number; used_mb: number; free_mb: number; util_pct: number | null; temp_c: number | null; power_w: number | null; power_limit_w: number | null; processes: GpuProcess[] }
export interface Budget { total_mb: number; used_mb: number; own_mb: number; other_mb: number; reserve_mb: number; free_for_models_mb: number }
export interface Resident { model: string; runtime: string; roles: string[]; vram_mb: number; load_s: number; loaded_at: number; last_used: number; jobs: number }
export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'expired'
export interface Job {
  id: number; kind: 'run' | 'lease'; app: string; priority: 'interactive' | 'batch'; status: JobStatus; submitted_at: number
  role: string | null; model: string | null; note: string | null; params: Record<string, unknown> | null; result?: unknown; error: string | null
  vram_gb: number | null; ttl_s: number | null; started_at: number | null; finished_at: number | null; heartbeat_at: number | null; load_s: number | null; run_s: number | null; wait_s?: number
}
export interface ServiceEvent { at: number; text: string }
export interface Status { gpu: Gpu | null; budget: Budget | null; resident: Resident[]; current: Job | null; lease: Job | null; queued: Job[]; events: ServiceEvent[] }
export interface Sample { at: number; util_pct: number | null; used_mb: number; own_mb: number }
export interface History { every_s: number | null; samples: Sample[] }
export interface Role { key: string; description: string; model: string }
export interface Model {
  key: string; runtime: string; name: string; roles: string[]; vram_gb: number; licence: string; source: string; used_by: string[]; options: Record<string, unknown>
  served: boolean; resident: boolean; missing: string[]; stats: { jobs: number; mean_run_s: number | null; last_at: number | null } | null
}
export interface Registry { roles: Role[]; models: Model[] }
export interface JobPage { items: Job[]; next: number | null }
export interface UsageRow { app: string; kind: 'run' | 'lease'; role: string | null; jobs: number; failed: number; gpu_s: number }
export interface Usage { hours: number; rows: UsageRow[] }
export interface Settings { data_dir: string; models_dir: string; vram_reserve_gb: number }
export interface JobFilters { status?: string; app?: string; role?: string; kind?: string }

// Where the app is served from: '' at the server's own root, '/subpath' behind a reverse proxy at that path. From the page's own URL (vite base './'), so one build works in both places.
export const BASE = typeof document === 'undefined' ? '' : new URL('.', document.baseURI).pathname.replace(/\/$/, '')
export const withBase = (u: string) => (u.startsWith('/') ? BASE + u : u)

export function jobsUrl(f: JobFilters, before?: number, limit = 100) {
  const p = new URLSearchParams()
  for (const k of ['status', 'app', 'role', 'kind'] as const) if (f[k]) p.set(k, f[k]!)
  if (before) p.set('before', String(before))
  p.set('limit', String(limit))
  return `/api/jobs?${p}`
}

const get = <T,>(u: string) => fetchJson<T>(withBase(u))
// Changes go through `send`, which turns the service's refusal (400 with `detail`, e.g. "models_dir must be an absolute path") into the Error's
// message, so the button says why; HttpError alone would only give the status.
async function send<T>(method: string, u: string, body?: unknown): Promise<T> {
  const r = await fetch(withBase(u), { method, headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
  const d = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error((d as { detail?: string }).detail || `${r.status} ${r.statusText}`)
  return d as T
}

export const api = {
  health: () => get<Health>('/api/health'),
  status: () => get<Status>('/api/status'),
  history: () => get<History>('/api/gpu/history'),
  registry: () => get<Registry>('/api/registry'),
  model: (key: string, action: 'load' | 'unload') => send<{ ok: boolean }>('POST', `/api/models/${encodeURIComponent(key)}/${action}`),
  jobs: (f: JobFilters, before?: number) => get<JobPage>(jobsUrl(f, before)),
  job: (id: number) => get<Job>(`/api/jobs/${id}`),
  cancel: (id: number) => send<Job>('DELETE', `/api/jobs/${id}`),
  usage: (hours: number) => get<Usage>(`/api/usage?hours=${hours}`),
  settings: () => get<Settings>('/api/settings'),
  saveSettings: (s: Partial<Settings>) => send<Settings>('PUT', '/api/settings', s),
}
