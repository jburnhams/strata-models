// Typed client for the service's API (src/stratamodels/server/app.py). Same origin; works at the root or behind a proxy at a path.
import { fetchJson } from 'strata-react'

export interface Health { ok: boolean; version: string; gpu: boolean }
export interface GpuProcess { pid: number; used_mb: number; command: string; ours: boolean }
export interface Gpu { name: string; total_mb: number; used_mb: number; free_mb: number; util_pct: number | null; temp_c: number | null; power_w: number | null; power_limit_w: number | null; processes: GpuProcess[] }
export interface Budget { total_mb: number; used_mb: number; own_mb: number; other_mb: number; reserve_mb: number; reserved_mb?: number; pending_mb?: number; free_for_models_mb: number }
export interface Resident { model: string; runtime: string; roles: string[]; vram_mb: number; load_s: number; loaded_at: number; last_used: number; jobs: number; busy?: boolean }
export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'expired'
export interface Job {
  id: number; kind: 'run' | 'lease'; app: string; priority: 'interactive' | 'batch'; status: JobStatus; submitted_at: number
  role: string | null; model: string | null; note: string | null; params: Record<string, unknown> | null; result?: unknown; error: string | null
  vram_gb: number | null; ttl_s: number | null; started_at: number | null; finished_at: number | null; heartbeat_at: number | null; load_s: number | null; run_s: number | null; wait_s?: number
}
export interface ServiceEvent { at: number; text: string }
/** A job running now, and its lane: `gpu`, or `cpu` for zero-VRAM models (recorded outputs, CPU models, cloud CLIs), which never wait for the card. */
export type RunningJob = Job & { lane?: 'gpu' | 'cpu' }
export interface Status { gpu: Gpu | null; budget: Budget | null; resident: Resident[]; current: Job | null; running?: RunningJob[]; slots?: { gpu: number; cpu: number }; lease: Job | null; leases?: Job[]; queued: Job[]; events: ServiceEvent[] }
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
export interface Settings { data_dir: string; models_dir: string; vram_reserve_gb: number; gpu_jobs: number; cpu_jobs: number; whole_card_gb: number }
export interface JobFilters { status?: string; app?: string; role?: string; kind?: string }

// ---- cloud APIs (spend.py): one budget per key, the ledger of every call --------------------------------------------------------------
export type Provider = 'claude' | 'gemini' | 'gemini-free' | 'nim'
export interface ProviderState {
  provider: Provider; label: string; budgeted: boolean; unit: 'usd'; budget: number | null; rollover: string | null; period_start: number | null; next_rollover: number | null
  spent: number; calls: number; reserved: number; left: number | null; used_share: number | null; warn: number | null
}
export interface Spend { providers: ProviderState[]; keys: Record<string, boolean>; warn_at: number[] }
export interface SpendPoint { t: number; provider: Provider; cost_usd: number; calls: number; failed: number; tokens: number }
export interface SpendRow { provider: Provider; model: string; app: string | null; calls: number; failed: number; tokens_in: number; tokens_out: number; cost_usd: number }
export interface SpendSeries { since: number; bucket_s: number; points: SpendPoint[]; breakdown: SpendRow[]; options: { models: string[]; apps: string[] } }
export interface ApiCall {
  id: number; at: number; provider: Provider; model: string; app: string | null; job_id: number | null; tokens_in: number; tokens_out: number; cache_read: number; cache_write: number
  cost_usd: number; ok: boolean; error: string | null; seconds: number | null; model_key: string | null; finish: string | null
  request?: { system?: string | null; prompt?: string; images?: string[]; schema?: unknown; max_tokens?: number; effort?: string | null } | null; response?: string | null
}
export interface ApiCallPage { calls: ApiCall[]; next: number | null }
export interface CallFilters { provider?: string; model?: string; app?: string; ok?: string; days?: number }
export interface BudgetChange { claude_budget_usd?: number; claude_rollover?: string; gemini_budget_usd?: number; gemini_rollover?: string }

export function spendUrl(path: string, f: CallFilters, extra: Record<string, string | number | undefined> = {}) {
  const p = new URLSearchParams()
  for (const k of ['provider', 'model', 'app', 'ok'] as const) if (f[k]) p.set(k, f[k]!)
  if (f.days) p.set('days', String(f.days))
  for (const [k, v] of Object.entries(extra)) if (v !== undefined) p.set(k, String(v))
  return `${path}?${p}`
}

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
  spend: () => get<Spend>('/api/spend'),
  spendSeries: (f: CallFilters) => get<SpendSeries>(spendUrl('/api/spend/series', { provider: f.provider, model: f.model, app: f.app }, { days: f.days ?? 30, bucket: (f.days ?? 30) <= 2 ? 'hour' : 'day' })),
  calls: (f: CallFilters, before?: number) => get<ApiCallPage>(spendUrl('/api/spend/calls', f, { before, limit: 50 })),
  call: (id: number) => get<ApiCall>(`/api/spend/calls/${id}`),
  saveBudgets: (b: BudgetChange) => send<Settings>('PUT', '/api/settings', b),
}

// ---- evals -------------------------------------------------------------------------------------------------------------------------------
export interface Stage { role: string; model: string | null; params: Record<string, unknown>; options: Record<string, unknown>; input: string; send: string; into: string; note: string }
export interface Candidate { name: string; model: string; params: Record<string, unknown>; options: Record<string, unknown>; note: string; stages?: Stage[] }
/** One step of a pipeline candidate's output: what it was sent and (for a stage) what it answered. The last step is the candidate's own job. */
export interface Step { role: string; model: string; sent: string; answer?: string; params?: Record<string, unknown>; run_s?: number | null; load_s?: number | null; tokens_out?: number | null }
export interface ItemOutputs { id: string; input: Record<string, unknown>; has_media: boolean; reference: Record<string, unknown> | null; outputs: Record<string, { output: unknown; error: string | null; run_s?: number | null; load_s?: number | null } | null> }
export interface Experiment {
  name: string; question?: string; role?: string; dataset?: string; metrics?: string[]; candidates?: Candidate[]; primary?: string; baseline?: string
  min_effect?: number; guards?: Record<string, number>; judge?: boolean; outputs?: number; total?: number; running?: boolean; error?: string
}
export interface DatasetInfo { name: string; version?: string; n?: number; description?: string; tags?: string[]; error?: string }
export interface Evals { experiments: Experiment[]; datasets: DatasetInfo[] }
export interface MetricInfo { name: string; shape: 'mean' | 'binary' | 'ratio' | 'dataset'; better: 'lower' | 'higher'; description: string; unit: string }
export interface Interval { value: number; low: number; high: number; n: number; p95?: number }
export interface Comparison { n: number; mean_a: number; mean_b: number; diff: number; low: number; high: number; p: number; p_holm?: number; improves: boolean | null; only_baseline?: number; only_candidate?: number }
export interface Verdict { verdict: 'adopt' | 'reject' | 'unclear'; why: string; gain: number }
export interface Analysis {
  experiment: Experiment & { candidates: Candidate[]; metrics: string[]; primary: string; baseline: string }
  dataset: DatasetInfo & { compared: number; every?: number; n: number; not_run?: string[] }
  candidates: Record<string, { done: number; compared?: number; sample?: number | string | null; failed: number; load_s: number | null; metrics: Record<string, Interval> }>
  comparisons: Record<string, Record<string, Comparison>>
  metrics: Record<string, MetricInfo>
  decision: { per_candidate: Record<string, Verdict>; recommendation: string; change: boolean }
  judgments?: { n: number; strength: Record<string, [number, number, number]> }
  judges?: Record<string, JudgeSummary> // model judges beside the people (evals.judges): agreement with people's judgments on the same pairs
  summary?: { computed_at: number; seconds: number; stale: boolean } // worked out ahead and kept; stale: a fresh one is on its way
}
export interface Share { value: number | null; low: number | null; high: number | null; n: number; k?: number }
export interface JudgeSummary {
  kind: 'text' | 'vision'; model: string | null; via: string; role: string | null; judged: number; with_people: number
  agree: Share; decided: Share; kappa: number | null; consistent: Share
  tally: { same_winner: number; opposite: number; judge_tie: number; person_tie: number; both_tie: number }
  disagreements: { item_id: string; a: string; b: string; person: string; judge: string; reason: string }[]
}
export interface JudgmentRow { a: string; b: string; winner: string; rater: string; judge: string | null; at: number; note: string | null; detail: { consistent?: boolean | null } | null; a_key: string | null; b_key: string | null; current: boolean }
export interface RunStatus { state: 'idle' | 'running' | 'done' | 'failed'; log: string[]; error?: string; result?: { ran: number; cached: number; failed: number } }
export interface Pair { similarity?: number; item_id: string; input: Record<string, unknown>; has_media: boolean; reference: Record<string, unknown> | null; x: unknown; y: unknown; token: string }

export const evalsApi = {
  list: () => get<Evals>('/api/evals'),
  analysis: (name: string) => get<Analysis>(`/api/evals/${encodeURIComponent(name)}`),
  status: (name: string) => get<RunStatus>(`/api/evals/${encodeURIComponent(name)}/status`),
  run: (name: string) => send<RunStatus>('POST', `/api/evals/${encodeURIComponent(name)}/run`),
  report: async (name: string) => { const r = await fetch(withBase(`/api/evals/${encodeURIComponent(name)}/report`)); if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.text() },
  pair: (name: string) => get<{ pair: Pair | null; remaining?: number }>(`/api/evals/${encodeURIComponent(name)}/pair`),
  outputs: (name: string, item?: string) => get<{ candidates: Candidate[]; items: ItemOutputs[] }>(`/api/evals/${encodeURIComponent(name)}/outputs${item ? `?item_id=${encodeURIComponent(item)}` : ''}`),
  judge: (name: string, token: string, winner: 'x' | 'y' | 'tie') => send<{ ok: boolean; item_id: string; x: string; y: string; winner: string }>('POST', `/api/evals/${encodeURIComponent(name)}/judgments`, { token, winner }),
  verdicts: (name: string, item: string) => get<{ item_id: string; judgments: JudgmentRow[] }>(`/api/evals/${encodeURIComponent(name)}/verdicts/${encodeURIComponent(item)}`),
  mediaUrl: (name: string, item: string, file?: string) => withBase(`/api/evals/${encodeURIComponent(name)}/media/${encodeURIComponent(item)}${file ? `?file=${encodeURIComponent(file)}` : ''}`),
}

// ---- image generation ----------------------------------------------------------------------------------------------------------------------
export interface GeneratedImage { id: number; app: string; model: string; prompt: string; finished_at: number; path: string; width: number; height: number; seed: number; steps: number; seconds: number; peak_vram_mb: number }
export interface ImagePage { items: GeneratedImage[]; next: number | null }
export interface GenerateParams { prompt: string; width?: number; height?: number; seed?: number; steps?: number; cfg_scale?: number; ref_images?: string[] }

export const imagesApi = {
  list: (before?: number) => get<ImagePage>(`/api/images?limit=24${before ? `&before=${before}` : ''}`),
  fileUrl: (path: string) => withBase(`/api/images/file?path=${encodeURIComponent(path)}`),
  upload: async (f: File) => {
    const r = await fetch(withBase(`/api/images/upload?name=${encodeURIComponent(f.name)}`), { method: 'POST', body: f })
    const d = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error((d as { detail?: string }).detail || `${r.status} ${r.statusText}`)
    return d as { path: string }
  },
  /** Queue an interactive job, then wait for it (long polls); throws with the job's error when it fails. */
  generate: async (model: string | undefined, params: GenerateParams) => {
    const j = await send<Job>('POST', '/api/jobs', { role: 'image-generate', params, app: 'strata-models web', priority: 'interactive', model: model || null })
    for (;;) {
      const r = await get<Job>(`/api/jobs/${j.id}?wait=30`)
      if (r.status === 'done') return r
      if (r.status === 'failed' || r.status === 'cancelled') throw new Error(r.error || `the job was ${r.status}`)
    }
  },
}
