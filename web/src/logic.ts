// Pure helpers the screens share: the words for a job, the VRAM breakdown, the Overview's one-line finding. Unit-tested in node.
import { format, type Point } from 'strata-react'
import type { Budget, Gpu, Job, ProviderState, Resident, SpendPoint, Status } from './api'

export const GB = 1024
export const gb = (mb: number) => mb / GB

/** What a job is, in a few words: "faces for bright-horizons", "GPU lease for strata-stills (arena build)". */
export function describeJob(j: Pick<Job, 'kind' | 'role' | 'app' | 'note'>) {
  return j.kind === 'lease' ? `GPU lease for ${j.app}${j.note ? ` (${j.note})` : ''}` : `${j.role} for ${j.app}`
}

export const STATUS_TONE = { queued: 'neutral', running: 'info', done: 'success', failed: 'danger', cancelled: 'neutral', expired: 'warning' } as const
/** A job's priority under today's names (rows from before 2026-11-14 say interactive or batch), and its word. */
export const PRIORITY_OF = { high: 'high', interactive: 'high', normal: 'normal', batch: 'normal', low: 'low' } as const
export const PRIORITY_WORD = { high: 'High', normal: 'Normal', low: 'Low' } as const
export const STATUS_WORD = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed', cancelled: 'Withdrawn', expired: 'Expired' } as const

/** A short readable name for another process on the card: the script, not the interpreter. */
export function processName(command: string) {
  const parts = command.split(' ').filter(Boolean)
  if (!parts.length) return 'Unknown process'
  const script = parts.find((p, i) => i > 0 && !p.startsWith('-') && /[./]/.test(p))
  const name = (script ?? parts[0]).split('/').pop()!
  return name.length > 40 ? name.slice(0, 39) + '…' : name
}

export interface Part { key: string; label: string; value: number }

/** The card's memory, in at most five parts (the composition bar has five colours; the free rest is its remainder): the two largest models, the
 * rest of the service (more models and the runtimes themselves), every other program on the card together, and the reserve. The parts and the
 * free rest add up to the total. Which programs those are is the Programs table's job. */
export function vramParts(gpu: Gpu, budget: Budget, resident: Resident[]): Part[] {
  const models = [...resident].sort((a, b) => b.vram_mb - a.vram_mb)
  const parts: Part[] = models.slice(0, 2).map(r => ({ key: `model:${r.model}`, label: r.model, value: r.vram_mb }))
  const named = parts.reduce((s, p) => s + p.value, 0)
  const restOfService = Math.max(budget.own_mb - named, 0)
  if (restOfService > 1) parts.push({ key: 'service', label: models.length > 2 ? `${models.length - 2} more models, runtimes` : 'Runtimes', value: restOfService })
  if (budget.other_mb > 1) parts.push({ key: 'others', label: 'Other programs and the desktop', value: budget.other_mb })
  const free = Math.max(gpu.total_mb - gpu.used_mb, 0)
  const reserve = Math.min(budget.reserve_mb, free)
  if (reserve > 0) parts.push({ key: 'reserve', label: 'Kept free for the desktop', value: reserve })
  return parts
}

/** The Overview's finding: what the GPU is doing now, in one sentence. */
export function finding(s: Status): string {
  if (!s.gpu) return 'No NVIDIA GPU found: models run on the CPU.'
  const waiting = s.queued.length ? `, ${s.queued.length} waiting` : ''
  if (s.lease && s.lease.vram_gb === null) return `The GPU is lent to ${s.lease.app}${s.lease.note ? ` (${s.lease.note})` : ''}${waiting}.`
  const shared = (s.leases ?? []).filter(l => l.vram_gb !== null).map(l => `${l.vram_gb}\u00a0GB to ${l.app}`).join(', ')
  const running = s.running ?? (s.current ? [s.current] : [])
  if (shared && !running.length) return `Lent ${shared}; the rest is free for other work${waiting}.`
  if (running.length > 2) return `Running ${running.length} jobs: ${describeJob(running[0])} and ${running.length - 1} more${waiting}.`
  if (running.length) return `Running ${running.map(describeJob).join(' and ')}${waiting}.`
  if (s.queued.length) return `${s.queued.length} waiting for the GPU.`
  const free = s.budget ? gb(s.budget.free_for_models_mb) : 0
  return s.resident.length ? `Idle, with ${s.resident.length === 1 ? s.resident[0].model : `${s.resident.length} models`} loaded and ${free.toFixed(1)} GB free for more.` : `Idle, nothing loaded: ${free.toFixed(1)} GB free for models.`
}

/** Seconds as people say them: "0.4 s", "12 s", "3 min", "2.1 h". */
export function seconds(s: number | null | undefined) {
  if (s === null || s === undefined) return '–'
  if (s < 10) return `${s.toFixed(1)}\u00a0s`
  if (s < 120) return `${Math.round(s)}\u00a0s`
  if (s < 7200) return `${Math.round(s / 60)}\u00a0min`
  return `${(s / 3600).toFixed(1)}\u00a0h`
}

// ---- cloud APIs ---------------------------------------------------------------------------------------------------------------------------
/** US dollars: whole cents from a cent up, four places below it (a single Haiku call is a fraction of a cent). */
export function usd(v: number | null | undefined) {
  if (v === null || v === undefined) return '\u2013'
  const a = Math.abs(v)
  return `$${a === 0 || a >= 0.01 ? v.toFixed(2) : v.toFixed(4)}`
}

/** The Cloud page's one-line answer: the paid key nearest its budget, else which paid keys have no budget yet. */
export function spendFinding(ps: ProviderState[]): string {
  const paid = ps.filter(p => p.budgeted)
  const set = paid.filter(p => p.budget).sort((a, b) => (b.used_share ?? 0) - (a.used_share ?? 0))
  if (set.length) {
    const p = set[0]
    const until = p.next_rollover ? ` until ${format.date(p.next_rollover * 1000)}` : ''
    return `${p.label}: ${usd(p.spent)} of ${usd(p.budget)} spent, ${usd(p.left)} left${until}.`
  }
  return paid.length ? `No budget is set for ${paid.map(p => p.label).join(' or ')}: their calls are refused until one is.` : 'No paid keys.'
}

/** One reading per bucket from `since` to `now`, zero where nothing was called, of one field summed over the providers shown. */
export function bucketSeries(points: SpendPoint[], since: number, bucket: number, now: number, field: 'cost_usd' | 'calls' | 'tokens'): Point[] {
  const sums = new Map<number, number>()
  for (const p of points) sums.set(p.t, (sums.get(p.t) ?? 0) + p[field])
  const out: Point[] = []
  for (let t = since; t <= now; t += bucket) out.push({ x: t * 1000, y: sums.get(t) ?? 0 })
  return out
}

/** Five evenly spaced ticks over a series' x range. */
export function ticksOf(series: Point[]) {
  if (!series.length) return []
  const a = series[0].x, b = series[series.length - 1].x
  return a === b ? [a] : [0, 1, 2, 3, 4].map(i => a + ((b - a) * i) / 4)
}
