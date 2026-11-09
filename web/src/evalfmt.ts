// How eval numbers are written on screen: the same rules as the Python write-up (report.py), so the page and the report read alike.
import type { Comparison, MetricInfo } from './api'

const bad = (x: number | null | undefined) => x === null || x === undefined || Number.isNaN(x)

export function value(m: MetricInfo, x: number | null | undefined) {
  if (bad(x)) return 'n/a'
  if (m.unit === '%') return `${(x! * 100).toFixed(2)}%`
  if (m.unit === 's') return `${x!.toFixed(3)} s`
  return x!.toFixed(4)
}

export function diff(m: MetricInfo, x: number | null | undefined) {
  if (bad(x)) return 'n/a'
  const s = x! >= 0 ? '+' : '−'
  if (m.unit === '%') return `${s}${Math.abs(x! * 100).toFixed(2)} pts`
  return `${s}${Math.abs(x!).toFixed(3)}${m.unit === 's' ? ' s' : ''}`
}

export const pvalue = (p: number | null | undefined) => (bad(p) ? 'n/a' : p! < 0.001 ? '< 0.001' : p!.toFixed(3))

export function reading(c: Comparison) { return c.improves === true ? 'Better' : c.improves === false ? 'Worse' : 'No clear difference' }

export const VERDICT_TONE = { adopt: 'success', reject: 'danger', unclear: 'neutral' } as const
export const VERDICT_WORD = { adopt: 'Adopt', reject: 'Reject', unclear: 'Unclear' } as const

/** A job output as text to read side by side: a transcript's segments joined, otherwise the JSON. */
export function outputText(o: unknown): string {
  if (o && typeof o === 'object' && Array.isArray((o as { segments?: unknown }).segments)) return ((o as { segments: { text?: string }[] }).segments).map(s => s.text ?? '').join(' ').trim()
  return JSON.stringify(o, null, 1)
}
