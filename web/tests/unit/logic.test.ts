import { describe, expect, it } from 'vitest'
import { bucketSeries, describeJob, finding, processName, seconds, spendFinding, ticksOf, usd, vramParts } from '../../src/logic'
import { jobsUrl, spendUrl } from '../../src/api'
import { byApp } from '../../src/components/UsagePage'
import { fixtures, spendFixtures } from '../utils/server'

describe('words for jobs and processes', () => {
  it('describes runs and leases', () => {
    expect(describeJob({ kind: 'run', role: 'faces', app: 'bh', note: null })).toBe('faces for bh')
    expect(describeJob({ kind: 'lease', role: null, app: 'stills', note: 'arena build' })).toBe('GPU lease for stills (arena build)')
    expect(describeJob({ kind: 'lease', role: null, app: 'stills', note: null })).toBe('GPU lease for stills')
  })
  it('names a process by its script', () => {
    expect(processName('/home/j/.venv/bin/python scripts/build_arena.py --picks x')).toBe('build_arena.py')
    expect(processName('Xwayland')).toBe('Xwayland')
    expect(processName('')).toBe('Unknown process')
    expect(processName('python ' + 'a'.repeat(50) + '.py').endsWith('…')).toBe(true)
  })
  it('writes durations as people say them', () => {
    expect(seconds(null)).toBe('–'); expect(seconds(0.42)).toBe('0.4 s'); expect(seconds(42)).toBe('42 s'); expect(seconds(600)).toBe('10 min'); expect(seconds(9000)).toBe('2.5 h')
  })
  it('builds job URLs with filters and a cursor', () => {
    expect(jobsUrl({ status: 'failed', app: 'bh' }, 40, 10)).toBe('/api/jobs?status=failed&app=bh&before=40&limit=10')
    expect(jobsUrl({})).toBe('/api/jobs?limit=100')
  })
})

describe('the VRAM breakdown and the finding', () => {
  it('splits the card into models, other processes, the desktop, the reserve and what is free', () => {
    const s = fixtures.status(); const parts = vramParts(s.gpu!, s.budget!, s.resident)
    expect(parts.map(p => p.key)).toEqual(['model:whisper-large-v3', 'model:buffalo_l', 'others', 'reserve'])
    expect(parts.reduce((a, p) => a + p.value, 0)).toBe(s.gpu!.used_mb + s.budget!.reserve_mb)
    const three = [...s.resident, { ...s.resident[1], model: 'whisper-small', vram_mb: 500 }]
    const many = vramParts({ ...s.gpu!, used_mb: 12288 }, { ...s.budget!, own_mb: 9000, other_mb: 3288 }, three)
    expect(many.map(p => p.label)).toEqual(['whisper-large-v3', 'buffalo_l', '1 more models, runtimes', 'Other programs and the desktop'])
    expect(vramParts(s.gpu!, { ...s.budget!, own_mb: 5000, other_mb: 0 }, [])[0].label).toBe('Runtimes')
  })
  it('says what the GPU is doing', () => {
    const s = fixtures.status()
    expect(finding(s)).toBe('Running transcribe for strata-360, 2 waiting.')
    expect(finding({ ...s, current: null })).toBe('2 waiting for the GPU.')
    expect(finding({ ...s, lease: { ...s.queued[1], status: 'running' }, queued: [] })).toBe('The GPU is lent to strata-stills (arena build).')
    expect(finding({ ...s, lease: { ...s.queued[1], note: null }, current: null, queued: [] })).toBe('The GPU is lent to strata-stills.')
    const sized = { ...s.queued[1], vram_gb: 6 }
    expect(finding({ ...s, lease: sized, leases: [sized], current: null, queued: [] })).toBe('Lent 6\u00a0GB to strata-stills; the rest is free for other work.')
    expect(finding({ ...s, lease: sized, leases: [sized], queued: [] })).toBe('Running transcribe for strata-360.')
    expect(finding({ ...s, current: null, queued: [] })).toBe('Idle, with 2 models loaded and 5.1 GB free for more.')
    expect(finding({ ...s, current: null, queued: [], resident: s.resident.slice(1) })).toBe('Idle, with buffalo_l loaded and 5.1 GB free for more.')
    expect(finding({ ...s, current: null, queued: [], resident: [] })).toBe('Idle, nothing loaded: 5.1 GB free for models.')
    expect(finding({ ...s, gpu: null })).toBe('No NVIDIA GPU found: models run on the CPU.')
    const cloud = { ...s.current!, id: 42, role: 'text', app: 'evals', lane: 'cpu' as const }
    expect(finding({ ...s, running: [s.current!, cloud] })).toBe('Running transcribe for strata-360 and text for evals, 2 waiting.')
    expect(finding({ ...s, running: [s.current!, cloud, { ...cloud, id: 43 }], queued: [] })).toBe('Running 3 jobs: transcribe for strata-360 and 2 more.')
    expect(finding({ ...s, running: [], queued: [] })).toBe('Idle, with 2 models loaded and 5.1 GB free for more.')
    expect(finding({ ...s, budget: null, current: null, queued: [], resident: [] })).toBe('Idle, nothing loaded: 0.0 GB free for models.')
  })
  it('totals usage by app, most first', () => {
    expect(byApp(fixtures.usage().rows)).toEqual([['strata-stills', 600], ['bright-horizons', 15]])
  })
})

import { diff, outputText, pvalue, reading, value } from '../../src/evalfmt'
describe('eval numbers', () => {
  const pct = { name: 'wer', shape: 'ratio', better: 'lower', description: '', unit: '%' } as const
  const sec = { ...pct, unit: 's' } as const
  const raw = { ...pct, unit: '' } as const
  it('writes values, differences and p as the report does', () => {
    expect(value(pct, 0.0312)).toBe('3.12%'); expect(value(sec, 0.5)).toBe('0.500 s'); expect(value(raw, 0.91234)).toBe('0.9123'); expect(value(pct, NaN)).toBe('n/a')
    expect(diff(pct, -0.002)).toBe('−0.20 pts'); expect(diff(sec, 0.1)).toBe('+0.100 s'); expect(diff(raw, 0.01)).toBe('+0.010'); expect(diff(pct, null)).toBe('n/a')
    expect(pvalue(0.0001)).toBe('< 0.001'); expect(pvalue(0.2)).toBe('0.200'); expect(pvalue(undefined)).toBe('n/a')
    const c = { n: 1, mean_a: 0, mean_b: 0, diff: 0, low: 0, high: 0, p: 1 }
    expect(reading({ ...c, improves: true })).toBe('Better'); expect(reading({ ...c, improves: false })).toBe('Worse'); expect(reading({ ...c, improves: null })).toBe('No clear difference')
    expect(outputText({ segments: [{ text: ' a' }, { text: 'b' }, {}] })).toBe('a b'); expect(outputText({ faces: [] })).toContain('faces')
  })
})

import { window_ } from '../../src/components/PhrasePlayer'
describe('the phrase window', () => {
  it('starts 2 s before and stops 1 s after, never before 0', () => {
    expect(window_([14.4, 16.2])).toEqual({ from: 12.4, to: 17.2 }); expect(window_([1, 2]).from).toBe(0); expect(window_(undefined)).toEqual({ from: 0, to: Infinity })
  })
})

describe('cloud spend', () => {
  it('writes dollars to the cent, and below a cent to four places', () => {
    expect([usd(62.5), usd(0), usd(0.00012), usd(0.01), usd(null), usd(undefined)]).toEqual(['$62.50', '$0.00', '$0.0001', '$0.01', '–', '–'])
  })
  it('finds the paid key nearest its budget, or the keys without one', () => {
    const ps = spendFixtures.spend().providers
    expect(spendFinding(ps)).toMatch(/^Claude API: \$62\.50 of \$100\.00 spent, \$37\.50 left until /)
    expect(spendFinding([{ ...ps[0], next_rollover: null }])).toBe('Claude API: $62.50 of $100.00 spent, $37.50 left.')
    expect(spendFinding([{ ...ps[0], budget: null }])).toBe('No budget is set for Claude API: their calls are refused until one is.')
    expect(spendFinding([ps[3]])).toBe('No paid keys.')
  })
  it('fills every bucket to now and sums the providers shown', () => {
    const pts = [{ t: 100, provider: 'claude' as const, cost_usd: 1, calls: 2, failed: 0, tokens: 9 }, { t: 100, provider: 'nim' as const, cost_usd: 0, calls: 3, failed: 0, tokens: 1 }]
    expect(bucketSeries(pts, 0, 100, 250, 'calls')).toEqual([{ x: 0, y: 0 }, { x: 100_000, y: 5 }, { x: 200_000, y: 0 }])
    expect(ticksOf([{ x: 0, y: 0 }, { x: 400, y: 1 }])).toEqual([0, 100, 200, 300, 400])
    expect(ticksOf([{ x: 5, y: 0 }])).toEqual([5])
    expect(ticksOf([])).toEqual([])
  })
  it('builds the spend URLs from the filters', () => {
    expect(spendUrl('/api/spend/calls', { provider: 'claude', ok: 'false', days: 7 }, { before: 9, limit: 50 })).toBe('/api/spend/calls?provider=claude&ok=false&days=7&before=9&limit=50')
    expect(spendUrl('/api/spend/series', {}, { days: undefined })).toBe('/api/spend/series?')
  })
})
