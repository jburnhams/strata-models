// The service as the web app sees it: fixtures plus msw handlers for every endpoint the app calls (one without a handler fails the test).
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { Analysis, Evals, Job, Model, Pair, Registry, RunStatus, Settings, Status, Usage } from '../../src/api'

export const NOW = 1_800_000_000
const job = (o: Partial<Job>): Job => ({ id: 1, kind: 'run', app: 'bright-horizons', priority: 'batch', status: 'done', submitted_at: NOW - 60, role: 'faces', model: 'buffalo_l', note: null, params: { path: '/p/a.jpg' }, error: null, vram_gb: null, ttl_s: null, started_at: NOW - 59, finished_at: NOW - 58, heartbeat_at: null, load_s: 0.7, run_s: 0.05, wait_s: 1, ...o })
export const fixtures = {
  status: (): Status => ({
    gpu: { name: 'NVIDIA GeForce RTX 3060', total_mb: 12288, used_mb: 6000, free_mb: 6288, util_pct: 42, temp_c: 57, power_w: 80, power_limit_w: 170, processes: [{ pid: 10, used_mb: 4300, command: 'python -m stratamodels serve', ours: true }, { pid: 11, used_mb: 1100, command: '/home/j/.venv/bin/python scripts/build_arena.py --picks x', ours: false }] },
    budget: { total_mb: 12288, used_mb: 6000, own_mb: 4300, other_mb: 1700, reserve_mb: 1024, free_for_models_mb: 5264 },
    resident: [{ model: 'whisper-large-v3', runtime: 'faster-whisper', roles: ['transcribe'], vram_mb: 3720, load_s: 2.9, loaded_at: NOW - 600, last_used: NOW - 5, jobs: 12 }, { model: 'buffalo_l', runtime: 'insightface', roles: ['faces'], vram_mb: 580, load_s: 0.7, loaded_at: NOW - 900, last_used: NOW - 300, jobs: 300 }],
    current: job({ id: 41, role: 'transcribe', model: 'whisper-large-v3', app: 'strata-360', status: 'running', finished_at: null, run_s: null }),
    lease: null,
    queued: [job({ id: 42, status: 'queued', started_at: null, finished_at: null }), job({ id: 43, kind: 'lease', app: 'strata-stills', note: 'arena build', role: null, model: null, status: 'queued', priority: 'interactive', started_at: null, finished_at: null })],
    events: [{ at: NOW - 600, text: 'loaded whisper-large-v3 in 2.9 s (3.6 GB)' }],
  }),
  registry: (): Registry => ({
    roles: [{ key: 'faces', description: 'Find faces', model: 'buffalo_l' }, { key: 'transcribe', description: 'Speech to text', model: 'whisper-large-v3' }, { key: 'vision-language', description: 'Pictures', model: 'qwen' }],
    models: [
      { key: 'buffalo_l', runtime: 'insightface', name: 'buffalo_l', roles: ['faces'], vram_gb: 1, licence: 'non-commercial', source: 'https://example.org/i', used_by: ['bright-horizons'], options: {}, served: true, resident: true, missing: [], stats: { jobs: 300, mean_run_s: 0.05, last_at: NOW } },
      { key: 'whisper-large-v3', runtime: 'faster-whisper', name: 'large-v3', roles: ['transcribe'], vram_gb: 4.5, licence: 'MIT', source: '', used_by: ['strata-360'], options: {}, served: true, resident: false, missing: [], stats: null },
      { key: 'qwen', runtime: 'llama-server', name: 'q', roles: ['vision-language'], vram_gb: 7, licence: 'Apache-2.0', source: '', used_by: ['strata-360'], options: {}, served: false, resident: false, missing: [], stats: null },
    ] as Model[],
  }),
  jobs: (): Job[] => [job({ id: 3, status: 'failed', error: 'no such image: /p/x.jpg', run_s: null }), job({ id: 2, kind: 'lease', app: 'strata-stills', note: 'arena build', role: null, model: null, status: 'running', finished_at: null }), job({ id: 1 })],
  usage: (): Usage => ({ hours: 24, rows: [{ app: 'bright-horizons', kind: 'run', role: 'faces', jobs: 300, failed: 1, gpu_s: 15 }, { app: 'strata-stills', kind: 'lease', role: null, jobs: 2, failed: 0, gpu_s: 600 }] }),
}
const iv = (value: number, d = 0.01) => ({ value, low: value - d, high: value + d, n: 200 })
const wer = { name: 'wer', shape: 'ratio', better: 'lower', description: 'Word error rate', unit: '%' } as const
const lat = { name: 'latency_s', shape: 'mean', better: 'lower', description: 'Seconds per item', unit: 's' } as const
export const evalFixtures = {
  list: (): Evals => ({
    experiments: [
      { name: 'transcribe-en', question: 'Which Whisper model for English?', role: 'transcribe', dataset: 'librispeech', candidates: [{ name: 'large-v3', model: 'whisper-large-v3', params: {}, options: {}, note: '' }, { name: 'turbo', model: 'whisper-large-v3-turbo', params: {}, options: {}, note: '' }], outputs: 400, total: 400, running: false },
      { name: 'faces-bh', question: 'Which face model?', role: 'faces', dataset: 'bh', candidates: [], outputs: 10, total: 300, running: true },
      { name: 'broken', error: 'unknown metrics' },
    ],
    datasets: [{ name: 'librispeech', version: 'abc123', n: 200, description: 'LibriSpeech test-clean' }, { name: 'bad', error: 'line 3' }],
  }),
  analysis: (): Analysis => ({
    experiment: { name: 'transcribe-en', question: 'Which Whisper model for English?', role: 'transcribe', dataset: 'librispeech', metrics: ['wer', 'latency_s'], candidates: [{ name: 'large-v3', model: 'whisper-large-v3', params: {}, options: {}, note: '' }, { name: 'turbo', model: 'whisper-large-v3-turbo', params: {}, options: {}, note: '' }], primary: 'wer', baseline: 'large-v3', min_effect: 0.005, guards: {}, judge: true },
    dataset: { name: 'librispeech', version: 'abc123', n: 200, compared: 200 },
    candidates: { 'large-v3': { done: 200, failed: 0, load_s: 3, metrics: { wer: iv(0.031), latency_s: iv(0.4, 0.05) } }, turbo: { done: 200, failed: 0, load_s: 2, metrics: { wer: iv(0.033), latency_s: iv(0.2, 0.02) } } },
    comparisons: { turbo: { wer: { n: 200, mean_a: 0.031, mean_b: 0.033, diff: 0.002, low: -0.001, high: 0.005, p: 0.2, p_holm: 0.2, improves: null }, latency_s: { n: 200, mean_a: 0.4, mean_b: 0.2, diff: -0.2, low: -0.25, high: -0.15, p: 0.0002, improves: true } } },
    metrics: { wer, latency_s: lat },
    decision: { per_candidate: { turbo: { verdict: 'unclear', why: 'difference -0.20 points, interval includes no change', gain: -0.002 } }, recommendation: 'large-v3', change: false },
    judgments: { n: 12, strength: { 'large-v3': [1.2, 0.8, 1.6], turbo: [0.8, 0.5, 1.1] } },
  }),
  pair: (): Pair => ({ item_id: 'u1', input: { language: 'en' }, has_media: true, reference: { text: 'ask not' }, x: { segments: [{ text: ' ask not' }] }, y: { segments: [{ text: ' ask knot' }] }, token: 'tok' }),
}

const defaultSettings: Settings = { data_dir: '/srv/sm', models_dir: '/home/j/models', vram_reserve_gb: 1 }

export const mock = { settings: { ...defaultSettings }, calls: [] as string[], status: fixtures.status(), pageSize: 100, run: { state: 'idle', log: [] } as RunStatus, analysis: evalFixtures.analysis() }
export const resetMock = () => { mock.settings = { ...defaultSettings }; mock.calls = []; mock.status = fixtures.status(); mock.pageSize = 100; mock.run = { state: 'idle', log: [] }; mock.analysis = evalFixtures.analysis() }

export const handlers = [
  http.get('/api/health', () => HttpResponse.json({ ok: true, version: '0.1.0', gpu: true })),
  http.get('/api/status', () => HttpResponse.json(mock.status)),
  http.get('/api/gpu/history', () => HttpResponse.json({ every_s: 5, samples: [0, 1, 2, 3].map(i => ({ at: NOW - 300 + i * 100, util_pct: i * 20, used_mb: 5000, own_mb: 4000 })) })),
  http.get('/api/registry', () => HttpResponse.json(fixtures.registry())),
  http.post('/api/models/:key/:action', ({ params }) => { mock.calls.push(`${params.action} ${params.key}`); return HttpResponse.json({ ok: true }, { status: 202 }) }),
  http.get('/api/jobs', ({ request }) => {
    const u = new URL(request.url); const before = Number(u.searchParams.get('before') || Infinity)
    let rows = fixtures.jobs().filter(j => j.id < before)
    for (const k of ['status', 'kind', 'app'] as const) { const v = u.searchParams.get(k); if (v) rows = rows.filter(j => j[k] === v) }
    const items = rows.slice(0, mock.pageSize)
    return HttpResponse.json({ items, next: rows.length > items.length ? items[items.length - 1].id : null })
  }),
  http.get('/api/jobs/:id', ({ params }) => { const j = fixtures.jobs().find(x => x.id === Number(params.id)); return j ? HttpResponse.json({ ...j, result: j.status === 'done' ? { faces: [] } : null }) : HttpResponse.json({ detail: 'no job' }, { status: 404 }) }),
  http.delete('/api/jobs/:id', ({ params }) => { mock.calls.push(`cancel ${params.id}`); return HttpResponse.json(fixtures.jobs()[0]) }),
  http.get('/api/usage', ({ request }) => HttpResponse.json({ ...fixtures.usage(), hours: Number(new URL(request.url).searchParams.get('hours')) })),
  http.get('/api/evals', () => HttpResponse.json(evalFixtures.list())),
  http.get('/api/evals/:name/status', () => HttpResponse.json(mock.run)),
  http.post('/api/evals/:name/run', ({ params }) => { mock.calls.push(`run ${params.name}`); mock.run = { state: 'running', log: ['transcribe-en: 0 of 400 outputs cached, 400 to run'] }; return HttpResponse.json(mock.run, { status: 202 }) }),
  http.get('/api/evals/:name/report', () => HttpResponse.text('# transcribe-en: Which Whisper model for English?\n\n## Answer\n')),
  http.get('/api/evals/:name/pair', () => HttpResponse.json({ pair: evalFixtures.pair() })),
  http.post('/api/evals/:name/judgments', async ({ request }) => { const b = (await request.json()) as { winner: string }; mock.calls.push(`judge ${b.winner}`); return HttpResponse.json({ ok: true, x: 'turbo', y: 'large-v3', winner: b.winner }, { status: 201 }) }),
  http.get('/api/evals/:name', ({ params }) => params.name === 'nope' ? HttpResponse.json({ detail: 'no experiment' }, { status: 404 }) : HttpResponse.json(mock.analysis)),
  http.get('/api/settings', () => HttpResponse.json(mock.settings)),
  http.put('/api/settings', async ({ request }) => {
    const body = (await request.json()) as Partial<Settings>
    if (body.models_dir && !body.models_dir.startsWith('/')) return HttpResponse.json({ detail: 'models_dir must be an absolute path' }, { status: 400 })
    mock.settings = { ...mock.settings, ...body }; return HttpResponse.json(mock.settings)
  }),
]

export const server = setupServer(...handlers)
export { http, HttpResponse }
