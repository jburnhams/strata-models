// The service as the web app sees it: fixtures plus msw handlers for every endpoint the app calls (one without a handler fails the test).
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { Job, Model, Registry, Settings, Status, Usage } from '../../src/api'

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
const defaultSettings: Settings = { data_dir: '/srv/sm', models_dir: '/home/j/models', vram_reserve_gb: 1 }

export const mock = { settings: { ...defaultSettings }, calls: [] as string[], status: fixtures.status(), pageSize: 100 }
export const resetMock = () => { mock.settings = { ...defaultSettings }; mock.calls = []; mock.status = fixtures.status(); mock.pageSize = 100 }

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
  http.get('/api/settings', () => HttpResponse.json(mock.settings)),
  http.put('/api/settings', async ({ request }) => {
    const body = (await request.json()) as Partial<Settings>
    if (body.models_dir && !body.models_dir.startsWith('/')) return HttpResponse.json({ detail: 'models_dir must be an absolute path' }, { status: 400 })
    mock.settings = { ...mock.settings, ...body }; return HttpResponse.json(mock.settings)
  }),
]

export const server = setupServer(...handlers)
export { http, HttpResponse }
