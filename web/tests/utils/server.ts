// The service as the web app sees it: fixtures plus msw handlers for every endpoint the app calls (one without a handler fails the test).
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { Analysis, ApiCall, Evals, Job, Model, Pair, ProviderState, Registry, RunStatus, Settings, Spend, SpendSeries, Status, Usage } from '../../src/api'

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
    roles: [{ key: 'image-generate', description: 'Pictures', model: 'flux3-klein' }, { key: 'faces', description: 'Find faces', model: 'buffalo_l' }, { key: 'transcribe', description: 'Speech to text', model: 'whisper-large-v3' }, { key: 'vision-language', description: 'Pictures', model: 'qwen' }],
    models: [
      { key: 'buffalo_l', runtime: 'insightface', name: 'buffalo_l', roles: ['faces'], vram_gb: 1, licence: 'non-commercial', source: 'https://example.org/i', used_by: ['bright-horizons'], options: {}, served: true, resident: true, missing: [], stats: { jobs: 300, mean_run_s: 0.05, last_at: NOW } },
      { key: 'whisper-large-v3', runtime: 'faster-whisper', name: 'large-v3', roles: ['transcribe'], vram_gb: 4.5, licence: 'MIT', source: '', used_by: ['strata-360'], options: {}, served: true, resident: false, missing: [], stats: null },
      { key: 'flux3-klein', runtime: 'sd-cpp', name: 'FLUX.3 [klein]', roles: ['image-generate'], vram_gb: 7, licence: 'FLUX Kommunity', source: '', used_by: [], options: { steps: 16 }, served: true, resident: false, missing: [], stats: null },
      { key: 'qwen', runtime: 'llama-server', name: 'q', roles: ['vision-language'], vram_gb: 7, licence: 'Apache-2.0', source: '', used_by: ['strata-360'], options: {}, served: false, resident: false, missing: [], stats: null },
    ] as Model[],
  }),
  jobs: (): Job[] => [job({ id: 3, status: 'failed', error: 'no such image: /p/x.jpg', run_s: null }), job({ id: 2, kind: 'lease', app: 'strata-stills', note: 'arena build', role: null, model: null, status: 'running', finished_at: null }), job({ id: 1 })],
  usage: (): Usage => ({ hours: 24, rows: [{ app: 'bright-horizons', kind: 'run', role: 'faces', jobs: 300, failed: 1, gpu_s: 15 }, { app: 'strata-stills', kind: 'lease', role: null, jobs: 2, failed: 0, gpu_s: 600 }] }),
}
const provider = (o: Partial<ProviderState>): ProviderState => ({ provider: 'claude', label: 'Claude API', budgeted: true, unit: 'usd', budget: 100, rollover: '2027-01-08', period_start: NOW - 86400 * 6, next_rollover: NOW + 86400 * 25, spent: 62.5, calls: 1200, reserved: 0, left: 37.5, used_share: 0.625, warn: 0.5, ...o })
const call = (o: Partial<ApiCall>): ApiCall => ({ id: 1, at: NOW - 120, provider: 'claude', model: 'claude-haiku-5-5', app: 'evals', job_id: 7, tokens_in: 900, tokens_out: 40, cache_read: 100, cache_write: 0, cost_usd: 0.00012, ok: true, error: null, seconds: 0.8, model_key: 'claude-haiku-5-5-api', finish: 'end_turn', ...o })
export const spendFixtures = {
  spend: (): Spend => ({
    providers: [
      provider({}),
      provider({ provider: 'gemini', label: 'Google AI Studio (paid key)', budget: null, rollover: null, next_rollover: null, period_start: null, spent: 0, calls: 0, left: 0, used_share: null, warn: 1 }),
      provider({ provider: 'gemini-free', label: 'Google AI Studio (free tier)', budgeted: false, budget: null, rollover: null, next_rollover: null, period_start: null, spent: 0, calls: 30, left: null, used_share: null, warn: null }),
      provider({ provider: 'nim', label: 'NVIDIA NIM (free)', budgeted: false, budget: null, rollover: null, next_rollover: null, period_start: null, spent: 0, calls: 5, left: null, used_share: null, warn: null }),
    ],
    keys: { claude: true, gemini: false, 'gemini-free': true, nim: true },
    warn_at: [0.5, 0.8, 1],
  }),
  series: (days = 30): SpendSeries => {
    const bucket = days <= 2 ? 3600 : 86400
    const since = Math.floor((Date.now() / 1000 - days * 86400) / bucket) * bucket
    return { since, bucket_s: bucket, points: [{ t: since + bucket, provider: 'claude', cost_usd: 1.5, calls: 10, failed: 1, tokens: 5000 }, { t: since + bucket, provider: 'nim', cost_usd: 0, calls: 4, failed: 0, tokens: 900 }], breakdown: [{ provider: 'claude', model: 'claude-haiku-5-5', app: 'evals', calls: 10, failed: 1, tokens_in: 4000, tokens_out: 1000, cost_usd: 1.5 }, { provider: 'nim', model: 'nvidia/nemotron', app: null, calls: 4, failed: 0, tokens_in: 800, tokens_out: 100, cost_usd: 0 }] }
  },
  calls: (): ApiCall[] => [call({ id: 3 }), call({ id: 2, provider: 'nim', model: 'nvidia/nemotron', app: null, job_id: null, ok: false, error: 'HTTP 429: slow down', cost_usd: 0, tokens_in: 0, tokens_out: 0, cache_read: 0, seconds: null, finish: null, model_key: null })],
  call: (id: number): ApiCall => ({ ...spendFixtures.calls().find(c => c.id === id)!, request: id === 3 ? { system: 'Be brief.', prompt: 'Describe the photo.', images: ['/p/a.jpg'], schema: { type: 'object' }, max_tokens: 512, effort: 'low' } : { prompt: 'hi' }, response: id === 3 ? '{"scene": "a beach"}' : null }),
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
    judges: { 'qwen-text': { kind: 'text', model: 'qwen3.5-9b-vl', via: 'queue', role: 'judge-text', judged: 14, with_people: 12, agree: { value: 0.75, low: 0.47, high: 0.91, n: 12, k: 9 }, decided: { value: 0.9, low: 0.6, high: 0.98, n: 10, k: 9 }, kappa: 0.58, consistent: { value: 0.86, low: 0.6, high: 0.96, n: 14, k: 12 }, tally: { same_winner: 9, opposite: 1, judge_tie: 1, person_tie: 1, both_tie: 0 }, disagreements: [] } },
  }),
  outputs: () => ({
    candidates: [{ name: 'large-v3', model: 'whisper-large-v3', params: {}, options: {}, note: '' }, { name: 'turbo', model: 'whisper-large-v3-turbo', params: {}, options: {}, note: '' }, { name: 'small', model: 'whisper-small', params: {}, options: {}, note: '' }],
    items: [
      { id: 'u1', input: { prompt: 'a cat' }, has_media: false, reference: null, outputs: {
        'large-v3': { output: { segments: [{ text: ' plain cat' }] }, error: null, run_s: 1.5 },
        turbo: { output: { segments: [{ text: ' rich cat' }], pipeline: [{ role: 'text', model: 'qwen3.5-9b-vl', sent: 'a cat', answer: 'a fluffy ginger cat asleep on a windowsill', run_s: 4.2 }, { role: 'transcribe', model: 'whisper-large-v3-turbo', sent: 'a fluffy ginger cat asleep on a windowsill', run_s: 1.1 }] }, error: null, run_s: 5.3 },
        small: { output: null, error: 'stage 1 (qwen3.5-9b-vl): out of memory' } } },
      { id: 'u2', input: { prompt: 'a dog' }, has_media: false, reference: null, outputs: { 'large-v3': { output: { segments: [{ text: ' dog' }] }, error: null }, turbo: null, small: null } },
    ],
  }),
  pair: (): Pair => ({ similarity: 0.62, item_id: 'u1', input: { language: 'en', clip: [14.4, 16.2] }, has_media: true, reference: { text: 'ask not' }, x: { segments: [{ text: ' ask not' }] }, y: { segments: [{ text: ' ask knot' }] }, token: 'tok' }),
}

const defaultSettings: Settings = { data_dir: '/srv/sm', models_dir: '/home/j/models', vram_reserve_gb: 1, gpu_jobs: 2, cpu_jobs: 4, whole_card_gb: 10 }

export const mock = { spend: spendFixtures.spend(), callPages: 1, settings: { ...defaultSettings }, calls: [] as string[], status: fixtures.status(), pageSize: 100, run: { state: 'idle', log: [] } as RunStatus, analysis: evalFixtures.analysis(), images: [] as unknown[] }
export const resetMock = () => { mock.spend = spendFixtures.spend(); mock.callPages = 1; mock.settings = { ...defaultSettings }; mock.calls = []; mock.status = fixtures.status(); mock.pageSize = 100; mock.run = { state: 'idle', log: [] }; mock.analysis = evalFixtures.analysis(); mock.images = [] }

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
  http.post('/api/jobs', async ({ request }) => { const b = (await request.json()) as { params: { prompt: string } }; mock.calls.push(`generate ${b.params.prompt}`); if (b.params.prompt === 'fail') return HttpResponse.json({ ...fixtures.jobs()[0], id: 98, status: 'queued' }, { status: 201 }); return HttpResponse.json({ ...fixtures.jobs()[0], id: 99, status: 'queued' }, { status: 201 }) }),
  http.get('/api/jobs/98', () => HttpResponse.json({ ...fixtures.jobs()[0], id: 98, status: 'failed', error: 'stable-diffusion.cpp is not installed' })),
  http.get('/api/jobs/99', () => HttpResponse.json({ ...fixtures.jobs()[0], id: 99, role: 'image-generate', model: 'flux3-klein', params: { prompt: 'a hare' }, status: 'done', result: { path: '/d/generated/a.png', width: 1024, height: 1024, seed: 5, steps: 16, seconds: 9.4, peak_vram_mb: 6200 } })),
  http.get('/api/images', () => HttpResponse.json({ items: mock.images, next: null })),
  http.post('/api/images/upload', () => { mock.calls.push('upload'); return HttpResponse.json({ path: '/d/uploads/r.png' }, { status: 201 }) }),
  http.get('/api/jobs/:id', ({ params }) => { const j = fixtures.jobs().find(x => x.id === Number(params.id)); return j ? HttpResponse.json({ ...j, result: j.status === 'done' ? { faces: [] } : null }) : HttpResponse.json({ detail: 'no job' }, { status: 404 }) }),
  http.delete('/api/jobs/:id', ({ params }) => { mock.calls.push(`cancel ${params.id}`); return HttpResponse.json(fixtures.jobs()[0]) }),
  http.get('/api/usage', ({ request }) => HttpResponse.json({ ...fixtures.usage(), hours: Number(new URL(request.url).searchParams.get('hours')) })),
  http.get('/api/evals', () => HttpResponse.json(evalFixtures.list())),
  http.get('/api/evals/:name/status', () => HttpResponse.json(mock.run)),
  http.post('/api/evals/:name/run', ({ params }) => { mock.calls.push(`run ${params.name}`); mock.run = { state: 'running', log: ['transcribe-en: 0 of 400 outputs cached, 400 to run'] }; return HttpResponse.json(mock.run, { status: 202 }) }),
  http.get('/api/evals/:name/report', () => HttpResponse.text('# transcribe-en: Which Whisper model for English?\n\n## Answer\n')),
  http.get('/api/evals/:name/pair', () => HttpResponse.json({ pair: evalFixtures.pair(), remaining: 12 })),
  http.get('/api/evals/:name/outputs', ({ request }) => { const item = new URL(request.url).searchParams.get('item_id'); const d = evalFixtures.outputs(); return HttpResponse.json(item ? { ...d, items: d.items.filter(i => i.id === item) } : d) }),
  http.get('/api/evals/:name/verdicts/:item', ({ params }) => HttpResponse.json({ item_id: params.item, judgments: [{ a: 'turbo', b: 'large-v3', winner: 'turbo', rater: 'qwen-text', judge: 'k1', at: 2, note: 'turbo keeps the last word', detail: { consistent: true }, a_key: 'a', b_key: 'b', current: true }, { a: 'turbo', b: 'large-v3', winner: 'large-v3', rater: 'qwen-text', judge: 'old', at: 1.5, note: 'an old judge on other answers', detail: null, a_key: 'z', b_key: 'b', current: false }, { a: 'turbo', b: 'large-v3', winner: 'tie', rater: 'jonathan', judge: null, at: 1, note: null, detail: null, a_key: 'a', b_key: 'b', current: true }] })),
  http.post('/api/evals/:name/judgments', async ({ request }) => { const b = (await request.json()) as { winner: string }; mock.calls.push(`judge ${b.winner}`); return HttpResponse.json({ ok: true, x: 'turbo', y: 'large-v3', winner: b.winner }, { status: 201 }) }),
  http.get('/api/evals/:name', ({ params }) => params.name === 'nope' ? HttpResponse.json({ detail: 'no experiment' }, { status: 404 }) : HttpResponse.json(mock.analysis)),
  http.get('/api/spend', () => HttpResponse.json(mock.spend)),
  http.get('/api/spend/series', ({ request }) => HttpResponse.json(spendFixtures.series(Number(new URL(request.url).searchParams.get('days'))))),
  http.get('/api/spend/calls', ({ request }) => {
    const q = new URL(request.url).searchParams
    if (q.get('before')) return HttpResponse.json({ calls: [{ ...spendFixtures.calls()[0], id: 1, model: 'claude-sonnet-5-5' }], next: null })
    const calls = spendFixtures.calls().filter(c => (!q.get('provider') || c.provider === q.get('provider')) && (!q.get('ok') || String(c.ok) === q.get('ok')))
    return HttpResponse.json({ calls, next: mock.callPages > 1 && calls.length ? 2 : null })
  }),
  http.get('/api/spend/calls/:id', ({ params }) => HttpResponse.json(spendFixtures.call(Number(params.id)))),
  http.get('/api/settings', () => HttpResponse.json(mock.settings)),
  http.put('/api/settings', async ({ request }) => {
    const body = (await request.json()) as Partial<Settings> & Record<string, unknown>
    if (body.claude_budget_usd !== undefined) {
      mock.spend = { ...mock.spend, providers: mock.spend.providers.map(p => p.provider === 'claude' ? { ...p, budget: Number(body.claude_budget_usd), rollover: String(body.claude_rollover || '') || null } : p) }
    }
    if (body.models_dir && !body.models_dir.startsWith('/')) return HttpResponse.json({ detail: 'models_dir must be an absolute path' }, { status: 400 })
    mock.settings = { ...mock.settings, ...body }; return HttpResponse.json(mock.settings)
  }),
]

export const server = setupServer(...handlers)
export { http, HttpResponse }
