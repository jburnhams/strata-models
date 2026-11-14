import { describe, expect, it } from 'vitest'
import App from '../../src/App'
import { mock, server, http, HttpResponse, fixtures } from '../utils/server'
import { screen, setup, waitFor, within } from '../utils/render'

describe('Overview', () => {
  it('says what the GPU is doing, how full it is and who is waiting', async () => {
    setup(<App />)
    expect(await screen.findByText('Running transcribe for strata-360, 2 waiting.', { selector: 'p, span, div' })).toBeInTheDocument()
    expect(screen.getByText('v0.1.0')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Where the memory is' })).toBeInTheDocument()
    const programs = screen.getByRole('grid', { name: 'Programs using the GPU' })
    expect(within(programs).getByText('build_arena.py')).toBeInTheDocument()
    expect(within(programs).getByText('strata-models')).toBeInTheDocument()
    const waiting = screen.getByRole('grid', { name: 'Jobs waiting for the GPU' })
    expect(within(waiting).getByText('GPU lease for strata-stills (arena build)')).toBeInTheDocument()
    expect(within(waiting).getByText('High')).toBeInTheDocument()
    expect(screen.getByText(/loaded whisper-large-v3/)).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'The last hour' })).toBeInTheDocument()
  })
  it('shows a held lease, an empty queue and a machine without a GPU', async () => {
    const s = fixtures.status()
    mock.status = { ...s, lease: { ...s.queued[1], status: 'running', started_at: Date.now() / 1000 - 30 }, current: null, queued: [], events: [] }
    const { unmount } = setup(<App />)
    expect(await screen.findByText(/Lent to/)).toBeInTheDocument()
    expect(screen.getByText('No one is waiting.')).toBeInTheDocument()
    expect(screen.getByText(/No models loaded or freed/)).toBeInTheDocument()
    unmount()
    mock.status = { ...s, gpu: null, budget: null, current: null, queued: [] }
    setup(<App />)
    expect(await screen.findByText('No NVIDIA GPU found: models run on the CPU.')).toBeInTheDocument()
    expect(screen.getByText('Nothing is running.')).toBeInTheDocument()
  })
  it('lists every job running at once, and a lease beside them', async () => {
    const s = fixtures.status()
    const cloud = { ...s.current!, id: 42, role: 'text', model: 'gemini-cli', app: 'evals', started_at: Date.now() / 1000 - 5, lane: 'cpu' as const }
    const sized = { ...s.queued[1], status: 'running' as const, vram_gb: 4, started_at: Date.now() / 1000 - 60 }
    mock.status = { ...s, running: [{ ...s.current!, started_at: Date.now() / 1000 - 10, lane: 'gpu' }, cloud], lease: sized, leases: [sized] }
    setup(<App />)
    expect(await screen.findByText(/on gemini-cli for/)).toHaveTextContent('(no GPU needed)')
    expect(screen.getByText(/on whisper-large-v3 for/)).not.toHaveTextContent('no GPU needed')
    expect(screen.getByText(/asking for 4 GB/)).toBeInTheDocument()
  })
})

describe('Models', () => {
  it('lists roles and models, and loads or frees one', async () => {
    const { user } = setup(<App />, { route: '/models' })
    expect(await screen.findByText(/3 of 4 roles are served here\. strata-360 still load/)).toBeInTheDocument()
    expect(screen.getAllByText('Not served yet').length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: 'Free buffalo_l' }))
    await user.click(screen.getByRole('button', { name: 'Load whisper-large-v3' }))
    await waitFor(() => expect(mock.calls).toEqual(['unload buffalo_l', 'load whisper-large-v3']))
  })
})

describe('Jobs', () => {
  it('lists jobs, opens one to show why it failed, filters by state and withdraws a lease', async () => {
    const { user } = setup(<App />, { route: '/jobs' })
    expect(await screen.findByText('1 job of the 3 shown failed; open one to see why.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /job #3/ }))
    expect(await screen.findByText('no such image: /p/x.jpg')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /job #1$/ }))
    expect(await screen.findByText(/"faces"/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'End lease job #2' }))
    await waitFor(() => expect(mock.calls).toEqual(['cancel 2']))
    await user.selectOptions(screen.getByLabelText('State'), 'done')
    expect(await screen.findByText('None of the 1 jobs shown failed.')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Kind'), 'lease')
    expect(await screen.findByText('No jobs match these filters')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('State'), '')
    await user.selectOptions(screen.getByLabelText('Kind'), '')
    expect(await screen.findByText(/of the 3 shown failed/)).toBeInTheDocument()
  })
  it('pages back through older jobs', async () => {
    mock.pageSize = 2
    const { user } = setup(<App />, { route: '/jobs' })
    await user.click(await screen.findByRole('button', { name: 'Show older jobs' }))
    expect(await screen.findByText('#1')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Show older jobs' })).not.toBeInTheDocument()
  })
  it('says when there are no jobs yet', async () => {
    server.use(http.get('/api/jobs', () => HttpResponse.json({ items: [], next: null })))
    setup(<App />, { route: '/jobs' })
    expect(await screen.findByText('No jobs yet')).toBeInTheDocument()
  })
})

describe('Usage and Settings', () => {
  it('shows GPU time by app for the chosen period', async () => {
    const { user } = setup(<App />, { route: '/usage' })
    expect(await screen.findByText(/strata-stills used the GPU most/)).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: 'Last 7 days' }))
    expect(await screen.findByRole('grid', { name: 'GPU use by app and role' })).toBeInTheDocument()
    server.use(http.get('/api/usage', () => HttpResponse.json({ hours: 24, rows: [] })))
    await user.click(screen.getByRole('tab', { name: 'Last 30 days' }))
    expect(await screen.findByText('Nothing used the GPU in this period')).toBeInTheDocument()
  })
  it('saves the models folder and the VRAM reserve', async () => {
    const { user } = setup(<App />, { route: '/settings' })
    const reserve = await screen.findByLabelText('VRAM kept free (GB)')
    await user.clear(reserve); await user.type(reserve, '2')
    await user.click(screen.getByRole('button', { name: 'Save settings' }))
    await waitFor(() => expect(mock.settings.vram_reserve_gb).toBe(2))
    const gpuJobs = screen.getByLabelText('GPU jobs at once')
    await user.clear(gpuJobs); await user.type(gpuJobs, '3')
    const cpuJobs = screen.getByLabelText('Jobs without the GPU at once')
    await user.clear(cpuJobs); await user.type(cpuJobs, '6')
    await user.click(screen.getByRole('button', { name: /Save settings|Saved/ }))
    await waitFor(() => expect(mock.settings.gpu_jobs).toBe(3))
    expect(mock.settings.cpu_jobs).toBe(6)
    const whole = screen.getByLabelText('Whole card above (GB)')
    await user.clear(whole); await user.type(whole, '9.5')
    await user.click(screen.getByRole('button', { name: /Save settings|Saved/ }))
    await waitFor(() => expect(mock.settings.whole_card_gb).toBe(9.5))
    const folder = screen.getByLabelText('Models folder')
    await user.clear(folder); await user.type(folder, 'relative')
    await user.click(screen.getByRole('button', { name: /Save settings|Saved/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('models_dir must be an absolute path')
  })
  it('sends an unknown address to the Overview', async () => {
    setup(<App />, { route: '/nowhere' })
    expect(await screen.findByRole('heading', { name: 'Overview' })).toBeInTheDocument()
  })
})
