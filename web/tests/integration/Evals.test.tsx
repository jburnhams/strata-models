import { describe, expect, it } from 'vitest'
import App from '../../src/App'
import { mock, server, http, HttpResponse, evalFixtures } from '../utils/server'
import { screen, setup, waitFor, within } from '../utils/render'

describe('Evals list', () => {
  it('shows each experiment with how many outputs are in, and the datasets', async () => {
    setup(<App />, { route: '/evals' })
    expect(await screen.findByText('1 of 2 experiments have every output in; 1 still need a run.')).toBeInTheDocument()
    expect(screen.getByText('All outputs in')).toBeInTheDocument()
    expect(screen.getByText('Running')).toBeInTheDocument()
    expect(screen.getByText('Broken spec')).toBeInTheDocument()
    expect(within(screen.getByRole('table', { name: 'Datasets' })).getByText('abc123')).toBeInTheDocument()
  })
  it('teaches what to do when there is nothing', async () => {
    server.use(http.get('/api/evals', () => HttpResponse.json({ experiments: [], datasets: [] })))
    setup(<App />, { route: '/evals' })
    expect(await screen.findByText('No experiments yet')).toBeInTheDocument()
  })
})

describe('An experiment', () => {
  it('leads with the decision, then results with intervals and comparisons', async () => {
    setup(<App />, { route: '/evals/transcribe-en' })
    expect(await screen.findByText(/Keep large-v3: no candidate is clearly better on wer/)).toBeInTheDocument()
    const results = screen.getByRole('table', { name: /Results on 200/ })
    expect(within(results).getByText('3.10%')).toBeInTheDocument()
    expect(within(results).getByText('Unclear')).toBeInTheDocument()
    const comp = screen.getByRole('table', { name: /Compared with large-v3/ })
    expect(within(comp).getByText('Better')).toBeInTheDocument()
    expect(within(comp).getByText('No clear difference')).toBeInTheDocument()
    expect(screen.getByText(/12\sblind judgments/)).toBeInTheDocument()
  })
  it('says when a candidate is to be adopted, and when nothing has run', async () => {
    const a = evalFixtures.analysis()
    mock.analysis = { ...a, decision: { per_candidate: { turbo: { verdict: 'adopt', why: 'wer better by 0.80 points', gain: 0.008 } }, recommendation: 'turbo', change: true }, judgments: undefined }
    const { unmount } = setup(<App />, { route: '/evals/transcribe-en' })
    expect(await screen.findByText('Change to turbo: wer better by 0.80 points.')).toBeInTheDocument()
    unmount(); mock.analysis = { ...a, dataset: { ...a.dataset, compared: 0 }, comparisons: {} }
    setup(<App />, { route: '/evals/transcribe-en' })
    expect(await screen.findByText('No outputs yet: run the experiment.')).toBeInTheDocument()
  })
  it('runs the missing outputs and shows the progress', async () => {
    const { user } = setup(<App />, { route: '/evals/transcribe-en' })
    await user.click(await screen.findByRole('button', { name: 'Run missing outputs' }))
    await waitFor(() => expect(mock.calls).toContain('run transcribe-en'))
    expect(await screen.findByText(/400 to run/)).toBeInTheDocument()
  })
  it('judges a blind pair and shows the write-up', async () => {
    const { user } = setup(<App />, { route: '/evals/transcribe-en?tab=judge' })
    expect(await screen.findByText('ask knot')).toBeInTheDocument()
    expect(screen.getByText('Reference: ask not')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'X is better' }))
    await user.click(await screen.findByRole('button', { name: 'About the same' }))
    await waitFor(() => expect(mock.calls).toEqual(['judge x', 'judge tie']))
    await user.click(screen.getByRole('tab', { name: 'Write-up' }))
    expect(await screen.findByText(/## Answer/)).toBeInTheDocument()
  })
  it('says when there is nothing to judge', async () => {
    server.use(http.get('/api/evals/:name/pair', () => HttpResponse.json({ pair: null })))
    setup(<App />, { route: '/evals/transcribe-en?tab=judge' })
    expect(await screen.findByText('Nothing to judge')).toBeInTheDocument()
  })
})

describe('Evals list, odd entries', () => {
  it('shows experiments without outputs and datasets without a version', async () => {
    server.use(http.get('/api/evals', () => HttpResponse.json({ experiments: [{ name: 'new', question: 'q?', role: 'faces', dataset: 'd', candidates: [], total: 10 }], datasets: [{ name: 'd' }] })))
    setup(<App />, { route: '/evals' })
    expect(await screen.findByText('0 of 10 outputs')).toBeInTheDocument()
    expect(screen.getByText('0 of 1 experiments have every output in; 1 still need a run.')).toBeInTheDocument()
  })
})

describe('Judging pictures', () => {
  it('shows the two pictures and the prompt', async () => {
    server.use(http.get('/api/evals/:name/pair', () => HttpResponse.json({ pair: { item_id: 'hands', input: { prompt: 'two hands' }, has_media: false, reference: null, x: { path: '/d/generated/a.png' }, y: { path: '/d/generated/b.png' }, token: 't' } })))
    setup(<App />, { route: '/evals/transcribe-en?tab=judge' })
    expect(await screen.findByRole('img', { name: 'Output X' })).toBeInTheDocument()
    expect(screen.getByText('Prompt: two hands')).toBeInTheDocument()
  })
})
