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
    expect(within(screen.getByRole('grid', { name: 'Datasets' })).getByText('abc123')).toBeInTheDocument()
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
    const results = screen.getByRole('grid', { name: /Results on each candidate/ })
    expect(within(results).getByText('3.10%')).toBeInTheDocument()
    expect(within(results).getByText('Unclear')).toBeInTheDocument()
    const comp = screen.getByRole('grid', { name: /Compared with large-v3/ })
    expect(within(comp).getByText('Better')).toBeInTheDocument()
    expect(within(comp).getByText('No clear difference')).toBeInTheDocument()
    expect(screen.getByText(/12\sblind judgments/)).toBeInTheDocument()
    const judges = screen.getByRole('grid', { name: 'Model judges against people' })
    expect(within(judges).getByText('qwen-text')).toBeInTheDocument()
    expect(within(judges).getByText('75 % (47 to 91) of 12')).toBeInTheDocument()
    expect(within(judges).getByText('0.58')).toBeInTheDocument()
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
  it('marks a candidate that has not run yet', async () => {
    const a = evalFixtures.analysis()
    mock.analysis = { ...a, dataset: { ...a.dataset, not_run: ['turbo'] }, decision: { ...a.decision, per_candidate: {} } }
    setup(<App />, { route: '/evals/transcribe-en' })
    await screen.findByText(/Keep large-v3/)
    expect(await screen.findByText('Not run yet')).toBeInTheDocument()
    expect(screen.getByText(/turbo not yet/)).toBeInTheDocument()
  })
  it('says when the results shown are being updated', async () => {
    mock.analysis = { ...evalFixtures.analysis(), summary: { computed_at: 1_700_000_000, seconds: 40, stale: true } }
    setup(<App />, { route: '/evals/transcribe-en' })
    expect(await screen.findByText(/Updating: these results are from/)).toBeInTheDocument()
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
    expect(screen.getByText('12.4 s to 17.2 s of the recording')).toBeInTheDocument()
    const audio = screen.getByLabelText('The whole recording') as HTMLAudioElement
    audio.play = () => Promise.resolve(); audio.dispatchEvent(new Event('loadedmetadata')); expect(audio.currentTime).toBeCloseTo(12.4)
    await user.click(screen.getByRole('button', { name: 'Play the phrase' })); expect(audio.currentTime).toBeCloseTo(12.4)
    let paused = false; audio.pause = () => { paused = true }; audio.currentTime = 17.5; audio.dispatchEvent(new Event('timeupdate')); expect(paused).toBe(true)
    expect(screen.getByText(/38% different · 12\spairs not judged yet, most different first/)).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'X is better' })[0])
    await user.click((await screen.findAllByRole('button', { name: 'About the same' }))[0])
    await waitFor(() => expect(mock.calls).toEqual(['judge x', 'judge tie']))
    await user.click(screen.getByRole('tab', { name: 'Write-up' }))
    expect(await screen.findByText(/## Answer/)).toBeInTheDocument()
  })
  it('shows the pair just judged unblinded, with its pipeline steps', async () => {
    const { user } = setup(<App />, { route: '/evals/transcribe-en?tab=judge' })
    await user.click((await screen.findAllByRole('button', { name: 'X is better' }))[0])
    const last = await screen.findByText('Last pair (u1): X was turbo, Y was large-v3; you chose x')
    await user.click(last)
    expect(await screen.findByText(/chose turbo\. turbo keeps the last word/)).toBeInTheDocument() // the model judge's verdict on the same pair, after the person's
    expect(screen.queryByText(/jonathan/)).not.toBeInTheDocument()
    expect(screen.queryByText(/an old judge/)).not.toBeInTheDocument() // a verdict on outputs or by a judge config no longer current is history
    expect(await screen.findByText('Pipeline: 2 steps')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Pipeline: 2 steps/ }))
    expect(screen.getAllByText('a fluffy ginger cat asleep on a windowsill').length).toBeGreaterThan(0)
  })
  it('judges two pictures in a blind comparison from the keyboard, without waiting for the results to refresh', async () => {
    server.use(
      http.get('/api/evals/:name/pair', () => HttpResponse.json({ pair: { ...evalFixtures.pair(), has_media: false, x: { path: '/o/x.png', width: 768, height: 512 }, y: { path: '/o/y.png' } }, remaining: 3 })),
      http.get('/api/evals/:name', async () => { await new Promise(r => setTimeout(r, 5000)); return HttpResponse.json(mock.analysis) }),
    )
    const { user } = setup(<App />, { route: '/evals/transcribe-en?tab=judge' })
    const compare = await screen.findByLabelText('Item u1, two outputs', {}, { timeout: 7000 })
    expect(within(compare).getByRole('img', { name: 'X' })).toBeInTheDocument()
    expect(within(compare).queryByText(/turbo|large-v3/)).not.toBeInTheDocument() // blind: sides are named X and Y only
    expect(screen.getByRole('radiogroup', { name: 'View' })).toBeInTheDocument()
    await user.keyboard(']')
    await waitFor(() => expect(mock.calls).toContain('judge y'))
    expect(await screen.findByText('Last pair (u1): X was turbo, Y was large-v3; you chose y')).toBeInTheDocument()
    await user.keyboard('t')
    await waitFor(() => expect(mock.calls).toEqual(['judge y', 'judge tie']))
    expect((await screen.findAllByRole('button', { name: 'X is better' }))[0]).toBeEnabled()
  }, 15000)
  it('does not take a judging key typed into a field', async () => {
    const { user } = setup(<App />, { route: '/evals/transcribe-en?tab=judge' })
    await screen.findByText('ask knot')
    const box = document.createElement('input'); document.body.append(box); box.focus()
    await user.keyboard('[t]')
    expect(mock.calls).toEqual([])
    box.remove()
  })
  it('says when a judgment was not saved', async () => {
    server.use(http.post('/api/evals/:name/judgments', () => HttpResponse.json({ detail: 'pair token expired' }, { status: 409 })))
    const { user } = setup(<App />, { route: '/evals/transcribe-en?tab=judge' })
    await user.click((await screen.findAllByRole('button', { name: 'Y is better' }))[0])
    expect(await screen.findByRole('alert')).toHaveTextContent(/Not saved/)
  })
  it('shows every output on the Outputs tab, each step of a pipeline, and failures', async () => {
    const { user } = setup(<App />, { route: '/evals/transcribe-en?tab=outputs' })
    expect(await screen.findByText('Prompt: a cat')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Pipeline: 2 steps/ }))
    expect(screen.getByText('Step 1: qwen3.5-9b-vl (text) · 4.2 s')).toBeInTheDocument()
    expect(screen.getByText('Failed: stage 1 (qwen3.5-9b-vl): out of memory')).toBeInTheDocument()
    expect(screen.getByText('5.3 s in all')).toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox', { name: 'Item' }), 'u2')
    expect(await screen.findByText('Prompt: a dog')).toBeInTheDocument()
    expect(screen.getAllByText('Not run yet.')).toHaveLength(2)
  })
  it('lays picture outputs out in one comparison grid, with a pipeline\'s steps under its tile, and steps through items by key', async () => {
    const d = evalFixtures.outputs()
    const pic = (n: string, extra = {}) => ({ output: { path: `/o/${n}.png`, width: 64, height: 64, ...extra }, error: null, run_s: 2 });
    (d.items[0] as { outputs: Record<string, unknown> }).outputs = { 'large-v3': pic('a'), turbo: pic('b', { pipeline: [{ role: 'text', model: 'qwen3.5-9b-vl', sent: 'a cat', answer: 'a ginger cat' }] }), small: { output: null, error: 'out of memory' } }
    server.use(http.get('/api/evals/:name/outputs', () => HttpResponse.json(d)))
    const { user } = setup(<App />, { route: '/evals/transcribe-en?tab=outputs' })
    const grid = await screen.findByLabelText(/Item u1: 2\spictures/)
    expect(within(grid).getByRole('img', { name: "turbo's picture" })).toBeInTheDocument()
    expect(within(grid).getAllByText('2.0 s in all').length).toBeGreaterThan(0) // not blind: each tile names its candidate and time
    await user.click(within(grid).getByRole('button', { name: /Pipeline: 1 steps/ }))
    expect(screen.getByText('a ginger cat')).toBeInTheDocument()
    expect(screen.getByText('Failed: out of memory')).toBeInTheDocument() // a failure stays a card beside the grid
    await user.keyboard('j')
    expect(await screen.findByText('Prompt: a dog')).toBeInTheDocument()
    await user.keyboard('k')
    expect(await screen.findByText('Prompt: a cat')).toBeInTheDocument()
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
    expect(await screen.findByRole('img', { name: 'X' })).toHaveAttribute('src', expect.stringContaining(encodeURIComponent('/d/generated/a.png')))
    expect(screen.getByText('Prompt: two hands')).toBeInTheDocument()
  })

  it('plays the two spoken outputs and shows the text they say', async () => {
    server.use(http.get('/api/evals/:name/pair', () => HttpResponse.json({ pair: { item_id: 'n001', input: { text: 'Two hundred kilometres behind me.' }, has_media: false, reference: { text: 'Two hundred kilometres behind me.' }, x: { path: '/d/generated/tts-a.wav', duration_s: 2 }, y: { path: '/d/generated/tts-b.wav', duration_s: 2.2 }, token: 't' } })))
    setup(<App />, { route: '/evals/transcribe-en?tab=judge' })
    await screen.findByText('Text: Two hundred kilometres behind me.')
    const audio = [...document.querySelectorAll('audio')]
    expect(audio.map(a => a.getAttribute('src'))).toEqual([expect.stringContaining(encodeURIComponent('/d/generated/tts-a.wav')), expect.stringContaining(encodeURIComponent('/d/generated/tts-b.wav'))])
    expect(within(screen.getByRole('region', { name: 'Output X' })).getAllByRole('button', { name: 'Play' }).length).toBeGreaterThan(0)
    expect(screen.getByText('Text: Two hundred kilometres behind me.')).toBeInTheDocument()
    expect(screen.queryByText(/^Reference:/)).not.toBeInTheDocument()
  })
})
