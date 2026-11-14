import { describe, expect, it } from 'vitest'
import App from '../../src/App'
import { mock, server, http, HttpResponse, spendFixtures } from '../utils/server'
import { screen, setup, waitFor, within } from '../utils/render'

describe('Cloud APIs', () => {
  it('says which paid key is nearest its budget and shows every key', async () => {
    setup(<App />, { route: '/cloud' })
    expect(await screen.findByText(/^Claude API: \$62\.50 of \$100\.00 spent, \$37\.50 left until/, { selector: 'p, span, div' })).toBeInTheDocument()
    expect(screen.getByRole('meter', { name: 'Claude API' })).toBeInTheDocument()
    expect(screen.getByText('No budget set: every call is refused')).toBeInTheDocument()
    expect(screen.getByText('Free, rate-limited: 5 calls so far.')).toBeInTheDocument()
    expect(screen.getByText('No key on this machine yet.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Cloud' })).toBeInTheDocument()
  })

  it('says when no paid key has a budget, and a held reservation and one-pot budget', async () => {
    const s = spendFixtures.spend()
    mock.spend = { ...s, providers: [{ ...s.providers[0], budget: null, warn: 1, used_share: null }, s.providers[1], s.providers[3]] }
    const { unmount } = setup(<App />, { route: '/cloud' })
    expect(await screen.findByText('No budget is set for Claude API or Google AI Studio (paid key): their calls are refused until one is.', { selector: 'p, span, div' })).toBeInTheDocument()
    unmount()
    mock.spend = { ...s, providers: [{ ...s.providers[0], next_rollover: null, reserved: 0.25 }] }
    setup(<App />, { route: '/cloud' })
    expect(await screen.findByText(/one budget, no rollover · 1,200 calls · \$0\.25 held for calls under way/)).toBeInTheDocument()
  })

  it('saves a budget and its rollover date, and says why a refusal failed', async () => {
    const { user } = setup(<App />, { route: '/cloud' })
    const claude = await screen.findByRole('group', { name: 'Claude API' })
    const budget = within(claude).getByLabelText('Budget (US$)')
    await user.clear(budget)
    await user.type(budget, '50')
    await user.click(screen.getByRole('button', { name: 'Save budgets' }))
    await waitFor(() => expect(mock.spend.providers[0].budget).toBe(50))
    expect(mock.spend.providers[0].rollover).toBe('2027-01-08')
    server.use(http.put('/api/settings', () => HttpResponse.json({ detail: 'claude_rollover must be a date, YYYY-MM-DD' }, { status: 400 })))
    await user.click(screen.getByRole('button', { name: 'Save budgets' }))
    expect(await screen.findByText('claude_rollover must be a date, YYYY-MM-DD')).toBeInTheDocument()
  })

  it('charts spend and calls, breaks them down and filters by key, model, app, outcome and period', async () => {
    const { user } = setup(<App />, { route: '/cloud' })
    expect(await screen.findByRole('img', { name: 'Spend per day' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Calls per day' })).toBeInTheDocument()
    const table = await screen.findByRole('grid', { name: 'Spend by key, model and app' })
    expect(within(table).getByText('$1.50')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Model'), 'claude-haiku-5-5')
    await user.selectOptions(screen.getByLabelText('App'), 'evals')
    await user.selectOptions(screen.getByLabelText('Period'), '1')
    expect(await screen.findByRole('img', { name: 'Calls per hour' })).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Period'), '30')
    await user.selectOptions(screen.getByLabelText('Key'), 'nim')
    await user.selectOptions(screen.getByLabelText('Outcome'), 'false')
    const log = await screen.findByRole('grid', { name: 'Request log' })
    await waitFor(() => expect(within(log).getAllByRole('row')).toHaveLength(2))
    expect(within(log).getByText('Failed')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Outcome'), 'true')
    expect(await screen.findByText('No calls match')).toBeInTheDocument()
  })

  it('opens a call to show what was sent and what came back', async () => {
    const { user } = setup(<App />, { route: '/cloud' })
    const log = await screen.findByRole('grid', { name: 'Request log' })
    await user.click(within(log).getByRole('button', { name: /call #3/ }))
    expect(await screen.findByText('Describe the photo.')).toBeInTheDocument()
    expect(screen.getByText('Be brief.')).toBeInTheDocument()
    expect(screen.getByText('{"scene": "a beach"}')).toBeInTheDocument()
    expect(screen.getByText('/p/a.jpg')).toBeInTheDocument()
    expect(screen.getByText('evals, job #7')).toBeInTheDocument()
    await user.click(within(log).getByRole('button', { name: /call #2/ }))
    expect(await screen.findByText('HTTP 429: slow down')).toBeInTheDocument()
    expect(screen.getByText('No answer.')).toBeInTheDocument()
  })

  it('loads older calls, and shows a failure to load them', async () => {
    mock.callPages = 2
    const { user, unmount } = setup(<App />, { route: '/cloud' })
    await user.click(await screen.findByRole('button', { name: 'Show older calls' }))
    expect(await screen.findByText('claude-sonnet-5-5')).toBeInTheDocument()
    unmount()
    server.use(http.get('/api/spend/calls', ({ request }) => new URL(request.url).searchParams.get('before') ? HttpResponse.json({ detail: 'boom' }, { status: 500 }) : HttpResponse.json({ calls: spendFixtures.calls(), next: 2 })))
    const r = setup(<App />, { route: '/cloud' })
    await r.user.click(await screen.findByRole('button', { name: 'Show older calls' }))
    expect(await screen.findByRole('alert')).toBeInTheDocument()
  })
})
