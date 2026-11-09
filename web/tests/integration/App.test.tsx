import { describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import App from '../../src/App'
import { api } from '../../src/api'
import { Card } from '../../src/components/ui'
import { setup, screen } from '../utils/render'
import { server } from '../utils/server'

const open = (hash: string) => { window.location.hash = hash; return setup(<App />) }

describe('App navigation and header', () => {
  it('shows the server version and navigates between Greetings and Settings', async () => {
    open('#/')
    expect(await screen.findByText('v0.1.0')).toBeInTheDocument()
    expect(await screen.findByText('Hello, World!')).toBeInTheDocument()

    window.location.hash = '#/settings'
    window.dispatchEvent(new HashChangeEvent('hashchange'))
    expect(await screen.findByText('Application settings')).toBeInTheDocument()
  })

  it('renders cleanly when the health check fails and exercises api.hello and Card link', async () => {
    server.use(http.get('/api/health', () => new HttpResponse(null, { status: 500 })))
    open('#/')
    expect(await screen.findByText('Hello, World!')).toBeInTheDocument()
    expect(screen.queryByText('v0.1.0')).toBeNull()

    const g = await api.hello('Ada', 'Welcome')
    expect(g.message).toBe('Welcome, Ada!')

    setup(<Card to="#/settings">Go settings</Card>)
    expect(screen.getByRole('link', { name: 'Go settings' })).toHaveAttribute('href', '#/settings')
  })
})
