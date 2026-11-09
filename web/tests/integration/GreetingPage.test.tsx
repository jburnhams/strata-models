import { describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import GreetingPage from '../../src/components/GreetingPage'
import { setup, screen } from '../utils/render'
import { server } from '../utils/server'

describe('GreetingPage', () => {
  it('lists existing greetings, adds a new greeting, and clears all', async () => {
    const { user } = setup(<GreetingPage />)
    expect(await screen.findByText('1 greeting recorded')).toBeInTheDocument()
    expect(screen.getByText('Hello, World!')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Name'), 'Grace')
    await user.type(screen.getByLabelText('Custom greeting'), 'Welcome')
    await user.click(screen.getByRole('button', { name: 'Send greeting' }))

    expect(await screen.findByText('Welcome, Grace!')).toBeInTheDocument()
    expect(screen.getByText('2 greetings recorded')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Clear all' }))
    expect(await screen.findByText('No greetings recorded yet.')).toBeInTheDocument()
  })

  it('defaults to World when submitted with an empty name and shows server errors', async () => {
    const { user } = setup(<GreetingPage />)
    await screen.findByText('1 greeting recorded')

    await user.click(screen.getByRole('button', { name: 'Send greeting' }))
    expect(await screen.findByText('2 greetings recorded')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Name'), 'bad')
    await user.click(screen.getByRole('button', { name: 'Send greeting' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('invalid name')

    server.use(http.delete('/api/messages', () => HttpResponse.json({ detail: 'cannot clear' }, { status: 500 })))
    await user.click(screen.getByRole('button', { name: 'Clear all' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('cannot clear')
  })

  it('shows an error alert when loading messages fails', async () => {
    server.use(http.get('/api/messages', () => HttpResponse.json({ detail: 'load failed' }, { status: 500 })))
    setup(<GreetingPage />)
    expect(await screen.findByRole('alert')).toHaveTextContent('load failed')
  })
})
