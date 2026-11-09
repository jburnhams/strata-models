import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { MessageRecord, Settings } from '../../src/api'

export const defaultSettings: Settings = { data_dir: '/tmp/appdata', greeting: 'Hello' }
export const initialMessages: MessageRecord[] = [
  { id: 1, name: 'World', greeting: 'Hello', message: 'Hello, World!', created: '2026-10-29T12:00:00Z' },
]

export const mock = {
  settings: { ...defaultSettings },
  messages: [...initialMessages],
}

export const resetMock = () => {
  mock.settings = { ...defaultSettings }
  mock.messages = [...initialMessages]
}

// Every endpoint the app calls has a handler here; one without a handler fails the test (setup.ts: onUnhandledFrame 'error').
export const handlers = [
  http.get('/api/health', () => HttpResponse.json({ ok: true, version: '0.1.0' })),
  http.get('/api/settings', () => HttpResponse.json(mock.settings)),
  http.put('/api/settings', async ({ request }) => {
    const body = (await request.json()) as Partial<Settings>
    if (body.data_dir && !body.data_dir.startsWith('/')) return HttpResponse.json({ detail: 'the data folder must be an absolute path' }, { status: 400 })
    mock.settings = { ...mock.settings, ...body }
    return HttpResponse.json(mock.settings)
  }),
  http.get('/api/hello', ({ request }) => {
    const u = new URL(request.url)
    const name = u.searchParams.get('name') || 'World'
    const greeting = u.searchParams.get('greeting') || mock.settings.greeting
    return HttpResponse.json({ name, greeting, message: `${greeting}, ${name}!` })
  }),
  http.get('/api/messages', () => HttpResponse.json(mock.messages)),
  http.post('/api/messages', async ({ request }) => {
    const b = (await request.json()) as { name?: string; greeting?: string }
    const name = (b.name || 'World').trim()
    if (name === 'bad') return HttpResponse.json({ detail: 'invalid name' }, { status: 400 })
    const greeting = (b.greeting || mock.settings.greeting).trim()
    const entry: MessageRecord = { id: mock.messages.length + 1, name, greeting, message: `${greeting}, ${name}!`, created: '2026-10-29T12:01:00Z' }
    mock.messages = [entry, ...mock.messages]
    return HttpResponse.json(entry)
  }),
  http.delete('/api/messages', () => {
    const cleared = mock.messages.length
    mock.messages = []
    return HttpResponse.json({ cleared })
  }),
]

export const server = setupServer(...handlers)
