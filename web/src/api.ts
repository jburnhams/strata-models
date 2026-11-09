// Typed client for the FastAPI server (src/app/server/app.py). Same-origin.
export interface Health { ok: boolean; version: string }
export interface Settings { data_dir: string; greeting: string }
export interface Greeting { name: string; greeting: string; message: string }
export interface MessageRecord extends Greeting { id: number; created: string }

// Where the app is served from: '' at the server's own root, '/subpath' behind a reverse proxy at that path. Taken from the page's own URL (index.html has relative asset paths: vite.config.ts base './'), so one build works in both places.
export const BASE = typeof document === 'undefined' ? '' : new URL('.', document.baseURI).pathname.replace(/\/$/, '')      // no page (the unit tests run in node): the root
export const withBase = (u: string) => (u.startsWith('/') ? BASE + u : u)

export function helloUrl(name: string, greeting?: string) {
  const params = new URLSearchParams({ name })
  if (greeting) params.set('greeting', greeting)
  return withBase('/api/hello?' + params.toString())
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(withBase(url), init)
  const d = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error((d as { detail?: string }).detail || r.statusText)
  return d as T
}
const send = (method: string, body: unknown): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  health: () => req<Health>('/api/health'),
  settings: () => req<Settings>('/api/settings'),
  saveSettings: (s: Partial<Settings>) => req<Settings>('/api/settings', send('PUT', s)),
  hello: (name: string, greeting?: string) => req<Greeting>(helloUrl(name, greeting).slice(BASE.length)),
  messages: () => req<MessageRecord[]>('/api/messages'),
  addMessage: (name: string, greeting?: string) => req<MessageRecord>('/api/messages', send('POST', greeting ? { name, greeting } : { name })),
  clearMessages: () => req<{ cleared: number }>('/api/messages', { method: 'DELETE' }),
}
