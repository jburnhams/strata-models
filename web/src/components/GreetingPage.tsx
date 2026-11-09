import { useState, type FormEvent } from 'react'
import { api, type MessageRecord } from '../api'
import { useLoad } from '../route'
import { Card, Err, btn, btnQuiet, input } from './ui'

export function describeCount(n: number) {
  return n === 0 ? 'No greetings recorded yet.' : `${n} greeting${n === 1 ? '' : 's'} recorded`
}

export default function GreetingPage() {
  const { data: messages, error, reload } = useLoad(() => api.messages(), [])
  const [name, setName] = useState('')
  const [greeting, setGreeting] = useState('')
  const [err, setErr] = useState<string | null>(null)

  async function onSubmit(e: FormEvent) {
    e.preventDefault(); setErr(null)
    try {
      await api.addMessage(name.trim() || 'World', greeting.trim() || undefined)
      setName(''); setGreeting(''); reload()
    } catch (x) { setErr(String((x as Error).message || x)) }
  }

  async function onClear() {
    setErr(null)
    try { await api.clearMessages(); reload() } catch (x) { setErr(String((x as Error).message || x)) }
  }

  return (
    <div>
      <Card className="mb-5">
        <h2 className="text-base font-medium mb-2">Say hello</h2>
        <form onSubmit={onSubmit} className="flex flex-wrap gap-2 items-center">
          <input aria-label="Name" className={`${input} flex-1 min-w-44`} placeholder="Name (default: World)" value={name} onChange={e => setName(e.target.value)} />
          <input aria-label="Custom greeting" className={`${input} w-44`} placeholder="Greeting (optional)" value={greeting} onChange={e => setGreeting(e.target.value)} />
          <button type="submit" className={btn}>Send greeting</button>
        </form>
        <Err msg={err || error} />
      </Card>

      <div className="flex items-center justify-between mb-2">
        <h2 className="text-base font-medium">{messages ? describeCount(messages.length) : 'Loading…'}</h2>
        {messages && messages.length > 0 && (
          <button type="button" onClick={onClear} className={btnQuiet}>Clear all</button>
        )}
      </div>

      <div className="space-y-2">
        {messages?.map((m: MessageRecord) => (
          <Card key={m.id} className="flex items-baseline justify-between gap-2">
            <span className="font-medium">{m.message}</span>
            <span className="text-xs text-stone-500 font-mono">{m.created}</span>
          </Card>
        ))}
      </div>
    </div>
  )
}
