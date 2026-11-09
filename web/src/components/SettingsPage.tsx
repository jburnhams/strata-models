import { useState, type FormEvent } from 'react'
import { api } from '../api'
import { href, useLoad } from '../route'
import { Card, Crumbs, Err, btn, input } from './ui'

export default function SettingsPage() {
  const { data, error, reload } = useLoad(() => api.settings(), [])
  const [greeting, setGreeting] = useState('')
  const [dataDir, setDataDir] = useState('')
  const [saved, setSaved] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function onSubmit(e: FormEvent) {
    e.preventDefault(); setErr(null); setSaved(false)
    try {
      await api.saveSettings({
        greeting: greeting.trim() || data?.greeting || 'Hello',
        data_dir: dataDir.trim() || data?.data_dir,
      })
      setSaved(true); reload()
    } catch (x) { setErr(String((x as Error).message || x)) }
  }

  return (
    <div>
      <Crumbs items={[{ label: 'Home', to: href.greeting() }, { label: 'Settings' }]} />
      <Card>
        <h2 className="text-base font-medium mb-3">Application settings</h2>
        {data && (
          <div className="text-sm text-stone-500 mb-3 space-y-1">
            <div>Current greeting: <span className="font-mono text-stone-800 dark:text-stone-200">{data.greeting}</span></div>
            <div>Current data folder: <span className="font-mono text-stone-800 dark:text-stone-200">{data.data_dir}</span></div>
          </div>
        )}
        <form onSubmit={onSubmit} className="space-y-3">
          <div className="flex flex-col gap-1">
            <label htmlFor="setting-greeting" className="text-sm">Default greeting</label>
            <input id="setting-greeting" className={input} placeholder={data?.greeting || 'Hello'} value={greeting} onChange={e => setGreeting(e.target.value)} />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="setting-datadir" className="text-sm">Data folder</label>
            <input id="setting-datadir" className={input} placeholder={data?.data_dir || '/path/to/data'} value={dataDir} onChange={e => setDataDir(e.target.value)} />
          </div>
          <div className="flex items-center gap-3">
            <button type="submit" className={btn}>Save settings</button>
            {saved && <span className="text-sm text-green-600 dark:text-green-400">Saved</span>}
          </div>
        </form>
        <Err msg={err || error} />
      </Card>
    </div>
  )
}
