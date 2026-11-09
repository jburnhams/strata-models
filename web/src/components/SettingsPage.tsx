// One question: "Where do the weights live, and how much VRAM is kept free?" Saved settings take effect at once (the reserve) or for the next load (the folder).
import { useState, type FormEvent } from 'react'
import { AsyncButton, ColorSchemeChoice, DataView, EmptyState, Field, Input, PageHeader, Skeleton, useAsyncAction, useLoad } from 'strata-react'
import { useQueryClient } from '@tanstack/react-query'
import { api, type Settings } from '../api'

function Form({ s }: { s: Settings }) {
  const qc = useQueryClient()
  const [models, setModels] = useState(s.models_dir), [reserve, setReserve] = useState(String(s.vram_reserve_gb))
  const save = useAsyncAction(async () => { await api.saveSettings({ models_dir: models.trim(), vram_reserve_gb: Number(reserve) }); await qc.invalidateQueries({ queryKey: ['settings'] }) })
  const onSubmit = (e: FormEvent) => { e.preventDefault(); save.run() }
  return (
    <form onSubmit={onSubmit} className="grid max-w-xl gap-4">
      <Field label="Models folder" hint="Every runtime downloads weights here and reads them from here. Applies to the next model loaded."><Input value={models} onChange={e => setModels(e.target.value)} /></Field>
      <Field label="VRAM kept free (GB)" hint="Left for the desktop and for programs that use the GPU outside strata-models."><Input inputMode="decimal" value={reserve} onChange={e => setReserve(e.target.value)} /></Field>
      <p className="type-body-sm text-muted-foreground">Service files: <span className="font-mono">{s.data_dir}</span></p>
      {save.error && <p role="alert" className="text-danger type-body-sm">{save.error}</p>}
      <div><AsyncButton type="submit" variant="default" phase={save.phase} error={save.error} labels={{ idle: 'Save settings', pending: 'Saving…', done: 'Saved' }} /></div>
    </form>
  )
}

export default function SettingsPage() {
  const load = useLoad({ queryKey: ['settings'], queryFn: api.settings })
  return (
    <div className="grid gap-8">
      <PageHeader title="Settings" />
      <DataView state={load} isEmpty={() => false} skeleton={<Skeleton className="h-40 w-full max-w-xl" />} empty={<EmptyState title="No settings" />}>
        {s => <Form s={s} />}
      </DataView>
      <section aria-labelledby="look" className="grid gap-3"><h2 id="look" className="type-heading">Appearance</h2><ColorSchemeChoice /></section>
    </div>
  )
}
