// One question: "Where do the weights live, how much VRAM is kept free, how many jobs run at once, how long an idle model stays loaded, how big a model gets the card to itself, and which faces are too small or doubtful to match?" Saved settings take effect at once (the reserve, the job counts, the whole-card size, the idle time, the face bar) or for the next load (the folder).
import { useState, type FormEvent } from 'react'
import { Alert, AsyncButton, ColorSchemeChoice, DataView, EmptyState, Field, Input, PageHeader, Skeleton, useAsyncAction, useLoad } from 'strata-react'
import { useQueryClient } from '@tanstack/react-query'
import { api, type Settings } from '../api'

function Form({ s }: { s: Settings }) {
  const qc = useQueryClient()
  const [models, setModels] = useState(s.models_dir), [reserve, setReserve] = useState(String(s.vram_reserve_gb))
  const [gpuJobs, setGpuJobs] = useState(String(s.gpu_jobs)), [cpuJobs, setCpuJobs] = useState(String(s.cpu_jobs)), [whole, setWhole] = useState(String(s.whole_card_gb)), [idle, setIdle] = useState(String(s.idle_unload_min))
  const [faceScore, setFaceScore] = useState(String(s.faces_min_score)), [facePx, setFacePx] = useState(String(s.faces_min_px))
  const save = useAsyncAction(async () => { await api.saveSettings({ models_dir: models.trim(), vram_reserve_gb: Number(reserve), gpu_jobs: Number(gpuJobs), cpu_jobs: Number(cpuJobs), whole_card_gb: Number(whole), idle_unload_min: Number(idle), faces_min_score: Number(faceScore), faces_min_px: Number(facePx) }); await qc.invalidateQueries({ queryKey: ['settings'] }) })
  const onSubmit = (e: FormEvent) => { e.preventDefault(); save.run() }
  return (
    <form onSubmit={onSubmit} className="grid max-w-xl gap-4">
      <Field label="Models folder" hint="Every runtime downloads weights here and reads them from here. Applies to the next model loaded."><Input value={models} onChange={e => setModels(e.target.value)} /></Field>
      <Field label="VRAM kept free (GB)" hint="Left for the desktop and for programs that use the GPU outside strata-models."><Input inputMode="decimal" value={reserve} onChange={e => setReserve(e.target.value)} /></Field>
      <Field label="GPU jobs at once" hint="Jobs on different models share the card when they fit in its memory together. 1 runs one at a time."><Input inputMode="numeric" value={gpuJobs} onChange={e => setGpuJobs(e.target.value)} /></Field>
      <Field label="Whole card above (GB)" hint="A model or lease that needs more than this waits for the card to be empty, then runs alone on it with the reserve ignored: a way to try models too big to budget for, at their own risk."><Input inputMode="decimal" value={whole} onChange={e => setWhole(e.target.value)} /></Field>
      <Field label="Jobs without the GPU at once" hint="Recorded outputs, CPU models and cloud calls: they run beside the GPU's jobs and never wait for the card."><Input inputMode="numeric" value={cpuJobs} onChange={e => setCpuJobs(e.target.value)} /></Field>
      <Field label="Unload idle models after (minutes)" hint="A model no job has used for this long gives back its GPU memory and RAM; the next job loads it again (3 to 10 s for most). 0 keeps models until the room is needed."><Input inputMode="decimal" value={idle} onChange={e => setIdle(e.target.value)} /></Field>
      <Field label="Faces: lowest detector score" hint="Faces the detector is less sure of are kept, but never searched, grouped, suggested or proposed, for every app. 0 to 1; 0 turns it off."><Input inputMode="decimal" value={faceScore} onChange={e => setFaceScore(e.target.value)} /></Field>
      <Field label="Faces: smallest size (pixels)" hint="The shorter side of a face's box in its picture. Smaller faces are kept but left out the same way. 0 turns it off."><Input inputMode="decimal" value={facePx} onChange={e => setFacePx(e.target.value)} /></Field>
      <p className="type-body-sm text-muted-foreground">Service files: <span className="font-mono">{s.data_dir}</span></p>
      {save.error && <Alert tone="danger" announce title="Not saved">{save.error}</Alert>}
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
