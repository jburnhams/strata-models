// One question: "Make me a picture of this." As simple as it can be: a prompt, Generate, and the picture, with sensible defaults (the role's
// model, FLUX.3 Klein; 1024x1024; a random seed; the model's own steps). Everything else is under Advanced, closed by default. Reference
// pictures (for edits) are one optional button. Past pictures follow as a gallery, newest first.
import { useState, type FormEvent } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, AsyncButton, FileButton, Button, Disclosure, Field, Input, NativeSelect, PageHeader, Textarea, Thumbnail, DataView, EmptyState, Skeleton, format, useAsyncAction, useLoad } from 'strata-react'
import { api, imagesApi, type GeneratedImage, type Job } from '../api'

export const SIZES = [
  { value: '1024x1024', label: 'Square, 1024 × 1024' },
  { value: '1216x832', label: 'Landscape, 1216 × 832' },
  { value: '832x1216', label: 'Portrait, 832 × 1216' },
  { value: '1344x768', label: 'Wide, 1344 × 768' },
  { value: '768x768', label: 'Small square, 768 × 768 (faster)' },
]

/** The params of a request from the form: only what was set, so the model's own defaults apply to the rest. */
export function paramsOf(f: { prompt: string; size: string; seed: string; steps: string; guidance: string; refs: string[] }) {
  const [w, h] = f.size.split('x').map(Number)
  const p: Record<string, unknown> = { prompt: f.prompt.trim(), width: w, height: h }
  if (f.seed.trim()) p.seed = Number(f.seed)
  if (f.steps.trim()) p.steps = Number(f.steps)
  if (f.guidance.trim()) p.cfg_scale = Number(f.guidance)
  if (f.refs.length) p.ref_images = f.refs
  return p
}

function Result({ job }: { job: Job }) {
  const r = job.result as GeneratedImage
  return (
    <figure className="grid gap-2">
      <img src={imagesApi.fileUrl(r.path)} alt={String((job.params as { prompt?: string })?.prompt ?? 'Generated picture')} width={r.width} height={r.height} className="h-auto max-w-full rounded-surface" />
      <figcaption className="type-caption text-muted-foreground">{job.model} · {r.width} × {r.height} · seed {r.seed} · {format.decimal(r.seconds)} s{r.peak_vram_mb ? ` · ${format.decimal(r.peak_vram_mb / 1024)} GB VRAM at peak` : ''}</figcaption>
    </figure>
  )
}

function Gallery({ refresh }: { refresh: number }) {
  const load = useLoad({ queryKey: ['images', refresh], queryFn: () => imagesApi.list() })
  return (
    <section aria-labelledby="gallery" className="grid gap-3">
      <h2 id="gallery" className="type-heading">Earlier pictures</h2>
      <DataView state={load} isEmpty={p => p.items.length === 0} skeleton={<Skeleton className="h-40 w-full" />} empty={<EmptyState title="No pictures yet">Write a prompt above and press Generate.</EmptyState>}>
        {p => (
          <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {p.items.map(i => (
              <li key={i.id} className="grid gap-1">
                <a href={imagesApi.fileUrl(i.path)} target="_blank" rel="noreferrer"><Thumbnail src={imagesApi.fileUrl(i.path)} alt={i.prompt} aspect={`${i.width}/${i.height}`} missing="Picture not found" /></a>
                <span className="type-caption text-muted-foreground line-clamp-2">{i.prompt}</span>
              </li>
            ))}
          </ul>
        )}
      </DataView>
    </section>
  )
}

export default function GeneratePage() {
  const qc = useQueryClient()
  const reg = useLoad({ queryKey: ['registry'], queryFn: api.registry })
  const models = reg.status === 'ready' ? reg.data.models.filter(m => m.roles.includes('image-generate')) : []
  const role = reg.status === 'ready' ? reg.data.roles.find(r => r.key === 'image-generate') : undefined
  const [prompt, setPrompt] = useState(''), [model, setModel] = useState(''), [size, setSize] = useState('1024x1024'), [seed, setSeed] = useState(''), [steps, setSteps] = useState(''), [guidance, setGuidance] = useState('')
  const [refs, setRefs] = useState<string[]>([]), [job, setJob] = useState<Job | null>(null), [n, setN] = useState(0)
  const gen = useAsyncAction(async () => { const j = await imagesApi.generate(model || undefined, paramsOf({ prompt, size, seed, steps, guidance, refs }) as never); setJob(j); setN(k => k + 1); await qc.invalidateQueries({ queryKey: ['images'] }) })
  const upload = useAsyncAction(async (files: File[]) => { const out: string[] = []; for (const f of files.slice(0, 10 - refs.length)) out.push((await imagesApi.upload(f)).path); setRefs(r => [...r, ...out]) })
  const onSubmit = (e: FormEvent) => { e.preventDefault(); if (prompt.trim()) gen.run() }
  const chosen = models.find(m => m.key === (model || role?.model))
  return (
    <div className="grid gap-8">
      <PageHeader title="Generate" scope="Pictures from a prompt, made on this machine's GPU" />
      <form onSubmit={onSubmit} className="grid max-w-3xl gap-4">
        <Field label="What should the picture show?"><Textarea rows={3} value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="A hare running through frosty grass at dawn, low sun, 35 mm photo" /></Field>
        <div className="flex flex-wrap items-center gap-3">
          <AsyncButton type="submit" variant="default" disabled={!prompt.trim()} phase={gen.phase} error={gen.error} labels={{ idle: 'Generate', pending: 'Generating…', done: 'Generated' }} />
          <FileButton accept={['image/png', 'image/jpeg', 'image/webp']} multiple disabled={refs.length >= 10} onFiles={f => upload.run(f)}>{refs.length ? `${format.count(refs.length, 'reference picture')} added` : 'Add a reference picture'}</FileButton>
          {refs.length > 0 && <Button size="sm" variant="ghost" type="button" onClick={() => setRefs([])}>Clear references</Button>}
        </div>
        {gen.error && <Alert tone="danger" announce>{gen.error}</Alert>}
        {upload.error && <Alert tone="danger" announce title="Reference not added">{upload.error}</Alert>}
        <Disclosure summary="Advanced">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Model" hint={chosen ? chosen.licence : undefined}><NativeSelect value={model} onChange={e => setModel(e.target.value)}><option value="">{role ? `Default (${role.model})` : 'Default'}</option>{models.map(m => <option key={m.key} value={m.key}>{m.name || m.key}</option>)}</NativeSelect></Field>
            <Field label="Size and shape"><NativeSelect value={size} onChange={e => setSize(e.target.value)}>{SIZES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}</NativeSelect></Field>
            <Field label="Seed" hint="Empty: a new picture every time. The same seed and prompt give the same picture."><Input inputMode="numeric" value={seed} onChange={e => setSeed(e.target.value)} /></Field>
            <Field label="Steps" hint={`Empty: the model's own (${String((chosen?.options as { steps?: number } | undefined)?.steps ?? '–')}). More is slower, rarely better.`}><Input inputMode="numeric" value={steps} onChange={e => setSteps(e.target.value)} /></Field>
            <Field label="Guidance" hint="Empty: the model's own. How closely to follow the prompt."><Input inputMode="decimal" value={guidance} onChange={e => setGuidance(e.target.value)} /></Field>
          </div>
        </Disclosure>
      </form>
      {job && <Result job={job} />}
      <Gallery refresh={n} />
    </div>
  )
}
