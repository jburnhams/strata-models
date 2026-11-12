// One question: "What should serve this role, and how sure are we?" Finding: the decision (change to X, or keep the baseline) with its reason.
// Then: every candidate's results with their intervals, the paired comparisons with the baseline, the human judgments, and the tabs to judge
// pairs blind and to read the standard write-up. Running fills in missing outputs through the GPU queue.
import { useParams } from 'react-router'
import PhrasePlayer from './PhrasePlayer'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, AsyncButton, Button, Compare, CompareGrid, CompareVideo, DataTable, DataView, Disclosure, EmptyState, Field, KeysLegend, MediaPlayer, NativeSelect, PageHeader, SPEECH_RATES, Skeleton, Status, Tabs, TableSkeleton, format, useAsyncAction, useHotkeys, useLoad, type Candidate, type Column, type CompareMode, type Region, type Shortcut, type VideoLoop, type VideoMedia, type VideoMode } from 'strata-react'
import { evalsApi, imagesApi, type Analysis, type JudgeSummary, type Pair, type Share, type Step } from '../api'
import { VideoCompare, videoOf, type VideoSide } from './VideoCompare'
import { VERDICT_TONE, VERDICT_WORD, diff, outputText, pvalue, reading, value } from '../evalfmt'
import { useEffect, useRef, useState, type RefObject } from 'react'

function finding(a: Analysis) {
  const d = a.decision
  if (a.dataset.compared === 0) return 'No outputs yet: run the experiment.'
  return d.change ? `Change to ${d.recommendation}: ${d.per_candidate[d.recommendation].why}.` : `Keep ${a.experiment.baseline}: no candidate is clearly better on ${a.experiment.primary} by enough.`
}

function Results({ a }: { a: Analysis }) {
  const metrics = a.experiment.metrics
  type Row = { name: string }
  const rows: Row[] = a.experiment.candidates.map(c => ({ name: c.name }))
  const columns: Column<Row>[] = [
    { key: 'name', header: 'Candidate', cell: r => <span className="grid"><span className="font-mono">{r.name}</span>{r.name === a.experiment.baseline && <span className="type-caption text-muted-foreground">baseline</span>}</span> },
    { key: 'items', header: 'Items', numeric: true, cell: r => { const c = a.candidates[r.name]; return c?.compared ?? '–' } },
    ...metrics.map(m => ({ key: m, header: m, numeric: true, cell: (r: Row) => { const v = a.candidates[r.name]?.metrics[m]; const info = a.metrics[m]; return v ? <span className="grid"><span>{value(info, v.value)}</span><span className="type-caption text-muted-foreground">{value(info, v.low)} to {value(info, v.high)}</span></span> : '–' } })),
    { key: 'verdict', header: 'Decision', cell: r => { const v = a.decision.per_candidate[r.name]; return v ? <Status tone={VERDICT_TONE[v.verdict]}>{VERDICT_WORD[v.verdict]}</Status> : <span className="text-muted-foreground">{a.dataset.not_run?.includes(r.name) ? 'Not run yet' : 'Baseline'}</span> } },
  ]
  type CRow = { key: string; cand: string; metric: string }
  const crows: CRow[] = Object.entries(a.comparisons).flatMap(([cand, comp]) => Object.keys(comp).map(metric => ({ key: `${cand}/${metric}`, cand, metric })))
  const ccols: Column<CRow>[] = [
    { key: 'cand', header: 'Candidate', cell: r => <span className="font-mono">{r.cand}</span> },
    { key: 'metric', header: 'Metric', cell: r => r.metric },
    { key: 'diff', header: 'Difference', numeric: true, cell: r => { const c = a.comparisons[r.cand][r.metric]; const m = a.metrics[r.metric]; return <span className="grid"><span>{diff(m, c.diff)}</span><span className="type-caption text-muted-foreground">{diff(m, c.low)} to {diff(m, c.high)}</span></span> } },
    { key: 'p', header: 'p', numeric: true, cell: r => { const c = a.comparisons[r.cand][r.metric]; return pvalue(c.p) + (c.p_holm !== undefined ? ` (Holm ${pvalue(c.p_holm)})` : '') } },
    { key: 'reading', header: 'Reading', cell: r => { const c = a.comparisons[r.cand][r.metric]; return <Status tone={c.improves === true ? 'success' : c.improves === false ? 'danger' : 'neutral'}>{reading(c)}</Status> } },
  ]
  return (
    <div className="grid gap-8">
      <section aria-labelledby="why" className="grid gap-2">
        <h2 id="why" className="type-heading">Each candidate</h2>
        <ul className="grid gap-1 type-body">{Object.entries(a.decision.per_candidate).map(([n, v]) => <li key={n}><span className="font-mono">{n}</span>: {VERDICT_WORD[v.verdict].toLowerCase()}, {v.why}.</li>)}</ul>
      </section>
      <DataTable caption="Results on each candidate's items (value, and its 95 % interval)" columns={columns} rows={rows} rowKey={r => r.name} />
      {crows.length > 0 && <DataTable caption={`Compared with ${a.experiment.baseline} on the same items (candidate minus baseline)`} columns={ccols} rows={crows} rowKey={r => r.key} />}
      {a.judgments && (
        <section aria-labelledby="judged" className="grid gap-2">
          <h2 id="judged" className="type-heading">Human judgments</h2>
          <p className="type-body">{format.count(a.judgments.n, 'blind judgment')}. Bradley–Terry strength, where 1 is average:</p>
          <ul className="grid gap-1 type-body">{Object.entries(a.judgments.strength).sort((x, y) => y[1][0] - x[1][0]).map(([n, s]) => <li key={n}><span className="font-mono">{n}</span>: {s[0].toFixed(2)} ({s[1].toFixed(2)} to {s[2].toFixed(2)})</li>)}</ul>
        </section>
      )}
      {a.judges && Object.keys(a.judges).length > 0 && <ModelJudges judges={a.judges} />}
      <p className="type-body-sm text-muted-foreground">Dataset {a.dataset.name}, version <span className="font-mono">{a.dataset.version}</span>: the baseline has {a.dataset.compared} of {a.dataset.n} items; each candidate is compared with it on the items both have (Items column: a sample has fewer, and wider intervals){a.dataset.not_run?.length ? ` (of those that have run: ${a.dataset.not_run.join(', ')} not yet)` : ''}. Intervals are 95 %; the Write-up tab says how each is computed.</p>
    </div>
  )
}

const share = (s: Share) => (!s.n || s.value === null ? 'n/a' : `${Math.round(s.value * 100)} % (${Math.round((s.low ?? 0) * 100)} to ${Math.round((s.high ?? 0) * 100)}) of ${s.n}`)

/** Model judges: their verdicts on the pairs people judged, and how often they gave the person's answer. */
function ModelJudges({ judges }: { judges: Record<string, JudgeSummary> }) {
  type JRow = JudgeSummary & { name: string }
  const cols: Column<JRow>[] = [
    { key: 'name', header: 'Judge', cell: r => <span className="grid"><span className="font-mono">{r.name}</span><span className="type-caption text-muted-foreground">{r.kind}, {r.model ?? '?'}{r.via !== 'queue' ? ` via ${r.via}` : ''}</span></span> },
    { key: 'judged', header: 'Verdicts', numeric: true, cell: r => `${r.judged} (${r.with_people} with people)` },
    { key: 'agree', header: 'Agrees with people', numeric: true, cell: r => share(r.agree) },
    { key: 'decided', header: 'Same winner', numeric: true, cell: r => share(r.decided) },
    { key: 'kappa', header: 'Kappa', numeric: true, cell: r => (r.kappa === null ? 'n/a' : r.kappa.toFixed(2)) },
    { key: 'consistent', header: 'Same either way round', numeric: true, cell: r => share(r.consistent) },
  ]
  return (
    <section aria-labelledby="model-judges" className="grid gap-2">
      <h2 id="model-judges" className="type-heading">Model judges</h2>
      <p className="type-body-sm text-muted-foreground">Verdicts by models on the same blind pairs, kept beside people's and never counted as theirs. Agrees: the person's answer, tie included. Same winner: of pairs where both picked one. Kappa: agreement beyond chance (1 perfect, 0 chance).</p>
      <DataTable caption="Model judges against people" columns={cols} rows={Object.entries(judges).map(([name, j]) => ({ ...j, name }))} rowKey={r => r.name} />
    </section>
  )
}

/** What the model judges said about the pair a person just judged (shown after, so they cannot sway the person). */
function PairVerdicts({ name, last }: { name: string; last: Pick<Last, 'item_id' | 'x' | 'y'> }) {
  const load = useLoad({ queryKey: ['verdicts', name, last.item_id], queryFn: () => evalsApi.verdicts(name, last.item_id), refetchInterval: 4000 })
  if (load.status !== 'ready') return null
  const mine = load.data.judgments.filter(v => v.judge && v.current && [v.a, v.b].includes(last.x) && [v.a, v.b].includes(last.y))
  if (!mine.length) return null
  return <ul className="mt-2 grid gap-1 type-body-sm">{mine.map(v => <li key={`${v.rater}-${v.at}`}><span className="font-mono">{v.rater}</span>: {v.winner === 'tie' ? 'about the same' : `chose ${v.winner}`}{v.detail?.consistent === false ? ' (changed its mind when X and Y were swapped)' : ''}{v.note ? `. ${v.note}` : ''}</li>)}</ul>
}

/** The picture of an image job's output, if it is one. */
export const imagePath = (o: unknown) => (o && typeof o === 'object' && typeof (o as { path?: unknown }).path === 'string' && /\.(png|jpe?g|webp)$/i.test((o as { path: string }).path) ? (o as { path: string }).path : null)

/** The sound of a speech job's output (speak), if it is one. */
export const audioPath = (o: unknown) => (o && typeof o === 'object' && typeof (o as { path?: unknown }).path === 'string' && /\.(wav|mp3|flac|ogg|m4a)$/i.test((o as { path: string }).path) ? (o as { path: string }).path : null)

/** A picture's natural size when the job reported it, so the comparison frame has its shape and 1:1 is exact before it loads. */
export const imageSize = (o: unknown): [number, number] | undefined => {
  const r = o as { width?: unknown; height?: unknown } | null
  return r && typeof r.width === 'number' && typeof r.height === 'number' ? [r.width, r.height] : undefined
}

/** A video side from an output: its whole frame and region crops as served files. */
export const videoSide = (id: string, label: string, o: unknown, note?: string): VideoSide | null => {
  const v = videoOf(o)
  return v ? { id, label, note, whole: imagesApi.fileUrl(v.path), regions: Object.fromEntries(v.crops.map(c => [c.id, imagesApi.fileUrl(c.path)])), fps: v.fps } : null
}

/** The tape itself, from the item's reference (a dataset builder's playable copy of the source and its region crops), if it has one. */
export const sourceSide = (name: string, itemId: string, reference: Record<string, unknown> | null): VideoSide | null => {
  const preview = reference?.preview
  if (typeof preview !== 'string') return null
  const crops = Array.isArray(reference?.crops) ? (reference.crops as { id?: unknown; path?: unknown }[]) : []
  return { id: 'source', label: 'Source', note: 'the tape, deinterlaced, enlarged nearest-neighbour', whole: evalsApi.mediaUrl(name, itemId, preview), regions: Object.fromEntries(crops.filter(c => typeof c.id === 'string' && typeof c.path === 'string').map(c => [c.id as string, evalsApi.mediaUrl(name, itemId, c.path as string)])) }
}

/** Region labels by id, from any output's or the reference's crops. */
export const regionLabels = (...os: unknown[]) => {
  const out: Record<string, string> = {}
  for (const o of os) for (const c of (o && typeof o === 'object' && Array.isArray((o as { crops?: unknown }).crops) ? (o as { crops: { id?: unknown; label?: unknown }[] }).crops : [])) if (typeof c.id === 'string' && typeof c.label === 'string') out[c.id] = c.label
  return out
}

/** An item's regions (boxes in the source's pixels, `input.regions`) as boxes in the output's frame, which is the source's `input.crop`
 * (the worker crops, then resizes): what `CompareVideo` enlarges. Without a crop the source's size is not known here, so none. */
export const regionsOf = (input: Record<string, unknown>): Region[] => {
  const crop = input.crop; const regions = input.regions
  if (!Array.isArray(crop) || crop.length !== 4 || !crop.every(v => typeof v === 'number') || !Array.isArray(regions)) return []
  const [cx, cy, cw, ch] = crop as number[]
  const out: Region[] = []
  for (const r of regions as { id?: unknown; label?: unknown; box?: unknown }[]) {
    if (!r || typeof r.id !== 'string' || !Array.isArray(r.box) || r.box.length !== 4) continue
    const [x, y, w, h] = r.box as number[]
    const x0 = Math.max(x, cx), y0 = Math.max(y, cy), x1 = Math.min(x + w, cx + cw), y1 = Math.min(y + h, cy + ch)
    if (x1 <= x0 || y1 <= y0) continue
    out.push({ id: r.id, label: typeof r.label === 'string' ? r.label : r.id, box: [x0 - cx, y0 - cy, x1 - x0, y1 - y0], of: [cw, ch] })
  }
  return out
}

/** A video output as one side of a blind `CompareVideo`, named by its label only. */
/** The whole clip as a loop range (0 to the shorter output's last frame): a pair plays round and round until it is answered. */
export const wholeLoop = (x: unknown, y: unknown, fps = 50): VideoLoop | null => {
  const d = [x, y].map(o => (o && typeof o === 'object' && typeof (o as { duration_s?: unknown }).duration_s === 'number' ? (o as { duration_s: number }).duration_s : null))
  if (d.some(v => v === null || v <= 0)) return null
  return { start: 0, end: Math.max(Math.min(...(d as number[])) - 1 / fps, 0) }
}

/** Press the comparison's own Play once its videos can play, once per pair (until strata-react#203 gives CompareVideo autoPlay). Muted
 * sound is what lets a browser start it without a click; a person who asked for less motion is left to press Play. */
export function useAutoPlay(ref: RefObject<HTMLElement | null>, key: string, enabled = true) {
  useEffect(() => {
    if (!enabled || typeof window === 'undefined' || window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    let done = false
    const id = window.setInterval(() => {
      const root = ref.current
      if (done || !root) return
      const vids = Array.from(root.querySelectorAll('video'))
      if (!vids.length || vids.some(v => v.readyState < 3)) return
      const play = Array.from(root.querySelectorAll('button')).find(b => (b.getAttribute('aria-label') ?? b.textContent ?? '').trim() === 'Play')
      if (play) { done = true; play.click(); window.clearInterval(id) }
    }, 200)
    const stop = window.setTimeout(() => window.clearInterval(id), 15000) // a clip that never loads is left paused
    return () => { window.clearInterval(id); window.clearTimeout(stop) }
  }, [ref, key, enabled])
}

/** useAutoPlay as an element, for places a hook cannot go (inside a DataView's render callback). */
function AutoPlay({ box, id }: { box: RefObject<HTMLElement | null>; id: string }) {
  useAutoPlay(box, id)
  return null
}

const videoCandidate = (k: 'x' | 'y', o: unknown): Candidate<VideoMedia> | null => {
  const v = videoOf(o)
  return v ? { id: k, label: k.toUpperCase(), media: { kind: 'video', src: imagesApi.fileUrl(v.path), width: v.width ?? 1440, height: v.height ?? 1080, fps: v.fps } } : null
}

function Output({ o, label }: { o: unknown; label: string }) {
  const img = imagePath(o); const snd = audioPath(o)
  if (img) return <img src={imagesApi.fileUrl(img)} alt={label} className="h-auto max-w-full rounded-control" />
  const vid = videoOf(o)
  if (vid) return <video controls muted playsInline preload="metadata" src={imagesApi.fileUrl(vid.path)} aria-label={label} className="h-auto max-w-full rounded-control" />
  if (snd) return <MediaPlayer label={`${label}, spoken`} src={imagesApi.fileUrl(snd)} rates={SPEECH_RATES} shortcuts={false} /> // two players on one screen: neither binds Space
  return <p className="type-body whitespace-pre-wrap">{outputText(o)}</p>
}

/** The steps of a pipeline candidate's output (a text model's expansion, then the picture), if it has them. */
export const pipelineOf = (o: unknown): Step[] | null => (o && typeof o === 'object' && Array.isArray((o as { pipeline?: unknown }).pipeline) ? (o as { pipeline: Step[] }).pipeline : null)

function Pipeline({ steps }: { steps: Step[] }) {
  return (
    <ol className="grid gap-2" aria-label="Pipeline steps">
      {steps.map((st, i) => (
        <li key={i} className="grid gap-1 rounded-control bg-muted p-2">
          <span className="type-caption text-muted-foreground">{`Step ${i + 1}: ${st.model} (${st.role})`}{typeof st.run_s === 'number' ? ` · ${st.run_s.toFixed(1)} s` : ''}</span>
          <p className="type-body-sm whitespace-pre-wrap"><span className="type-label">Sent: </span>{st.sent}</p>
          {st.answer !== undefined && <p className="type-body-sm whitespace-pre-wrap"><span className="type-label">Answered: </span>{st.answer}</p>}
        </li>
      ))}
    </ol>
  )
}

const PipelineSteps = ({ steps }: { steps: Step[] }) => <Disclosure summary={`Pipeline: ${steps.length} steps`}><Pipeline steps={steps} /></Disclosure>

type Out = { output: unknown; error: string | null; run_s?: number | null } | null

/** One candidate's output for one item, with its pipeline's steps when it has them (shown only outside blind judging). */
function FullOutput({ name, out }: { name: string; out: Out }) {
  const steps = pipelineOf(out?.output)
  return (
    <section aria-label={name} className="grid content-start gap-2 rounded-surface border border-border p-3">
      <h3 className="type-label font-mono">{name}</h3>
      {out === null ? <p className="type-body-sm text-muted-foreground">Not run yet.</p> : out.error ? <p className="type-body-sm text-danger">{`Failed: ${out.error}`}</p> : <Output o={out.output} label={name} />}
      {typeof out?.run_s === 'number' && <span className="type-caption text-muted-foreground">{`${out.run_s.toFixed(1)} s in all`}</span>}
      {steps && <PipelineSteps steps={steps} />}
    </section>
  )
}

/** Candidates whose output is a picture, as the tiles of a comparison grid (one zoom and pan for all, full screen); the rest stay cards. */
function candidatesOf(outputs: Record<string, Out>, names: string[]) {
  const pictures: Candidate[] = []; const others: string[] = []
  for (const n of names) {
    const o = outputs[n]; const img = o && !o.error ? imagePath(o.output) : null
    if (img) pictures.push({ id: n, label: n, note: typeof o!.run_s === 'number' ? `${o!.run_s.toFixed(1)} s in all` : undefined, media: { kind: 'image', src: imagesApi.fileUrl(img), alt: `${n}'s picture`, size: imageSize(o!.output) } })
    else others.push(n)
  }
  return { pictures, others }
}

/** Every candidate's output for each item, a pipeline's every step included: the view after (not during) blind judging. */
function Outputs({ name }: { name: string }) {
  const load = useLoad({ queryKey: ['outputs', name], queryFn: () => evalsApi.outputs(name) })
  const [item, setItem] = useState<string | null>(null)
  const items = load.status === 'ready' ? load.data.items : []
  const at = Math.max(0, items.findIndex(x => x.id === item))
  const go = (k: number) => { if (items.length) setItem(items[(at + k + items.length) % items.length].id) }
  const keys: Shortcut[] = [{ keys: ['j'], label: 'Next item', run: () => go(1), repeat: true }, { keys: ['k'], label: 'Previous item', run: () => go(-1), repeat: true }]
  useHotkeys(keys, { enabled: items.length > 1 })
  return (
    <DataView state={load} isEmpty={d => d.items.length === 0} skeleton={<Skeleton className="h-96 w-full" />} empty={<EmptyState title="No items" />}>
      {d => {
        const it = d.items[at] ?? d.items[0]
        const names = d.candidates.map(c => c.name)
        const videos = names.map(n => { const o = it.outputs[n]; return o && !o.error ? videoSide(n, n, o.output, typeof o.run_s === 'number' ? `${o.run_s.toFixed(1)} s` : undefined) : null }).filter((v): v is VideoSide => v !== null)
        const source = videos.length ? sourceSide(name, it.id, it.reference) : null
        const { pictures, others } = candidatesOf(it.outputs, names.filter(n => !videos.some(v => v.id === n)))
        return (
          <div className="grid gap-4">
            <p className="type-body-sm text-muted-foreground">Every candidate&apos;s output, and for a pipeline what each step was sent and answered. The Judge tab shows only the end results, so look here after judging.</p>
            <div className="flex flex-wrap items-end gap-4">
              <Field label="Item"><NativeSelect value={it.id} onChange={e => setItem(e.target.value)}>{d.items.map(x => <option key={x.id} value={x.id}>{x.id}</option>)}</NativeSelect></Field>
              {d.items.length > 1 && <KeysLegend shortcuts={keys} label="Item keys" />}
            </div>
            {typeof it.input.prompt === 'string' && <p className="type-body">Prompt: {it.input.prompt}</p>}
            {typeof it.input.text === 'string' && <p className="type-body">Text: {it.input.text}</p>}
            {videos.length > 0 && <VideoCompare key={it.id} label={`Item ${it.id}: ${format.count(videos.length, 'video')}`} sides={source ? [source, ...videos] : videos} regionLabels={regionLabels(it.reference, ...names.map(n => it.outputs[n]?.output))} fps={videos.find(v => v.fps)?.fps} keys={false} autoPlay />}
            {pictures.length > 0 && <CompareGrid key={it.id} label={`Item ${it.id}: ${format.count(pictures.length, 'picture')}`} items={pictures} blind={false} missing="Picture not found" columns={pictures.length >= 4 ? 4 : pictures.length >= 3 ? 3 : 2}
              actions={c => { const steps = pipelineOf(it.outputs[c.id]?.output); return steps ? <PipelineSteps steps={steps} /> : null }} />}
            {others.length > 0 && <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{others.map(c => <FullOutput key={c} name={c} out={it.outputs[c] ?? null} />)}</div>}
          </div>
        )
      }}
    </DataView>
  )
}

/** The pair just judged, unblinded: which candidate each side was, what the model judges said, and its every step (loaded when opened). */
function LastPair({ name, last }: { name: string; last: Last }) {
  return (
    <Disclosure key={`${last.item_id}-${last.x}-${last.y}`} className="rounded-surface border border-border p-3" summary={`Last pair (${last.item_id}): X was ${last.x}, Y was ${last.y}; ${last.winner === 'tie' ? 'about the same' : `you chose ${last.winner}`}`}>
      <LastPairBody name={name} last={last} />
    </Disclosure>
  )
}

function LastPairBody({ name, last }: { name: string; last: Last }) {
  const load = useLoad({ queryKey: ['outputs', name, last.item_id], queryFn: () => evalsApi.outputs(name, last.item_id) })
  return (
    <>
      <PairVerdicts name={name} last={last} />
      <DataView state={load} isEmpty={d => d.items.length === 0} skeleton={<Skeleton className="h-24 w-full" />} empty={null}>
        {d => <div className="mt-2 grid gap-4 md:grid-cols-2">{[last.x, last.y].map(c => <FullOutput key={c} name={c} out={d.items[0].outputs[c] ?? null} />)}</div>}
      </DataView>
    </>
  )
}

type Last = { item_id: string; x: string; y: string; winner: string }
type Answer = 'x' | 'y' | 'tie'

/** The judging keys. Not 1 to 5, x, f, + - 0 or the arrows: the picture comparison binds those for its views, zoom and full screen. */
export const JUDGE_KEYS: Record<Answer, string[]> = { x: ['['], y: [']'], tie: ['t'] }
const ANSWER_WORD: Record<Answer, string> = { x: 'X is better', y: 'Y is better', tie: 'About the same' }

function Judge({ name }: { name: string }) {
  const qc = useQueryClient(); const [n, setN] = useState(0); const [last, setLast] = useState<Last | null>(null)
  const [mode, setMode] = useState<CompareMode>('side')
  const [vmode, setVmode] = useState<VideoMode>('side')
  const [loop, setLoop] = useState<{ item: string; range: VideoLoop } | null>(null) // a loop set on one pair is kept for the next pair of the same item
  const videoBox = useRef<HTMLDivElement>(null)
  const load = useLoad({ queryKey: ['pair', name, n], queryFn: () => evalsApi.pair(name), staleTime: Infinity })
  const answer = useAsyncAction(async (p: Pair, w: Answer) => { const j = await evalsApi.judge(name, p.token, w); setLast({ item_id: j.item_id ?? p.item_id, x: j.x, y: j.y, winner: j.winner }); setN(k => k + 1); void qc.invalidateQueries({ queryKey: ['eval', name] }) }, { settleMs: 0 }) // the results refresh behind: the next pair must not wait for them
  const pair = load.status === 'ready' ? load.data.pair : null
  const busy = answer.phase === 'pending'
  const choose = (w: Answer) => { if (pair && !busy) answer.run(pair, w) }
  const keys: Shortcut[] = (['x', 'tie', 'y'] as const).map(w => ({ keys: JUDGE_KEYS[w], label: ANSWER_WORD[w], run: () => choose(w), disabled: busy }))
  const videoPair = pair !== null && videoOf(pair.x) !== null && videoOf(pair.y) !== null
  useHotkeys(keys, { enabled: pair !== null && !videoPair }) // a video pair binds them in CompareVideo, with its own keys and one legend
  return (
    <DataView state={load} isEmpty={d => d.pair === null} skeleton={<Skeleton className="h-48 w-full" />} empty={<EmptyState title="Nothing to judge">Either the outputs are not in yet (run the experiment), or every candidate gave the same output for every item, so there is nothing to choose between.</EmptyState>}>
      {({ pair, remaining }) => {
        const p = pair!; const pics = imagePath(p.x) && imagePath(p.y)
        const vx = videoCandidate('x', p.x); const vy = videoCandidate('y', p.y); const src = vx && vy ? sourceSide(name, p.item_id, p.reference) : null
        const answers = <div className="flex flex-wrap items-center gap-3">{(['x', 'tie', 'y'] as const).map(w => <Button key={w} onClick={() => choose(w)} disabled={busy}>{ANSWER_WORD[w]}</Button>)}</div>
        const side = (k: 'x' | 'y'): Candidate => ({ id: k, label: k.toUpperCase(), media: { kind: 'image', src: imagesApi.fileUrl(imagePath(p[k])!), alt: `Output ${k.toUpperCase()}`, size: imageSize(p[k]) } })
        return (
          <div className="grid gap-4">
            <p className="type-body">Which output is better for item <span className="font-mono">{p.item_id}</span>? The candidates are hidden until you answer.</p>
            {p.similarity !== undefined && <p className="type-body-sm text-muted-foreground">{`${format.percent(1 - p.similarity)} different`}{remaining !== undefined ? ` · ${format.count(remaining, 'pair')} not judged yet, most different first` : ''}</p>}
            {p.has_media && !imagePath(p.x) && !vx && <PhrasePlayer key={p.item_id} src={evalsApi.mediaUrl(name, p.item_id)} clip={Array.isArray(p.input.clip) ? p.input.clip as [number, number] : undefined} />}
            {typeof p.input.prompt === 'string' && <p className="type-body">Prompt: {p.input.prompt}</p>}
            {typeof p.input.text === 'string' && <p className="type-body">Text: {p.input.text}</p>}
            {p.reference && 'text' in p.reference && p.reference.text !== p.input.text && <p className="type-body-sm text-muted-foreground">Reference: {String(p.reference.text)}</p>}
            {vx && vy
              ? <>
                  <div ref={videoBox}>
                    <CompareVideo key={p.token} label={`Item ${p.item_id}, two outputs`} a={vx} b={vy} blind missing="Video not found" mode={vmode} onModeChange={setVmode} regions={regionsOf(p.input)} pixelated="always" audio="none"
                      loop={(loop?.item === p.item_id && loop.range.start !== null && loop.range.end !== null ? loop.range : null) ?? wholeLoop(p.x, p.y, vx.media.fps) ?? undefined} onLoopChange={range => setLoop({ item: p.item_id, range })} extraShortcuts={keys} footer={answers} maxHeight="viewport" />
                  </div>
                  <AutoPlay box={videoBox} id={p.token} />
                  {src && <Disclosure summary="The tape itself (the source, deinterlaced, enlarged nearest-neighbour)"><video controls muted playsInline preload="metadata" src={src.whole} aria-label="The source" className="h-auto max-w-full rounded-control" style={{ imageRendering: 'pixelated' }} /></Disclosure>}
                </>
              : pics
              ? <Compare key={p.token} label={`Item ${p.item_id}, two outputs`} a={side('x')} b={side('y')} blind missing="Picture not found" mode={mode} onModeChange={setMode} />
              : <div className="grid gap-4 md:grid-cols-2">
                  {(['x', 'y'] as const).map(k => (
                    <section key={k} aria-label={`Output ${k.toUpperCase()}`} className="grid content-start gap-2 rounded-surface border border-border p-3">
                      <h3 className="type-label">Output {k.toUpperCase()}</h3>
                      <Output o={p[k]} label={`Output ${k.toUpperCase()}`} />
                    </section>
                  ))}
                </div>}
            {!(vx && vy) && <div className="flex flex-wrap items-center gap-3">
              {answers}
              <KeysLegend shortcuts={keys} label="Judging keys" />
            </div>}
            {answer.error && <Alert tone="danger" announce title="Not saved">{answer.error}</Alert>}
            {last && <LastPair name={name} last={last} />}
          </div>
        )
      }}
    </DataView>
  )
}

function WriteUp({ name }: { name: string }) {
  const load = useLoad({ queryKey: ['report', name], queryFn: () => evalsApi.report(name) })
  return (
    <DataView state={load} isEmpty={t => !t} skeleton={<Skeleton className="h-96 w-full" />} empty={<EmptyState title="No write-up" />}>
      {t => <div className="grid gap-3"><p className="type-body-sm text-muted-foreground">The standard write-up, generated from the results. Save it with <code className="font-mono">./strata-models eval report {name} --out FILE</code> and fill in the two sections left for a person.</p><pre className="overflow-x-auto rounded-control bg-muted p-4 font-mono type-caption whitespace-pre-wrap">{t}</pre></div>}
    </DataView>
  )
}

function RunButton({ name }: { name: string }) {
  const qc = useQueryClient()
  const status = useLoad({ queryKey: ['eval-status', name], queryFn: () => evalsApi.status(name), refetchInterval: q => (q.state.data?.state === 'running' ? 1000 : false) })
  const run = useAsyncAction(async () => { await evalsApi.run(name); await qc.invalidateQueries({ queryKey: ['eval-status', name] }) })
  const st = status.status === 'ready' ? status.data : undefined
  const running = st?.state === 'running'
  return (
    <div className="grid justify-items-end gap-1">
      <AsyncButton variant="default" phase={running ? 'pending' : run.phase} error={run.error} onClick={() => run.run()} labels={{ idle: 'Run missing outputs', pending: 'Running…', done: 'Started' }} />
      {st && st.log.length > 0 && <span className="type-caption text-muted-foreground">{st.state === 'failed' ? st.error : st.log[st.log.length - 1]}</span>}
    </div>
  )
}

export default function EvalPage() {
  const { name = '' } = useParams()
  const load = useLoad({ queryKey: ['eval', name], queryFn: () => evalsApi.analysis(name), refetchInterval: q => (q.state.data?.summary?.stale ? 3000 : 60_000) })
  return (
    <div className="grid gap-6">
      <PageHeader title={name} scope={load.status === 'ready' ? load.data.experiment.question : undefined} actions={<RunButton name={name} />} finding={load.status === 'ready' ? { text: finding(load.data) } : undefined} />
      {load.status === 'ready' && load.data.summary?.stale && <p className="type-caption text-muted-foreground">Updating: these results are from {new Date(load.data.summary.computed_at * 1000).toLocaleTimeString()}, new outputs or judgments are being added.</p>}
      <DataView state={load} isEmpty={() => false} skeleton={<TableSkeleton rows={4} columns={5} />} empty={null}>
        {a => (
          <Tabs param="tab" label="Experiment view" items={[{ value: 'results', label: 'Results' }, { value: 'judge', label: 'Judge' }, { value: 'outputs', label: 'Outputs' }, { value: 'writeup', label: 'Write-up' }]}>
            {tab => tab === 'judge' ? <Judge name={name} /> : tab === 'outputs' ? <Outputs name={name} /> : tab === 'writeup' ? <WriteUp name={name} /> : <Results a={a} />}
          </Tabs>
        )}
      </DataView>
    </div>
  )
}
