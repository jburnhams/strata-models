// One question: "What should serve this role, and how sure are we?" Finding: the decision (change to X, or keep the baseline) with its reason.
// Then: every candidate's results with their intervals, the paired comparisons with the baseline, the human judgments, and the tabs to judge
// pairs blind and to read the standard write-up. Running fills in missing outputs through the GPU queue.
import { useParams } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { AsyncButton, Button, DataTable, DataView, EmptyState, PageHeader, Skeleton, Status, Tabs, TableSkeleton, format, useAsyncAction, useLoad, type Column } from 'strata-react'
import { evalsApi, imagesApi, type Analysis, type Pair } from '../api'
import { VERDICT_TONE, VERDICT_WORD, diff, outputText, pvalue, reading, value } from '../evalfmt'
import { useState } from 'react'

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
    ...metrics.map(m => ({ key: m, header: m, numeric: true, cell: (r: Row) => { const v = a.candidates[r.name]?.metrics[m]; const info = a.metrics[m]; return v ? <span className="grid"><span>{value(info, v.value)}</span><span className="type-caption text-muted-foreground">{value(info, v.low)} to {value(info, v.high)}</span></span> : '–' } })),
    { key: 'verdict', header: 'Decision', cell: r => { const v = a.decision.per_candidate[r.name]; return v ? <Status tone={VERDICT_TONE[v.verdict]}>{VERDICT_WORD[v.verdict]}</Status> : <span className="text-muted-foreground">Baseline</span> } },
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
      <DataTable caption={`Results on ${format.count(a.dataset.compared, 'item')} (value, and its 95 % interval)`} columns={columns} rows={rows} rowKey={r => r.name} />
      {crows.length > 0 && <DataTable caption={`Compared with ${a.experiment.baseline} on the same items (candidate minus baseline)`} columns={ccols} rows={crows} rowKey={r => r.key} />}
      {a.judgments && (
        <section aria-labelledby="judged" className="grid gap-2">
          <h2 id="judged" className="type-heading">Human judgments</h2>
          <p className="type-body">{format.count(a.judgments.n, 'blind judgment')}. Bradley–Terry strength, where 1 is average:</p>
          <ul className="grid gap-1 type-body">{Object.entries(a.judgments.strength).sort((x, y) => y[1][0] - x[1][0]).map(([n, s]) => <li key={n}><span className="font-mono">{n}</span>: {s[0].toFixed(2)} ({s[1].toFixed(2)} to {s[2].toFixed(2)})</li>)}</ul>
        </section>
      )}
      <p className="type-body-sm text-muted-foreground">Dataset {a.dataset.name}, version <span className="font-mono">{a.dataset.version}</span>: {a.dataset.compared} of {a.dataset.n} items have every candidate&apos;s output. Intervals are 95 %; the Write-up tab says how each is computed.</p>
    </div>
  )
}

/** The picture of an image job's output, if it is one. */
export const imagePath = (o: unknown) => (o && typeof o === 'object' && typeof (o as { path?: unknown }).path === 'string' && /\.(png|jpe?g|webp)$/i.test((o as { path: string }).path) ? (o as { path: string }).path : null)

function Judge({ name }: { name: string }) {
  const qc = useQueryClient(); const [n, setN] = useState(0)
  const load = useLoad({ queryKey: ['pair', name, n], queryFn: () => evalsApi.pair(name), staleTime: Infinity })
  const answer = useAsyncAction(async (p: Pair, w: 'x' | 'y' | 'tie') => { await evalsApi.judge(name, p.token, w); setN(k => k + 1); await qc.invalidateQueries({ queryKey: ['eval', name] }) }, { settleMs: 0 })
  return (
    <DataView state={load} isEmpty={d => d.pair === null} skeleton={<Skeleton className="h-48 w-full" />} empty={<EmptyState title="Nothing to judge yet">Run the experiment first: pairs need two candidates&apos; outputs for the same item.</EmptyState>}>
      {({ pair }) => (
        <div className="grid gap-4">
          <p className="type-body">Which output is better for item <span className="font-mono">{pair!.item_id}</span>? The candidates are hidden until you answer.</p>
          {pair!.has_media && !imagePath(pair!.x) && <audio controls src={evalsApi.mediaUrl(name, pair!.item_id)} className="w-full" />}
          {typeof pair!.input.prompt === 'string' && <p className="type-body">Prompt: {pair!.input.prompt}</p>}
          {pair!.reference && 'text' in pair!.reference && <p className="type-body-sm text-muted-foreground">Reference: {String(pair!.reference.text)}</p>}
          <div className="grid gap-4 md:grid-cols-2">
            {(['x', 'y'] as const).map(k => (
              <section key={k} aria-label={`Output ${k.toUpperCase()}`} className="grid gap-2 rounded-surface border border-border p-3">
                <h3 className="type-label">Output {k.toUpperCase()}</h3>
                {imagePath(pair![k]) ? <img src={imagesApi.fileUrl(imagePath(pair![k])!)} alt={`Output ${k.toUpperCase()}`} className="h-auto max-w-full rounded-control" /> : <p className="type-body whitespace-pre-wrap">{outputText(pair![k])}</p>}
                <Button onClick={() => answer.run(pair!, k)} disabled={answer.phase === 'pending'}>{`${k.toUpperCase()} is better`}</Button>
              </section>
            ))}
          </div>
          <div><Button onClick={() => answer.run(pair!, 'tie')} disabled={answer.phase === 'pending'}>About the same</Button></div>
          {answer.error && <p role="alert" className="text-danger type-body-sm">{answer.error}</p>}
        </div>
      )}
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
  const load = useLoad({ queryKey: ['eval', name], queryFn: () => evalsApi.analysis(name), refetchInterval: 60_000 })
  return (
    <div className="grid gap-6">
      <PageHeader title={name} scope={load.status === 'ready' ? load.data.experiment.question : undefined} actions={<RunButton name={name} />} finding={load.status === 'ready' ? { text: finding(load.data) } : undefined} />
      <DataView state={load} isEmpty={() => false} skeleton={<TableSkeleton rows={4} columns={5} />} empty={null}>
        {a => (
          <Tabs param="tab" label="Experiment view" items={[{ value: 'results', label: 'Results' }, { value: 'judge', label: 'Judge' }, { value: 'writeup', label: 'Write-up' }]}>
            {tab => tab === 'judge' ? <Judge name={name} /> : tab === 'writeup' ? <WriteUp name={name} /> : <Results a={a} />}
          </Tabs>
        )}
      </DataView>
    </div>
  )
}
