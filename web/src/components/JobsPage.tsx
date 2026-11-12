// One question: "What ran on the GPU, for whom, and why did it fail?" Finding: the failures among the jobs shown, or that none failed.
// Filters live in the URL (a failure is a link to share). Pages arrive by cursor (older than the last job shown). A row opens in place to show its
// params, result and error, read only when opened.
import { useSearchParams } from 'react-router'
import { Alert, Button, DataTable, DataView, EmptyState, Field, NativeSelect, PageHeader, RowAction, Skeleton, TableSkeleton, format, useAsyncAction, useExpanded, useInfiniteLoad, useLoad, type Column } from 'strata-react'
import { useQueryClient } from '@tanstack/react-query'
import { api, type Job, type JobFilters, type JobPage } from '../api'
import { describeJob, seconds } from '../logic'
import JobStatus from './JobStatus'

const STATUSES = ['queued', 'running', 'done', 'failed', 'cancelled', 'expired']

function Detail({ id }: { id: number }) {
  const load = useLoad({ queryKey: ['job', id], queryFn: () => api.job(id) })
  return (
    <DataView state={load} isEmpty={() => false} skeleton={<Skeleton className="h-24 w-full" />} empty={null}>
      {j => (
        <div className="grid gap-3 type-body-sm">
          {j.error && <p className="text-danger">{j.error}</p>}
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
            <dt className="text-muted-foreground">Model</dt><dd>{j.model ?? '–'}</dd>
            <dt className="text-muted-foreground">Submitted</dt><dd>{format.dateTime(j.submitted_at * 1000)}</dd>
            <dt className="text-muted-foreground">Waited</dt><dd>{seconds(j.wait_s)}</dd>
            <dt className="text-muted-foreground">Loading the model</dt><dd>{seconds(j.load_s)}</dd>
          </dl>
          {j.params && Object.keys(j.params).length > 0 && <pre className="overflow-x-auto rounded-control bg-muted p-3 font-mono type-caption">{JSON.stringify(j.params, null, 1)}</pre>}
          {j.result !== null && j.result !== undefined && <pre className="max-h-80 overflow-auto rounded-control bg-muted p-3 font-mono type-caption">{JSON.stringify(j.result, null, 1).slice(0, 20000)}</pre>}
        </div>
      )}
    </DataView>
  )
}

function Withdraw({ j }: { j: Job }) {
  const qc = useQueryClient()
  const a = useAsyncAction(async () => { await api.cancel(j.id); await qc.invalidateQueries({ queryKey: ['jobs'] }) })
  if (j.status !== 'queued' && !(j.kind === 'lease' && j.status === 'running')) return null
  return <RowAction label={j.kind === 'lease' && j.status === 'running' ? 'End lease' : 'Withdraw'} subject={`job #${j.id}`} onClick={() => a.run()} disabled={a.phase === 'pending'} />
}

const columns: Column<Job>[] = [
  { key: 'id', header: 'Job', numeric: true, cell: j => `#${j.id}` },
  { key: 'when', header: 'Submitted', cell: j => <span className="tabular-nums">{format.dateTime(j.submitted_at * 1000)}</span> },
  { key: 'what', header: 'What', cell: j => describeJob(j) },
  { key: 'status', header: 'State', cell: j => <JobStatus status={j.status} /> },
  { key: 'run', header: 'Took', numeric: true, cell: j => seconds(j.kind === 'lease' && j.started_at ? (j.finished_at ?? Date.now() / 1000) - j.started_at : j.run_s) },
  { key: 'action', header: 'Action', hideHeader: true, cell: j => <Withdraw j={j} /> },
]

export default function JobsPage() {
  const [params, setParams] = useSearchParams()
  const filters: JobFilters = { status: params.get('status') || undefined, app: params.get('app') || undefined, kind: params.get('kind') || undefined }
  const setFilter = (k: string, v: string) => setParams(p => { const n = new URLSearchParams(p); if (v) n.set(k, v); else n.delete(k); return n })
  const load = useInfiniteLoad<Job, JobPage, number>({ queryKey: ['jobs', filters], queryFn: ({ cursor }) => api.jobs(filters, cursor), items: p => p.items, itemKey: j => String(j.id), cursorAfter: j => j.id, hasMore: p => p.next !== null })
  const expanded = useExpanded({ param: 'open' })
  const failed = load.status === 'ready' ? load.data.filter(j => j.status === 'failed').length : 0
  return (
    <div className="grid gap-6">
      <PageHeader title="Jobs" scope="Every request to the GPU, newest first" finding={load.status === 'ready' && load.data.length ? { text: failed ? `${format.count(failed, 'job')} of the ${load.data.length} shown failed; open one to see why.` : `None of the ${load.data.length} jobs shown failed.` } : undefined} />
      <div className="flex flex-wrap gap-4">
        <Field label="State"><NativeSelect value={filters.status ?? ''} onChange={e => setFilter('status', e.target.value)}><option value="">Any state</option>{STATUSES.map(s => <option key={s} value={s}>{s}</option>)}</NativeSelect></Field>
        <Field label="Kind"><NativeSelect value={filters.kind ?? ''} onChange={e => setFilter('kind', e.target.value)}><option value="">Jobs and leases</option><option value="run">Model jobs</option><option value="lease">GPU leases</option></NativeSelect></Field>
      </div>
      <DataView state={load} isEmpty={rows => rows.length === 0} skeleton={<TableSkeleton rows={8} columns={6} />} empty={<EmptyState title={params.toString() ? 'No jobs match these filters' : 'No jobs yet'}>{params.toString() ? 'Clear a filter to see more.' : 'Apps send jobs through the strata-models client; they show here as they arrive.'}</EmptyState>}>
        {rows => (
          <div className="grid gap-3">
            <DataTable caption="Jobs" columns={columns} rows={rows} rowKey={j => String(j.id)} expansion={{ expanded: expanded.expanded, onChange: expanded.onChange, rowLabel: j => `job #${j.id}`, detail: j => <Detail id={j.id} /> }} />
            {load.more.hasMore && <Button onClick={load.more.loadMore} disabled={load.more.loadingMore}>{load.more.loadingMore ? 'Loading older jobs…' : 'Show older jobs'}</Button>}
            {load.more.error && <Alert tone="danger" announce>{load.more.error}</Alert>}
          </div>
        )}
      </DataView>
    </div>
  )
}
