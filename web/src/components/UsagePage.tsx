// One question: "Which apps use the GPU most?" Finding: the app with the most GPU time in the period, and its share.
// The period is a tab in the URL; the breakdown is a composition bar of GPU time by app, then the table by app and role with failures.
import { CompositionBar, DataTable, DataView, EmptyState, PageHeader, TableSkeleton, Tabs, format, useLoad, type Column } from 'strata-react'
import { api, type UsageRow } from '../api'
import { seconds } from '../logic'

const columns: Column<UsageRow>[] = [
  { key: 'app', header: 'App', cell: r => r.app, sort: (a, b) => a.app.localeCompare(b.app) },
  { key: 'what', header: 'What', cell: r => r.kind === 'lease' ? 'GPU leases' : r.role },
  { key: 'jobs', header: 'Jobs', numeric: true, cell: r => format.integer(r.jobs), sort: (a, b) => a.jobs - b.jobs },
  { key: 'failed', header: 'Failed', numeric: true, cell: r => format.integer(r.failed), sort: (a, b) => a.failed - b.failed },
  { key: 'gpu', header: 'GPU time', numeric: true, cell: r => seconds(r.gpu_s), sort: (a, b) => a.gpu_s - b.gpu_s },
]

export function byApp(rows: UsageRow[]) {
  const m = new Map<string, number>()
  for (const r of rows) m.set(r.app, (m.get(r.app) ?? 0) + r.gpu_s)
  return [...m.entries()].sort((a, b) => b[1] - a[1])
}

function Period({ hours }: { hours: number }) {
  const load = useLoad({ queryKey: ['usage', hours], queryFn: () => api.usage(hours) })
  return (
    <DataView state={load} isEmpty={u => u.rows.length === 0} skeleton={<TableSkeleton rows={5} columns={5} />} empty={<EmptyState title="Nothing used the GPU in this period" />}>
      {u => {
        const apps = byApp(u.rows); const total = apps.reduce((s, [, v]) => s + v, 0)
        return (
          <div className="grid gap-8">
            {total > 0 && <p className="type-body">{apps[0][0]} used the GPU most: {seconds(apps[0][1])}, {format.percent(apps[0][1] / total)} of the {seconds(total)} in all.</p>}
            {total > 0 && <div className="max-w-2xl"><CompositionBar label="GPU time by app" parts={apps.map(([app, v]) => ({ key: app, label: app, value: v }))} formatValue={v => seconds(v)} partHeader="App" /></div>}
            <DataTable caption="GPU use by app and role" columns={columns} rows={u.rows} rowKey={r => `${r.app}/${r.kind}/${r.role}`} sortParam="sort" />
          </div>
        )
      }}
    </DataView>
  )
}

export default function UsagePage() {
  return (
    <div className="grid gap-6">
      <PageHeader title="Usage" scope="GPU time per app: model run time, plus the time leases were held" />
      <Tabs param="period" label="Period" items={[{ value: '24', label: 'Last 24 hours' }, { value: '168', label: 'Last 7 days' }, { value: '720', label: 'Last 30 days' }]}>
        {v => <Period key={v} hours={Number(v)} />}
      </Tabs>
    </div>
  )
}
