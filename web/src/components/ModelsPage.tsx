// One question: "Which model serves each role, and is it ready?" Finding: how many roles are served here, and how many apps still load models themselves.
// The roles table first (what apps ask for), then every model with its state and a Load / Free action, then the models still loaded by apps on their own
// (the to-do list for moving them over).
import { AsyncButton, DataTable, DataView, EmptyState, PageHeader, Status, TableSkeleton, format, useAsyncAction, useLoad, type Column } from 'strata-react'
import { useQueryClient } from '@tanstack/react-query'
import { api, type Model, type Registry, type Role } from '../api'
import { seconds } from '../logic'

function modelState(m: Model) {
  if (!m.served) return <Status tone="neutral">Not served yet</Status>
  if (m.missing.length) return <Status tone="warning">Needs libraries</Status>
  return m.resident ? <Status tone="success">Loaded</Status> : <Status tone="neutral">Not loaded</Status>
}

function LoadButton({ m }: { m: Model }) {
  const qc = useQueryClient()
  const action = m.resident ? 'unload' : 'load'
  const a = useAsyncAction(async () => { await api.model(m.key, action); await new Promise(r => setTimeout(r, 1500)); await qc.invalidateQueries({ queryKey: ['registry'] }) })
  if (!m.served || m.missing.length) return null
  return <AsyncButton size="sm" phase={a.phase} error={a.error} onClick={() => a.run()} labels={m.resident ? { idle: `Free ${m.key}`, pending: 'Freeing…', done: 'Freed' } : { idle: `Load ${m.key}`, pending: 'Loading…', done: 'Asked' }} />
}

function roleColumns(models: Map<string, Model>): Column<Role>[] {
  return [
    { key: 'role', header: 'Role', cell: r => <span className="grid"><span className="font-mono">{r.key}</span><span className="text-muted-foreground">{r.description}</span></span> },
    { key: 'model', header: 'Model', cell: r => r.model },
    { key: 'state', header: 'State', cell: r => modelState(models.get(r.model)!) },
    { key: 'apps', header: 'Used by', cell: r => models.get(r.model)!.used_by.join(', ') || '–' },
  ]
}

const modelColumns: Column<Model>[] = [
  { key: 'model', header: 'Model', cell: m => <span className="grid"><span className="font-mono">{m.key}</span><span className="text-muted-foreground">{m.runtime} · {m.name}</span></span>, sort: (a, b) => a.key.localeCompare(b.key) },
  { key: 'state', header: 'State', cell: m => <span className="grid gap-1">{modelState(m)}{m.missing.length > 0 && <span className="type-caption text-muted-foreground">pip install {m.missing.join(' ')}</span>}</span> },
  { key: 'vram', header: 'VRAM', numeric: true, cell: m => `${format.decimal(m.vram_gb)}\u00a0GB`, sort: (a, b) => a.vram_gb - b.vram_gb },
  { key: 'jobs', header: 'Jobs', numeric: true, cell: m => format.integer(m.stats?.jobs ?? 0), sort: (a, b) => (a.stats?.jobs ?? 0) - (b.stats?.jobs ?? 0) },
  { key: 'mean', header: 'Mean run', numeric: true, cell: m => seconds(m.stats?.mean_run_s) },
  { key: 'licence', header: 'Licence', cell: m => m.source ? <a className="underline" href={m.source} target="_blank" rel="noreferrer">{m.licence || 'source'}</a> : m.licence },
  { key: 'action', header: 'Action', hideHeader: true, cell: m => <LoadButton m={m} /> },
]

function finding(r: Registry) {
  const served = r.roles.filter(x => r.models.find(m => m.key === x.model)?.served).length
  const elsewhere = new Set(r.models.filter(m => !m.served).flatMap(m => m.used_by))
  return `${served} of ${r.roles.length} roles are served here.` + (elsewhere.size ? ` ${[...elsewhere].sort().join(', ')} still load ${elsewhere.size === 1 ? 'some models' : 'models'} themselves.` : '')
}

export default function ModelsPage() {
  const load = useLoad({ queryKey: ['registry'], queryFn: api.registry, refetchInterval: 10_000 })
  return (
    <div className="grid gap-8">
      <PageHeader title="Models" scope="What each role runs on (models.toml)" finding={load.status === 'ready' ? { text: finding(load.data) } : undefined} />
      <DataView state={load} isEmpty={r => r.roles.length === 0} skeleton={<TableSkeleton rows={6} columns={4} />} empty={<EmptyState title="No roles in the registry">Add a role to <code className="font-mono">models.toml</code> in the data folder.</EmptyState>}>
        {r => {
          const byKey = new Map(r.models.map(m => [m.key, m]))
          return (
            <div className="grid gap-10">
              <DataTable caption="Roles apps ask for" columns={roleColumns(byKey)} rows={r.roles} rowKey={x => x.key} />
              <DataTable caption="Every model" columns={modelColumns} rows={r.models} rowKey={m => m.key} sortParam="sort" />
            </div>
          )
        }}
      </DataView>
    </div>
  )
}
