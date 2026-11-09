// One question: "Which decisions about models have evidence behind them, and which are waiting for a run?" Finding: how many experiments
// have every output in, and how many still need running. Each experiment is a link to its results; datasets are listed below with their version.
import { DataTable, DataView, EmptyState, LinkCard, LinkCardList, PageHeader, Status, TableSkeleton, format, useLoad, type Column } from 'strata-react'
import { evalsApi, type DatasetInfo, type Evals } from '../api'

const datasetColumns: Column<DatasetInfo>[] = [
  { key: 'name', header: 'Dataset', cell: d => <span className="grid"><span className="font-mono">{d.name}</span><span className="text-muted-foreground">{d.description || d.error || ''}</span></span> },
  { key: 'n', header: 'Items', numeric: true, cell: d => (d.n === undefined ? '–' : format.integer(d.n)) },
  { key: 'version', header: 'Version', cell: d => <span className="font-mono">{d.version ?? '–'}</span> },
]

function finding(e: Evals) {
  const ok = e.experiments.filter(x => !x.error)
  const complete = ok.filter(x => x.total && x.outputs === x.total).length
  return `${complete} of ${ok.length} experiments have every output in` + (ok.length - complete ? `; ${ok.length - complete} still need a run.` : '.')
}

export default function EvalsPage() {
  const load = useLoad({ queryKey: ['evals'], queryFn: evalsApi.list, refetchInterval: 10_000 })
  return (
    <div className="grid gap-8">
      <PageHeader title="Evals" scope="Experiments that decide which model, settings or prompt serves a role" finding={load.status === 'ready' && load.data.experiments.length ? { text: finding(load.data) } : undefined} />
      <DataView state={load} isEmpty={e => e.experiments.length === 0 && e.datasets.length === 0} skeleton={<TableSkeleton rows={4} columns={3} />}
        empty={<EmptyState title="No experiments yet">Write a spec in <code className="font-mono">evals/specs/</code> and a dataset in the data folder&apos;s <code className="font-mono">evals/datasets/</code>; see the repository&apos;s docs/evals.md.</EmptyState>}>
        {e => (
          <div className="grid gap-10">
            <LinkCardList label="Experiments">
              {e.experiments.map(x => (
                <LinkCard key={x.name} to={`/evals/${encodeURIComponent(x.name)}`} title={x.name} description={x.error ?? x.question ?? ''}
                  marks={x.error ? <Status tone="danger">Broken spec</Status> : <>
                    {x.running ? <Status tone="info">Running</Status> : x.outputs === x.total ? <Status tone="success">All outputs in</Status> : <Status tone="neutral">{`${x.outputs ?? 0} of ${x.total ?? 0} outputs`}</Status>}
                    <span className="type-caption text-muted-foreground">{x.role} · {x.candidates?.length ?? 0} candidates · {x.dataset}</span>
                  </>} />
              ))}
            </LinkCardList>
            <section aria-labelledby="datasets" className="grid gap-3">
              <h2 id="datasets" className="type-heading">Datasets</h2>
              <DataTable caption="Datasets" columns={datasetColumns} rows={e.datasets} rowKey={d => d.name} />
            </section>
          </div>
        )}
      </DataView>
    </div>
  )
}
