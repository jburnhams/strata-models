// One question: "Is the GPU free, and who is using it?" Finding: what it is doing now (a job, a lease, idle with what loaded).
// Then in the order the next questions come: how busy and how full (metrics, then where the memory is), the last hour, what is waiting, what just happened.
// Polled every 2 s; a failed poll keeps the last readings with a note of their age, so a restarting service does not blank the page.
import { CompositionBar, DataTable, DataView, EmptyState, LineChart, Metric, MetricStrip, PageHeader, Skeleton, StaleNotice, Status, format, useLoad, type Column } from 'strata-react'
import { api, type GpuProcess, type Job, type Status as S } from '../api'
import { PRIORITY_OF, PRIORITY_WORD, describeJob, finding, gb, processName, seconds, vramParts } from '../logic'

const queueColumns: Column<Job>[] = [
  { key: 'id', header: 'Job', numeric: true, cell: j => `#${j.id}` },
  { key: 'what', header: 'What', cell: j => describeJob(j) },
  { key: 'priority', header: 'Priority', cell: j => { const p = PRIORITY_OF[j.priority]; return <Status tone={p === 'high' ? 'info' : 'neutral'}>{PRIORITY_WORD[p]}</Status> } },
  { key: 'waiting', header: 'Waiting', numeric: true, cell: j => seconds(Date.now() / 1000 - j.submitted_at) },
]

const processColumns: Column<GpuProcess>[] = [
  { key: 'name', header: 'Program', cell: p => <span className="grid"><span>{p.ours ? 'strata-models' : processName(p.command)}</span><span className="type-caption text-muted-foreground font-mono">{p.command || `pid ${p.pid}`}</span></span> },
  { key: 'pid', header: 'Process', numeric: true, cell: p => String(p.pid) },
  { key: 'mem', header: 'Memory', numeric: true, cell: p => `${format.decimal(gb(p.used_mb))}\u00a0GB`, sort: (a, b) => a.used_mb - b.used_mb },
]

function Now({ s }: { s: S }) {
  const running = s.running ?? (s.current ? [s.current] : [])
  const jobs = s.lease && s.lease.vram_gb === null ? [s.lease] : [...(s.lease ? [s.lease] : []), ...running]
  if (!jobs.length) return <p className="type-body text-muted-foreground">Nothing is running.</p>
  return (
    <ul className="grid gap-1">
      {jobs.map(j => {
        const since = j.started_at ? seconds(Date.now() / 1000 - j.started_at) : '–'
        return (
          <li key={j.id} className="type-body">
            {j.kind === 'lease' ? <>Lent to <strong>{j.app}</strong>{j.note ? ` for ${j.note}` : ''}, for {since} so far{j.vram_gb !== null ? `, asking for ${j.vram_gb} GB` : ', the whole card'}.</>
              : <>Running <strong>{j.role}</strong> on {j.model} for <strong>{j.app}</strong>, for {since} so far{'lane' in j && j.lane === 'cpu' ? ' (no GPU needed)' : ''}.</>}
          </li>
        )
      })}
    </ul>
  )
}

function UtilChart() {
  const load = useLoad({ queryKey: ['history'], queryFn: api.history, refetchInterval: 10_000 })
  if (load.status !== 'ready' || load.data.samples.length < 2) return null
  const pts = load.data.samples.map(p => ({ x: p.at * 1000, y: p.util_pct ?? 0 }))
  const first = pts[0].x, last = pts[pts.length - 1].x, step = Math.max((last - first) / 4, 1)
  return (
    <section aria-labelledby="hour" className="grid gap-3">
      <h2 id="hour" className="type-heading">The last hour</h2>
      <LineChart label="GPU busy over the last hour" series={pts} seriesLabel="Busy" formatX={x => format.clock(x)} formatY={y => `${Math.round(y)}%`} xTicks={[0, 1, 2, 3, 4].map(i => first + i * step)} yMin={0} yMax={100} height={160} />
    </section>
  )
}

function OverviewSkeleton() {
  return <div className="grid gap-8"><Skeleton className="h-20 w-full" /><Skeleton className="h-28 w-full" /><Skeleton className="h-40 w-full" /></div>
}

export default function OverviewPage() {
  const load = useLoad({ queryKey: ['status'], queryFn: api.status, refetchInterval: 2000 })
  return (
    <div className="grid gap-8">
      <PageHeader title="Overview" scope="The GPU now, checked every 2 seconds" finding={load.status === 'ready' ? { text: finding(load.data) } : undefined} />
      <DataView
        state={load}
        isEmpty={() => false}
        skeleton={<OverviewSkeleton />}
        empty={<EmptyState title="No readings yet" />}
        staleNotice={stale => <StaleNotice stale={stale} shown={when => `These readings are from ${when}: the service is not answering.`} />}
      >
        {s => (
          <div className="grid gap-10">
            {s.gpu && s.budget && (
              <>
                <MetricStrip label={s.gpu.name}>
                  <Metric label="Busy" value={s.gpu.util_pct === null ? '–' : format.integer(s.gpu.util_pct)} unit="%" />
                  <Metric label="Memory in use" value={format.decimal(gb(s.gpu.used_mb))} unit={`of ${format.decimal(gb(s.gpu.total_mb))} GB`} />
                  <Metric label="Free for models" value={format.decimal(gb(s.budget.free_for_models_mb))} unit="GB" note={`${format.decimal(gb(s.budget.reserve_mb))} GB kept for the desktop`} />
                  <Metric label="Temperature" value={s.gpu.temp_c === null ? '–' : format.integer(s.gpu.temp_c)} unit="°C" note={s.gpu.power_w !== null ? `${format.integer(s.gpu.power_w)} W` : undefined} />
                </MetricStrip>
                <section aria-labelledby="now" className="grid gap-3">
                  <h2 id="now" className="type-heading">Now</h2>
                  <Now s={s} />
                </section>
                <section aria-labelledby="memory" className="grid max-w-2xl gap-3">
                  <h2 id="memory" className="type-heading">Where the memory is</h2>
                  <CompositionBar label={`${format.decimal(gb(s.gpu.total_mb))} GB on the card`} parts={vramParts(s.gpu, s.budget, s.resident)} total={s.gpu.total_mb} restLabel="Free for models" formatValue={v => `${format.decimal(gb(v))}\u00a0GB`} partHeader="Held by" />
                </section>
                <section aria-labelledby="programs" className="grid gap-3">
                  <h2 id="programs" className="type-heading">Programs on the card</h2>
                  <DataTable caption="Programs using the GPU" columns={processColumns} rows={s.gpu.processes} rowKey={p => String(p.pid)} />
                </section>
              </>
            )}
            {!s.gpu && <Now s={s} />}
            <UtilChart />
            <section aria-labelledby="queue" className="grid gap-3">
              <h2 id="queue" className="type-heading">Waiting</h2>
              {s.queued.length ? <DataTable caption="Jobs waiting for the GPU" columns={queueColumns} rows={s.queued} rowKey={j => String(j.id)} />
                : <p className="type-body text-muted-foreground">No one is waiting.</p>}
            </section>
            <section aria-labelledby="events" className="grid gap-3">
              <h2 id="events" className="type-heading">Recent events</h2>
              {s.events.length ? (
                <ul className="grid gap-1 type-body-sm">
                  {[...s.events].reverse().slice(0, 12).map((e, i) => <li key={i}><span className="font-mono text-muted-foreground tabular-nums">{format.clock(e.at * 1000)}</span> {e.text}</li>)}
                </ul>
              ) : <p className="type-body text-muted-foreground">No models loaded or freed since the service started.</p>}
            </section>
          </div>
        )}
      </DataView>
    </div>
  )
}
