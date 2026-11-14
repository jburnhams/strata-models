// One question: "What are the cloud API keys spending, and on what?" Finding: the paid key nearest its budget, with what is left until it rolls over.
// Each key has a meter (budget used, warning at 50 % and 80 %) or, for the free keys, its calls; the budgets are set here. Filters live in the URL
// (provider, model, app, outcome, period) and narrow the charts (spend, calls per day or hour) and the request log, whose rows open to show
// exactly what was sent and what came back, read only when opened.
import { useState } from 'react'
import { useSearchParams } from 'react-router'
import { Alert, AsyncButton, Button, DataTable, DataView, EmptyState, Field, Input, LineChart, Meter, NativeSelect, PageHeader, Skeleton, Status, TableSkeleton, format, useAsyncAction, useExpanded, useInfiniteLoad, useLoad, type Column } from 'strata-react'
import { useQueryClient } from '@tanstack/react-query'
import { api, type ApiCall, type ApiCallPage, type BudgetChange, type CallFilters, type ProviderState, type SpendRow, type SpendSeries } from '../api'
import { bucketSeries, seconds, spendFinding, ticksOf, usd } from '../logic'

const PERIODS = [{ value: '1', label: 'Last 24 hours' }, { value: '7', label: 'Last 7 days' }, { value: '30', label: 'Last 30 days' }, { value: '90', label: 'Last 90 days' }]
const THRESHOLDS = [{ at: 0.5, tone: 'warning' as const, label: 'Half spent' }, { at: 0.8, tone: 'danger' as const, label: 'Nearly spent' }]

function Budget({ p, hasKey }: { p: ProviderState; hasKey: boolean }) {
  const until = p.next_rollover ? `until ${format.date(p.next_rollover * 1000)}` : 'one budget, no rollover'
  const noKey = !hasKey && <p className="type-caption text-muted-foreground">No key on this machine yet.</p>
  if (!p.budgeted) {
    return (
      <div className="grid gap-1 rounded-control border p-4">
        <p className="type-body font-medium">{p.label}</p>
        <p className="type-body-sm">Free, rate-limited: {format.count(p.calls, 'call')} so far.</p>
        {noKey}
      </div>
    )
  }
  return (
    <div className="grid gap-2 rounded-control border p-4">
      {p.budget ? (
        <Meter label={p.label} value={p.spent} max={p.budget} thresholds={THRESHOLDS} valueText={`${usd(p.spent)} of ${usd(p.budget)} spent`} detail={`${usd(p.left)} left ${until} · ${format.count(p.calls, 'call')}${p.reserved ? ` · ${usd(p.reserved)} held for calls under way` : ''}`} />
      ) : (
        <>
          <p className="type-body font-medium">{p.label}</p>
          <Status tone="danger">No budget set: every call is refused</Status>
        </>
      )}
      {noKey}
    </div>
  )
}

function BudgetForm({ ps }: { ps: ProviderState[] }) {
  const qc = useQueryClient()
  const paid = ps.filter(p => p.budgeted)
  const [vals, setVals] = useState(() => Object.fromEntries(paid.map(p => [p.provider, { budget: String(p.budget ?? 0), rollover: p.rollover ?? '' }])))
  const save = useAsyncAction(async () => {
    const b: Record<string, number | string> = {} // BudgetChange, keyed by provider
    for (const p of paid) { b[`${p.provider}_budget_usd`] = Number(vals[p.provider].budget); b[`${p.provider}_rollover`] = vals[p.provider].rollover }
    await api.saveBudgets(b as BudgetChange)
    await qc.invalidateQueries({ queryKey: ['spend'] })
  })
  const set = (k: string, f: 'budget' | 'rollover', v: string) => setVals(s => ({ ...s, [k]: { ...s[k], [f]: v } }))
  return (
    <form className="grid gap-4" onSubmit={e => { e.preventDefault(); save.run() }}>
      <h2 className="type-heading-sm">Budgets</h2>
      <p className="type-body-sm text-muted-foreground">What each paid key may spend per period. The rollover date repeats monthly; leave it empty for one budget that only goes down. A call whose worst case does not fit what is left is refused before it is sent.</p>
      <div className="flex flex-wrap gap-6">
        {paid.map(p => (
          <fieldset key={p.provider} className="flex flex-wrap gap-3">
            <legend className="type-body-sm font-medium">{p.label}</legend>
            <Field label="Budget (US$)"><Input type="number" min={0} step="0.01" value={vals[p.provider].budget} onChange={e => set(p.provider, 'budget', e.target.value)} /></Field>
            <Field label="Rolls over on"><Input type="date" value={vals[p.provider].rollover} onChange={e => set(p.provider, 'rollover', e.target.value)} /></Field>
          </fieldset>
        ))}
      </div>
      <div><AsyncButton type="submit" phase={save.phase} labels={{ idle: 'Save budgets', pending: 'Saving…', done: 'Saved' }} /></div>
      {save.error && <Alert tone="danger" announce>{save.error}</Alert>}
    </form>
  )
}

const breakdownColumns: Column<SpendRow>[] = [
  { key: 'provider', header: 'Key', cell: r => r.provider },
  { key: 'model', header: 'Model', cell: r => <span className="font-mono">{r.model}</span>, sort: (a, b) => a.model.localeCompare(b.model) },
  { key: 'app', header: 'App', cell: r => r.app ?? '–' },
  { key: 'calls', header: 'Calls', numeric: true, cell: r => format.integer(r.calls), sort: (a, b) => a.calls - b.calls },
  { key: 'failed', header: 'Failed', numeric: true, cell: r => format.integer(r.failed), sort: (a, b) => a.failed - b.failed },
  { key: 'tokens', header: 'Tokens in / out', numeric: true, cell: r => `${format.compact(r.tokens_in)} / ${format.compact(r.tokens_out)}` },
  { key: 'cost', header: 'Cost', numeric: true, cell: r => usd(r.cost_usd), sort: (a, b) => a.cost_usd - b.cost_usd },
]

function Charts({ s }: { s: SpendSeries }) {
  const now = Date.now() / 1000
  const hourly = s.bucket_s < 86400
  const fx = (x: number) => (hourly ? format.clock(x) : format.shortDay(x))
  const cost = bucketSeries(s.points, s.since, s.bucket_s, now, 'cost_usd')
  const calls = bucketSeries(s.points, s.since, s.bucket_s, now, 'calls')
  const per = hourly ? 'hour' : 'day'
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <LineChart label={`Spend per ${per}`} series={cost} seriesLabel="Spend" formatX={fx} formatY={usd} xTicks={ticksOf(cost)} xLabel={hourly ? 'Hour' : 'Day'} end="none" readout height={180} />
      <LineChart label={`Calls per ${per}`} series={calls} seriesLabel="Calls" formatX={fx} formatY={y => format.integer(y)} xTicks={ticksOf(calls)} xLabel={hourly ? 'Hour' : 'Day'} end="none" readout height={180} />
    </div>
  )
}

function CallDetail({ id }: { id: number }) {
  const load = useLoad({ queryKey: ['call', id], queryFn: () => api.call(id) })
  return (
    <DataView state={load} isEmpty={() => false} skeleton={<Skeleton className="h-24 w-full" />} empty={null}>
      {c => (
        <div className="grid gap-3 type-body-sm">
          {c.error && <p className="text-danger">{c.error}</p>}
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
            <dt className="text-muted-foreground">Model</dt><dd className="font-mono">{c.model}{c.model_key ? ` (${c.model_key})` : ''}</dd>
            <dt className="text-muted-foreground">For</dt><dd>{c.app ?? '–'}{c.job_id ? `, job #${c.job_id}` : ''}</dd>
            <dt className="text-muted-foreground">Tokens</dt><dd>{format.integer(c.tokens_in)} in, {format.integer(c.cache_read)} from cache, {format.integer(c.cache_write)} written to cache, {format.integer(c.tokens_out)} out</dd>
            <dt className="text-muted-foreground">Ended</dt><dd>{c.finish ?? '–'}</dd>
            {c.request?.effort && <><dt className="text-muted-foreground">Effort</dt><dd>{c.request.effort}</dd></>}
            {c.request?.max_tokens && <><dt className="text-muted-foreground">Max tokens</dt><dd>{format.integer(c.request.max_tokens)}</dd></>}
            {c.request?.images?.length ? <><dt className="text-muted-foreground">Pictures</dt><dd className="font-mono break-all">{c.request.images.join(', ')}</dd></> : null}
          </dl>
          {c.request?.system && <><h3 className="type-body-sm font-medium">System prompt</h3><pre className="max-h-60 overflow-auto whitespace-pre-wrap rounded-control bg-muted p-3 font-mono type-caption">{c.request.system}</pre></>}
          {c.request?.prompt && <><h3 className="type-body-sm font-medium">Prompt</h3><pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-control bg-muted p-3 font-mono type-caption">{c.request.prompt}</pre></>}
          {c.request?.schema ? <><h3 className="type-body-sm font-medium">Answer schema</h3><pre className="max-h-60 overflow-auto rounded-control bg-muted p-3 font-mono type-caption">{JSON.stringify(c.request.schema, null, 1)}</pre></> : null}
          <h3 className="type-body-sm font-medium">Answer</h3>
          {c.response ? <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-control bg-muted p-3 font-mono type-caption">{c.response}</pre> : <p className="text-muted-foreground">No answer.</p>}
        </div>
      )}
    </DataView>
  )
}

const callColumns: Column<ApiCall>[] = [
  { key: 'when', header: 'When', cell: c => <span className="tabular-nums">{format.dateTime(c.at * 1000)}</span> },
  { key: 'provider', header: 'Key', cell: c => c.provider },
  { key: 'model', header: 'Model', cell: c => <span className="font-mono">{c.model}</span> },
  { key: 'app', header: 'App', cell: c => c.app ?? '–' },
  { key: 'ok', header: 'Outcome', cell: c => <Status tone={c.ok ? 'success' : 'danger'}>{c.ok ? 'Answered' : 'Failed'}</Status> },
  { key: 'tokens', header: 'Tokens in / out', numeric: true, cell: c => `${format.integer(c.tokens_in + c.cache_read + c.cache_write)} / ${format.integer(c.tokens_out)}` },
  { key: 'took', header: 'Took', numeric: true, cell: c => seconds(c.seconds) },
  { key: 'cost', header: 'Cost', numeric: true, cell: c => usd(c.cost_usd) },
]

function Log({ filters }: { filters: CallFilters }) {
  const load = useInfiniteLoad<ApiCall, ApiCallPage, number>({ queryKey: ['calls', filters], queryFn: ({ cursor }) => api.calls(filters, cursor), items: p => p.calls, itemKey: c => String(c.id), cursorAfter: c => c.id, hasMore: p => p.next !== null })
  const expanded = useExpanded({ param: 'open' })
  return (
    <DataView state={load} isEmpty={rows => rows.length === 0} skeleton={<TableSkeleton rows={6} columns={8} />} empty={<EmptyState title="No calls match">Calls to the cloud APIs show here as they are made.</EmptyState>}>
      {rows => (
        <div className="grid gap-3">
          <DataTable caption="Request log" columns={callColumns} rows={rows} rowKey={c => String(c.id)} expansion={{ expanded: expanded.expanded, onChange: expanded.onChange, rowLabel: c => `call #${c.id}`, detail: c => <CallDetail id={c.id} /> }} />
          {load.more.hasMore && <Button onClick={load.more.loadMore} disabled={load.more.loadingMore}>{load.more.loadingMore ? 'Loading older calls…' : 'Show older calls'}</Button>}
          {load.more.error && <Alert tone="danger" announce>{load.more.error}</Alert>}
        </div>
      )}
    </DataView>
  )
}

export default function CloudPage() {
  const [params, setParams] = useSearchParams()
  const days = Number(params.get('days') || '30')
  const filters: CallFilters = { provider: params.get('provider') || undefined, model: params.get('model') || undefined, app: params.get('app') || undefined, ok: params.get('ok') || undefined, days }
  const setFilter = (k: string, v: string) => setParams(p => { const n = new URLSearchParams(p); if (v) n.set(k, v); else n.delete(k); n.delete('open'); return n })
  const spend = useLoad({ queryKey: ['spend'], queryFn: api.spend, refetchInterval: 30_000 })
  const series = useLoad({ queryKey: ['spend-series', filters.provider, filters.model, filters.app, days], queryFn: () => api.spendSeries(filters) })
  const models = series.status === 'ready' ? series.data.options.models : []
  const apps = series.status === 'ready' ? series.data.options.apps : []
  return (
    <div className="grid gap-8">
      <PageHeader title="Cloud APIs" scope="What the Claude, Google and NVIDIA keys spend, and every request they made" finding={spend.status === 'ready' ? { text: spendFinding(spend.data.providers) } : undefined} />
      <DataView state={spend} isEmpty={() => false} skeleton={<Skeleton className="h-28 w-full" />} empty={null}>
        {s => (
          <div className="grid gap-6">
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{s.providers.map(p => <Budget key={p.provider} p={p} hasKey={!!s.keys[p.provider]} />)}</div>
            <BudgetForm key={s.providers.map(p => `${p.budget}/${p.rollover}`).join()} ps={s.providers} />
          </div>
        )}
      </DataView>
      <div className="flex flex-wrap gap-4">
        <Field label="Period"><NativeSelect value={String(days)} onChange={e => setFilter('days', e.target.value === '30' ? '' : e.target.value)}>{PERIODS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}</NativeSelect></Field>
        <Field label="Key"><NativeSelect value={filters.provider ?? ''} onChange={e => setFilter('provider', e.target.value)}><option value="">Every key</option>{(spend.status === 'ready' ? spend.data.providers : []).map(p => <option key={p.provider} value={p.provider}>{p.label}</option>)}</NativeSelect></Field>
        <Field label="Model"><NativeSelect value={filters.model ?? ''} onChange={e => setFilter('model', e.target.value)}><option value="">Every model</option>{models.map(m => <option key={m} value={m}>{m}</option>)}</NativeSelect></Field>
        <Field label="App"><NativeSelect value={filters.app ?? ''} onChange={e => setFilter('app', e.target.value)}><option value="">Every app</option>{apps.map(a => <option key={a} value={a}>{a}</option>)}</NativeSelect></Field>
        <Field label="Outcome"><NativeSelect value={filters.ok ?? ''} onChange={e => setFilter('ok', e.target.value)}><option value="">Any outcome</option><option value="true">Answered</option><option value="false">Failed</option></NativeSelect></Field>
      </div>
      <DataView state={series} isEmpty={() => false} skeleton={<Skeleton className="h-48 w-full" />} empty={null}>
        {s => (
          <div className="grid gap-6">
            <Charts s={s} />
            {s.breakdown.length > 0 && <DataTable caption="Spend by key, model and app" columns={breakdownColumns} rows={s.breakdown} rowKey={r => `${r.provider}/${r.model}/${r.app}`} sortParam="sort" />}
          </div>
        )}
      </DataView>
      <section className="grid gap-3">
        <h2 className="type-heading-sm">Request log</h2>
        <Log filters={filters} />
      </section>
    </div>
  )
}
