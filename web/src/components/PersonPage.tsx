// One question: "Are these all the same person, and where were they?" Every face confirmed as this person (and anyone merged into them), best
// detected first, as crops cut by the service; a face opened (Enter, a double click, or "Show in its photo") shows in its photo above the
// grid, zoomed on it, with who is beside it. Faces that are someone else are chosen and taken off in one decision, with Undo while it waits
// (useUndoable): the decision is recorded as "not this person", so the face does not come back to them in a group. Faces an app suggested
// (an app's labels turned into proposals by the service) wait below, a page at a time: confirmed as theirs, chosen or all at once, or
// "not this person"; each with Undo while it waits. Thousands are too many to look through, so they come in rounds: the most certain first,
// each scored afresh against the faces already confirmed (before three are, against the tightest group of the suggestions), every tile
// with its likeness; every face starts chosen, and what is left unticked can be said not to be them. The whole list stays a click away.
import { useState, type FormEvent } from 'react'
import { useParams } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { AsyncButton, BulkBar, Button, DataView, EmptyState, Field, Input, PageHeader, SelectableGrid, TableSkeleton, TextLink, Thumbnail, UndoBar, format, useAsyncAction, useLoad, useSearchParam, useSelection, useUndoable } from 'strata-react'
import { facesApi, personName, type Face, type Person } from '../faces'
import { FaceInPhoto } from './FaceInPhoto'

const isId = (raw: string) => /^\d+$/.test(raw)

function Rename({ person }: { person: Person }) {
  const qc = useQueryClient()
  const [name, setName] = useState(person.name ?? '')
  const save = useAsyncAction(async () => { await facesApi.rename(person.id, name.trim()); await qc.invalidateQueries({ queryKey: ['faces'] }) }, { validate: () => ({ name: name.trim() ? undefined : 'A name cannot be empty.' }) })
  const onSubmit = (e: FormEvent) => { e.preventDefault(); save.run() }
  return (
    <form onSubmit={onSubmit} className="max-w-md">
      <Field label="Name" error={save.fieldErrors.name ?? save.error} action={<AsyncButton type="submit" phase={save.phase} error={save.error} labels={{ idle: 'Rename', pending: 'Renaming…', done: 'Renamed' }} />}>
        <Input value={name} onChange={e => { setName(e.target.value); save.clearFieldError('name') }} />
      </Field>
    </form>
  )
}

type Open = { open: string; setOpen: (v: string | null) => void }

function InPhoto({ open, setOpen, names }: Open & { names: Map<number, Person> }) {
  return (
    <section aria-labelledby="in-photo" className="grid gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-3"><h2 id="in-photo" className="type-heading">In its photo</h2><Button variant="ghost" size="sm" onPress={() => setOpen(null)}>Close the photo</Button></div>
      <FaceInPhoto faceId={Number(open)} names={names} />
    </section>
  )
}

function Faces({ person, faces, open, setOpen }: Open & { person: Person; faces: Face[] }) {
  const qc = useQueryClient()
  const who = personName(person)
  const undoable = useUndoable()
  const shown = faces.filter(f => !undoable.isRemoved(String(f.id)))
  const selection = useSelection(shown, f => String(f.id))
  const takeOff = () => {
    const ids = selection.selectedRows.map(f => f.id)
    const n = format.count(ids.length, 'face')
    selection.clear()
    if (ids.includes(Number(open))) setOpen(null)
    undoable.schedule({
      id: `not-${ids.join('-')}`, ids: ids.map(String), message: `Took ${n} off ${who}.`, failure: `${n} are back with ${who}: the change did not go through.`,
      commit: async () => { await facesApi.assign(ids, person.id, 'rejected'); await qc.invalidateQueries({ queryKey: ['faces'] }) },
    })
  }
  return (
    <div className="grid gap-6">
      <BulkBar count={selection.count} noun="face" onClear={selection.clear}>
        {selection.count === 1 && <Button onPress={() => setOpen(String(selection.selectedRows[0].id))}>Show in its photo</Button>}
        <Button variant="default" onPress={takeOff}>{`Not ${who} (${format.integer(selection.count)})`}</Button>
      </BulkBar>
      <SelectableGrid
        label={`Faces of ${who}`} rows={shown} rowKey={f => String(f.id)} selection={{ ...selection, rowLabel: f => `Face ${f.id}` }} selectionBehavior="replace"
        renderImage={f => <Thumbnail src={facesApi.cropUrl(f.id)} alt="" missing="No picture" className="size-full" />}
        onAction={f => setOpen(String(f.id))} tileSize="sm"
      />
      <UndoBar undoable={undoable} />
    </div>
  )
}

const ROUND = 48
const likeness = (d: number) => `${format.share(Math.max(0, 1 - d))} alike`

function Round({ person, open, setOpen }: Open & { person: Person }) {
  const qc = useQueryClient()
  const who = personName(person)
  const load = useLoad({ queryKey: ['faces', 'review', person.id], queryFn: () => facesApi.review(person.id, ROUND) })
  const undoable = useUndoable()
  const rows = load.status === 'ready' ? load.data.items.filter(f => !undoable.isRemoved(String(f.id))) : []
  const selection = useSelection(rows, f => String(f.id), () => true)
  const yes = selection.selectedRows.map(f => f.id)
  const no = rows.filter(f => !selection.selected.has(String(f.id))).map(f => f.id)
  const decide = (rest: boolean) => {
    const ids = rest ? [...yes, ...no] : yes
    const n = format.count(yes.length, 'face')
    if (ids.includes(Number(open))) setOpen(null)
    undoable.schedule({
      id: `round-${ids.join('-')}`, ids: ids.map(String),
      message: rest && no.length ? `Confirmed ${n} as ${who}, and ${format.count(no.length, 'face')} as not them.` : `Confirmed ${n} as ${who}.`,
      failure: `That round is back: confirming it as ${who} did not go through.`,
      commit: async () => {
        if (yes.length) await facesApi.assign(yes, person.id, 'confirmed')
        if (rest && no.length) await facesApi.assign(no, person.id, 'rejected')
        await qc.invalidateQueries({ queryKey: ['faces'] })
      },
    })
  }
  const r = load.status === 'ready' ? load.data : undefined
  const basis = r?.basis === 'confirmed' ? `Scored against ${format.count(r.basis_faces, 'confirmed face')}.` : r?.basis === 'group' ? `Nothing confirmed yet: these are from the tightest group of ${format.count(r.basis_faces, 'suggestion')}, all photos an app says ${who} is in.` : ''
  return (
    <section aria-labelledby="round" className="grid gap-4">
      <h2 id="round" className="type-heading">{`Is this ${who}? ${format.count(person.suggested, 'face')} suggested`}</h2>
      {r && <p className="type-body-sm max-w-prose text-muted-foreground">{`${basis} ${format.integer(r.near['0.3'] ?? 0)} are very close and ${format.integer(r.near['0.4'] ?? 0)} close. Untick any that are someone else.`}</p>}
      <DataView state={load} isEmpty={() => rows.length === 0} skeleton={<TableSkeleton rows={2} columns={6} />} empty={<EmptyState title="Nothing left to confirm">{`Every face suggested for ${who} has been decided.`}</EmptyState>}>
        {() => (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="default" disabled={yes.length === 0} onPress={() => decide(false)}>{`Confirm ${format.integer(yes.length)}`}</Button>
              {no.length > 0 && <Button onPress={() => decide(true)}>{`Confirm ${format.integer(yes.length)}, the other ${format.integer(no.length)} are not ${who}`}</Button>}
              {selection.count === 1 && <Button variant="ghost" onPress={() => setOpen(String(yes[0]))}>Show in its photo</Button>}
            </div>
            <SelectableGrid
              label={`Is this ${who}?`} rows={rows} rowKey={f => String(f.id)} selection={{ ...selection, rowLabel: f => `Face ${f.id}, ${likeness(f.distance)}` }}
              renderImage={f => <Thumbnail src={facesApi.cropUrl(f.id)} alt="" missing="No picture" className="size-full" />}
              detail={f => likeness(f.distance)} onAction={f => setOpen(String(f.id))} tileSize="sm"
            />
          </>
        )}
      </DataView>
      <UndoBar undoable={undoable} />
    </section>
  )
}

const PAGE = 200

function Suggested({ person, open, setOpen }: Open & { person: Person }) {
  const qc = useQueryClient()
  const who = personName(person)
  const [page, setPage] = useState(0)
  const load = useLoad({ queryKey: ['faces', 'suggested-of', person.id, page], queryFn: () => facesApi.facesOf(person.id, PAGE, 'suggested', page * PAGE) })
  const undoable = useUndoable()
  const shown = load.status === 'ready' ? load.data.filter(f => !undoable.isRemoved(String(f.id))) : []
  const selection = useSelection(shown, f => String(f.id))
  const decide = (ids: number[], status: 'confirmed' | 'rejected') => {
    const n = format.count(ids.length, 'face')
    selection.clear()
    if (ids.includes(Number(open))) setOpen(null)
    undoable.schedule(status === 'confirmed'
      ? { id: `yes-${ids.join('-')}`, ids: ids.map(String), message: `Confirmed ${n} as ${who}.`, failure: `${n} are suggested again: confirming them as ${who} did not go through.`, commit: async () => { await facesApi.assign(ids, person.id, 'confirmed'); await qc.invalidateQueries({ queryKey: ['faces'] }) } }
      : { id: `no-${ids.join('-')}`, ids: ids.map(String), message: `Took ${n} off ${who}'s suggestions.`, failure: `${n} are suggested again: the change did not go through.`, commit: async () => { await facesApi.assign(ids, person.id, 'rejected'); await qc.invalidateQueries({ queryKey: ['faces'] }) } })
  }
  const all = useAsyncAction(async () => decide(await facesApi.allSuggested(person.id), 'confirmed'))
  const ids = selection.selectedRows.map(f => f.id)
  const pages = Math.max(1, Math.ceil(person.suggested / PAGE))
  return (
    <section aria-labelledby="suggested" className="grid gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 id="suggested" className="type-heading">{`Every suggestion: ${format.count(person.suggested, 'face')}`}</h2>
        <AsyncButton phase={all.phase} error={all.error} onPress={() => all.run()} labels={{ idle: `Confirm all ${format.integer(person.suggested)}`, pending: 'Gathering…', done: 'Confirmed' }} />
      </div>
      <p className="type-body-sm max-w-prose text-muted-foreground">{`An app's labels say ${who} is in these photos, and these are the faces closest to theirs. Choose any that are someone else and say so, or confirm the ones that are ${who}.`}</p>
      <DataView state={load} isEmpty={() => shown.length === 0} skeleton={<TableSkeleton rows={2} columns={6} />} empty={<EmptyState title="Nothing left to confirm on this page">{page > 0 ? 'Go back a page, or come back once the list has refreshed.' : 'Every suggestion here has been decided.'}</EmptyState>}>
        {() => (
          <>
            <BulkBar count={selection.count} noun="face" onClear={selection.clear}>
              {selection.count === 1 && <Button onPress={() => setOpen(String(ids[0]))}>Show in its photo</Button>}
              <Button variant="default" onPress={() => decide(ids, 'confirmed')}>{`Confirm selected (${format.integer(selection.count)})`}</Button>
              <Button onPress={() => decide(ids, 'rejected')}>{`Not this person (${format.integer(selection.count)})`}</Button>
            </BulkBar>
            <SelectableGrid
              label={`Faces suggested for ${who}`} rows={shown} rowKey={f => String(f.id)} selection={{ ...selection, rowLabel: f => `Face ${f.id}` }}
              renderImage={f => <Thumbnail src={facesApi.cropUrl(f.id)} alt="" missing="No picture" className="size-full" />}
              onAction={f => setOpen(String(f.id))} tileSize="sm"
            />
          </>
        )}
      </DataView>
      {pages > 1 && (
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" disabled={page === 0} onPress={() => { selection.clear(); setPage(page - 1) }}>Previous page</Button>
          <span className="type-caption text-muted-foreground">{`Page ${format.integer(page + 1)} of ${format.integer(pages)}`}</span>
          <Button size="sm" disabled={page >= pages - 1} onPress={() => { selection.clear(); setPage(page + 1) }}>Next page</Button>
        </div>
      )}
      <UndoBar undoable={undoable} />
    </section>
  )
}

export default function PersonPage() {
  const id = Number(useParams().id)
  const [open, setOpen] = useSearchParam<string>('face', { default: '', parse: isId })
  const person = useLoad({ queryKey: ['faces', 'person', id], queryFn: () => facesApi.person(id) })
  const [browse, setBrowse] = useState(false)
  const [all, setAll] = useState(false) // the best 500 at first; the service gives up to 5000
  const faces = useLoad({ queryKey: ['faces', 'faces-of', id, all], queryFn: () => facesApi.facesOf(id, all ? 5000 : 500) })
  const people = useLoad({ queryKey: ['faces', 'people', ''], queryFn: () => facesApi.people() })
  const names = new Map((people.status === 'ready' ? people.data : []).map(p => [p.id, p]))
  const p = person.status === 'ready' ? person.data : undefined
  const apps = p ? Object.keys(p.apps).sort() : []
  return (
    <div className="grid gap-8">
      <PageHeader title={p ? personName(p) : 'Person'} scope={<TextLink to="/faces">Every person</TextLink>} finding={p ? { text: `${format.count(p.faces, 'face')} confirmed${p.suggested ? `, ${format.integer(p.suggested)} suggested` : ''}${apps.length ? `, named in ${apps.join(', ')}` : ''}.` } : undefined} />
      {open && <InPhoto open={open} setOpen={setOpen} names={names} />}
      <DataView state={person} isEmpty={() => false} skeleton={<TableSkeleton rows={1} columns={2} />} empty={<EmptyState title="No such person" />}>
        {pp => <Rename key={pp.id} person={pp} />}
      </DataView>
      <DataView state={faces} isEmpty={f => f.length === 0} skeleton={<TableSkeleton rows={3} columns={6} />} empty={<EmptyState title="No faces confirmed yet">{p?.suggested ? 'Confirm the suggested faces below.' : <>Name a group of faces on <TextLink to="/faces?tab=groups">Unnamed groups</TextLink>.</>}</EmptyState>}>
        {f => (p ? (
          <div className="grid gap-4">
            {f.length < p.faces && (
              <p className="type-body-sm text-muted-foreground">
                {`Showing the best ${format.integer(f.length)} of ${format.count(p.faces, 'face')}. `}
                {!all && <Button variant="ghost" size="sm" onPress={() => setAll(true)}>Show them all</Button>}
              </p>
            )}
            <Faces person={p} faces={f} open={open} setOpen={setOpen} />
          </div>
        ) : null)}
      </DataView>
      {p && p.suggested > 0 && <Round key={p.id} person={p} open={open} setOpen={setOpen} />}
      {p && p.suggested > 0 && (browse
        ? <Suggested key={`all-${p.id}`} person={p} open={open} setOpen={setOpen} />
        : <div><Button variant="ghost" onPress={() => setBrowse(true)}>{`Browse every suggestion (${format.integer(p.suggested)})`}</Button></div>)}
    </div>
  )
}
