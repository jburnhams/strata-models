// One question: "Are these all the same person, and where were they?" Every face confirmed as this person (and anyone merged into them), best
// detected first, as crops cut by the service; a face opened (Enter, a double click, or "Show in its photo") shows in its photo above the
// grid, zoomed on it, with who is beside it. Faces that are someone else are chosen and taken off in one decision, with Undo while it waits
// (useUndoable): the decision is recorded as "not this person", so the face does not come back to them in a group.
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

function Faces({ person, faces, names }: { person: Person; faces: Face[]; names: Map<number, Person> }) {
  const qc = useQueryClient()
  const who = personName(person)
  const [open, setOpen] = useSearchParam<string>('face', { default: '', parse: isId })
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
      {open && (
        <section aria-labelledby="in-photo" className="grid gap-3">
          <div className="flex flex-wrap items-baseline justify-between gap-3"><h2 id="in-photo" className="type-heading">In its photo</h2><Button variant="ghost" size="sm" onPress={() => setOpen(null)}>Close the photo</Button></div>
          <FaceInPhoto faceId={Number(open)} names={names} />
        </section>
      )}
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

export default function PersonPage() {
  const id = Number(useParams().id)
  const person = useLoad({ queryKey: ['faces', 'person', id], queryFn: () => facesApi.person(id) })
  const [all, setAll] = useState(false) // the best 500 at first; the service gives up to 5000
  const faces = useLoad({ queryKey: ['faces', 'faces-of', id, all], queryFn: () => facesApi.facesOf(id, all ? 5000 : 500) })
  const people = useLoad({ queryKey: ['faces', 'people', ''], queryFn: () => facesApi.people() })
  const names = new Map((people.status === 'ready' ? people.data : []).map(p => [p.id, p]))
  const p = person.status === 'ready' ? person.data : undefined
  const apps = p ? Object.keys(p.apps).sort() : []
  return (
    <div className="grid gap-8">
      <PageHeader title={p ? personName(p) : 'Person'} scope={<TextLink to="/faces">Every person</TextLink>} finding={p ? { text: `${format.count(p.faces, 'face')} confirmed${apps.length ? `, named in ${apps.join(', ')}` : ''}.` } : undefined} />
      <DataView state={person} isEmpty={() => false} skeleton={<TableSkeleton rows={1} columns={2} />} empty={<EmptyState title="No such person" />}>
        {pp => <Rename key={pp.id} person={pp} />}
      </DataView>
      <DataView state={faces} isEmpty={f => f.length === 0} skeleton={<TableSkeleton rows={3} columns={6} />} empty={<EmptyState title="No faces confirmed yet">Name a group of faces on <TextLink to="/faces?tab=groups">Unnamed groups</TextLink>.</EmptyState>}>
        {f => (p ? (
          <div className="grid gap-4">
            {f.length < p.faces && (
              <p className="type-body-sm text-muted-foreground">
                {`Showing the best ${format.integer(f.length)} of ${format.count(p.faces, 'face')}. `}
                {!all && <Button variant="ghost" size="sm" onPress={() => setAll(true)}>Show them all</Button>}
              </p>
            )}
            <Faces person={p} faces={f} names={names} />
          </div>
        ) : null)}
      </DataView>
    </div>
  )
}
