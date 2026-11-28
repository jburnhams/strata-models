// One question: "Who is in the pictures, and what is left to decide?" One face store serves every app (docs/database.md#faces); this screen
// is where a person makes its decisions. People: everyone named, by how many faces are theirs. Unnamed groups: faces nobody has decided
// about, grouped by likeness, one group at a time (name it, add it to someone, or say it is not faces). Possible duplicates: two people whose
// faces sit close, to merge or keep apart; the service only suggests. Faces an app suggested for someone are decided on that person's page,
// so they are not counted as unnamed. Tags: which tags keep their pictures out of searches and groups unless
// an app names them. Every decision commits after an Undo window (useUndoable), and every one can be undone later through the store.
import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { AsyncButton, BulkBar, Button, DataTable, DataView, EmptyState, Field, Input, NativeSelect, PageHeader, SelectableGrid, Status, Switch, TableSkeleton, Tabs, TextLink, Thumbnail, UndoBar, format, useAsyncAction, useLoad, useSearchParam, useSelection, useUndoable, type Column } from 'strata-react'
import { facesApi, personName, type Group, type Person, type Suggestion, type Tag } from '../faces'
import { SourcePicker, useSource, withSource } from './SourcePicker'

const tabs = [
  { value: 'people', label: 'People' },
  { value: 'groups', label: 'Unnamed groups' },
  { value: 'duplicates', label: 'Possible duplicates' },
  { value: 'tags', label: 'Tags' },
]

const usePeople = () => useLoad({ queryKey: ['faces', 'people', ''], queryFn: () => facesApi.people() })
const byName = (a: Person, b: Person) => personName(a).localeCompare(personName(b))
const crop = (faceId: number) => <Thumbnail src={facesApi.cropUrl(faceId)} alt="" missing="No picture" className="size-full" />

// ---- people --------------------------------------------------------------------------------------------------------------------------------
const peopleColumns = (source: string): Column<Person>[] => [
  { key: 'name', header: 'Person', cell: p => <TextLink to={withSource(`/faces/people/${p.id}`, source)}>{personName(p)}</TextLink>, sort: byName },
  { key: 'faces', header: 'Faces', numeric: true, cell: p => format.integer(p.faces), sort: (a, b) => a.faces - b.faces },
  { key: 'suggested', header: 'Suggested', numeric: true, cell: p => (p.suggested ? format.integer(p.suggested) : format.missing), sort: (a, b) => a.suggested - b.suggested },
  { key: 'apps', header: 'Named in', cell: p => Object.keys(p.apps).sort().join(', ') || format.missing },
]

function People() {
  const [q, setQ] = useSearchParam<string>('q', { default: '' })
  const [text, setText] = useState(q)
  useEffect(() => { const t = setTimeout(() => setQ(text.trim() || null), 250); return () => clearTimeout(t) }, [text, setQ])
  const [source] = useSource()
  const load = useLoad({ queryKey: ['faces', 'people', q, source], queryFn: () => facesApi.people(q, source) })
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-end gap-4">
        <Field label="Find a person" className="max-w-sm"><Input type="search" value={text} onChange={e => setText(e.target.value)} /></Field>
        <SourcePicker />
      </div>
      <DataView state={load} isEmpty={p => p.length === 0} skeleton={<TableSkeleton rows={6} columns={3} />} empty={<EmptyState title={q ? `Nobody called “${q}”` : source ? `Nobody has faces from ${source}` : 'Nobody is named yet'}>{q ? 'Try part of the name.' : <>Name a group of faces on <TextLink to={withSource('/faces?tab=groups', source)}>Unnamed groups</TextLink>.</>}</EmptyState>}>
        {p => <DataTable caption={source ? `People with faces from ${source}` : 'People'} columns={peopleColumns(source)} rows={[...p].sort((a, b) => b.faces - a.faces || b.suggested - a.suggested || byName(a, b))} rowKey={x => String(x.id)} sortParam="sort" />}
      </DataView>
    </div>
  )
}

// ---- unnamed groups --------------------------------------------------------------------------------------------------------------------------
function GroupReview({ group, index, total, people, onDecide }: { group: Group; index: number; total: number; people: Person[]; onDecide: (what: { ids: number[]; message: string; failure: string; commit: () => Promise<unknown> }) => void }) {
  const rows = group.faces.map(id => ({ id }))
  const selection = useSelection(rows, r => String(r.id), () => true)
  const [name, setName] = useState('')
  const [into, setInto] = useState('')
  const ids = selection.selectedRows.map(r => r.id)
  const n = format.count(ids.length, 'face')
  const target = people.find(p => String(p.id) === into)
  const named = useAsyncAction(async () => {
    const who = name.trim()
    onDecide({ ids, message: `Named ${n} ${who}.`, failure: `${n} are back in the group: naming them ${who} did not go through.`, commit: async () => facesApi.assign(ids, (await facesApi.create(who)).id) })
  }, { validate: () => ({ name: name.trim() ? undefined : 'Give the person a name.' }) })
  const add = () => target && onDecide({ ids, message: `Added ${n} to ${personName(target)}.`, failure: `${n} are back in the group: adding them to ${personName(target)} did not go through.`, commit: () => facesApi.assign(ids, target.id) })
  const notFaces = () => onDecide({ ids, message: `Marked ${n} as not faces.`, failure: `${n} are back in the group: the change did not go through.`, commit: () => facesApi.assign(ids, null, 'rejected') })
  return (
    <section aria-labelledby="group" className="grid gap-4">
      <h2 id="group" className="type-heading">{`Group ${format.integer(index + 1)} of ${format.integer(total)}: ${format.count(group.size, 'face')}`}</h2>
      <p className="type-body-sm text-muted-foreground">Every face starts chosen. Untick any that are someone else, then say who the rest are.</p>
      <SelectableGrid label={`Faces in group ${index + 1}`} rows={rows} rowKey={r => String(r.id)} selection={{ ...selection, rowLabel: r => `Face ${r.id}` }} renderImage={r => crop(r.id)} tileSize="sm" />
      <BulkBar count={selection.count} noun="face" onClear={selection.clear}>
        <form className="flex flex-wrap items-end gap-3" onSubmit={e => { e.preventDefault(); named.run() }}>
          <Field label="A new person" error={named.fieldErrors.name} action={<AsyncButton type="submit" variant="default" phase={named.phase} labels={{ idle: `Name ${format.integer(ids.length)}`, pending: 'Naming…', done: 'Named' }} />}>
            <Input value={name} onChange={e => { setName(e.target.value); named.clearFieldError('name') }} />
          </Field>
        </form>
        {people.length > 0 && (
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Or someone named already"><NativeSelect value={into} onChange={e => setInto(e.target.value)}><option value="">Choose a person</option>{people.map(p => <option key={p.id} value={p.id}>{personName(p)}</option>)}</NativeSelect></Field>
            <Button disabled={!target} onPress={add}>{target ? `Add to ${personName(target)}` : 'Add to them'}</Button>
          </div>
        )}
        <Button variant="ghost" className="self-end" onPress={notFaces}>Not faces</Button>
      </BulkBar>
    </section>
  )
}

function Groups() {
  const [source] = useSource()
  return (
    <div className="grid gap-4">
      <SourcePicker />
      <SourceGroups key={source} source={source} />
    </div>
  )
}

function SourceGroups({ source }: { source: string }) {
  const qc = useQueryClient()
  const load = useLoad({ queryKey: ['faces', 'groups', source], queryFn: () => facesApi.groups(source), staleTime: 5 * 60_000 })
  const people = usePeople()
  const undoable = useUndoable()
  const [at, setAt] = useState(0)
  const undecided = (all: Group[]) => all.map(g => ({ faces: g.faces.filter(f => !undoable.isRemoved(String(f))) })).filter(g => g.faces.length > 0).map(g => ({ ...g, size: g.faces.length }))
  return (
    <div className="grid gap-4">
      <DataView state={load} isEmpty={g => undecided(g).length === 0} skeleton={<TableSkeleton rows={3} columns={6} />} empty={<EmptyState title={source ? `Every face from ${source} has been decided` : 'Every face has been decided'}>New pictures an app indexes bring new groups here.</EmptyState>}>
        {all => {
          const left = undecided(all)
          const i = Math.min(at, left.length - 1)
          const g = left[i]
          // A decision takes the faces it names out of view, not the group: the faces left unticked stay to be decided.
          const decide = (d: { ids: number[]; message: string; failure: string; commit: () => Promise<unknown> }) => undoable.schedule({ id: `group-${d.ids.join('-')}`, ids: d.ids.map(String), message: d.message, failure: d.failure, commit: async () => { await d.commit(); await qc.invalidateQueries({ queryKey: ['faces'] }) } })
          return (
            <>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" disabled={i === 0} onPress={() => setAt(i - 1)}>Previous group</Button>
                <Button size="sm" disabled={i >= left.length - 1} onPress={() => setAt(i + 1)}>Next group</Button>
              </div>
              <GroupReview key={g.faces.join('-')} group={g} index={i} total={left.length} people={people.status === 'ready' ? [...people.data].sort(byName) : []} onDecide={decide} />
            </>
          )
        }}
      </DataView>
      <UndoBar undoable={undoable} />
    </div>
  )
}

// ---- possible duplicates -----------------------------------------------------------------------------------------------------------------------
function Side({ person, id }: { person: Person | undefined; id: number }) {
  const faces = useLoad({ queryKey: ['faces', 'faces-of', id, 4], queryFn: () => facesApi.facesOf(id, 4) })
  return (
    <div className="grid content-start gap-2">
      <TextLink to={`/faces/people/${id}`}>{personName(person, id)}</TextLink>
      <span className="type-caption text-muted-foreground">{person ? format.count(person.faces, 'face') : format.missing}{person && Object.keys(person.apps).length ? ` · ${Object.keys(person.apps).sort().join(', ')}` : ''}</span>
      <div className="grid grid-cols-4 gap-1">{faces.status === 'ready' && faces.data.map(f => <span key={f.id}>{crop(f.id)}</span>)}</div>
    </div>
  )
}

function Duplicates() {
  const qc = useQueryClient()
  const load = useLoad({ queryKey: ['faces', 'suggestions'], queryFn: facesApi.suggestions })
  const people = usePeople()
  const undoable = useUndoable()
  const names = new Map((people.status === 'ready' ? people.data : []).map(p => [p.id, p]))
  const key = (s: Suggestion) => `${s.a}-${s.b}`
  const decide = (s: Suggestion, same: boolean) => {
    const a = names.get(s.a), b = names.get(s.b)
    // The one with more faces keeps the record; the other's faces are read through it from now on (unmerge gives them back).
    const [from, into] = (a?.faces ?? 0) >= (b?.faces ?? 0) ? [s.b, s.a] : [s.a, s.b]
    const [nf, ni] = [personName(names.get(from), from), personName(names.get(into), into)]
    undoable.schedule(same
      ? { id: key(s), message: `Merged ${nf} into ${ni}.`, failure: `${nf} and ${ni} are apart again: the merge did not go through.`, commit: async () => { await facesApi.merge(from, into); await qc.invalidateQueries({ queryKey: ['faces'] }) } }
      : { id: key(s), message: `Kept ${nf} and ${ni} apart.`, failure: `${nf} and ${ni} are back on the list: the change did not go through.`, commit: async () => { await facesApi.notSame(s.a, s.b); await qc.invalidateQueries({ queryKey: ['faces', 'suggestions'] }) } })
  }
  return (
    <div className="grid gap-4">
      <DataView state={load} isEmpty={l => l.filter(s => !undoable.isRemoved(key(s))).length === 0} skeleton={<TableSkeleton rows={3} columns={3} />} empty={<EmptyState title="Nobody looks like anyone else">Two people whose faces sit close show here, to merge or keep apart.</EmptyState>}>
        {l => (
          <ul className="grid gap-4" aria-label="Possible duplicates">
            {l.filter(s => !undoable.isRemoved(key(s))).map(s => (
              <li key={key(s)} className="grid gap-3 rounded-surface border border-border p-4 md:grid-cols-[1fr_1fr_auto]">
                <Side person={names.get(s.a)} id={s.a} />
                <Side person={names.get(s.b)} id={s.b} />
                <div className="grid content-start gap-2">
                  <Status tone={s.distance < 0.35 ? 'success' : 'neutral'}>{`${format.share(Math.max(0, 1 - s.distance))} alike`}</Status>
                  <Button onPress={() => decide(s, true)}>{`Merge ${personName(names.get(s.a), s.a)} and ${personName(names.get(s.b), s.b)}`}</Button>
                  <Button variant="ghost" onPress={() => decide(s, false)}>Different people</Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </DataView>
      <UndoBar undoable={undoable} />
    </div>
  )
}

// ---- tags ------------------------------------------------------------------------------------------------------------------------------------
function OptIn({ t }: { t: Tag }) {
  const qc = useQueryClient()
  const set = useAsyncAction(async (on: boolean) => { await facesApi.setTag(t.tag, on); await qc.invalidateQueries({ queryKey: ['faces'] }) })
  return (
    <span className="grid gap-1">
      <Switch label={`Only when asked for: ${t.tag}`} on={t.opt_in} disabled={set.phase === 'pending'} onChange={on => set.run(on)} />
      {set.error && <span role="alert" className="type-caption text-danger">{set.error}</span>}
    </span>
  )
}

// An opt-in tag can still lend the faces a person named to every search (owner, 2026-11-28): a family tape's named faces help
// recognise those people in other apps, while its unnamed faces (other people's children) stay out. Meaningless for a tag searched anyway.
function ShareNamed({ t }: { t: Tag }) {
  const qc = useQueryClient()
  const set = useAsyncAction(async (on: boolean) => { await facesApi.setTag(t.tag, t.opt_in, on); await qc.invalidateQueries({ queryKey: ['faces'] }) })
  if (!t.opt_in) return <span className="type-body-sm text-muted-foreground">Searched anyway</span>
  return (
    <span className="grid gap-1">
      <Switch label={`Named faces searched everywhere: ${t.tag}`} on={t.share_named} disabled={set.phase === 'pending'} onChange={on => set.run(on)} />
      {set.error && <span role="alert" className="type-caption text-danger">{set.error}</span>}
    </span>
  )
}

const tagColumns: Column<Tag>[] = [
  { key: 'tag', header: 'Tag', cell: t => <span className="font-mono">{t.tag}</span>, sort: (a, b) => a.tag.localeCompare(b.tag) },
  { key: 'images', header: 'Pictures', numeric: true, cell: t => format.integer(t.images), sort: (a, b) => a.images - b.images },
  { key: 'opt', header: 'Searched', cell: t => <OptIn t={t} /> },
  { key: 'shared', header: 'Named faces', cell: t => <ShareNamed t={t} /> },
]

function Tags() {
  const load = useLoad({ queryKey: ['faces', 'tags'], queryFn: facesApi.tags })
  return (
    <div className="grid gap-4">
      <p className="type-body-sm max-w-prose text-muted-foreground">Every picture carries its app's name and the app's own tags. A tag that is only searched when asked for keeps its pictures out of every search and group unless an app names it, for pictures that should not turn up anywhere else.</p>
      <DataView state={load} isEmpty={t => t.length === 0} skeleton={<TableSkeleton rows={4} columns={3} />} empty={<EmptyState title="No tags yet">Apps tag their pictures when they index them.</EmptyState>}>
        {t => <DataTable caption="Tags" columns={tagColumns} rows={t} rowKey={x => x.tag} />}
      </DataView>
    </div>
  )
}

export default function FacesPage() {
  const people = usePeople()
  const suggested = people.status === 'ready' ? people.data.reduce((n, p) => n + p.suggested, 0) : 0
  const finding = people.status === 'ready' ? { text: `${format.count(people.data.length, 'person', 'people')} named, with ${format.count(people.data.reduce((n, p) => n + p.faces, 0), 'face')} confirmed${suggested ? ` and ${format.integer(suggested)} suggested to confirm` : ''}.` } : undefined
  return (
    <div className="grid gap-8">
      <PageHeader title="Faces" scope="Every app's pictures, in one store" finding={finding} />
      <Tabs param="tab" label="Faces" items={tabs}>
        {v => (v === 'groups' ? <Groups /> : v === 'duplicates' ? <Duplicates /> : v === 'tags' ? <Tags /> : <People />)}
      </Tabs>
    </div>
  )
}
