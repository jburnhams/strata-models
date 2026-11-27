// One question: "Which pictures am I deciding about?" A source is a tag of the face store: every app's name (immich, whatsapp, ...) and the
// tags an app gave (a video eval's stills, one clip of it). Held in the address (`?source=`), so People, Unnamed groups and a person's
// page all show the same source, and a link keeps it.
import { Field, NativeSelect, format, useLoad, useSearchParam } from 'strata-react'
import { facesApi } from '../faces'

export const useSource = () => useSearchParam<string>('source', { default: '' })

export function SourcePicker() {
  const [source, setSource] = useSource()
  const tags = useLoad({ queryKey: ['faces', 'tags'], queryFn: facesApi.tags, staleTime: 5 * 60_000 })
  const items = tags.status === 'ready' ? [...tags.data].sort((a, b) => a.tag.localeCompare(b.tag)) : []
  const known = items.some(t => t.tag === source)
  return (
    <Field label="Source" hint="Only faces from these pictures: an app, or a tag an app gave them." className="max-w-sm">
      <NativeSelect value={source} onChange={e => setSource(e.target.value || null)}>
        <option value="">Every source</option>
        {source && !known && <option value={source}>{source}</option>}
        {items.map(t => <option key={t.tag} value={t.tag}>{`${t.tag} (${format.count(t.images, 'picture')})`}</option>)}
      </NativeSelect>
    </Field>
  )
}

/** A link's `?source=` part, so a person opened from a list keeps the source chosen there. */
export const withSource = (path: string, source: string) => (source ? `${path}${path.includes('?') ? '&' : '?'}source=${encodeURIComponent(source)}` : path)
