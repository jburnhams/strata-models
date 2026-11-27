// Which pictures' faces to decide: everywhere, or one source (a tag; every picture carries its app's name as one, so an app is a source too,
// and an app's own tags narrow it further, such as one video's stills). Kept in the URL (?source=) so a link opens on the same source.
import { Field, NativeSelect, format, useLoad, useSearchParam } from 'strata-react'
import { facesApi } from '../faces'

export const useSource = () => useSearchParam<string>('source', { default: '' })

export function SourcePicker({ source, setSource }: { source: string; setSource: (v: string) => void }) {
  const tags = useLoad({ queryKey: ['faces', 'tags'], queryFn: facesApi.tags })
  const items = tags.status === 'ready' ? [...tags.data].sort((a, b) => a.tag.localeCompare(b.tag)) : []
  return (
    <Field label="Pictures from" className="max-w-sm">
      <NativeSelect value={source} onChange={e => setSource(e.target.value)}>
        <option value="">Everywhere</option>
        {source && !items.some(t => t.tag === source) && <option value={source}>{source}</option>}
        {items.map(t => <option key={t.tag} value={t.tag}>{`${t.tag} (${format.count(t.images, 'picture')})`}</option>)}
      </NativeSelect>
    </Field>
  )
}
