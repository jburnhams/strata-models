// The face store as the Faces screens see it: three people (one with three suggested faces), one photo with three faces on it, two unnamed
// groups, one possible duplicate and two tags, plus msw handlers for every face endpoint the app calls. Writes are recorded in `faceMock.calls` and change the state, so a test
// sees what a person would after the request.
import { http, HttpResponse } from 'msw'
import type { Face, FaceImage, Group, Person, Suggestion, Tag } from '../../src/faces'

const person = (o: Partial<Person>): Person => ({ id: 1, name: 'Ada', fields: {}, apps: {}, merged_into: null, merged: [], faces: 0, suggested: 0, ...o })
const face = (o: Partial<Face>): Face => ({ id: 11, image_id: 100, box: [100, 100, 300, 340], det_score: 0.9, found_by: 'detector', identity_id: null, status: null, ...o })

const initial = () => ({
  people: [person({ id: 1, name: 'Ada', faces: 3, apps: { 'bright-horizons': { app_ref: 'c1', fields: { room: 'Robins' } } } }), person({ id: 2, name: null, faces: 1 }), person({ id: 3, name: 'Cy', faces: 2, suggested: 3, apps: { 'strata-360': { app_ref: null, fields: {} } } })],
  faces: {
    1: [face({ id: 11, identity_id: 1, status: 'confirmed' }), face({ id: 12, image_id: 101, identity_id: 1, status: 'confirmed' }), face({ id: 13, image_id: 102, identity_id: 1, status: 'confirmed' })],
    2: [face({ id: 14, image_id: 101, identity_id: 2, status: 'confirmed' })],
    3: [face({ id: 15, image_id: 103, identity_id: 3, status: 'confirmed' }), face({ id: 16, image_id: 103, identity_id: 3, status: 'confirmed' })],
  } as Record<number, Face[]>,
  suggested: { 3: [face({ id: 61, image_id: 100, identity_id: 3, status: 'suggested' }), face({ id: 62, image_id: 101, identity_id: 3, status: 'suggested' }), face({ id: 63, image_id: 102, identity_id: 3, status: 'suggested' })] } as Record<number, Face[]>,
  images: {
    100: { id: 100, app: 'bright-horizons', ref: 'r1', path: '/p/a.jpg', width: 2000, height: 1000, taken_at: 1_700_000_000, medium: 'faces/medium/100.webp', tags: ['bright-horizons'], faces: [face({ id: 11, identity_id: 1, status: 'confirmed' }), face({ id: 21, box: [900, 200, 1100, 450] }), face({ id: 22, box: [1500, 600, 1560, 660], status: 'rejected' })] },
    101: { id: 101, app: 'strata-360', ref: 'r2', path: null, width: null, height: null, taken_at: null, medium: null, tags: [], faces: [] },
    102: { id: 102, app: 'strata-360', ref: 'r3', path: null, width: 640, height: 480, taken_at: null, medium: null, tags: [], faces: [face({ id: 13, image_id: 102, box: [10, 10, 100, 120], identity_id: 1, status: 'confirmed' }), face({ id: 23, image_id: 102, box: [300, 10, 400, 120], identity_id: 3, status: 'confirmed' })] },
  } as Record<number, FaceImage>,
  groups: [{ faces: [31, 32, 33], size: 3 }, { faces: [41, 42], size: 2 }] as Group[],
  suggestions: [{ a: 1, b: 3, distance: 0.3, apps_a: ['bright-horizons'], apps_b: ['strata-360'] }] as Suggestion[],
  tags: [{ tag: 'bright-horizons', images: 40, opt_in: false, note: '' }, { tag: 'nursery', images: 5, opt_in: true, note: 'children' }] as Tag[],
  calls: [] as string[],
  fail: null as string | null, // the path of a write that answers 500
})

export const faceMock = initial()
export const resetFaces = () => Object.assign(faceMock, initial())

const failing = (path: string) => (faceMock.fail === path ? HttpResponse.json({ detail: 'the face store is busy' }, { status: 500 }) : null)
const notFound = (what: string) => HttpResponse.json({ detail: `no such ${what}` }, { status: 404 })

export const faceHandlers = [
  http.get('/api/identities', ({ request }) => {
    const q = (new URL(request.url).searchParams.get('q') || '').toLowerCase()
    return HttpResponse.json({ items: faceMock.people.filter(p => !q || (p.name || '').toLowerCase().includes(q)) })
  }),
  http.post('/api/identities', async ({ request }) => {
    const b = (await request.json()) as { name: string }
    faceMock.calls.push(`create ${b.name}`)
    const p = person({ id: 50 + faceMock.people.length, name: b.name })
    faceMock.people.push(p)
    return HttpResponse.json(p, { status: 201 })
  }),
  http.get('/api/identities/merge-suggestions', () => HttpResponse.json({ model: 'buffalo_l', items: faceMock.suggestions })),
  http.post('/api/identities/not-same', async ({ request }) => {
    const b = (await request.json()) as { a: number; b: number }
    faceMock.calls.push(`not-same ${b.a} ${b.b}`)
    faceMock.suggestions = faceMock.suggestions.filter(s => !(s.a === b.a && s.b === b.b))
    return HttpResponse.json({ a: b.a, b: b.b })
  }),
  http.get('/api/identities/:id', ({ params }) => { const p = faceMock.people.find(x => x.id === Number(params.id)); return p ? HttpResponse.json(p) : notFound('person') }),
  http.patch('/api/identities/:id', async ({ params, request }) => {
    const f = failing('rename'); if (f) return f
    const b = (await request.json()) as { name: string }
    faceMock.calls.push(`rename ${params.id} ${b.name}`)
    faceMock.people = faceMock.people.map(p => (p.id === Number(params.id) ? { ...p, name: b.name } : p))
    return HttpResponse.json(faceMock.people.find(p => p.id === Number(params.id)))
  }),
  http.get('/api/identities/:id/review', ({ params, request }) => {
    const q = new URL(request.url).searchParams
    const id = Number(params.id), limit = Number(q.get('limit') || 48), min = q.get('min_likeness'), tag = q.get('tag')
    faceMock.calls.push(`review ${id} up to ${limit}${min ? ` at ${min}` : ''}${tag ? ` from ${tag}` : ''}`)
    const scored = (faceMock.suggested[id] ?? []).map((f, i) => ({ ...f, distance: 0.1 + i * 0.15 }))
    const counts = Object.fromEntries([0.5, 0.6, 0.7, 0.8, 0.9].map(x => [x.toFixed(2), scored.filter(f => 1 - f.distance >= x - 1e-9).length]))
    const items = scored.filter(f => !min || 1 - f.distance >= Number(min) - 1e-9).slice(0, limit)
    const confirmed = (faceMock.faces[id] ?? []).length
    return HttpResponse.json({ basis: items.length ? (confirmed >= 3 ? 'confirmed' : 'group') : null, basis_faces: confirmed >= 3 ? confirmed : items.length, suggested: items.length, confirmed, near: { '0.3': items.filter(f => f.distance < 0.3).length, '0.4': items.filter(f => f.distance < 0.4).length, '0.5': items.length }, counts, left_out: 0, items })
  }),
  http.get('/api/identities/:id/faces', ({ params, request }) => {
    const q = new URL(request.url).searchParams
    if (faceMock.fail === 'faces' && q.get('offset')) return HttpResponse.json({ detail: 'the face store is busy' }, { status: 500 })
    const limit = Number(q.get('limit') || 500), offset = Number(q.get('offset') || 0), status = q.get('status') || 'confirmed'
    faceMock.calls.push(status === 'confirmed' ? `faces of ${params.id} up to ${limit}` : `${status} of ${params.id} from ${offset} up to ${limit}`)
    const from = status === 'suggested' ? faceMock.suggested : faceMock.faces
    return HttpResponse.json({ items: (from[Number(params.id)] ?? []).slice(offset, offset + limit) })
  }),
  http.post('/api/identities/:id/merge', async ({ params, request }) => {
    const b = (await request.json()) as { into: number; app: string }
    faceMock.calls.push(`merge ${params.id} into ${b.into} by ${b.app}`)
    faceMock.suggestions = []
    return HttpResponse.json(faceMock.people.find(p => p.id === b.into))
  }),
  http.post('/api/faces/assign', async ({ request }) => {
    const f = failing('assign'); if (f) return f
    const b = (await request.json()) as { face_ids: number[]; identity_id: number | null; status: string; app: string }
    faceMock.calls.push(`assign ${b.face_ids.join(',')} to ${b.identity_id} ${b.status} by ${b.app}`)
    for (const k of Object.keys(faceMock.faces)) faceMock.faces[Number(k)] = faceMock.faces[Number(k)].filter(x => !(b.status === 'rejected' && b.face_ids.includes(x.id)))
    for (const k of Object.keys(faceMock.suggested)) faceMock.suggested[Number(k)] = faceMock.suggested[Number(k)].filter(x => !b.face_ids.includes(x.id))
    faceMock.people = faceMock.people.map(p => ({ ...p, suggested: (faceMock.suggested[p.id] ?? []).length || (p.id in faceMock.suggested ? 0 : p.suggested) }))
    faceMock.groups = faceMock.groups.map(g => ({ faces: g.faces.filter(x => !b.face_ids.includes(x)), size: g.faces.filter(x => !b.face_ids.includes(x)).length })).filter(g => g.size > 0)
    return HttpResponse.json({ batch: 9 })
  }),
  http.post('/api/faces/cluster', async ({ request }) => {
    const b = (await request.json()) as { app: string; suggested?: boolean; tags?: string[] }
    faceMock.calls.push(`cluster by ${b.app}${b.suggested === false ? ' without suggested' : ''}${b.tags ? ` from ${b.tags.join(',')}` : ''}`)
    return HttpResponse.json({ model: 'buffalo_l', clusters: b.tags ? faceMock.groups.slice(0, 1) : faceMock.groups })
  }),
  http.get('/api/faces/tags', () => HttpResponse.json({ items: faceMock.tags })),
  http.put('/api/faces/tags/:tag', async ({ params, request }) => {
    const f = failing('tag'); if (f) return f
    const b = (await request.json()) as { opt_in: boolean }
    faceMock.calls.push(`tag ${params.tag} ${b.opt_in}`)
    faceMock.tags = faceMock.tags.map(t => (t.tag === params.tag ? { ...t, opt_in: b.opt_in } : t))
    return HttpResponse.json({ tag: params.tag, opt_in: b.opt_in })
  }),
  http.get('/api/faces/images/:id', ({ params }) => { const i = faceMock.images[Number(params.id)]; return i ? HttpResponse.json(i) : notFound('picture') }),
  http.get('/api/faces/:id', ({ params }) => {
    const f = Object.values(faceMock.faces).flat().find(x => x.id === Number(params.id))
    return f ? HttpResponse.json(f) : notFound('face')
  }),
]
