// The face store's API (src/stratamodels/faces/api.py, docs/database.md#faces) as the web app uses it: people, their faces, the groups
// nobody has named, the pairs of people that look like one person, and the opt-in tags. Everything the app writes is in the app's own name
// (APP), so a decision made here is told apart from one an app made.
import { withBase } from './api'

export const APP = 'strata-models'

export interface Face { id: number; image_id: number; box: [number, number, number, number]; det_score: number; found_by: string; identity_id: number | null; status: 'confirmed' | 'rejected' | 'suggested' | null }
export interface FaceImage { id: number; app: string; ref: string; path: string | null; width: number | null; height: number | null; taken_at: number | null; medium: string | null; tags: string[]; faces: Face[] }
export interface Person { id: number; name: string | null; fields: Record<string, unknown>; apps: Record<string, { app_ref: string | null; fields: Record<string, unknown> }>; merged_into: number | null; merged: number[]; faces: number; suggested: number }
/** The next round of a person's suggested faces, most certain first: `distance` to their model, built from their confirmed faces
 * (`basis` confirmed) or, before 3 are confirmed, from the tightest group of their suggestions (`basis` group). */
export interface Review { basis: 'confirmed' | 'group' | null; basis_faces: number; suggested: number; confirmed: number; near: Record<string, number>; counts: Record<string, number>; left_out?: number; items: (Face & { distance: number })[] }
export interface Group { faces: number[]; size: number }
export interface Suggestion { a: number; b: number; distance: number; apps_a: string[]; apps_b: string[] }
export interface Tag { tag: string; images: number; opt_in: boolean; note: string }

async function call<T>(method: string, u: string, body?: unknown): Promise<T> {
  const r = await fetch(withBase(u), { method, headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
  const d = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error((d as { detail?: string }).detail || `${r.status} ${r.statusText}`)
  return d as T
}

const src = (source: string) => (source ? `&tag=${encodeURIComponent(source)}` : '')

export const facesApi = {
  /** People, or with `source` (a tag: an app's name or a tag it gave) only those with faces there, counting those faces. */
  people: (q = '', source = '') => call<{ items: Person[] }>('GET', `/api/identities?limit=500${q ? `&q=${encodeURIComponent(q)}` : ''}${src(source)}`).then(r => r.items),
  person: (id: number) => call<Person>('GET', `/api/identities/${id}`),
  /** A person's faces in one state, best detected first: `confirmed` (theirs) or `suggested` (an app's proposal waiting for a person). */
  facesOf: (id: number, limit = 500, status: 'confirmed' | 'suggested' = 'confirmed', offset = 0) =>
    call<{ items: Face[] }>('GET', `/api/identities/${id}/faces?limit=${limit}${status === 'confirmed' ? '' : `&status=${status}`}${offset ? `&offset=${offset}` : ''}`).then(r => r.items),
  /** Every face suggested for a person (the service answers up to 5000 a call). */
  allSuggested: async (id: number): Promise<number[]> => {
    const out: number[] = []
    for (let offset = 0; ; offset += 5000) {
      const page = await facesApi.facesOf(id, 5000, 'suggested', offset)
      out.push(...page.map(f => f.id))
      if (page.length < 5000) return out
    }
  },
  /** The next round of a person's suggestions (`limit` most certain), or with `minLikeness` every one at least that alike. */
  review: (id: number, limit = 48, minLikeness?: number, source = '') => call<Review>('GET', `/api/identities/${id}/review?limit=${limit}${minLikeness === undefined ? '' : `&min_likeness=${minLikeness}`}${src(source)}`),
  rename: (id: number, name: string) => call<Person>('PATCH', `/api/identities/${id}`, { name }),
  create: (name: string) => call<Person>('POST', '/api/identities', { name }),
  /** One decision on some faces: theirs (`confirmed`), not theirs (`rejected` with the person), or not a face at all (`rejected`, nobody). */
  assign: (faceIds: number[], identityId: number | null, status: 'confirmed' | 'rejected' = 'confirmed') => call<{ batch: number }>('POST', '/api/faces/assign', { face_ids: faceIds, identity_id: identityId, app: APP, status }),
  face: (id: number) => call<Face>('GET', `/api/faces/${id}`),
  image: (id: number) => call<FaceImage>('GET', `/api/faces/images/${id}`),
  /** Faces nobody has decided about, grouped; faces waiting on a suggestion are decided on their person's page, so they are left out. */
  groups: (source = '') => call<{ clusters: Group[] }>('POST', '/api/faces/cluster', { app: APP, suggested: false, ...(source ? { tags: [source] } : {}) }).then(r => r.clusters),
  suggestions: () => call<{ items: Suggestion[] }>('GET', '/api/identities/merge-suggestions').then(r => r.items),
  merge: (id: number, into: number) => call<Person>('POST', `/api/identities/${id}/merge`, { into, app: APP }),
  notSame: (a: number, b: number) => call<unknown>('POST', '/api/identities/not-same', { a, b }),
  tags: () => call<{ items: Tag[] }>('GET', '/api/faces/tags').then(r => r.items),
  setTag: (tag: string, optIn: boolean) => call<unknown>('PUT', `/api/faces/tags/${encodeURIComponent(tag)}`, { opt_in: optIn }),
  cropUrl: (faceId: number, px = 192) => withBase(`/api/faces/${faceId}/crop?px=${px}`),
  pictureUrl: (imageId: number) => withBase(`/api/faces/images/${imageId}/picture`),
}

/** A person's name as a sentence uses it: an unnamed person is still someone ("Person 12"). */
export const personName = (p: Pick<Person, 'id' | 'name'> | undefined, id?: number) => p?.name || `Person ${p?.id ?? id}`
