// The face store's API (src/stratamodels/faces/api.py, docs/database.md#faces) as the web app uses it: people, their faces, the groups
// nobody has named, the pairs of people that look like one person, and the opt-in tags. Everything the app writes is in the app's own name
// (APP), so a decision made here is told apart from one an app made.
import { withBase } from './api'

export const APP = 'strata-models'

export interface Face { id: number; image_id: number; box: [number, number, number, number]; det_score: number; found_by: string; identity_id: number | null; status: 'confirmed' | 'rejected' | 'suggested' | null }
export interface FaceImage { id: number; app: string; ref: string; path: string | null; width: number | null; height: number | null; taken_at: number | null; medium: string | null; tags: string[]; faces: Face[] }
export interface Person { id: number; name: string | null; fields: Record<string, unknown>; apps: Record<string, { app_ref: string | null; fields: Record<string, unknown> }>; merged_into: number | null; merged: number[]; faces: number }
export interface Group { faces: number[]; size: number }
export interface Suggestion { a: number; b: number; distance: number; apps_a: string[]; apps_b: string[] }
export interface Tag { tag: string; images: number; opt_in: boolean; note: string }

async function call<T>(method: string, u: string, body?: unknown): Promise<T> {
  const r = await fetch(withBase(u), { method, headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
  const d = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error((d as { detail?: string }).detail || `${r.status} ${r.statusText}`)
  return d as T
}

export const facesApi = {
  people: (q = '') => call<{ items: Person[] }>('GET', `/api/identities?limit=500${q ? `&q=${encodeURIComponent(q)}` : ''}`).then(r => r.items),
  person: (id: number) => call<Person>('GET', `/api/identities/${id}`),
  facesOf: (id: number, limit = 500) => call<{ items: Face[] }>('GET', `/api/identities/${id}/faces?limit=${limit}`).then(r => r.items),
  rename: (id: number, name: string) => call<Person>('PATCH', `/api/identities/${id}`, { name }),
  create: (name: string) => call<Person>('POST', '/api/identities', { name }),
  /** One decision on some faces: theirs (`confirmed`), not theirs (`rejected` with the person), or not a face at all (`rejected`, nobody). */
  assign: (faceIds: number[], identityId: number | null, status: 'confirmed' | 'rejected' = 'confirmed') => call<{ batch: number }>('POST', '/api/faces/assign', { face_ids: faceIds, identity_id: identityId, app: APP, status }),
  face: (id: number) => call<Face>('GET', `/api/faces/${id}`),
  image: (id: number) => call<FaceImage>('GET', `/api/faces/images/${id}`),
  groups: () => call<{ clusters: Group[] }>('POST', '/api/faces/cluster', { app: APP }).then(r => r.clusters),
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
