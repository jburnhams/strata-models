// One question: "Where is this face, and who is beside it?" A face opened from a grid lands zoomed on that face in its photo (strata-react's
// BoxedPicture, which every app that shows a face uses), with every face the photo holds outlined and listed beside it by the person's name,
// so choosing one in the list or the picture moves to it. The photo is the service's kept 1600 px copy: an app's original may be on another
// machine, and the copy is what is shown when it is not.
import { BoxList, BoxedPicture, DataView, EmptyState, ThumbnailSkeleton, format, useBoxes, useLoad, type PictureBox } from 'strata-react'
import { facesApi, personName, type FaceImage, type Person } from '../faces'

function boxesOf(img: FaceImage, names: Map<number, Person>): PictureBox[] {
  return img.faces
    .filter(f => f.status !== 'rejected' || f.identity_id !== null) // a "not a face" decision is not drawn
    .map(f => {
      const [x0, y0, x1, y1] = f.box
      const who = f.status === 'confirmed' && f.identity_id !== null ? personName(names.get(f.identity_id), f.identity_id) : 'Not named yet'
      return { id: String(f.id), label: who, x: x0, y: y0, w: x1 - x0, h: y1 - y0, tone: f.status === 'confirmed' ? 'success' : 'neutral' }
    })
}

function Photo({ img, faceId, names }: { img: FaceImage; faceId: number; names: Map<number, Person> }) {
  const state = useBoxes(String(faceId))
  const boxes = boxesOf(img, names)
  const taken = img.taken_at ? format.dateTime(new Date(img.taken_at * 1000)) : null
  const label = `Photo from ${img.app}${taken ? `, ${taken}` : ''}`
  return (
    <div className="grid gap-4 md:grid-cols-[1fr_14rem]">
      <BoxedPicture label={label} src={img.medium ? facesApi.pictureUrl(img.id) : null} alt="" missing="No copy of this photo is kept" of={{ w: img.width ?? 1, h: img.height ?? 1 }} boxes={boxes} state={state} boxesLabel="Faces in this photo" maxHeight="60svh" />
      <BoxList label="Faces in this photo" boxes={boxes} state={state} />
    </div>
  )
}

export function FaceInPhoto({ faceId, names }: { faceId: number; names: Map<number, Person> }) {
  const load = useLoad({ queryKey: ['faces', 'photo-of', faceId], queryFn: async () => facesApi.image((await facesApi.face(faceId)).image_id) })
  return (
    <DataView state={load} isEmpty={img => !img.width || !img.height} skeleton={<ThumbnailSkeleton aspect={1.5} className="w-full" />} empty={<EmptyState title="This photo's size is not known">An app indexed it without its size, so its faces cannot be placed on it.</EmptyState>}>
      {img => <Photo key={faceId} img={img} faceId={faceId} names={names} />}
    </DataView>
  )
}
