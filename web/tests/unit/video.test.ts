import { describe, expect, it } from 'vitest'
import { sharedRegions, srcFor, videoOf, WHOLE, type VideoSide } from '../../src/components/VideoCompare'
import { regionLabels, sourceSide, videoSide } from '../../src/components/EvalPage'

describe('video outputs', () => {
  it('reads a video-enhance output and its crops, and nothing else', () => {
    expect(videoOf({ path: '/g/v.mp4', width: 1440, height: 1080, fps: 50, crops: [{ id: 'face', label: 'Face', path: '/g/v.face.mp4' }, { id: 'nolabel', path: '/g/v.n.mp4' }, { id: 3 }, null] }))
      .toEqual({ path: '/g/v.mp4', width: 1440, height: 1080, fps: 50, crops: [{ id: 'face', label: 'Face', path: '/g/v.face.mp4' }, { id: 'nolabel', label: 'nolabel', path: '/g/v.n.mp4' }] })
    expect(videoOf({ path: '/g/v.MOV' })).toEqual({ path: '/g/v.MOV', width: undefined, height: undefined, fps: undefined, crops: [] })
    expect(videoOf({ path: '/g/p.png' })).toBeNull()
    expect(videoOf(null)).toBeNull()
    expect(videoOf('x')).toBeNull()
  })
  it('offers only the regions every side has, and falls back to the whole frame', () => {
    const a: VideoSide = { id: 'a', label: 'A', whole: 'a.mp4', regions: { face: 'a.face.mp4', text: 'a.text.mp4' } }
    const b: VideoSide = { id: 'b', label: 'B', whole: 'b.mp4', regions: { face: 'b.face.mp4' } }
    expect(sharedRegions([a, b], { face: 'Face' })).toEqual([{ value: 'face', label: 'Face' }])
    expect(sharedRegions([a], {})).toEqual([{ value: 'face', label: 'face' }, { value: 'text', label: 'text' }])
    expect(sharedRegions([], {})).toEqual([])
    expect(srcFor(a, 'face')).toBe('a.face.mp4')
    expect(srcFor(b, 'text')).toBe('b.mp4')
    expect(srcFor(a, WHOLE)).toBe('a.mp4')
  })
  it('makes sides from outputs and the tape from the reference, with region labels', () => {
    expect(videoSide('x', 'X', { path: '/g/x.mp4', crops: [{ id: 'face', label: 'Face', path: '/g/x.face.mp4' }] }, '3 s')).toEqual({ id: 'x', label: 'X', note: '3 s', whole: '/api/images/file?path=%2Fg%2Fx.mp4', regions: { face: '/api/images/file?path=%2Fg%2Fx.face.mp4' } })
    expect(videoSide('x', 'X', { segments: [] })).toBeNull()
    const src = sourceSide('vhs', 'c1', { preview: '/m/c1.source.mp4', crops: [{ id: 'face', path: '/m/c1.source.face.mp4' }, { id: 1 }] })
    expect(src?.whole).toBe('/api/evals/vhs/media/c1?file=%2Fm%2Fc1.source.mp4')
    expect(src?.regions).toEqual({ face: '/api/evals/vhs/media/c1?file=%2Fm%2Fc1.source.face.mp4' })
    expect(sourceSide('vhs', 'c1', { preview: '/m/p.mp4' })?.regions).toEqual({})
    expect(sourceSide('vhs', 'c1', { text: 'x' })).toBeNull()
    expect(sourceSide('vhs', 'c1', null)).toBeNull()
    expect(regionLabels({ crops: [{ id: 'face', label: 'Face' }, { id: 'x' }] }, null, 'x', { crops: [{ id: 'text', label: 'Text' }] })).toEqual({ face: 'Face', text: 'Text' })
  })
})

describe('regions for CompareVideo', () => {
  it('moves source-pixel boxes into the cropped frame, clipped, and leaves out what is outside', async () => {
    const { regionsOf } = await import('../../src/components/EvalPage')
    expect(regionsOf({ crop: [14, 0, 692, 568], regions: [{ id: 'face', label: 'Face', box: [200, 90, 260, 195] }, { id: 'edge', box: [0, 500, 100, 200] }, { id: 'out', box: [710, 0, 10, 10] }, { box: [1, 2, 3, 4] }, null] }))
      .toEqual([{ id: 'face', label: 'Face', box: [186, 90, 260, 195], of: [692, 568] }, { id: 'edge', label: 'edge', box: [0, 500, 86, 68], of: [692, 568] }])
    expect(regionsOf({ regions: [{ id: 'face', box: [0, 0, 10, 10] }] })).toEqual([]) // no crop: the source's size is not known
    expect(regionsOf({ crop: [0, 0, 720, 576] })).toEqual([])
  })
})

describe('the whole clip as a loop', () => {
  it('runs from 0 to the shorter output\'s last frame, or nothing when a length is missing', async () => {
    const { wholeLoop } = await import('../../src/components/EvalPage')
    const w = wholeLoop({ duration_s: 8 }, { duration_s: 7.98 }, 50)
    expect(w?.start).toBe(0)
    expect(w?.end).toBeCloseTo(7.96, 6)
    expect(wholeLoop({ duration_s: 0.01 }, { duration_s: 8 }, 50)).toEqual({ start: 0, end: 0 })
    expect(wholeLoop({ duration_s: 8 }, {}, 50)).toBeNull()
    expect(wholeLoop(null, { duration_s: 0 })).toBeNull()
  })
})

describe('the loop a judge sees', () => {
  it('keeps a half-set range (a new in-point mid edit), and falls back to the whole clip when none or cleared', async () => {
    const { judgeLoop } = await import('../../src/components/EvalPage')
    const whole = { start: 0, end: 7.96 }
    expect(judgeLoop({ start: 5, end: null }, whole)).toEqual({ start: 5, end: null })
    expect(judgeLoop({ start: null, end: 6 }, whole)).toEqual({ start: null, end: 6 })
    expect(judgeLoop({ start: 5, end: 6 }, whole)).toEqual({ start: 5, end: 6 })
    expect(judgeLoop({ start: null, end: null }, whole)).toEqual(whole)
    expect(judgeLoop(null, whole)).toEqual(whole)
    expect(judgeLoop(null, null)).toBeUndefined()
  })
})
