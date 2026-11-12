import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../../src/App'
import { mock, server, http, HttpResponse, evalFixtures } from '../utils/server'
import { act, screen, setup, waitFor, within } from '../utils/render'

// jsdom has <video> but no media pipeline: play and pause are "not implemented" and the times are fixed. Give each element a time that the
// test moves by hand, a duration and an `ended` flag, so the shared clock can be watched pulling followers back and looping.
type Fake = { t: number; ended: boolean; playing: boolean; rate: number }
const state = new WeakMap<HTMLMediaElement, Fake>()
const of = (e: HTMLMediaElement) => { let s = state.get(e); if (!s) { s = { t: 0, ended: false, playing: false, rate: 1 }; state.set(e, s) } return s }
let frames: FrameRequestCallback[] = []
beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) { of(this).playing = true; return Promise.resolve() })
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) { of(this).playing = false })
  Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', { configurable: true, get() { return of(this).t }, set(v: number) { of(this).t = v; of(this).ended = false } })
  Object.defineProperty(HTMLMediaElement.prototype, 'duration', { configurable: true, get: () => 8 })
  Object.defineProperty(HTMLMediaElement.prototype, 'ended', { configurable: true, get() { return of(this).ended } })
  Object.defineProperty(HTMLMediaElement.prototype, 'playbackRate', { configurable: true, get() { return of(this).rate }, set(v: number) { of(this).rate = v } })
  Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', { configurable: true, get: () => 960 })
  Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', { configurable: true, get: () => 660 })
  frames = []
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => { frames.push(cb); return frames.length })
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })
const tick = () => act(() => { const due = frames; frames = []; for (const f of due) f(0) })
const load = (vs: HTMLVideoElement[]) => { for (const v of vs) v.dispatchEvent(new Event('loadedmetadata')) }

const crops = (k: string) => [{ id: 'face', label: 'Face', path: `/g/${k}.face.mp4` }, { id: 'text', label: 'Card text', path: `/g/${k}.text.mp4` }]
const video = (k: string) => ({ path: `/g/${k}.mp4`, width: 1440, height: 1080, fps: 50, crops: crops(k), stills: [] })
const reference = { note: 'tape', preview: '/m/c.source.mp4', crops: [{ id: 'face', label: 'Face', path: '/m/c.source.face.mp4' }, { id: 'text', label: 'Card text', path: '/m/c.source.text.mp4' }] }

describe('Judging video', () => {
  it('plays X, Y and the source on one clock, by region, and answers by key', async () => {
    server.use(http.get('/api/evals/:name/pair', () => HttpResponse.json({ pair: { similarity: 0.1, item_id: 'c', input: { crop: [14, 0, 692, 568] }, has_media: true, reference, x: video('x'), y: video('y'), token: 't' }, remaining: 4 })))
    const { user, container } = setup(<App />, { route: '/evals/vhs?tab=judge' })
    const cmp = await screen.findByLabelText('Item c, two outputs')
    const vids = () => Array.from(container.querySelectorAll('video'))
    expect(vids().map(v => v.getAttribute('src'))).toEqual(['/api/images/file?path=%2Fg%2Fx.mp4', '/api/images/file?path=%2Fg%2Fy.mp4', '/api/evals/vhs/media/c?file=%2Fm%2Fc.source.mp4'])
    expect(within(cmp).getByText('Source')).toBeInTheDocument()
    expect(within(cmp).queryByText(/x\.mp4|turbo/)).not.toBeInTheDocument() // blind: X and Y only
    expect(screen.queryByRole('button', { name: /Play the phrase/ })).not.toBeInTheDocument() // no phrase player for a video item
    load(vids())
    await user.click(within(cmp).getByRole('radio', { name: 'Face' }))
    expect(vids()[0].getAttribute('src')).toBe('/api/images/file?path=%2Fg%2Fx.face.mp4')
    expect(vids()[2].getAttribute('src')).toBe('/api/evals/vhs/media/c?file=%2Fm%2Fc.source.face.mp4')
    await user.keyboard('g') // next region
    await waitFor(() => expect(vids()[1].getAttribute('src')).toBe('/api/images/file?path=%2Fg%2Fy.text.mp4'))
    load(vids())
    await user.click(within(cmp).getByRole('button', { name: 'Play' }))
    expect(vids().every(v => of(v).playing)).toBe(true)
    of(vids()[0]).t = 2.0; of(vids()[1]).t = 1.0; of(vids()[2]).t = 2.01
    tick()
    expect(of(vids()[1]).t).toBe(2.0) // a follower that drifted is pulled back to the leader
    expect(of(vids()[2]).t).toBe(2.01) // within a frame: left alone
    of(vids()[0]).ended = true
    tick()
    expect(vids().map(v => of(v).t)).toEqual([0, 0, 0]) // the end loops back to the start
    await user.click(within(cmp).getByRole('button', { name: 'Pause' }))
    expect(vids().some(v => of(v).playing)).toBe(false)
    await user.click(within(cmp).getByRole('button', { name: /forward one frame/i }))
    expect(vids().map(v => of(v).t)).toEqual([0.02, 0.02, 0.02])
    await user.selectOptions(within(cmp).getByRole('combobox', { name: 'Playback speed' }), '0.25')
    expect(vids().every(v => of(v).rate === 0.25)).toBe(true)
    ;(document.activeElement as HTMLElement).blur() // keys typed into the speed menu are the menu's
    await user.keyboard('v') // flip view: one frame, every video stacked, one shown
    const showing = await within(cmp).findByRole('radiogroup', { name: 'Showing' })
    load(vids())
    expect(vids().filter(v => !v.className.includes('invisible'))).toHaveLength(1)
    expect(vids().every(v => of(v).t === 0.02)).toBe(true) // a remounted video is put back on the clock
    await user.keyboard('b')
    expect(within(showing).getByRole('radio', { name: 'Y' })).toBeChecked()
    await user.click(within(showing).getByRole('radio', { name: 'Source' }))
    expect(vids()[2].className).not.toContain('invisible')
    await user.keyboard(']')
    await waitFor(() => expect(mock.calls).toContain('judge y'))
  })
})

describe('Video outputs', () => {
  it('shows the source and every candidate on one clock, the item keys left to the page', async () => {
    const d = evalFixtures.outputs()
    const item = d.items[0] as { input: Record<string, unknown>; reference: unknown; outputs: Record<string, unknown> }
    item.reference = reference
    item.outputs = { 'large-v3': { output: video('a'), error: null, run_s: 20.5 }, turbo: { output: video('b'), error: null }, small: { output: null, error: 'out of memory' } }
    server.use(http.get('/api/evals/:name/outputs', () => HttpResponse.json(d)))
    const { user, container } = setup(<App />, { route: '/evals/vhs?tab=outputs' })
    const cmp = await screen.findByLabelText(/Item u1: 2\svideos/)
    expect(within(cmp).getByText('large-v3')).toBeInTheDocument() // not blind
    expect(within(cmp).getByText('· 20.5 s')).toBeInTheDocument()
    expect(container.querySelectorAll('video')).toHaveLength(3)
    expect(screen.getByText('Failed: out of memory')).toBeInTheDocument()
    expect(screen.queryByLabelText('Player keys')).not.toBeInTheDocument()
    await user.click(within(cmp).getByRole('radio', { name: 'Card text' }))
    expect(container.querySelector('video')?.getAttribute('src')).toBe('/api/evals/vhs/media/u1?file=%2Fm%2Fc.source.text.mp4')
    await user.keyboard('j') // the item key, not the player's
    expect(await screen.findByText('Prompt: a dog')).toBeInTheDocument()
  })
  it('plays a video output in the unblinded last pair', async () => {
    server.use(
      http.get('/api/evals/:name/pair', () => HttpResponse.json({ pair: { item_id: 'u1', input: {}, has_media: false, reference: null, x: video('x'), y: video('y'), token: 't' } })),
      http.get('/api/evals/:name/outputs', () => { const d = evalFixtures.outputs(); (d.items[0] as { outputs: Record<string, unknown> }).outputs = { turbo: { output: video('b'), error: null }, 'large-v3': { output: { segments: [{ text: 'x' }] }, error: null } }; return HttpResponse.json(d) }),
    )
    const { user } = setup(<App />, { route: '/evals/vhs?tab=judge' })
    await user.click(await screen.findByRole('button', { name: 'X is better' }))
    await user.click(await screen.findByText(/Last pair \(u1\)/))
    expect(await screen.findByLabelText('turbo', { selector: 'video' })).toHaveAttribute('controls')
  })
})
