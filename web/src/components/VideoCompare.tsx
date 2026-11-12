import { useCallback, useEffect, useRef, useState } from 'react'
import { KeysLegend, MediaStage, RateChoice, SeekBar, TimeReadout, ToggleGroup, Transport, transportShortcuts, useHotkeys, type Shortcut } from 'strata-react'

// Video outputs (video-enhance evals) on one clock, until strata-react's Compare takes video (jburnhams/strata-react#156: synced compare,
// #157 frame stepping and loops, #158 region crops, #159 the grid). Plain <video> elements: the first one leads, the others are pulled back
// to its time whenever they drift by more than a frame, and every seek, step and rate change goes to all of them. Regions are the
// worker's crops (the same box of each output, enlarged nearest-neighbour), so "Face" shows the face of every side at once.

export type Crop = { id: string; label: string; path: string }
export type VideoOut = { path: string; width?: number; height?: number; fps?: number; crops: Crop[] }

/** The video of a video-enhance output (path to an .mp4/.webm/.mov and its region crops), if it is one. */
export const videoOf = (o: unknown): VideoOut | null => {
  const r = o as { path?: unknown; width?: unknown; height?: unknown; fps?: unknown; crops?: unknown } | null
  if (!r || typeof r !== 'object' || typeof r.path !== 'string' || !/\.(mp4|webm|mov)$/i.test(r.path)) return null
  const crops = Array.isArray(r.crops) ? (r.crops as Crop[]).filter(c => c && typeof c.id === 'string' && typeof c.path === 'string').map(c => ({ id: c.id, label: typeof c.label === 'string' ? c.label : c.id, path: c.path })) : []
  return { path: r.path, width: typeof r.width === 'number' ? r.width : undefined, height: typeof r.height === 'number' ? r.height : undefined, fps: typeof r.fps === 'number' ? r.fps : undefined, crops }
}

/** One video on the clock: its whole frame and its regions, by region id, as URLs. */
export type VideoSide = { id: string; label: string; note?: string; whole: string; regions: Record<string, string> }

export const WHOLE = 'whole'

/** The regions every side has (the same id in each), in the first side's order, so a region chip never shows one side blank. */
export const sharedRegions = (sides: VideoSide[], labels: Record<string, string>) => {
  const first = sides[0] ? Object.keys(sides[0].regions) : []
  return first.filter(id => sides.every(s => id in s.regions)).map(id => ({ value: id, label: labels[id] ?? id }))
}

/** Where a side's video for the chosen region is: the whole frame when the side has no such region. */
export const srcFor = (s: VideoSide, region: string) => (region !== WHOLE && s.regions[region]) || s.whole

/** Several <video>s kept on one clock. */
export function useVideoClock(fps: number) {
  const els = useRef<(HTMLVideoElement | null)[]>([])
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [rate, setRateState] = useState(1)
  const at = useRef(0)
  const all = () => els.current.filter((e): e is HTMLVideoElement => e !== null)
  const seek = useCallback((t: number) => {
    const d = els.current[0]?.duration
    const to = Math.max(0, Math.min(t, Number.isFinite(d) && d ? d - 0.001 : t))
    at.current = to; setTime(to)
    for (const e of all()) e.currentTime = to
  }, [])
  const pause = useCallback(() => { for (const e of all()) e.pause(); setPlaying(false) }, [])
  const play = useCallback(() => { for (const e of all()) { e.playbackRate = rate; void e.play().catch(() => undefined) } setPlaying(true) }, [rate])
  const toggle = useCallback(() => (playing ? pause() : play()), [playing, pause, play])
  const step = useCallback((frames: number) => { pause(); seek(at.current + frames / fps) }, [pause, seek, fps])
  const setRate = useCallback((r: number) => { setRateState(r); for (const e of all()) e.playbackRate = r }, [])
  useEffect(() => {
    if (!playing) return
    let id = 0
    const tick = () => {
      const lead = els.current[0]
      if (lead) {
        const t = lead.currentTime; at.current = t; setTime(t)
        for (const e of all().slice(1)) if (Math.abs(e.currentTime - t) > 1.5 / fps) e.currentTime = t // a follower that drifted: back on the leader's frame
        if (lead.ended) { seek(0); for (const e of all()) void e.play().catch(() => undefined) } // loop: judging means watching the same seconds again
      }
      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [playing, fps, seek])
  /** A video (re)loaded, after a region change say: put it on the clock's time and state. */
  const attach = (i: number) => (e: HTMLVideoElement | null) => { els.current[i] = e }
  const loaded = (i: number) => (ev: React.SyntheticEvent<HTMLVideoElement>) => {
    const e = ev.currentTarget
    if (i === 0 && Number.isFinite(e.duration)) setDuration(e.duration)
    e.currentTime = at.current; e.playbackRate = rate
    if (playing) void e.play().catch(() => undefined)
  }
  return { time, duration, playing, rate, seek, toggle, pause, step, setRate, attach, loaded }
}

const VIEW_LABEL = { side: 'Side by side', flip: 'Flip' } as const
type View = keyof typeof VIEW_LABEL

/** Two or more videos of the same clip compared on one clock: side by side or flipped in one frame, whole or by region. */
export function VideoCompare({ label, sides, regionLabels = {}, fps = 50, keys = true, columns }: {
  label: string
  sides: VideoSide[]
  regionLabels?: Record<string, string>
  fps?: number
  /** Bind the player's keys (Space/K play, J/L jump, arrows a frame, G region, V view, B flip); off where the page binds the same keys. */
  keys?: boolean
  columns?: number
}) {
  const clock = useVideoClock(fps)
  const [region, setRegion] = useState(WHOLE)
  const [view, setView] = useState<View>('side')
  const [shown, setShown] = useState(0)
  const [aspect, setAspect] = useState(4 / 3)
  const regions = [{ value: WHOLE, label: 'Whole frame' }, ...sharedRegions(sides, regionLabels)]
  const cycle = () => setRegion(r => regions[(regions.findIndex(x => x.value === r) + 1) % regions.length].value)
  const flip = () => setShown(s => (s + 1) % Math.max(sides.length, 1))
  const shortcuts: Shortcut[] = [
    ...transportShortcuts({ toggle: clock.toggle, seek: clock.seek, time: clock.time, duration: clock.duration, frame: 1 / fps, jump: 1 }),
    { keys: ['g'], label: 'Next region', run: cycle, disabled: regions.length < 2 },
    { keys: ['v'], label: 'Side by side or flip', run: () => setView(v => (v === 'side' ? 'flip' : 'side')) },
    { keys: ['b'], label: 'Flip to the next video', run: flip, disabled: view !== 'flip' },
  ]
  useHotkeys(shortcuts, { enabled: keys && sides.length > 0 })
  const video = (s: VideoSide, i: number, visible = true) => (
    <video key={`${s.id}-${region}`} ref={clock.attach(i)} src={srcFor(s, region)} muted={i !== 0} playsInline preload="auto" aria-label={`${s.label}, ${regions.find(r => r.value === region)?.label ?? ''}`}
      onLoadedMetadata={e => { clock.loaded(i)(e); if (i === 0 && e.currentTarget.videoWidth) setAspect(e.currentTarget.videoWidth / e.currentTarget.videoHeight) }}
      className={`absolute inset-0 h-full w-full object-contain ${visible ? '' : 'invisible'}`} style={{ imageRendering: 'pixelated' }} />
  )
  const cols = columns ?? (sides.length >= 4 ? 4 : sides.length >= 3 ? 3 : 2)
  return (
    <section aria-label={label} className="grid gap-3">
      <div className="flex flex-wrap items-center gap-3">
        {regions.length > 1 && <ToggleGroup label="Region" options={regions} value={region} onChange={setRegion} size="sm" />}
        <ToggleGroup label="View" options={(Object.keys(VIEW_LABEL) as View[]).map(v => ({ value: v, label: VIEW_LABEL[v] }))} value={view} onChange={v => setView(v as View)} size="sm" />
      </div>
      {view === 'side'
        ? <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(cols, sides.length)}, minmax(0, 1fr))` }}>
            {sides.map((s, i) => (
              <figure key={s.id} className="grid content-start gap-1">
                <MediaStage aspect={aspect} label={s.label}><div className="relative h-full w-full">{video(s, i)}</div></MediaStage>
                <figcaption className="type-caption"><span className="type-label">{s.label}</span>{s.note ? <span className="text-muted-foreground">{` · ${s.note}`}</span> : null}</figcaption>
              </figure>
            ))}
          </div>
        : <figure className="grid gap-1">
            <MediaStage aspect={aspect} label={`${label}: ${sides[shown]?.label ?? ''}`}><div className="relative h-full w-full">{sides.map((s, i) => video(s, i, i === shown))}</div></MediaStage>
            <figcaption className="flex flex-wrap items-center gap-2 type-caption">
              <ToggleGroup label="Showing" options={sides.map((s, i) => ({ value: String(i), label: s.label }))} value={String(shown)} onChange={v => setShown(Number(v))} size="sm" />
              {sides[shown]?.note && <span className="text-muted-foreground">{sides[shown].note}</span>}
            </figcaption>
          </figure>}
      <Transport playing={clock.playing} onPlayPause={clock.toggle} onToStart={() => clock.seek(0)} onStepBack={() => clock.step(-1)} onStepForward={() => clock.step(1)}>
        <TimeReadout time={clock.time} duration={clock.duration || null} precision="frames" fps={fps} />
        <SeekBar time={clock.time} duration={clock.duration || null} onSeek={clock.seek} precision="frames" fps={fps} step={1 / fps} bigStep={1} label="Position" className="min-w-40 flex-1" />
        <RateChoice rate={clock.rate} onRateChange={clock.setRate} rates={[0.25, 0.5, 1]} />
      </Transport>
      {keys && <KeysLegend shortcuts={shortcuts} label="Player keys" />}
    </section>
  )
}
