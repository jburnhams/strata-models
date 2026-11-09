// The whole recording, cued to the phrase being judged: the player starts 2 s before it (so it is heard in context and the judge can scrub
// further either way), and "Play the phrase" plays from there to 1 s after it ends, as often as wanted.
import { useEffect, useRef } from 'react'
import { Button } from 'strata-react'

export const LEAD_S = 2
export const TAIL_S = 1

/** Where the player is cued to, and where "Play the phrase" stops. */
export function window_(clip: [number, number] | undefined) {
  return clip ? { from: Math.max(clip[0] - LEAD_S, 0), to: clip[1] + TAIL_S } : { from: 0, to: Infinity }
}

export default function PhrasePlayer({ src, clip }: { src: string; clip?: [number, number] }) {
  const ref = useRef<HTMLAudioElement>(null)
  const stopAt = useRef(Infinity)
  const w = window_(clip)
  useEffect(() => {
    const a = ref.current
    if (!a) return
    const cue = () => { a.currentTime = w.from }
    const stop = () => { if (a.currentTime >= stopAt.current) { a.pause(); stopAt.current = Infinity } }
    a.addEventListener('loadedmetadata', cue); a.addEventListener('timeupdate', stop)
    return () => { a.removeEventListener('loadedmetadata', cue); a.removeEventListener('timeupdate', stop) }
  }, [src, w.from])
  const play = () => { const a = ref.current; if (!a) return; a.currentTime = w.from; stopAt.current = w.to; void a.play() }
  return (
    <div className="grid gap-2">
      <audio ref={ref} controls preload="metadata" src={src} className="w-full" aria-label="The whole recording" />
      {clip && <div className="flex flex-wrap items-center gap-3"><Button type="button" onClick={play}>Play the phrase</Button><span className="type-caption text-muted-foreground tabular-nums">{`${w.from.toFixed(1)} s to ${Number.isFinite(w.to) ? w.to.toFixed(1) : '–'} s of the recording`}</span></div>}
    </div>
  )
}
