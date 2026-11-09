"""Whisper through faster-whisper (CTranslate2): float16 on the GPU, int8 on the CPU when there is none.

params: path (an audio or video file ffmpeg can read); language (ISO code, else detected); task (transcribe | translate); word_timestamps (default
true); vad_filter (default false: callers that cut by pauses do it themselves); beam_size (default 5); condition_on_previous_text (default false:
fewer runaway repeats); initial_prompt (names, jargon); clip (start, end seconds)."""
import os

from stratamodels.runtimes import RunError

KEYS = ('language', 'task', 'word_timestamps', 'vad_filter', 'beam_size', 'condition_on_previous_text', 'initial_prompt', 'temperature')


def load(model, models_dir):
    import ctranslate2
    from faster_whisper import WhisperModel
    cuda = ctranslate2.get_cuda_device_count() > 0
    m = WhisperModel(model.name, device='cuda' if cuda else 'cpu', compute_type=model.options.get('compute_type', 'float16') if cuda else 'int8', download_root=os.path.join(models_dir, 'whisper'))
    return dict(model=m)


def _r(v): return round(float(v), 3)


def run(handle, params):
    path = params.get('path')
    if not isinstance(path, str) or not os.path.isfile(path): raise RunError(f'no such audio file: {path!r}')
    unknown = set(params) - set(KEYS) - {'path', 'clip'}
    if unknown: raise RunError(f'unknown params {sorted(unknown)}')
    kw = dict(word_timestamps=True, vad_filter=False, beam_size=5, condition_on_previous_text=False)
    kw.update({k: params[k] for k in KEYS if k in params})
    if params.get('clip'): kw['clip_timestamps'] = [float(v) for v in params['clip']]
    segs, info = handle['model'].transcribe(path, **kw)
    out = []
    for s in segs:
        d = dict(start=_r(s.start), end=_r(s.end), text=s.text, avg_logprob=_r(s.avg_logprob), no_speech_prob=_r(s.no_speech_prob))
        if s.words: d['words'] = [dict(start=_r(w.start), end=_r(w.end), word=w.word, probability=_r(w.probability)) for w in s.words]
        out.append(d)
    return dict(language=info.language, language_probability=_r(info.language_probability), duration=_r(info.duration), segments=out)


def unload(handle):
    handle.pop('model', None)
    import gc; gc.collect()
