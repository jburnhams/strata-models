#!/usr/bin/env bash
# Make the venv a venv-worker model runs in: <models>/venvs/NAME (models dir: STRATAMODELS_MODELS, default ~/models). Needs uv.
#   scripts/setup_worker_venv.sh kokoro|chatterbox|qwen3-tts|whisper|dfn|clearvoice|resemble|audio|audio-quality
# Each pins what its model needs; they are kept apart from the service's venv on purpose (Chatterbox wants torch 2.6, for one).
set -euo pipefail
name=${1:?give the venv name: kokoro, chatterbox, qwen3-tts, whisper, dfn, clearvoice, resemble, audio or audio-quality}
models=${STRATAMODELS_MODELS:-$HOME/models}; venv=$models/venvs/$name
case $name in
  kokoro) pkgs="kokoro-onnx soundfile numpy" ;;          # model files: <models>/tts/kokoro/{kokoro-v1.0.onnx,voices-v1.0.bin}
  chatterbox) pkgs="chatterbox-tts soundfile setuptools<81" ;;   # its watermarker (resemble-perth) imports pkg_resources, gone from setuptools 81
  qwen3-tts) pkgs="qwen-tts soundfile" ;;
  whisper) pkgs="faster-whisper numpy nvidia-cublas-cu12 nvidia-cudnn-cu12" ;;   # CTranslate2 is built for CUDA 12; its libraries live here, apart from the service's CUDA 13
  # Sound (workers/audio.py, audio_quality.py): numpy, scipy and soundfile in each. torch from PyTorch's CUDA 12.8 index (its own CUDA libraries).
  dfn) pkgs="--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.* torchaudio==2.8.* numpy scipy soundfile deepfilterlib==0.5.6 loguru appdirs packaging requests" ;;
  clearvoice) pkgs="clearvoice scipy soundfile" ;;       # pins numpy<2 and an old opencv: the reason it has a venv of its own
  resemble) pkgs="resemble-enhance scipy soundfile" ;;   # pins torch 2.1.1, so Python 3.11
  audio) pkgs="numpy scipy soundfile noisereduce" ;;     # the CPU engines: none, afftdn (ffmpeg), noisereduce
  audio-quality) pkgs="--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.* torchaudio==2.8.* onnxruntime pesq pystoi numpy scipy soundfile" ;;   # CPU only
  *) echo "no recipe for $name" >&2; exit 2 ;;
esac
py=3.12; [ "$name" = resemble ] && py=3.11
mkdir -p "$models/venvs"; uv venv -q -p $py "$venv"; VIRTUAL_ENV=$venv uv pip install $pkgs
if [ "$name" = dfn ]; then
  # deepfilternet 0.5.6 pins numpy<2 (it runs on 2) and imports torchaudio.backend.common, gone since torchaudio 2.1: install it without its
  # pins and give it a stand-in for the one type it imports (strata-360's scripts/patch_deepfilternet.py, the same patch).
  VIRTUAL_ENV=$venv uv pip install --no-deps deepfilternet==0.5.6
  "$venv/bin/python" - <<'PY'
import importlib.util, os
p = os.path.join(importlib.util.find_spec("df").submodule_search_locations[0], "io.py")
s = open(p).read()
old = "from torchaudio.backend.common import AudioMetaData"
if old in s and "stand-in" not in s:
    s = s.replace(old, "try:\n    from torchaudio.backend.common import AudioMetaData\nexcept Exception:  # stand-in (strata-models setup): df's own file loaders are not used\n    from dataclasses import dataclass\n    @dataclass\n    class AudioMetaData:\n        sample_rate: int = 0\n        num_frames: int = 0\n        num_channels: int = 0\n        bits_per_sample: int = 0\n        encoding: str = ''")
    open(p, "w").write(s)
    print("patched", p)
PY
fi
if [ "$name" = audio-quality ]; then   # DNSMOS P.835 (Microsoft DNS Challenge); SQUIM's weights come through torchaudio on first use
  mkdir -p "$models/audio-quality"; f=$models/audio-quality/sig_bak_ovr.onnx
  [ -f "$f" ] || { curl -fsSL -o "$f.part" https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx && mv "$f.part" "$f"; }
fi
if [ "$name" = kokoro ]; then
  mkdir -p "$models/tts/kokoro"; cd "$models/tts/kokoro"
  for f in kokoro-v1.0.onnx voices-v1.0.bin; do [ -f $f ] || { curl -sSL -o $f.part https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f && mv $f.part $f; }; done
fi
echo "ready: $venv"
