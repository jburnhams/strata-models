#!/usr/bin/env bash
# Make the venv a venv-worker model runs in: <models>/venvs/NAME (models dir: STRATAMODELS_MODELS, default ~/models). Needs uv.
#   scripts/setup_worker_venv.sh kokoro|chatterbox|qwen3-tts
# Each pins what its model needs; they are kept apart from the service's venv on purpose (Chatterbox wants torch 2.6, for one).
set -euo pipefail
name=${1:?give the venv name: kokoro, chatterbox or qwen3-tts}
models=${STRATAMODELS_MODELS:-$HOME/models}; venv=$models/venvs/$name
case $name in
  kokoro) pkgs="kokoro-onnx soundfile numpy" ;;          # model files: <models>/tts/kokoro/{kokoro-v1.0.onnx,voices-v1.0.bin}
  chatterbox) pkgs="chatterbox-tts soundfile" ;;
  qwen3-tts) pkgs="qwen-tts soundfile" ;;
  *) echo "no recipe for $name" >&2; exit 2 ;;
esac
mkdir -p "$models/venvs"; uv venv -q -p 3.12 "$venv"; VIRTUAL_ENV=$venv uv pip install $pkgs
if [ "$name" = kokoro ]; then
  mkdir -p "$models/tts/kokoro"; cd "$models/tts/kokoro"
  for f in kokoro-v1.0.onnx voices-v1.0.bin; do [ -f $f ] || { curl -sSL -o $f.part https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f && mv $f.part $f; }; done
fi
echo "ready: $venv"
