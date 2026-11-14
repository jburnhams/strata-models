#!/usr/bin/env bash
# Make the venv a venv-worker model runs in: <models>/venvs/NAME (models dir: STRATAMODELS_MODELS, default ~/models). Needs uv (and git, curl).
#   scripts/setup_worker_venv.sh kokoro|chatterbox|qwen3-tts|whisper|dfn|clearvoice|resemble|audio|audio-quality|video|seedvr2|flashvsr-plus|vapoursynth
# Each pins what its model needs; they are kept apart from the service's venv on purpose (Chatterbox wants torch 2.6, for one).
set -euo pipefail
name=${1:?give the venv name: kokoro, chatterbox, qwen3-tts, whisper, dfn, clearvoice, resemble, audio, audio-quality, video, seedvr2, flashvsr-plus or vapoursynth}
models=${STRATAMODELS_MODELS:-$HOME/models}; venv=$models/venvs/$name
TORCH_INDEX=https://download.pytorch.org/whl/cu128   # CUDA 12.8 wheels: they bring their own CUDA libraries, any recent driver runs them
py=3.12
fetch() { [ -f "$2" ] || { mkdir -p "$(dirname "$2")"; curl -fSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }   # url file: skipped when present
fetch_sha() { fetch "$1" "$2"; echo "$3  $2" | sha256sum -c --quiet || { rm -f "$2"; echo "checksum mismatch: $2" >&2; exit 1; }; }   # url file sha256
checkout() { [ -d "$2/.git" ] || git clone -q "$1" "$2"; git -C "$2" fetch -q origin "$3" 2>/dev/null || true; git -C "$2" checkout -q "$3"; }   # repo dir commit
case $name in
  kokoro) pkgs="kokoro-onnx soundfile numpy" ;;          # model files: <models>/tts/kokoro/{kokoro-v1.0.onnx,voices-v1.0.bin}
  chatterbox) pkgs="chatterbox-tts soundfile setuptools<81" ;;   # its watermarker (resemble-perth) imports pkg_resources, gone from setuptools 81
  qwen3-tts) pkgs="qwen-tts soundfile" ;;
  whisper) pkgs="faster-whisper numpy nvidia-cublas-cu12 nvidia-cudnn-cu12" ;;   # CTranslate2 is built for CUDA 12; its libraries live here, apart from the service's CUDA 13
  video) pkgs="torch spandrel numpy" ;;                  # workers/video.py, engines ffmpeg and spandrel; weights below, into <models>/spandrel/
  seedvr2) pkgs="" ;;                                    # SeedVR2 through numz/ComfyUI-SeedVR2_VideoUpscaler's CLI (Apache-2.0), cloned into <models>/src/seedvr2
  flashvsr-plus) pkgs=""; py=3.10 ;;                     # FlashVSR+ (OpenImagingLab, Apache-2.0), cloned into <models>/src/flashvsr-plus; wants Python 3.10 and torch <2.8
  # VapourSynth R82 and its plugins as PyPI wheels (no system packages, no sudo) with vsjetpack (MIT): QTGMC (vsdeinterlace.QTempGaussMC)
  # and SMDegrain-style motion-compensated denoise (vsdenoise.mc_degrain, on mvutensils: vsjetpack 3 calls core.mvu) for workers/video.py's
  # VapourSynth stage, which runs its vspipe
  vapoursynth) py=3.13; pkgs="vapoursynth vsjetpack vapoursynth-mvtools vapoursynth-mvutensils vapoursynth-znedi3 vapoursynth-zsmooth vapoursynth-bestsource vapoursynth-bwdif vapoursynth-resize2 vapoursynth-akarin vapoursynth-vszip numpy" ;;
  # Sound (workers/audio.py, audio_quality.py): numpy, scipy and soundfile in each. torch from PyTorch's CUDA 12.8 index (its own CUDA libraries).
  dfn) pkgs="--index-strategy unsafe-best-match --extra-index-url $TORCH_INDEX torch==2.8.* torchaudio==2.8.* numpy scipy soundfile deepfilterlib==0.5.6 loguru appdirs packaging requests" ;;
  clearvoice) pkgs="clearvoice scipy soundfile" ;;       # pins numpy<2 and an old opencv: the reason it has a venv of its own
  resemble) pkgs="resemble-enhance scipy soundfile"; py=3.11 ;;   # pins torch 2.1.1, so Python 3.11
  audio) pkgs="numpy scipy soundfile noisereduce" ;;     # the CPU engines: none, afftdn (ffmpeg), noisereduce
  audio-quality) pkgs="--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.* torchaudio==2.8.* onnxruntime pesq pystoi numpy scipy soundfile" ;;   # CPU only
  *) echo "no recipe for $name" >&2; exit 2 ;;
esac
mkdir -p "$models/venvs"; uv venv -q -p $py "$venv"
case $name in
  video)
    VIRTUAL_ENV=$venv uv pip install --index-strategy unsafe-best-match --extra-index-url $TORCH_INDEX $pkgs
    d=$models/spandrel   # the per-frame models of the video-enhance eval (the licence of each is in models.toml)
    fetch https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth "$d/RealESRGAN_x4plus.pth"
    fetch https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth "$d/RealESRGAN_x2plus.pth"
    fetch https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth "$d/realesr-general-x4v3.pth"
    fetch https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth "$d/4xNomosWebPhoto_RealPLKSR.pth"
    fetch https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth "$d/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
    # 1x VHS cleanup models (OpenModelDB; the cleanup stage's `model`): BleedOut Compact (zarxrax, WTFPL), BaldrickVHSFix v0.2 (nimrodzorg,
    # CC-BY-NC-SA-4.0) and BroadcastToStudio Lite (saurusx, CC-BY-NC-SA-4.0): non-commercial is fine for a private archive
    omdb=https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o
    fetch_sha $omdb/1x-BleedOut-Compact.pth "$d/1x-BleedOut-Compact.pth" 86a9c0bbeb130aa01b7d5c3ea32691a521a4de65771d7133fa57a58ac4f5485c
    fetch_sha $omdb/1x-BaldrickVHSFixV0-2.pth "$d/1x-BaldrickVHSFixV0-2.pth" 9c945b10ab0d45f8f53d2e18f9bfe1017dbc3c6e7233d657cb4845de81cd5ea1
    fetch_sha $omdb/1x-BroadcastToStudioLite.pth "$d/1x-BroadcastToStudioLite.pth" 912ff494e454a135ba2341d522c38089736654daf32bd6c7bb81549bb9512d38
    ;;
  seedvr2)
    src=$models/src/seedvr2; checkout https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler "$src" 4490bd1f482e026674543386bb2a4d176da245b9   # v2.5.24
    VIRTUAL_ENV=$venv uv pip install --index-strategy unsafe-best-match --extra-index-url $TORCH_INDEX torch torchvision spandrel -r "$src/requirements.txt"   # spandrel: a cleanup stage's 1x model before the engine
    d=$models/seedvr2   # the CLI's --model_dir (workers/video.py passes it); it would download these itself, slowly, on the first run
    for f in seedvr2_ema_3b_fp8_e4m3fn.safetensors ema_vae_fp16.safetensors; do fetch "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/$f" "$d/$f"; done
    ;;
  flashvsr-plus)
    src=$models/src/flashvsr-plus; checkout https://github.com/OpenImagingLab/FlashVSR_Plus "$src" b548a2cc2c929d3d4676f2973abe27876ec4dee7
    VIRTUAL_ENV=$venv uv pip install --index-strategy unsafe-best-match --extra-index-url $TORCH_INDEX "torch>=2.5,<2.8" spandrel -r "$src/requirements.txt"   # spandrel: a cleanup stage's 1x model
    VIRTUAL_ENV=$venv uv pip install --no-deps -e "$src"
    # The attention kernel: without BSA or this one, FlashVSR+ falls back to PyTorch SDPA, which refuses a 1536 x 1152 clip ("would
    # allocate too many mask entries"). Its own Triton kernel (the README's pick for consumer cards) needs flash-attn's utilities: the
    # prebuilt wheel for this venv (cp310, torch 2.7, CUDA 12, C++11 ABI), so nothing is compiled; the kernel is pure Triton.
    VIRTUAL_ENV=$venv uv pip install --no-deps "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    VIRTUAL_ENV=$venv uv pip install "setuptools>=68" wheel   # --no-build-isolation builds with the venv's own backend, and uv venv seeds none (the kernel's pyproject uses setuptools.build_meta)
    VIRTUAL_ENV=$venv uv pip install --no-deps --no-build-isolation -e "$src/operators/flashvsr_attn"
    for f in TCDecoder.ckpt turbo/dit.safetensors turbo/lq_proj_in.safetensors base/dit.safetensors base/lq_proj_in.safetensors; do   # into the repo's ckpts/, where the CLI looks
      fetch "https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/main/$f" "$src/ckpts/$f"
    done
    ;;
  *) VIRTUAL_ENV=$venv uv pip install $pkgs ;;
esac
# a pip VapourSynth finds its Python through ~/.config/vapoursynth/vapoursynth.toml, keyed by this venv's own libvsscript (a system install keeps its entry)
if [ "$name" = vapoursynth ]; then "$venv/bin/vapoursynth" config; fi
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
