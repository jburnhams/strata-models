#!/usr/bin/env bash
# Make the venv a venv-worker model runs in: <models>/venvs/NAME (models dir: STRATAMODELS_MODELS, default ~/models). Needs uv (and git, curl).
#   scripts/setup_worker_venv.sh kokoro|chatterbox|qwen3-tts|whisper|video|seedvr2|flashvsr-plus
# Each pins what its model needs; they are kept apart from the service's venv on purpose (Chatterbox wants torch 2.6, for one).
set -euo pipefail
name=${1:?give the venv name: kokoro, chatterbox, qwen3-tts, whisper, video, seedvr2 or flashvsr-plus}
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
    VIRTUAL_ENV=$venv uv pip install --index-strategy unsafe-best-match --extra-index-url $TORCH_INDEX torch torchvision -r "$src/requirements.txt"
    d=$models/seedvr2   # the CLI's --model_dir (workers/video.py passes it); it would download these itself, slowly, on the first run
    for f in seedvr2_ema_3b_fp8_e4m3fn.safetensors ema_vae_fp16.safetensors; do fetch "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/$f" "$d/$f"; done
    ;;
  flashvsr-plus)
    src=$models/src/flashvsr-plus; checkout https://github.com/OpenImagingLab/FlashVSR_Plus "$src" b548a2cc2c929d3d4676f2973abe27876ec4dee7
    VIRTUAL_ENV=$venv uv pip install --index-strategy unsafe-best-match --extra-index-url $TORCH_INDEX "torch>=2.5,<2.8" -r "$src/requirements.txt"
    VIRTUAL_ENV=$venv uv pip install --no-deps -e "$src"
    for f in TCDecoder.ckpt turbo/dit.safetensors turbo/lq_proj_in.safetensors base/dit.safetensors base/lq_proj_in.safetensors; do   # into the repo's ckpts/, where the CLI looks
      fetch "https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/main/$f" "$src/ckpts/$f"
    done
    ;;
  *) VIRTUAL_ENV=$venv uv pip install $pkgs ;;
esac
if [ "$name" = kokoro ]; then
  mkdir -p "$models/tts/kokoro"; cd "$models/tts/kokoro"
  for f in kokoro-v1.0.onnx voices-v1.0.bin; do [ -f $f ] || { curl -sSL -o $f.part https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f && mv $f.part $f; }; done
fi
echo "ready: $venv"
