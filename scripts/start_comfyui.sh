#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
COMFYUI="$INSTALL_ROOT/ComfyUI"
PYTHON="$INSTALL_ROOT/.venv/bin/python"

[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON" >&2; exit 1; }
[[ -f "$COMFYUI/main.py" ]] || { echo "Missing ComfyUI: $COMFYUI" >&2; exit 1; }

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$COMFYUI"
exec "$PYTHON" -u main.py \
    --listen "${COMFY_LISTEN:-127.0.0.1}" \
    --port "${COMFY_PORT:-8188}" \
    --default-device 0 \
    --force-fp16 \
    --fp16-unet \
    --fp16-vae \
    --fp16-text-enc \
    --use-split-cross-attention \
    --highvram \
    --disable-dynamic-vram \
    --disable-async-offload \
    --preview-method none \
    --output-directory "$COMFYUI/output" \
    "$@"
