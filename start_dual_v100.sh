#!/usr/bin/env bash
# Start the packaged MiniMax H3 v0.2 route on two V100 GPUs.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${H3_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" && -x "$ROOT_DIR/../.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/../.venv/bin/python"
fi

[[ -x "$PYTHON_BIN" ]] || {
    echo "Python venv not found. Create .venv or set H3_PYTHON." >&2
    exit 1
}

export H3_QWEN32_Q2_MODE="${H3_QWEN32_Q2_MODE:-mp}"
export H3_QWEN32_Q2_MP="${H3_QWEN32_Q2_MP:-1}"
export H3_QWEN32_Q2_TP="${H3_QWEN32_Q2_TP:-0}"
export H3_QWEN32_MP_DEVICES="${H3_QWEN32_MP_DEVICES:-cuda:0,cuda:1}"
export H3_QWEN32_MP_SPLIT="${H3_QWEN32_MP_SPLIT:-auto}"
export H3_QWEN32_MP_RESIDENCY="${H3_QWEN32_MP_RESIDENCY:-evict}"
export H3_QWEN32_MP_PREFETCH="${H3_QWEN32_MP_PREFETCH:-1}"
export H3_QWEN32_MP_PREFETCH_MAX_MIB="${H3_QWEN32_MP_PREFETCH_MAX_MIB:-256}"
export H3_CLIPPROJ_OFFLOAD_BEFORE_DIT="${H3_CLIPPROJ_OFFLOAD_BEFORE_DIT:-1}"

export H3_NO_HOST_MMAP="${H3_NO_HOST_MMAP:-1}"
export H3_TP_INT8_CONVROT_PATH="${H3_TP_INT8_CONVROT_PATH:-online}"
# Compact all Q/K/V views before long-sequence SDPA.  Q-only leaves the fused
# projection alive and OOMs 720p/243f on 16 GiB V100; short 832x480 A/B runs
# may explicitly set H3_TP_COMPACT_QKV=q.
export H3_TP_COMPACT_QKV="${H3_TP_COMPACT_QKV:-all}"
export H3_V100_FP32_TC="${H3_V100_FP32_TC:-1}"

export H3_ASYNC_VAE_LOAD="${H3_ASYNC_VAE_LOAD:-1}"
export H3_ASYNC_VAE_PREFETCH_MIB="${H3_ASYNC_VAE_PREFETCH_MIB:-0,0}"
export H3_ASYNC_VAE_OUTPUT_DTYPE="${H3_ASYNC_VAE_OUTPUT_DTYPE:-fp16}"
export H3_VAE_MP="${H3_VAE_MP:-1}"
export H3_VAE_SPLIT="${H3_VAE_SPLIT:-18}"
export H3_VAE_OUTPUT_DEVICE="${H3_VAE_OUTPUT_DEVICE:-cpu}"
export H3_VAE_INT8_TILE_BATCH="${H3_VAE_INT8_TILE_BATCH:-1}"
export H3_VAE_INT8_SM70_W8A16="${H3_VAE_INT8_SM70_W8A16:-1}"

MODEL_DIR="${H3_MODEL_DIR:-$ROOT_DIR/models}"
OUTPUT_DIR="${H3_OUTPUT_DIR:-$ROOT_DIR/output}"
mkdir -p "$OUTPUT_DIR"

exec "$PYTHON_BIN" -u "$ROOT_DIR/main.py" \
    --listen "${H3_LISTEN:-127.0.0.1}" \
    --port "${H3_PORT:-8188}" \
    --default-device 0 \
    --force-fp16 \
    --fp16-unet \
    --fp16-text-enc \
    --models-directory "$MODEL_DIR" \
    --disable-pinned-memory \
    --use-pytorch-cross-attention \
    --enable-dynamic-vram \
    --fast-disk \
    --disable-async-offload \
    --cache-none \
    --preview-method none \
    --output-directory "$OUTPUT_DIR" \
    "$@"
