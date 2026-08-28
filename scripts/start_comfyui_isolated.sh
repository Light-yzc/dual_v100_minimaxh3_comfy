#!/usr/bin/env bash
set -euo pipefail

# Start ComfyUI outside an IDE's cgroup.  On a small-RAM workstation a large
# model mmap otherwise counts against VS Code/Electron and systemd-oomd can
# kill the editor during model load.  This service is intentionally bounded:
# it may fail a model load, but it must not take down the desktop.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
UNIT_NAME="${H3_SYSTEMD_UNIT:-minimax-h3-comfy}"
MEMORY_HIGH="${H3_MEMORY_HIGH:-6500M}"
MEMORY_MAX="${H3_MEMORY_MAX:-7G}"
MEMORY_SWAP_MAX="${H3_MEMORY_SWAP_MAX:-256M}"

command -v systemd-run >/dev/null || {
    echo "systemd-run is required for the protected ComfyUI launcher." >&2
    exit 1
}

if ! systemctl --user show-environment >/dev/null 2>&1; then
    echo "No reachable user systemd manager; rerun with H3_ALLOW_UNISOLATED=1 only if this host is externally supervised." >&2
    exit 1
fi

if systemctl --user is-active --quiet "$UNIT_NAME.service"; then
    echo "ComfyUI service is already running: $UNIT_NAME.service" >&2
    echo "Logs: journalctl --user -fu $UNIT_NAME.service" >&2
    exit 1
fi

# Pass only settings that affect this deployment; the service gets its own
# cgroup rather than inheriting the VS Code / terminal cgroup.
ENV_ARGS=(
    "--setenv=H3_ISOLATED_SERVICE=1"
    "--setenv=INSTALL_ROOT=$INSTALL_ROOT"
    "--setenv=H3_FINITE_TRACE=${H3_FINITE_TRACE:-0}"
    "--setenv=H3_FP32_RESIDUAL=${H3_FP32_RESIDUAL:-1}"
    "--setenv=H3_FP32_MLP=${H3_FP32_MLP:-1}"
    "--setenv=H3_FP32_MLP_CHUNK_ROWS=${H3_FP32_MLP_CHUNK_ROWS:-2048}"
    "--setenv=H3_FP32_ATTN_OUT=${H3_FP32_ATTN_OUT:-1}"
    "--setenv=H3_V100_FP32_TC=${H3_V100_FP32_TC:-1}"
    "--setenv=H3_CONDITIONING_TRACE=${H3_CONDITIONING_TRACE:-0}"
    "--setenv=H3_CONDITIONING_DUMP=${H3_CONDITIONING_DUMP:-}"
    "--setenv=H3_CONDITIONING_DUMP_FULL=${H3_CONDITIONING_DUMP_FULL:-0}"
    # Layer-MP is the safe Qwen32 default; output-row TP remains explicit.
    "--setenv=H3_QWEN32_Q2_TP=${H3_QWEN32_Q2_TP:-0}"
    "--setenv=H3_QWEN32_Q2_MODE=${H3_QWEN32_Q2_MODE:-mp}"
    "--setenv=H3_QWEN32_Q2_MP=${H3_QWEN32_Q2_MP:-1}"
    "--setenv=H3_QWEN32_MP_DEVICES=${H3_QWEN32_MP_DEVICES:-cuda:0,cuda:1}"
    "--setenv=H3_QWEN32_MP_SPLIT=${H3_QWEN32_MP_SPLIT:-auto}"
    "--setenv=H3_QWEN32_OUTPUT_DEVICE=${H3_QWEN32_OUTPUT_DEVICE:-cuda:1}"
    "--setenv=H3_QWEN32_RESIDENCY=${H3_QWEN32_RESIDENCY:-evict}"
    "--setenv=H3_QWEN32_KEEP_LAYERS=${H3_QWEN32_KEEP_LAYERS:-0}"
    "--setenv=H3_QWEN32_STAGING_MIB=${H3_QWEN32_STAGING_MIB:-4}"
    "--setenv=H3_QWEN32_CACHE_MAX_MIB=${H3_QWEN32_CACHE_MAX_MIB:-256}"
    "--setenv=H3_QWEN32_MP_PREFETCH=${H3_QWEN32_MP_PREFETCH:-${H3_QWEN32_PREFETCH:-0}}"
    "--setenv=H3_QWEN32_MP_PREFETCH_MAX_MIB=${H3_QWEN32_MP_PREFETCH_MAX_MIB:-${H3_QWEN32_PREFETCH_MAX_MIB:-256}}"
    "--setenv=H3_ASYNC_VAE_LOAD=${H3_ASYNC_VAE_LOAD:-0}"
    "--setenv=H3_ASYNC_VAE_SAFETY_MIB=${H3_ASYNC_VAE_SAFETY_MIB:-1024}"
    "--setenv=H3_ASYNC_VAE_STAGING_MIB=${H3_ASYNC_VAE_STAGING_MIB:-4}"
    "--setenv=H3_ASYNC_VAE_PREFETCH_MIB=${H3_ASYNC_VAE_PREFETCH_MIB:-1962,1787}"
    # No VAE defaults here.  ``start_comfyui.sh`` owns every VAE default; this
    # launcher only forwards what the caller actually set (see the pass-through
    # list below).  Injecting a fixed H3_VAE_SPLIT would look "explicitly set"
    # downstream and would permanently pin the VAE to one layout, disabling the
    # sampling/decode stage rebalance.
)
for variable_name in \
    COMFY_LISTEN COMFY_PORT CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES \
    PYTORCH_CUDA_ALLOC_CONF H3_ATTENTION_BACKEND H3_VRAM_MODE \
    H3_V100_ATTENTION H3_V100_ATTENTION_STRICT \
    H3_V100_ATTN_BLOCK_M H3_V100_ATTN_BLOCK_N H3_V100_ATTN_WARPS H3_V100_ATTN_STAGES \
    H3_V100_RMS_ROPE H3_V100_RMS_ROPE_WARPS H3_TP_Q4_DEQUANT H3_TP_Q4_DEQUANT_STRICT H3_TP_TE_SPEED \
    H3_TP_COMPACT_QKV H3_TP_COMPACT_QKV_MIN_SEQUENCE \
    H3_TP_RESULTS_DIR H3_TP_PROFILE H3_TP_STAGE_PROFILE H3_TP_TIMEOUT H3_TP_STAGING_MIB \
    H3_TP_MLP_REDUCE_ROWS \
    H3_TP_FUSED_FP32_OPS H3_TP_FP32_OPS_WARPS \
    H3_RESERVE_VRAM H3_PARALLEL_MODE \
    H3_QWEN_SPLIT H3_VAE_SPLIT H3_VAE_DIT_SPLIT H3_VAE_DECODE_SPLIT \
    H3_VAE_REBALANCE_SAFETY_MIB H3_VAE_MP_PIPELINE H3_VAE_MP_PIPELINE_DEPTH \
    H3_VAE_OUTPUT_DEVICE H3_VAE_INT8_SM70_W8A16 H3_VAE_INT8_TILE_BATCH H3_MP_DEVICES \
    H3_PP_SPLIT H3_PP_DEVICE H3_PARALLEL_PROFILE \
    H3_FORCE_FP16_VAE H3_MODEL_DIR H3_DISABLE_PINNED_MEMORY H3_NO_HOST_MMAP \
    H3_CLIPPROJ_OFFLOAD_BEFORE_DIT \
    H3_MECHANICAL_ROOT H3_COMFY_CACHE_DIR H3_COMFY_OUTPUT_DIR H3_TP_RESULTS_DIR \
    H3_AUDIO_VAE_COMPILE H3_AUDIO_VAE_COMPILE_VALIDATE H3_AUDIO_VAE_COMPILE_MODE \
    TORCHINDUCTOR_COMPILE_THREADS MAX_JOBS TORCHINDUCTOR_CACHE_DIR \
    XDG_CACHE_HOME \
    OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS \
    TOKENIZERS_PARALLELISM MALLOC_ARENA_MAX
do
    if [[ -v "$variable_name" ]]; then
        ENV_ARGS+=("--setenv=$variable_name=${!variable_name}")
    fi
done

systemd-run --user \
    --unit="$UNIT_NAME" \
    --collect \
    --working-directory="$REPO_ROOT" \
    --property=MemoryAccounting=yes \
    --property="MemoryHigh=$MEMORY_HIGH" \
    --property="MemoryMax=$MEMORY_MAX" \
    --property="MemorySwapMax=$MEMORY_SWAP_MAX" \
    --property=OOMPolicy=kill \
    --property=Nice=10 \
    --property=CPUWeight=25 \
    --property=IOWeight=100 \
    "${ENV_ARGS[@]}" \
    "$SCRIPT_DIR/start_comfyui.sh" "$@"

echo "ComfyUI started in $UNIT_NAME.service (MemoryHigh=$MEMORY_HIGH, MemoryMax=$MEMORY_MAX)."
echo "Logs: $SCRIPT_DIR/start_comfyui.sh logs"
echo "Stop: $SCRIPT_DIR/start_comfyui.sh stop"
