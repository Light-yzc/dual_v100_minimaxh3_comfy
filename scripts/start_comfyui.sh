#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
COMFYUI="$INSTALL_ROOT/ComfyUI"
PYTHON="$INSTALL_ROOT/.venv/bin/python"
MODEL_DIR="${H3_MODEL_DIR:-/mnt/GALAX/minimax-h3/models}"
UNIT_NAME="${H3_SYSTEMD_UNIT:-minimax-h3-comfy}"

# This file is both the low-level ComfyUI launcher (no subcommand) and the
# small service controller users normally need.  Keep the management verbs
# here so words such as ``start`` never reach main.py as an unknown argument.
usage() {
    cat <<EOF
Usage: $0 {start|stop|restart|logs|status} [options]

  start    start the protected user service (extra options go to ComfyUI)
  stop     stop the user service
  restart  restart it, or start it when it is not active
  logs     follow the service journal (Ctrl-C only stops log viewing)
  status   show service status and recent journal lines

No subcommand starts ComfyUI directly with the configured safe profile.
Qwen32 default: layer-MP (mode=mp, MP=1); TP requires explicit mode=tp and TP=1.
Qwen32 layer prefetch is experimental and off by default; set H3_QWEN32_MP_PREFETCH=1.
EOF
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        start)
            shift
            exec "$SCRIPT_DIR/start_comfyui_isolated.sh" "$@"
            ;;
        stop)
            shift
            if [[ $# -gt 0 ]]; then
                echo "stop does not accept extra arguments: $*" >&2
                usage >&2
                exit 2
            fi
            if systemctl --user is-active --quiet "$UNIT_NAME.service"; then
                systemctl --user stop "$UNIT_NAME.service"
            else
                echo "ComfyUI service is not running: $UNIT_NAME.service"
            fi
            exit 0
            ;;
        restart)
            shift
            if [[ $# -gt 0 ]]; then
                echo "restart does not accept extra arguments: $*" >&2
                usage >&2
                exit 2
            fi
            if systemctl --user is-active --quiet "$UNIT_NAME.service"; then
                # Stop first so environment overrides on this invocation are
                # passed to the newly-created transient unit.  systemctl
                # restart would preserve the old unit environment, making a
                # command such as H3_TP_TE_SPEED=0 ... restart ineffective.
                systemctl --user stop "$UNIT_NAME.service"
            fi
            exec "$SCRIPT_DIR/start_comfyui_isolated.sh"
            ;;
        logs|log)
            shift
            exec journalctl --user -fu "$UNIT_NAME.service" "$@"
            ;;
        status)
            shift
            if [[ $# -gt 0 ]]; then
                echo "status does not accept extra arguments: $*" >&2
                usage >&2
                exit 2
            fi
            exec systemctl --user --no-pager --full status "$UNIT_NAME.service"
            ;;
        help|-h|--help)
            usage
            exit 0
            ;;
    esac
fi

# This workstation keeps the large, read-only model payload on /mnt/GALAX
# (SSD) and generated/cache data on /home/regen (mechanical disk).  Keep these
# defaults explicit so a future change to --models-directory cannot silently
# move compiler cache or video output onto the model disk.
MECHANICAL_ROOT="${H3_MECHANICAL_ROOT:-/home/regen}"
COMFY_CACHE_DIR="${H3_COMFY_CACHE_DIR:-$MECHANICAL_ROOT/minimax-h3/ComfyUI/.torchinductor-cache}"
COMFY_OUTPUT_DIR="${H3_COMFY_OUTPUT_DIR:-$MECHANICAL_ROOT/minimax-h3/ComfyUI/output}"
TP_RESULTS_DIR="${H3_TP_RESULTS_DIR:-$REPO_ROOT/results}"

[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON" >&2; exit 1; }
[[ -f "$COMFYUI/main.py" ]] || { echo "Missing ComfyUI: $COMFYUI" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || {
    echo "Missing model directory: $MODEL_DIR" >&2
    echo "Set H3_MODEL_DIR to an existing ComfyUI models directory to override it." >&2
    exit 1
}

# A terminal started by VS Code is charged to VS Code's systemd cgroup.  Large
# GGUF mmap/page-cache activity would then make systemd-oomd kill the editor
# instead of the inference process.  The public launcher therefore hands off
# to a sibling, memory-bounded user service by default.  Set
# H3_ALLOW_UNISOLATED=1 only for an intentionally headless / externally
# supervised launch.
if [[ "${H3_ISOLATED_SERVICE:-0}" != "1" && "${H3_ALLOW_UNISOLATED:-0}" != "1" ]]; then
    exec "$SCRIPT_DIR/start_comfyui_isolated.sh" "$@"
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export H3_NO_HOST_MMAP="${H3_NO_HOST_MMAP:-1}"
# ClipProj encodes on the resident Qwen path, then returns the encoder and
# projection cache to CPU immediately before the DiT sampler starts.  Set this
# to 0 only when deliberately testing the legacy always-resident behaviour.
export H3_CLIPPROJ_OFFLOAD_BEFORE_DIT="${H3_CLIPPROJ_OFFLOAD_BEFORE_DIT:-1}"

# The 32B Q2 encoder uses the complete-layer MP route by default.  It keeps
# each layer on one V100 and leaves the output-row TP implementation as an
# explicit experiment: set H3_QWEN32_Q2_MODE=tp together with
# H3_QWEN32_Q2_TP=1 when a workflow specifically needs that route.
export H3_QWEN32_Q2_TP="${H3_QWEN32_Q2_TP:-0}"
export H3_QWEN32_Q2_MODE="${H3_QWEN32_Q2_MODE:-mp}"
export H3_QWEN32_Q2_MP="${H3_QWEN32_Q2_MP:-1}"
export H3_QWEN32_MP_DEVICES="${H3_QWEN32_MP_DEVICES:-cuda:0,cuda:1}"
export H3_QWEN32_MP_SPLIT="${H3_QWEN32_MP_SPLIT:-auto}"
export H3_QWEN32_OUTPUT_DEVICE="${H3_QWEN32_OUTPUT_DEVICE:-cuda:1}"
export H3_QWEN32_RESIDENCY="${H3_QWEN32_RESIDENCY:-evict}"
export H3_QWEN32_KEEP_LAYERS="${H3_QWEN32_KEEP_LAYERS:-0}"
export H3_QWEN32_STAGING_MIB="${H3_QWEN32_STAGING_MIB:-4}"
export H3_QWEN32_CACHE_MAX_MIB="${H3_QWEN32_CACHE_MAX_MIB:-256}"
export H3_QWEN32_MP_PREFETCH="${H3_QWEN32_MP_PREFETCH:-${H3_QWEN32_PREFETCH:-0}}"
export H3_QWEN32_MP_PREFETCH_MAX_MIB="${H3_QWEN32_MP_PREFETCH_MAX_MIB:-${H3_QWEN32_PREFETCH_MAX_MIB:-256}}"
export H3_ASYNC_VAE_LOAD="${H3_ASYNC_VAE_LOAD:-0}"
export H3_ASYNC_VAE_SAFETY_MIB="${H3_ASYNC_VAE_SAFETY_MIB:-1024}"
export H3_ASYNC_VAE_STAGING_MIB="${H3_ASYNC_VAE_STAGING_MIB:-4}"
export H3_ASYNC_VAE_PREFETCH_MIB="${H3_ASYNC_VAE_PREFETCH_MIB:-1962,1787}"

require_boolean_env() {
    local variable_name="$1"
    case "${!variable_name}" in
        0|1) ;;
        *)
            echo "Unsupported $variable_name=${!variable_name}; use 0 or 1" >&2
            exit 2
            ;;
    esac
}

require_integer_range() {
    local variable_name="$1"
    local minimum="$2"
    local maximum="$3"
    local value="${!variable_name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]] || (( 10#$value < minimum || 10#$value > maximum )); then
        echo "Unsupported $variable_name=$value; use an integer from $minimum through $maximum" >&2
        exit 2
    fi
}

require_boolean_env H3_QWEN32_Q2_TP
require_boolean_env H3_QWEN32_Q2_MP
require_boolean_env H3_QWEN32_MP_PREFETCH
require_boolean_env H3_ASYNC_VAE_LOAD
require_boolean_env H3_CLIPPROJ_OFFLOAD_BEFORE_DIT
case "$H3_QWEN32_Q2_MODE" in
    tp|mp) ;;
    *)
        echo "Unsupported H3_QWEN32_Q2_MODE=$H3_QWEN32_Q2_MODE; use tp or mp" >&2
        exit 2
        ;;
esac
echo "[DualV100] Qwen32 route: mode=$H3_QWEN32_Q2_MODE mp=$H3_QWEN32_Q2_MP tp=$H3_QWEN32_Q2_TP devices=$H3_QWEN32_MP_DEVICES split=$H3_QWEN32_MP_SPLIT prefetch=$H3_QWEN32_MP_PREFETCH; ClipProj pre-DiT offload=$H3_CLIPPROJ_OFFLOAD_BEFORE_DIT" >&2
if [[ ! "$H3_QWEN32_MP_DEVICES" =~ ^[^,]+,[^,]+$ ]]; then
    echo "Unsupported H3_QWEN32_MP_DEVICES=$H3_QWEN32_MP_DEVICES; use device0,device1" >&2
    exit 2
fi
if [[ "$H3_QWEN32_MP_SPLIT" != "auto" && ! "$H3_QWEN32_MP_SPLIT" =~ ^[0-9]+$ ]]; then
    echo "Unsupported H3_QWEN32_MP_SPLIT=$H3_QWEN32_MP_SPLIT; use auto or an integer" >&2
    exit 2
fi
case "$H3_QWEN32_RESIDENCY" in
    evict|partial|full) ;;
    *)
        echo "Unsupported H3_QWEN32_RESIDENCY=$H3_QWEN32_RESIDENCY; use evict, partial, or full" >&2
        exit 2
        ;;
esac
require_integer_range H3_QWEN32_KEEP_LAYERS 0 50
require_integer_range H3_QWEN32_STAGING_MIB 1 64
require_integer_range H3_QWEN32_CACHE_MAX_MIB 0 16384
require_integer_range H3_QWEN32_MP_PREFETCH_MAX_MIB 1 4096
require_integer_range H3_ASYNC_VAE_SAFETY_MIB 1 16384
require_integer_range H3_ASYNC_VAE_STAGING_MIB 4 8
if [[ ! "$H3_ASYNC_VAE_PREFETCH_MIB" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo "Unsupported H3_ASYNC_VAE_PREFETCH_MIB=$H3_ASYNC_VAE_PREFETCH_MIB; use GPU0_MiB,GPU1_MiB" >&2
    exit 2
fi
IFS=, read -r async_vae_prefetch_gpu0 async_vae_prefetch_gpu1 <<< "$H3_ASYNC_VAE_PREFETCH_MIB"
if (( 10#$async_vae_prefetch_gpu0 > 16384 || 10#$async_vae_prefetch_gpu1 > 16384 )); then
    echo "Unsupported H3_ASYNC_VAE_PREFETCH_MIB=$H3_ASYNC_VAE_PREFETCH_MIB; each cap must be <= 16384 MiB" >&2
    exit 2
fi

# The V100 has no native INT8 Tensor Core path for this checkpoint. Keep INT8
# weights resident and dequantize one Linear at a time for FP16 Tensor Core
# GEMM.  That bounded temporary is the reason the INT8 decoder needs ~0.3 GiB
# of decode headroom where the FP16 decoder needs >11 GiB: the FP16 path keeps
# every layer's activation live through ComfyUI's ordinary Linear.
export H3_VAE_INT8_SM70_W8A16="${H3_VAE_INT8_SM70_W8A16:-1}"
# Measured on 448x256x21, INT8 + layer-MP, same tiled_decode path:
#   tile_batch=1  0.362 s  decode delta 306/262 MiB
#   tile_batch=2  0.394 s  decode delta 438/414 MiB
# Batching tiles concatenates them on the batch axis, so activations, the
# attention temporary and the cross-card handoff all scale with it.  Under
# layer-MP that costs both time and memory, so the default is now 1.  Set 2
# only for a single-card decode where the weight-dequant amortization wins.
export H3_VAE_INT8_TILE_BATCH="${H3_VAE_INT8_TILE_BATCH:-1}"
# Sampling and decode want opposite layouts.  DiT needs cuda:0 clear (a
# decode-optimal 24/12 split parked ~3.4 GiB of idle FP16 decoder weights
# there and pushed the 1280x736 QKV projection into OOM), while decode wants
# the heavier half on cuda:0 because the layer-MP decoder is serial.  The VAE
# now loads at H3_VAE_DIT_SPLIT and moves the boundary blocks over NVLink
# before the first decode, then back before the next sampler entry.
# Set H3_VAE_SPLIT to pin both stages to one layout and disable the move.
export H3_VAE_DIT_SPLIT="${H3_VAE_DIT_SPLIT:-18}"
export H3_VAE_DECODE_SPLIT="${H3_VAE_DECODE_SPLIT:-24}"
# Whole-card headroom the rebalance admission check must preserve on each
# card.  When cuda:0 cannot take the decode layout the manager degrades
# toward the sampling layout instead of letting a later allocation fail
# inside a collective.
export H3_VAE_REBALANCE_SAFETY_MIB="${H3_VAE_REBALANCE_SAFETY_MIB:-1024}"
# Keep the long video output in the host buffer.  This is explicit rather than
# inheriting ComfyUI's mutable intermediate-device policy, which can point at
# GPU1 after a MultiGPU node has run.
export H3_VAE_OUTPUT_DEVICE="${H3_VAE_OUTPUT_DEVICE:-cpu}"
validate_vae_split() {
    local variable_name="$1"
    case "${!variable_name}" in
        auto|balanced|default) ;;
        ''|*[!0-9]*)
            echo "Unsupported $variable_name=${!variable_name}; use 1-35 or auto" >&2
            exit 2
            ;;
        *)
            require_integer_range "$variable_name" 1 35
            ;;
    esac
}
validate_vae_split H3_VAE_DIT_SPLIT
validate_vae_split H3_VAE_DECODE_SPLIT
if [[ -n "${H3_VAE_SPLIT:-}" ]]; then
    validate_vae_split H3_VAE_SPLIT
    export H3_VAE_SPLIT
fi
require_integer_range H3_VAE_REBALANCE_SAFETY_MIB 0 16384
require_integer_range H3_VAE_INT8_TILE_BATCH 1 8
require_boolean_env H3_VAE_INT8_SM70_W8A16
if [[ -n "${H3_VAE_SPLIT:-}" ]]; then
    echo "[DualV100] VAE layout: pinned split=$H3_VAE_SPLIT (stage rebalance disabled)" >&2
else
    echo "[DualV100] VAE layout: dit_split=$H3_VAE_DIT_SPLIT decode_split=$H3_VAE_DECODE_SPLIT safety=${H3_VAE_REBALANCE_SAFETY_MIB}MiB; tile_batch=$H3_VAE_INT8_TILE_BATCH w8a16=$H3_VAE_INT8_SM70_W8A16" >&2
fi
case "$H3_VAE_OUTPUT_DEVICE" in
    cpu|host|auto|cuda:[0-9]*) ;;
    *)
        echo "Unsupported H3_VAE_OUTPUT_DEVICE=$H3_VAE_OUTPUT_DEVICE; use cpu, auto, or cuda:N" >&2
        exit 2
        ;;
esac
# BigVGAN's validated audio compile path is opt-in.  Eager audio decode is the
# default so different output shapes do not trigger extra specialized graphs or
# cold compilation.  Set H3_AUDIO_VAE_COMPILE=1 to opt in.  Keep the compiler
# single-threaded and its persistent code cache with ComfyUI, separate from the
# model store.  The first compiled request performs a strict eager-vs-compiled
# waveform gate; set H3_AUDIO_VAE_COMPILE_VALIDATE=0 only for experiments.
export H3_AUDIO_VAE_COMPILE="${H3_AUDIO_VAE_COMPILE:-0}"
export H3_AUDIO_VAE_COMPILE_VALIDATE="${H3_AUDIO_VAE_COMPILE_VALIDATE:-1}"
export H3_AUDIO_VAE_COMPILE_MODE="${H3_AUDIO_VAE_COMPILE_MODE:-default}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MECHANICAL_ROOT/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$COMFY_CACHE_DIR}"
export H3_TP_RESULTS_DIR="${H3_TP_RESULTS_DIR:-$TP_RESULTS_DIR}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$COMFY_OUTPUT_DIR" "$H3_TP_RESULTS_DIR"

# V100 cannot safely accumulate the BF16-trained H3 residual/MLP range in
# native FP16 at 832x480. Keep only the numerically sensitive islands in FP32;
# the GGUF weights, attention and most activations remain FP16. The MLP row
# chunk bounds the temporary FP32 activation without a full-model FP32 run.
# Finite tracing is diagnostic-only and stays off by default because it
# synchronizes every checked tensor.
export H3_FINITE_TRACE="${H3_FINITE_TRACE:-0}"
export H3_FP32_RESIDUAL="${H3_FP32_RESIDUAL:-1}"
export H3_FP32_MLP="${H3_FP32_MLP:-1}"
export H3_FP32_MLP_CHUNK_ROWS="${H3_FP32_MLP_CHUNK_ROWS:-2048}"
export H3_FP32_ATTN_OUT="${H3_FP32_ATTN_OUT:-1}"
# Preserve FP32 outputs while letting V100 Tensor Cores execute H3's two wide
# row projections. FC2 input is protected by exact power-of-two row scaling.
export H3_V100_FP32_TC="${H3_V100_FP32_TC:-1}"

# The fused partial RMS-RoPE path is H3/SM70-specific and separately gated.
# Global attention stays on PyTorch efficient SDPA: the experimental Triton
# online-softmax kernel is retained for reproducible research but is slower on
# Volta and therefore never enabled by this launcher by default.
export H3_V100_ATTENTION="${H3_V100_ATTENTION:-pytorch}"
export H3_V100_RMS_ROPE="${H3_V100_RMS_ROPE:-pytorch}"
export H3_TP_Q4_DEQUANT="${H3_TP_Q4_DEQUANT:-eager}"
export H3_TP_Q4_DEQUANT_STRICT="${H3_TP_Q4_DEQUANT_STRICT:-0}"
# Deployment-level kill switch for the experimental TP-aware TE-Speed node.
# The node itself remains opt-in; setting this to 0 forces all workflows back
# to the exact full 50-layer TP route without editing their JSON.
export H3_TP_TE_SPEED="${H3_TP_TE_SPEED:-1}"
# On SM70, making only Q contiguous selects a substantially faster efficient
# SDPA path at production sequence lengths without changing any arithmetic.
# K/V stay as fused-QKV views to avoid ~0.5 GiB of unnecessary transient VRAM.
# Tiny smoke sequences retain the original path because their copy overhead is
# larger than the kernel saving.
export H3_TP_COMPACT_QKV="${H3_TP_COMPACT_QKV:-q}"
export H3_TP_COMPACT_QKV_MIN_SEQUENCE="${H3_TP_COMPACT_QKV_MIN_SEQUENCE:-4096}"

# ComfyUI otherwise reserves roughly 40% of this host's RAM for pinned CPU
# buffers. Keep the safe profile as the default; opt in only with explicit
# H3_DISABLE_PINNED_MEMORY=0 when there is enough spare host memory.
case "${H3_DISABLE_PINNED_MEMORY:-1}" in
    1) PINNED_MEMORY_ARGS=(--disable-pinned-memory) ;;
    0) PINNED_MEMORY_ARGS=() ;;
    *)
        echo "Unsupported H3_DISABLE_PINNED_MEMORY=${H3_DISABLE_PINNED_MEMORY}; use 0 or 1" >&2
        exit 2
        ;;
esac

# Let each H3 VAE select its declared working dtype.  The video decoder
# selects FP16 on V100, while the H3 audio VAE explicitly requires FP32.
# A global --fp16-vae overrides that declaration and can degrade/break audio.
VAE_ARGS=()
if [[ "${H3_FORCE_FP16_VAE:-0}" == "1" ]]; then
    VAE_ARGS+=(--fp16-vae)
fi

# PyTorch's memory-efficient SDPA backend is available on SM70 and avoids the
# quadratic attention matrix for the 1 MP H3 packed sequence.  Flash/Sage are
# intentionally not selected on V100.  Keep split attention as an explicit
# fallback for a driver/PyTorch regression: H3_ATTENTION_BACKEND=split.
case "${H3_ATTENTION_BACKEND:-pytorch}" in
    pytorch) ATTENTION_ARGS=(--use-pytorch-cross-attention) ;;
    split) ATTENTION_ARGS=(--use-split-cross-attention) ;;
    *)
        echo "Unsupported H3_ATTENTION_BACKEND=${H3_ATTENTION_BACKEND}; use pytorch or split" >&2
        exit 2
        ;;
esac

# The old default forced --highvram together with --disable-dynamic-vram.
# That maps both large GGUF files through the launcher cgroup and is unsafe on
# this 14 GiB RAM host.  DynamicVRAM + --fast-disk makes the NVMe model store
# eligible for ComfyUI's file-backed loading path.  The resident profile is
# opt-in once a staged Qwen -> DiT workflow has been proven on this host.
case "${H3_VRAM_MODE:-safe}" in
    safe)
        VRAM_ARGS=(--enable-dynamic-vram --fast-disk --disable-async-offload)
        ;;
    resident)
        VRAM_ARGS=(--highvram --enable-dynamic-vram --fast-disk --disable-async-offload)
        ;;
    legacy-static)
        [[ "${H3_ALLOW_UNSAFE_STATIC:-0}" == "1" ]] || {
            echo "legacy-static can exhaust host RAM; set H3_ALLOW_UNSAFE_STATIC=1 only on a headless host." >&2
            exit 2
        }
        VRAM_ARGS=(--highvram --disable-dynamic-vram --disable-async-offload)
        ;;
    *)
        echo "Unsupported H3_VRAM_MODE=${H3_VRAM_MODE}; use safe, resident, or legacy-static" >&2
        exit 2
        ;;
esac

if [[ -n "${H3_RESERVE_VRAM:-}" ]]; then
    VRAM_ARGS+=(--reserve-vram "$H3_RESERVE_VRAM")
fi

cd "$COMFYUI"
exec "$PYTHON" -u main.py \
    --listen "${COMFY_LISTEN:-127.0.0.1}" \
    --port "${COMFY_PORT:-8188}" \
    --default-device 0 \
    --force-fp16 \
    --fp16-unet \
    --fp16-text-enc \
    --models-directory "$MODEL_DIR" \
    --extra-model-paths-config "$REPO_ROOT/configs/extra_model_paths.yaml.example" \
    "${PINNED_MEMORY_ARGS[@]}" \
    "${VAE_ARGS[@]}" \
    "${ATTENTION_ARGS[@]}" \
    "${VRAM_ARGS[@]}" \
    --preview-method none \
    --output-directory "$COMFY_OUTPUT_DIR" \
    "$@"
