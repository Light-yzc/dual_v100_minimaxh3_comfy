#!/usr/bin/env bash
set -euo pipefail

# Run this after starting scripts/start_comfyui_v100_fast.sh. It first proves
# the patched FP16 path is finite, then times the static-VRAM 480p sample and
# the separate GPU1 video-and-audio VAE decode stage.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
COMFYUI="$INSTALL_ROOT/ComfyUI"
PYTHON="$INSTALL_ROOT/.venv/bin/python"
SERVER="${COMFY_SERVER:-http://127.0.0.1:8188}"
MODEL_DIR="${H3_MODEL_DIR:-/mnt/GALAX/minimax-h3/models}"

[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON" >&2; exit 1; }
[[ -f "$COMFYUI/main.py" ]] || { echo "Missing ComfyUI: $COMFYUI" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "Missing model directory: $MODEL_DIR" >&2; exit 1; }

required_models=(
    "$MODEL_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_Q4_0.gguf"
    "$MODEL_DIR/text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
    "$MODEL_DIR/vae/minimax_h3_video_vae_fp16.safetensors"
    "$MODEL_DIR/vae/minimax_h3_audio_vae_fp32.safetensors"
    "$MODEL_DIR/loras/minimax_h3_turbo_v4_step600_ema.safetensors"
)
for model in "${required_models[@]}"; do
    [[ -f "$model" ]] || { echo "Missing required model: $model" >&2; exit 1; }
done

run_workflow() {
    local workflow="$1"
    echo "== $(basename "$workflow") =="
    "$PYTHON" "$SCRIPT_DIR/submit_workflow.py" \
        "$workflow" \
        --server "$SERVER" \
        --wait
}

run_workflow "$REPO_ROOT/workflows/static-smoke-448x256-1step.json"
"$PYTHON" "$SCRIPT_DIR/check_latent.py" "$COMFYUI/output/h3_static_smoke_448x256_latent.pt"

run_workflow "$REPO_ROOT/workflows/turbo-5s-832x480-stage1-static.json"
"$PYTHON" "$SCRIPT_DIR/check_latent.py" "$COMFYUI/output/h3_5s_832x480_latent.pt"

# The denoiser's GGUF/Qwen pages can remain in the service cgroup after
# /free. Reclaim only this ComfyUI cgroup's file cache before staging the
# 5.2-GiB video VAE; this keeps the hard 6-GiB host guard useful without
# dropping caches for unrelated applications.
"$PYTHON" "$SCRIPT_DIR/release_comfy_models.py" --server "$SERVER" --settle-seconds 2
comfy_pid="$(pgrep -f '[p]ython.*main\.py' | while read -r process_id; do
    if [[ "$(readlink -f "/proc/$process_id/cwd" 2>/dev/null || true)" == "$COMFYUI" ]]; then
        printf '%s\n' "$process_id"
        break
    fi
done)"
if [[ -n "$comfy_pid" ]]; then
    cgroup_path="$(awk -F: '$1 == "0" {print $3}' "/proc/$comfy_pid/cgroup")"
    reclaim_path="/sys/fs/cgroup${cgroup_path}/memory.reclaim"
    if [[ -w "$reclaim_path" ]]; then
        printf '%s\n' "${H3_CGROUP_RECLAIM:-3G}" >"$reclaim_path"
        echo "Reclaimed ${H3_CGROUP_RECLAIM:-3G} from ComfyUI cgroup file cache."
    fi
fi

run_workflow "$REPO_ROOT/workflows/turbo-5s-832x480-stage2.json"

video_path="$(find "$COMFYUI/output/video" -type f -name 'MiniMax_H3_dual_v100_turbo_5s_832x480*.mp4' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
[[ -n "$video_path" ]] || { echo "Stage2 did not produce an H3 MP4." >&2; exit 1; }
ffprobe -v error -show_entries stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate,channels -of json "$video_path"
echo "Generated video: $video_path"
