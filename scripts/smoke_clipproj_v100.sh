#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
MODEL_DIR="${H3_MODEL_DIR:-/mnt/GALAX/minimax-h3/models}"
SERVER="${COMFY_SERVER:-http://127.0.0.1:8188}"
WORKFLOW="${H3_SMOKE_WORKFLOW:-$REPO_ROOT/workflows/clipproj-smoke-448x256-1step.json}"
MAX_RSS_GROWTH_MB="${H3_SMOKE_MAX_RSS_GROWTH_MB:-4096}"
MECHANICAL_ROOT="${H3_MECHANICAL_ROOT:-/home/regen}"
LOG_PATH="${H3_SMOKE_LOG:-$MECHANICAL_ROOT/minimax-h3/logs/clipproj-smoke-$(date +%Y%m%d-%H%M%S).log}"

ENCODER="$MODEL_DIR/text_encoders/qwen3vl_4b_int8_convrot.safetensors"
PROJECTION="$MODEL_DIR/clip_projections/mmh3-4b-ClipProj-v3.1.safetensors"

[[ -f "$WORKFLOW" ]] || { echo "Missing workflow: $WORKFLOW" >&2; exit 1; }
[[ -f "$ENCODER" ]] || { echo "Missing encoder: $ENCODER" >&2; exit 1; }
[[ -f "$PROJECTION" ]] || { echo "Missing projection: $PROJECTION" >&2; exit 1; }

mkdir -p "$(dirname -- "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

find_comfy_pid() {
    local pid
    pid="$(systemctl --user show -p MainPID --value minimax-h3-comfy.service 2>/dev/null || true)"
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
        echo "$pid"
        return
    fi
    pgrep -f "$INSTALL_ROOT/ComfyUI/main.py" | head -n 1 || true
}

rss_mb() {
    local pid="$1"
    awk '/^VmRSS:/ {printf "%.1f", $2 / 1024}' "/proc/$pid/status"
}

model_map_hits() {
    local pid="$1"
    rg -F -e "$ENCODER" -e "$PROJECTION" "/proc/$pid/maps" 2>/dev/null || true
}

gpu_snapshot() {
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null || true
}

pid="$(find_comfy_pid)"
[[ "$pid" =~ ^[1-9][0-9]*$ ]] || {
    echo "ComfyUI is not running. Start it with scripts/start_comfyui.sh first." >&2
    exit 1
}

echo "ClipProj smoke: workflow=$WORKFLOW"
echo "ComfyUI pid=$pid; log=$LOG_PATH"
echo "GPU before:"
gpu_snapshot
rss_before="$(rss_mb "$pid")"
echo "RSS before: ${rss_before} MiB"

run_id=1
while [[ "$run_id" -le 2 ]]; do
    echo "--- submission $run_id ---"
    "$REPO_ROOT/scripts/submit_workflow.py" "$WORKFLOW" \
        --server "$SERVER" --wait --timeout "${H3_SMOKE_TIMEOUT:-1800}"
    sleep 2
    rss_now="$(rss_mb "$pid")"
    echo "RSS after submission $run_id: ${rss_now} MiB"
    echo "GPU after submission $run_id:"
    gpu_snapshot
    echo "Model mappings after submission $run_id (must be empty):"
    map_hits="$(model_map_hits "$pid")"
    if [[ -n "$map_hits" ]]; then
        echo "$map_hits"
        echo "FAIL: an encoder/projection file is still mmap'ed" >&2
        exit 2
    fi
    run_id=$((run_id + 1))
done

rss_after="$(rss_mb "$pid")"
rss_growth="$(awk -v a="$rss_before" -v b="$rss_after" 'BEGIN {print b-a}')"
echo "RSS growth across both submissions: ${rss_growth} MiB"
if awk -v growth="$rss_growth" -v limit="$MAX_RSS_GROWTH_MB" 'BEGIN {exit !(growth > limit)}'; then
    echo "FAIL: RSS grew beyond ${MAX_RSS_GROWTH_MB} MiB" >&2
    exit 2
fi

echo "PASS: ClipProj conditioning + 1-step DiT latent smoke completed twice."
echo "PASS: no encoder/projection mmap was present and RSS stayed below the guard."
