#!/usr/bin/env bash
set -euo pipefail

# End-to-end 1 MP / 5 second MiniMax-H3 benchmark for the V100 GGUF path.
# Start ComfyUI first with scripts/start_comfyui_v100_fast.sh, then run this
# script. Sampling and decode are intentionally submitted as separate jobs so
# GPU 1 can release the Qwen encoder before it loads the VAE.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
COMFYUI="$INSTALL_ROOT/ComfyUI"
PYTHON="$INSTALL_ROOT/.venv/bin/python"
SERVER="${COMFY_SERVER:-http://127.0.0.1:8188}"
MODEL_DIR="${H3_MODEL_DIR:-/mnt/GALAX/minimax-h3/models}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BENCHMARK_DIR="$COMFYUI/output/benchmarks"
TELEMETRY="$BENCHMARK_DIR/h3_1mp_${STAMP}_gpu.csv"

[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON" >&2; exit 1; }
[[ -f "$COMFYUI/main.py" ]] || { echo "Missing ComfyUI: $COMFYUI" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "Missing model directory: $MODEL_DIR" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }

verify_model() {
  local path="$1"
  local size="$2"
  local hash="$3"
  [[ -f "$path" ]] || { echo "Missing model: $path" >&2; return 1; }
  [[ "$(stat -c '%s' "$path")" == "$size" ]] || { echo "Wrong/incomplete model size: $path" >&2; return 1; }
  # Hashing the whole H3 set here rereads ~28 GB before every benchmark.  The
  # downloader already verifies hashes atomically; normal benchmark runs only
  # need the cheap size guard.  Opt into a full integrity pass explicitly.
  if [[ "${H3_VERIFY_MODEL_HASHES:-0}" == "1" ]]; then
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$hash" ]] || {
      echo "Checksum mismatch: $path" >&2
      return 1
    }
  fi
}

find_comfy_pid() {
  local process_id
  for process_id in $(pgrep -f '[p]ython.*main\.py' || true); do
    if [[ "$(readlink -f "/proc/$process_id/cwd" 2>/dev/null || true)" == "$COMFYUI" ]]; then
      printf '%s\n' "$process_id"
      return 0
    fi
  done
  return 1
}

guard_host_before_benchmark() {
  local server_pid available_kib min_available_kib swap_used_kib max_swap_kib
  server_pid="$(find_comfy_pid || true)"
  [[ -n "$server_pid" ]] || {
    echo "ComfyUI is not running at $COMFYUI; start it with scripts/start_comfyui_v100_fast.sh first." >&2
    return 1
  }

  if grep -q '/app-code-' "/proc/$server_pid/cgroup"; then
    echo "Refusing to benchmark a ComfyUI process inside VS Code's cgroup." >&2
    echo "Use scripts/start_comfyui_v100_fast.sh so systemd-oomd can kill Comfy, not VS Code." >&2
    return 1
  fi

  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  min_available_kib=$(( ${H3_MIN_MEM_AVAILABLE_MIB:-2500} * 1024 ))
  if (( available_kib < min_available_kib )); then
    echo "Only $((available_kib / 1024)) MiB host RAM is available; need at least $((min_available_kib / 1024)) MiB before a 1 MP run." >&2
    return 1
  fi

  swap_used_kib="$(free -k | awk '/^Swap:/ {print $3}')"
  max_swap_kib=$(( ${H3_MAX_SWAP_USED_MIB:-1024} * 1024 ))
  if (( swap_used_kib > max_swap_kib )); then
    echo "Swap is already $((swap_used_kib / 1024)) MiB used; wait for host memory recovery before benchmarking." >&2
    return 1
  fi

  echo "Host guard: Comfy PID=$server_pid, available RAM=$((available_kib / 1024)) MiB, swap used=$((swap_used_kib / 1024)) MiB"
}

verify_model "$MODEL_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_Q4_0.gguf" 11377542880 50891b806d6d700f4f20931791ca42a083dd9148609838268ccdc782bf899c1c
verify_model "$MODEL_DIR/text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf" 8487968160 5bbc11d0b3ef197c98df2ce8f05de8fbb8eb5917cd91c33d0b59f93759b34914
verify_model "$MODEL_DIR/vae/minimax_h3_video_vae_fp16.safetensors" 5207808496 7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522
verify_model "$MODEL_DIR/vae/minimax_h3_audio_vae_fp32.safetensors" 605254808 8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48
verify_model "$MODEL_DIR/loras/minimax_h3_turbo_v4_step600_ema.safetensors" 779849816 5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3
guard_host_before_benchmark

mkdir -p "$BENCHMARK_DIR"
nvidia-smi --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.current.graphics,clocks.current.memory \
  --format=csv -l 1 >"$TELEMETRY" 2>&1 &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT

run_timed() {
  local label="$1"
  local workflow="$2"
  local start end elapsed
  start="$(date +%s%N)"
  "$PYTHON" "$SCRIPT_DIR/submit_workflow.py" "$workflow" --server "$SERVER" --wait
  end="$(date +%s%N)"
  elapsed=$(( (end - start) / 1000000 ))
  printf '%s: %.3f seconds\n' "$label" "$(awk "BEGIN { print $elapsed / 1000 }")"
}

echo "== H3 1 MP benchmark: 1344x768, 124 frames, Turbo 4-step =="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,pstate --format=csv,noheader

run_timed "stage1 sampling" "$REPO_ROOT/workflows/turbo-5s-1344x768-stage1-static.json"
# The peer-bridge store node verifies both nested latent tensors are finite
# before it accepts them; no CPU/.pt round trip is used here.
echo "== Releasing Qwen/DiT before GPU1 VAE decode =="
"$PYTHON" "$SCRIPT_DIR/release_comfy_models.py" --server "$SERVER"
run_timed "stage2 video+audio decode" "$REPO_ROOT/workflows/turbo-5s-1344x768-stage2.json"

echo "GPU telemetry: $TELEMETRY"
