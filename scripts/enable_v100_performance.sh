#!/usr/bin/env bash
set -euo pipefail

# Apply only factory-supported application clocks; this is not an overclock.
GPU_INDEXES="${GPU_INDEXES:-0 1}"
V100_MEMORY_CLOCK="${V100_MEMORY_CLOCK:-877}"
V100_GRAPHICS_CLOCK="${V100_GRAPHICS_CLOCK:-1530}"

command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required" >&2
    exit 1
}

if [[ "${EUID}" -eq 0 ]]; then
    NVIDIA_SMI=(nvidia-smi)
elif command -v sudo >/dev/null; then
    NVIDIA_SMI=(sudo nvidia-smi)
else
    echo "This profile needs root or sudo to enable persistence and application clocks." >&2
    exit 1
fi

gpu_count="$(nvidia-smi -L | wc -l)"
if [[ "$gpu_count" -lt 2 ]]; then
    echo "Need two NVIDIA GPUs; found $gpu_count." >&2
    exit 1
fi

for gpu in $GPU_INDEXES; do
    "${NVIDIA_SMI[@]}" -i "$gpu" -pm 1
    "${NVIDIA_SMI[@]}" -i "$gpu" -ac "$V100_MEMORY_CLOCK,$V100_GRAPHICS_CLOCK"
done

echo "== Applied V100 fast profile =="
nvidia-smi --query-gpu=index,name,persistence_mode,pstate,power.limit,clocks.applications.graphics,clocks.applications.memory,clocks.max.graphics,clocks.max.memory --format=csv,noheader

echo "== NVLink topology =="
nvidia-smi topo -m

echo "Reset factory application clocks later with:"
for gpu in $GPU_INDEXES; do
    echo "  sudo nvidia-smi -i $gpu -rac"
done
