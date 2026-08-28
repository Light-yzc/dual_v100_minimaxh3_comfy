#!/usr/bin/env bash
set -euo pipefail

# Run a full one-shot topology/P2P/NCCL validation, then continuously exercise
# bidirectional peer transfers.  This is a post-reboot stability gate, not a
# replacement for hardware/power diagnostics when the PCIe fabric faults.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
PYTHON="$INSTALL_ROOT/.venv/bin/python"
SOAK_SECONDS="${NVLINK_SOAK_SECONDS:-600}"

[[ -x "$PYTHON" ]] || { echo "Missing Python environment: $PYTHON" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }

mapfile -t v100s < <(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -E '^Tesla V100')
if [[ "${#v100s[@]}" -ne 2 ]]; then
    echo "Need two healthy Tesla V100 GPUs before the soak test." >&2
    exit 1
fi

topology="$(nvidia-smi topo -m)"
# In the two-GPU table, the GPU0 row starts with its self-column (`X`) and
# then the GPU1 link.  Match both fields so we do not reject a valid NV6 row.
if ! grep -Eq '^GPU0[[:space:]]+X[[:space:]]+NV[0-9]+' <<<"$topology"; then
    echo "GPU0/GPU1 do not have an active NVLink topology." >&2
    printf '%s\n' "$topology" >&2
    exit 1
fi

echo "== One-shot NVLink/P2P/NCCL baseline =="
INSTALL_ROOT="$INSTALL_ROOT" "$SCRIPT_DIR/check_nvlink.sh"

echo "== ${SOAK_SECONDS}s bidirectional CUDA P2P soak =="
NVLINK_SOAK_SECONDS="$SOAK_SECONDS" "$PYTHON" "$SCRIPT_DIR/soak_cuda_p2p.py"

echo "== GPU state after soak =="
nvidia-smi --query-gpu=index,name,pstate,power.draw,power.limit,memory.used,memory.total,clocks.sm,clocks.mem --format=csv,noheader
