#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
PYTHON="$INSTALL_ROOT/.venv/bin/python"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON" >&2; exit 1; }

echo "== GPUs =="
nvidia-smi -L

echo "== Topology =="
nvidia-smi topo -m

echo "== NVLink status =="
nvidia-smi nvlink --status

echo "== CUDA peer copy =="
"$PYTHON" "$SCRIPT_DIR/test_cuda_peer.py"

echo "== NCCL all-reduce =="
NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
    NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}" \
    "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    "$SCRIPT_DIR/test_nccl.py"
