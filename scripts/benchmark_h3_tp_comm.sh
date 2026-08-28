#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
PYTHON="$INSTALL_ROOT/.venv/bin/python"

[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON" >&2; exit 1; }

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$SCRIPT_DIR/benchmark_h3_tp_comm.py" "$@"
