#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reapply the factory V100 profile after a reboot or driver reset.  The actual
# server is then started by start_comfyui.sh in an isolated user service so an
# IDE cgroup cannot be killed by a large GGUF mapping.
"$SCRIPT_DIR/enable_v100_performance.sh"
exec "$SCRIPT_DIR/start_comfyui.sh" "$@"
