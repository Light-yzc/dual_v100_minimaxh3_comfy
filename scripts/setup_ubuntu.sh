#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

COMFY_COMMIT="2a68ce33b4c9ea6ee4283e618a74560cefb32694"
GGUF_COMMIT="72c8990f22b86b06a4c9f4cad628d18825160f79"
MULTIGPU_COMMIT="b51c99a525e9607e43545ee2a8b7694c74a4775a"
TURBO_COMMIT="4274783a23afcfdbea3b4876cb79effd6c510785"

if [[ "${INSTALL_SYSTEM_DEPS:-0}" == "1" ]]; then
    sudo apt-get update
    sudo apt-get install -y git python3 python3-venv python3-dev build-essential ffmpeg
fi

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo "$PYTHON_BIN is required" >&2; exit 1; }

mkdir -p "$INSTALL_ROOT"

clone_at_commit() {
    local url="$1"
    local target="$2"
    local commit="$3"
    if [[ ! -d "$target/.git" ]]; then
        git clone --filter=blob:none "$url" "$target"
    fi
    git -C "$target" fetch --depth 1 origin "$commit"
    git -C "$target" checkout --detach "$commit"
}

clone_at_commit \
    https://github.com/Comfy-Org/ComfyUI.git \
    "$INSTALL_ROOT/ComfyUI" \
    "$COMFY_COMMIT"

mkdir -p "$INSTALL_ROOT/ComfyUI/custom_nodes"
clone_at_commit \
    https://github.com/molbal/ComfyUI-GGUF.git \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" \
    "$GGUF_COMMIT"
clone_at_commit \
    https://github.com/pollockjj/ComfyUI-MultiGPU.git \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MultiGPU" \
    "$MULTIGPU_COMMIT"
clone_at_commit \
    https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" \
    "$TURBO_COMMIT"

PATCH_FILE="$REPO_ROOT/patches/comfyui-minimax-h3-v100-fp16-rmsnorm.patch"
if git -C "$INSTALL_ROOT/ComfyUI" apply --unidiff-zero --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "FP16 RMSNorm patch is already applied"
else
    git -C "$INSTALL_ROOT/ComfyUI" apply --unidiff-zero --check "$PATCH_FILE"
    git -C "$INSTALL_ROOT/ComfyUI" apply --unidiff-zero "$PATCH_FILE"
fi

install -D -m 0644 \
    "$REPO_ROOT/custom_nodes/DualV100/__init__.py" \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/DualV100/__init__.py"
install -D -m 0644 \
    "$REPO_ROOT/custom_nodes/DualV100/h3_latent_io.py" \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/DualV100/h3_latent_io.py"

if [[ ! -x "$INSTALL_ROOT/.venv/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$INSTALL_ROOT/.venv"
fi

PYTHON="$INSTALL_ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install -r "$INSTALL_ROOT/ComfyUI/requirements.txt"
"$PYTHON" -m pip install -r "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt"

mkdir -p \
    "$INSTALL_ROOT/ComfyUI/models/diffusion_models" \
    "$INSTALL_ROOT/ComfyUI/models/text_encoders" \
    "$INSTALL_ROOT/ComfyUI/models/vae" \
    "$INSTALL_ROOT/ComfyUI/models/loras" \
    "$INSTALL_ROOT/ComfyUI/output"

"$PYTHON" -m py_compile \
    "$INSTALL_ROOT/ComfyUI/comfy/ldm/minimax/model.py" \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/DualV100/__init__.py" \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/DualV100/h3_latent_io.py"

"$PYTHON" - <<'PY'
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}, indent=2))
PY

echo "Installed into: $INSTALL_ROOT"
echo "Weights were not downloaded. See README.md for the required filenames."
