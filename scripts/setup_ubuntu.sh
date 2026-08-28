#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
MODEL_DIR="${H3_MODEL_DIR:-/mnt/GALAX/minimax-h3/models}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
SOURCE_FETCH="${SOURCE_FETCH:-git}"

# PyTorch 2.8.0 CUDA 12.6 publishes Linux wheels for CPython 3.9 through 3.13.
# Ubuntu 26.04 defaults to Python 3.14, which cannot install this pinned stack.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in \
        "$HOME/.local/bin/python3.13" \
        python3.13 \
        "$HOME/.local/bin/python3.12" \
        python3.12 \
        python3.11 \
        python3.10 \
        python3.9 \
        python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

COMFY_COMMIT="2a68ce33b4c9ea6ee4283e618a74560cefb32694"
GGUF_COMMIT="72c8990f22b86b06a4c9f4cad628d18825160f79"
MULTIGPU_COMMIT="b51c99a525e9607e43545ee2a8b7694c74a4775a"
TURBO_COMMIT="4274783a23afcfdbea3b4876cb79effd6c510785"
CLIPPROJ_COMMIT="c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c"

if [[ "${INSTALL_SYSTEM_DEPS:-0}" == "1" ]]; then
    sudo apt-get update
    sudo apt-get install -y git python3 python3-venv python3-dev build-essential ffmpeg
fi

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo "$PYTHON_BIN is required" >&2; exit 1; }

case "$SOURCE_FETCH" in
    git|archive) ;;
    *)
        echo "SOURCE_FETCH must be 'git' or 'archive', got: $SOURCE_FETCH" >&2
        exit 1
        ;;
esac
if [[ "$SOURCE_FETCH" == "archive" ]]; then
    command -v wget >/dev/null || { echo "wget is required when SOURCE_FETCH=archive" >&2; exit 1; }
    command -v tar >/dev/null || { echo "tar is required when SOURCE_FETCH=archive" >&2; exit 1; }
    command -v patch >/dev/null || { echo "patch is required when SOURCE_FETCH=archive" >&2; exit 1; }
    command -v mktemp >/dev/null || { echo "mktemp is required when SOURCE_FETCH=archive" >&2; exit 1; }
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
    3.9|3.10|3.11|3.12|3.13) ;;
    *)
        cat >&2 <<EOF
PyTorch 2.8.0 + CUDA 12.6 requires CPython 3.9 through 3.13; selected $PYTHON_BIN is Python $PYTHON_VERSION.
On Ubuntu 26.04, install a supported interpreter and rerun, for example:
  uv python install 3.13
  PYTHON_BIN=\$HOME/.local/bin/python3.13 INSTALL_ROOT=\$HOME/minimax-h3 ./scripts/setup_ubuntu.sh
EOF
        exit 1
        ;;
esac

mkdir -p "$INSTALL_ROOT"

sync_python_package() {
    local package="$1"
    local source_dir="$REPO_ROOT/custom_nodes/$package"
    local target_dir="$INSTALL_ROOT/ComfyUI/custom_nodes/$package"
    local source_file

    [[ -d "$source_dir" ]] || {
        echo "Missing repository custom-node package: $source_dir" >&2
        exit 1
    }
    mkdir -p "$target_dir"
    for source_file in "$source_dir"/*.py; do
        install -D -m 0644 "$source_file" "$target_dir/$(basename "$source_file")"
    done
}

sync_workflows() {
    local source_dir="$REPO_ROOT/workflows"
    local target_dir="$INSTALL_ROOT/ComfyUI/user/default/workflows"
    local source_file

    # Workflows are user-facing deployment assets, not model payloads.  Copy
    # only the curated H3 V100 entrypoints and never delete user workflows.
    # The int8-* pair is the current production route: INT8 video VAE plus the
    # full video/audio output chain, in both reference-image and first/last
    # frame modes.  The multimode entries are kept for the FP16 comparison.
    mkdir -p "$target_dir"
    for source_file in \
        "$source_dir"/H3-V100-09-int8-ref2v-832x480-124f-4step.json \
        "$source_dir"/H3-V100-09-int8-ref2v-832x480-124f-4step-ui.json \
        "$source_dir"/H3-V100-10-int8-fl2v-832x480-124f-4step.json \
        "$source_dir"/H3-V100-10-int8-fl2v-832x480-124f-4step-ui.json \
        "$source_dir"/H3-V100-11-int8-ref2v-smoke-448x256-1step.json \
        "$source_dir"/H3-V100-11-int8-fl2v-smoke-448x256-1step.json \
        "$source_dir"/h3-v100-multimode-adjustable-resident.json \
        "$source_dir"/h3-v100-multimode-adjustable-resident-ui.json \
        "$source_dir"/h3-v100-multimode-smoke-448x256-1step.json \
        "$source_dir"/H3-V100-START-HERE.md; do
        [[ -f "$source_file" ]] || continue
        install -D -m 0644 "$source_file" "$target_dir/$(basename "$source_file")"
    done
}

sync_repo_custom_nodes() {
    sync_python_package NoHostMMap
    sync_python_package DualV100
    sync_workflows
}

# Hardware iteration should not fetch sources, reapply upstream patches, or
# reinstall Python packages merely to deploy repository-owned node changes.
# This mode is intentionally additive: it never deletes deployment-local files.
if [[ "${SYNC_ONLY:-0}" == "1" ]]; then
    PYTHON="$INSTALL_ROOT/.venv/bin/python"
    [[ -d "$INSTALL_ROOT/ComfyUI/custom_nodes" ]] || {
        echo "ComfyUI installation is missing under $INSTALL_ROOT" >&2
        exit 1
    }
    [[ -x "$PYTHON" ]] || {
        echo "Python environment is missing: $PYTHON" >&2
        exit 1
    }
    sync_repo_custom_nodes
    "$PYTHON" -m py_compile \
        "$INSTALL_ROOT/ComfyUI/custom_nodes/DualV100/"*.py \
        "$INSTALL_ROOT/ComfyUI/custom_nodes/NoHostMMap/"*.py
    echo "Synchronized repository custom nodes into: $INSTALL_ROOT/ComfyUI/custom_nodes"
    exit 0
fi

clone_at_commit() {
    local url="$1"
    local target="$2"
    local commit="$3"
    shift 3
    local target_parent
    target_parent="$(dirname "$target")"

    # An interrupted archive stream can leave a marker and only the first few
    # files behind. Check the backend files this deployment actually imports
    # before treating an archive checkout as usable.
    source_tree_is_complete() {
        local path="$1"
        shift
        local required_file
        for required_file in "$@"; do
            [[ -f "$path/$required_file" ]] || return 1
        done
    }

    if [[ "$SOURCE_FETCH" == "git" ]]; then
        if [[ ! -d "$target/.git" ]]; then
            git clone --filter=blob:none "$url" "$target"
        fi
        git -C "$target" fetch --depth 1 origin "$commit"
        git -C "$target" checkout --detach "$commit"
        source_tree_is_complete "$target" "$@" || {
            echo "Checkout is missing required source files: $target" >&2
            exit 1
        }
        return
    fi

    if [[ -f "$target/.minimax-source-commit" ]]; then
        local installed_commit
        read -r installed_commit < "$target/.minimax-source-commit"
        if [[ "$installed_commit" == "$commit" ]] && source_tree_is_complete "$target" "$@"; then
            echo "Archive checkout is already present: $target"
            return
        fi
        if [[ "$installed_commit" == "$commit" ]]; then
            local backup_target
            backup_target="$(mktemp -d "$target_parent/.minimax-incomplete.XXXXXX")"
            rmdir "$backup_target"
            mv "$target" "$backup_target"
            echo "Moved incomplete archive checkout to: $backup_target" >&2
        fi
    fi
    if [[ -e "$target" ]] && [[ -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to replace non-empty archive checkout: $target" >&2
        exit 1
    fi

    local repository="${url#https://github.com/}"
    local staging
    repository="${repository%.git}"
    mkdir -p "$target_parent"
    if [[ -d "$target" ]]; then
        rmdir "$target"
    fi
    staging="$(mktemp -d "$target_parent/.minimax-source.XXXXXX")"
    wget --progress=dot:giga -O - \
        "https://codeload.github.com/$repository/tar.gz/$commit" \
        | tar -xzf - --strip-components=1 -C "$staging"
    source_tree_is_complete "$staging" "$@" || {
        echo "Archive extraction is missing required source files; staged copy left at: $staging" >&2
        exit 1
    }
    printf '%s\n' "$commit" > "$staging/.minimax-source-commit"
    mv "$staging" "$target"
}

clone_at_commit \
    https://github.com/Comfy-Org/ComfyUI.git \
    "$INSTALL_ROOT/ComfyUI" \
    "$COMFY_COMMIT" \
    main.py \
    requirements.txt \
    comfy/ldm/minimax/model.py

mkdir -p "$INSTALL_ROOT/ComfyUI/custom_nodes"
clone_at_commit \
    https://github.com/molbal/ComfyUI-GGUF.git \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" \
    "$GGUF_COMMIT" \
    nodes.py \
    loader.py \
    ops.py \
    quant_ops.py

sync_python_package NoHostMMap

GGUF_PATCH_FILE="$REPO_ROOT/patches/comfyui-gguf-no-host-mmap.patch"
if [[ "$SOURCE_FETCH" == "archive" ]]; then
    if patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" -p1 --fuzz=5 --reverse --dry-run < "$GGUF_PATCH_FILE" >/dev/null 2>&1; then
        echo "GGUF no-host-mmap patch is already applied"
    else
        patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" -p1 --fuzz=5 --dry-run < "$GGUF_PATCH_FILE"
        patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" -p1 --fuzz=5 < "$GGUF_PATCH_FILE"
    fi
elif git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" apply --reverse --check "$GGUF_PATCH_FILE" >/dev/null 2>&1; then
    echo "GGUF no-host-mmap patch is already applied"
else
    git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" apply --check "$GGUF_PATCH_FILE"
    git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" apply "$GGUF_PATCH_FILE"
fi

clone_at_commit \
    https://github.com/pollockjj/ComfyUI-MultiGPU.git \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MultiGPU" \
    "$MULTIGPU_COMMIT" \
    device_utils.py \
    model_management_mgpu.py \
    p2p_registry.py \
    clip_dynamic_load_list_guard.py \
    nodes.py \
    wrappers.py \
    distorch_2.py \
    checkpoint_multigpu.py
clone_at_commit \
    https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" \
    "$TURBO_COMMIT" \
    __init__.py

TURBO_PATCH_FILE="$REPO_ROOT/patches/comfyui-minimax-h3-turbo.patch"
if [[ "$SOURCE_FETCH" == "archive" ]]; then
    if patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" -p1 --fuzz=5 --reverse --dry-run < "$TURBO_PATCH_FILE" >/dev/null 2>&1; then
        echo "Turbo zero-strength patch is already applied"
    else
        patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" -p1 --fuzz=5 --dry-run < "$TURBO_PATCH_FILE"
        patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" -p1 --fuzz=5 < "$TURBO_PATCH_FILE"
    fi
elif git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" apply --reverse --check "$TURBO_PATCH_FILE" >/dev/null 2>&1; then
    echo "Turbo zero-strength patch is already applied"
else
    git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" apply --check "$TURBO_PATCH_FILE"
    git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" apply "$TURBO_PATCH_FILE"
fi

TP_TURBO_PATCH_FILE="$REPO_ROOT/patches/comfyui-minimax-h3-tp-turbo.patch"
if [[ "$SOURCE_FETCH" == "archive" ]]; then
    if patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" -p1 --reverse --dry-run < "$TP_TURBO_PATCH_FILE" >/dev/null 2>&1; then
        echo "Turbo persistent-TP delegation patch is already applied"
    else
        patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" -p1 --dry-run < "$TP_TURBO_PATCH_FILE"
        patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" -p1 < "$TP_TURBO_PATCH_FILE"
    fi
elif git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" apply --reverse --check "$TP_TURBO_PATCH_FILE" >/dev/null 2>&1; then
    echo "Turbo persistent-TP delegation patch is already applied"
else
    git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" apply --check "$TP_TURBO_PATCH_FILE"
    git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo" apply "$TP_TURBO_PATCH_FILE"
fi

clone_at_commit \
    https://github.com/nicolab28/ComfyUI-ClipProj.git \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj" \
    "$CLIPPROJ_COMMIT" \
    __init__.py \
    clipproj_nodes.py \
    clipproj_projection.py \
    clipproj_pinning.py \
    pyproject.toml

CLIPPROJ_PATCH_FILE="$REPO_ROOT/patches/comfyui-clipproj-v100-no-host-mmap.patch"
if [[ "${SOURCE_FETCH}" == "archive" ]]; then
    if patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj" -p1 --fuzz=5 --reverse --dry-run < "$CLIPPROJ_PATCH_FILE" >/dev/null 2>&1; then
        echo "ClipProj V100 no-host-mmap patch is already applied"
    else
        patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj" -p1 --fuzz=5 --dry-run < "$CLIPPROJ_PATCH_FILE"
        patch -d "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj" -p1 --fuzz=5 < "$CLIPPROJ_PATCH_FILE"
    fi
elif git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj" apply --reverse --check "$CLIPPROJ_PATCH_FILE" >/dev/null 2>&1; then
    echo "ClipProj V100 no-host-mmap patch is already applied"
else
    git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj" apply --check "$CLIPPROJ_PATCH_FILE"
    git -C "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj" apply "$CLIPPROJ_PATCH_FILE"
fi

PATCH_FILE="$REPO_ROOT/patches/comfyui-minimax-h3-v100-runtime.patch"
if [[ "$SOURCE_FETCH" == "archive" ]]; then
    if patch -d "$INSTALL_ROOT/ComfyUI" -p1 --reverse --dry-run < "$PATCH_FILE" >/dev/null 2>&1; then
        echo "FP16 RMSNorm patch is already applied"
    else
        patch -d "$INSTALL_ROOT/ComfyUI" -p1 --dry-run < "$PATCH_FILE"
        patch -d "$INSTALL_ROOT/ComfyUI" -p1 < "$PATCH_FILE"
    fi
elif git -C "$INSTALL_ROOT/ComfyUI" apply --unidiff-zero --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "FP16 RMSNorm patch is already applied"
else
    git -C "$INSTALL_ROOT/ComfyUI" apply --unidiff-zero --check "$PATCH_FILE"
    git -C "$INSTALL_ROOT/ComfyUI" apply --unidiff-zero "$PATCH_FILE"
fi

KEYFRAME_GEOMETRY_PATCH_FILE="$REPO_ROOT/patches/comfyui-minimax-h3-keyframe-geometry.patch"
if [[ "$SOURCE_FETCH" == "archive" ]]; then
    if patch -d "$INSTALL_ROOT/ComfyUI" -p1 --reverse --dry-run < "$KEYFRAME_GEOMETRY_PATCH_FILE" >/dev/null 2>&1; then
        echo "H3 first/last keyframe geometry patch is already applied"
    else
        patch -d "$INSTALL_ROOT/ComfyUI" -p1 --dry-run < "$KEYFRAME_GEOMETRY_PATCH_FILE"
        patch -d "$INSTALL_ROOT/ComfyUI" -p1 < "$KEYFRAME_GEOMETRY_PATCH_FILE"
    fi
elif git -C "$INSTALL_ROOT/ComfyUI" apply --reverse --check "$KEYFRAME_GEOMETRY_PATCH_FILE" >/dev/null 2>&1; then
    echo "H3 first/last keyframe geometry patch is already applied"
else
    git -C "$INSTALL_ROOT/ComfyUI" apply --check "$KEYFRAME_GEOMETRY_PATCH_FILE"
    git -C "$INSTALL_ROOT/ComfyUI" apply "$KEYFRAME_GEOMETRY_PATCH_FILE"
fi

AUDIO_COMPILE_PATCH_FILE="$REPO_ROOT/patches/comfyui-minimax-h3-audio-compile.patch"
if [[ "$SOURCE_FETCH" == "archive" ]]; then
    if patch -d "$INSTALL_ROOT/ComfyUI" -p1 --fuzz=5 --reverse --dry-run < "$AUDIO_COMPILE_PATCH_FILE" >/dev/null 2>&1; then
        echo "H3 audio compile patch is already applied"
    else
        patch -d "$INSTALL_ROOT/ComfyUI" -p1 --fuzz=5 --dry-run < "$AUDIO_COMPILE_PATCH_FILE"
        patch -d "$INSTALL_ROOT/ComfyUI" -p1 --fuzz=5 < "$AUDIO_COMPILE_PATCH_FILE"
    fi
elif git -C "$INSTALL_ROOT/ComfyUI" apply --reverse --check "$AUDIO_COMPILE_PATCH_FILE" >/dev/null 2>&1; then
    echo "H3 audio compile patch is already applied"
else
    git -C "$INSTALL_ROOT/ComfyUI" apply --check "$AUDIO_COMPILE_PATCH_FILE"
    git -C "$INSTALL_ROOT/ComfyUI" apply "$AUDIO_COMPILE_PATCH_FILE"
fi

sync_python_package DualV100

if [[ ! -x "$INSTALL_ROOT/.venv/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$INSTALL_ROOT/.venv"
fi

PYTHON="$INSTALL_ROOT/.venv/bin/python"
VENV_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$VENV_VERSION" != "$PYTHON_VERSION" ]]; then
    cat >&2 <<EOF
Existing virtual environment uses Python $VENV_VERSION, but the selected interpreter is Python $PYTHON_VERSION.
Use a new INSTALL_ROOT or recreate $INSTALL_ROOT/.venv with the selected interpreter.
EOF
    exit 1
fi
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install -r "$INSTALL_ROOT/ComfyUI/requirements.txt"
"$PYTHON" -m pip install -r "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt"

mkdir -p \
    "$MODEL_DIR/diffusion_models" \
    "$MODEL_DIR/text_encoders" \
    "$MODEL_DIR/clip_projections" \
    "$MODEL_DIR/vae" \
    "$MODEL_DIR/loras" \
    "$INSTALL_ROOT/ComfyUI/output"

"$PYTHON" -m py_compile \
    "$INSTALL_ROOT/ComfyUI/comfy/ldm/minimax/model.py" \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/DualV100/"*.py \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/NoHostMMap/"*.py \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj/clipproj_nodes.py" \
    "$INSTALL_ROOT/ComfyUI/custom_nodes/ComfyUI-ClipProj/clipproj_projection.py" \
    "$REPO_ROOT/scripts/audit_qwen32_q2_tp_layout.py" \
    "$REPO_ROOT/scripts/test_h3_async_vae.py" \
    "$REPO_ROOT/scripts/test_h3_async_vae_bridge.py"

"$PYTHON" - <<'PY'
import json
import platform
import torch

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}, indent=2))
PY

echo "Installed into: $INSTALL_ROOT"
echo "Models directory: $MODEL_DIR"
echo "Weights were not downloaded. Run scripts/download_h3_clipproj_models.sh for the 4B ClipProj profile."
