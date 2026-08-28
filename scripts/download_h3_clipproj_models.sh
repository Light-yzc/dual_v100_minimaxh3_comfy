#!/usr/bin/env bash
set -euo pipefail

# Download the small-encoder H3 profile.  The default destination deliberately
# stays on the fast NVMe mount; override H3_MODEL_DIR only when that is
# intentional.  This script never opens a checkpoint through a model library.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/home/regen/minimax-h3}"
MODEL_DIR="${H3_MODEL_DIR:-/mnt/GALAX/minimax-h3/models}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

QWEN4B_REPO="${QWEN4B_REPO:-Qwen/Qwen3-VL-4B-Instruct-GGUF}"
QWEN4B_FILE="${QWEN4B_FILE:-Qwen3VL-4B-Instruct-Q4_K_M.gguf}"
QWEN4B_SIZE="${QWEN4B_SIZE:-2497281664}"
QWEN4B_SHA256="${QWEN4B_SHA256:-66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a}"

MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Qwen3VL-4B-Instruct-F16.gguf}"
MMPROJ_SIZE="${MMPROJ_SIZE:-836180256}"
MMPROJ_SHA256="${MMPROJ_SHA256:-256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331}"

PROJECTION_REPO="${PROJECTION_REPO:-NicoLab28/ClipProj-MiniMax-H3}"
PROJECTION_FILE="${PROJECTION_FILE:-mmh3-4b-ClipProj-v3.1.safetensors}"
PROJECTION_SIZE="${PROJECTION_SIZE:-26256128}"
PROJECTION_SHA256="${PROJECTION_SHA256:-0184e5c8d666a131962506d21949c2d8a8c6f33445b7b5e347e9a7e0a5baa819}"

QWEN_DIR="${H3_QWEN_Q4_DIR:-$(dirname -- "$MODEL_DIR")/experimental/qwen3vl_q4}"
PROJECTION_DIR="$MODEL_DIR/clip_projections"
mkdir -p "$QWEN_DIR" "$PROJECTION_DIR"

command -v curl >/dev/null || {
    echo "curl is required" >&2
    exit 1
}

download_file() {
    local repo="$1"
    local filename="$2"
    local destination="$3"
    local url="$HF_ENDPOINT/$repo/resolve/main/$filename"

    if [[ -f "$destination" ]]; then
        local existing_size
        existing_size="$(stat -c '%s' "$destination")"
        if [[ "$existing_size" == "$4" ]]; then
            echo "Already complete: $destination"
            return
        fi
        echo "Resuming partial file: $destination ($existing_size / $4 bytes)"
    fi

    if command -v aria2c >/dev/null; then
        aria2c \
            --continue=true \
            --allow-overwrite=false \
            --auto-file-renaming=false \
            --file-allocation=none \
            --max-connection-per-server="${H3_DOWNLOAD_CONNECTIONS:-8}" \
            --split="${H3_DOWNLOAD_CONNECTIONS:-8}" \
            --min-split-size=16M \
            --summary-interval=10 \
            --console-log-level=notice \
            --dir="$(dirname -- "$destination")" \
            --out="$(basename -- "$destination")" \
            "$url"
    else
        curl -4 -L --fail --retry 5 --retry-delay 3 --continue-at - \
            --output "$destination" "$url"
    fi
}

verify_file() {
    local path="$1"
    local expected_size="$2"
    local expected_sha="$3"
    local actual_size
    actual_size="$(stat -c '%s' "$path")"
    [[ "$actual_size" == "$expected_size" ]] || {
        echo "Size mismatch for $path: got $actual_size, expected $expected_size" >&2
        exit 2
    }
    if [[ -n "$expected_sha" ]]; then
        local actual_sha
        actual_sha="$(sha256sum "$path" | awk '{print $1}')"
        [[ "$actual_sha" == "$expected_sha" ]] || {
            echo "SHA256 mismatch for $path: got $actual_sha, expected $expected_sha" >&2
            exit 2
        }
    fi
    echo "Verified: $path ($actual_size bytes)"
}

python_for_header() {
    if [[ -x "$INSTALL_ROOT/.venv/bin/python" ]]; then
        printf '%s\n' "$INSTALL_ROOT/.venv/bin/python"
    else
        command -v python3
    fi
}

inspect_qwen_gguf() {
    local encoder_path="$1"
    local mmproj_path="$2"
    local python_bin
    python_bin="$(python_for_header)"
    PYTHONPATH="$REPO_ROOT:$INSTALL_ROOT/ComfyUI" "$python_bin" - "$encoder_path" "$mmproj_path" <<'PY'
import sys

from custom_nodes.NoHostMMap.gguf_reader import NoMmapGGUFReader

encoder_path, mmproj_path = sys.argv[1:]


def architecture(reader):
    field = reader.get_field("general.architecture")
    if field is None:
        return None
    return bytes(field.parts[field.data[-1]]).decode("utf-8")


encoder = NoMmapGGUFReader(encoder_path)
if architecture(encoder) != "qwen3vl":
    raise SystemExit(f"not a Qwen3-VL GGUF: {encoder_path}")
token = next((item for item in encoder.tensors if item.name == "token_embd.weight"), None)
shape = tuple(int(value) for value in reversed(token.shape)) if token is not None else ()
blocks = {
    int(item.name.split(".", 2)[1])
    for item in encoder.tensors
    if item.name.startswith("blk.") and item.name.split(".", 2)[1].isdigit()
}
if shape != (151936, 2560) or blocks != set(range(36)):
    raise SystemExit(
        f"unexpected Qwen3-VL-4B geometry: embedding={shape}, blocks={len(blocks)}"
    )

mmproj = NoMmapGGUFReader(mmproj_path)
names = {item.name for item in mmproj.tensors}
required = {"v.patch_embd.weight", "mm.2.weight"}
if architecture(mmproj) != "clip" or not required.issubset(names):
    raise SystemExit(f"not the matching Qwen3-VL vision/mmproj GGUF: {mmproj_path}")
print("Header OK: Qwen3-VL-4B Q4_K_M + FP16 vision/mmproj, 36 language blocks")
PY
}

inspect_projection_header() {
    local path="$1"
    local python_bin
    python_bin="$(python_for_header)"
    PYTHONPATH="$REPO_ROOT" "$python_bin" - "$path" <<'PY'
import json
import struct
import sys

path = sys.argv[1]
with open(path, "rb", buffering=0) as handle:
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise SystemExit(f"incomplete safetensors header: {path}")
    header_size = struct.unpack("<Q", prefix)[0]
    header = json.loads(handle.read(header_size).decode("utf-8"))

input_stats = header.get("mean_in")
output_stats = header.get("mean_out")
matrix = header.get("W")
if (
    not input_stats
    or input_stats.get("shape") != [2560]
    or not output_stats
    or output_stats.get("shape") != [5120]
    or not matrix
    or matrix.get("shape") != [2560, 5120]
):
    raise SystemExit(f"not an H3 4B->5120 ridge projection: {path}")
print("Header OK: ClipProj ridge 2560 -> 5120")
PY
}

encoder_path="$QWEN_DIR/$QWEN4B_FILE"
mmproj_path="$QWEN_DIR/$MMPROJ_FILE"
projection_path="$PROJECTION_DIR/$PROJECTION_FILE"

download_file "$QWEN4B_REPO" "$QWEN4B_FILE" "$encoder_path" "$QWEN4B_SIZE"
verify_file "$encoder_path" "$QWEN4B_SIZE" "$QWEN4B_SHA256"

download_file "$QWEN4B_REPO" "$MMPROJ_FILE" "$mmproj_path" "$MMPROJ_SIZE"
verify_file "$mmproj_path" "$MMPROJ_SIZE" "$MMPROJ_SHA256"
inspect_qwen_gguf "$encoder_path" "$mmproj_path"

download_file "$PROJECTION_REPO" "$PROJECTION_FILE" "$projection_path" "$PROJECTION_SIZE"
verify_file "$projection_path" "$PROJECTION_SIZE" "$PROJECTION_SHA256"
inspect_projection_header "$projection_path"

echo "H3 ClipProj 4B Q4 profile is ready: $QWEN_DIR"
echo "Ridge projection is ready: $projection_path"
echo "Use workflows/clipproj-4b-q4-tp-turbo-smoke-448x256-1step.json for the guarded smoke test."
