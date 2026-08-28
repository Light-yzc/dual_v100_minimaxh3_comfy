#!/usr/bin/env bash
set -euo pipefail

# Download the MiniMax-H3 set selected for two 16 GB V100s.
#
# This intentionally uses a V100-fit GGUF pair, not the official 21/27 GB
# int8 checkpoints: the DiT runs on GPU 0 and the Qwen encoder runs on GPU 1.
# Every artifact is downloaded to a .part file, SHA-256 checked, then renamed
# atomically, so a half-downloaded model is never offered to ComfyUI.

INSTALL_ROOT="${INSTALL_ROOT:-$HOME/minimax-h3}"
MODEL_DIR="${H3_MODEL_DIR:-/mnt/GALAX/minimax-h3/models}"
MIRROR_BASE="${H3_MIRROR:-https://hf-mirror.com}"

command -v wget >/dev/null || {
  echo "wget is required (install it with: sudo apt install wget)" >&2
  exit 1
}
command -v sha256sum >/dev/null || {
  echo "sha256sum is required" >&2
  exit 1
}

download_one() {
  local url="$1"
  local destination="$2"
  local expected_size="$3"
  local expected_sha256="$4"
  local partial="${destination}.part"
  local actual_sha256
  local actual_size

  mkdir -p "$(dirname "$destination")"

  if [[ -f "$destination" ]]; then
    actual_size="$(stat -c '%s' "$destination")"
    actual_sha256="$(sha256sum "$destination" | awk '{print $1}')"
    if [[ "$actual_size" == "$expected_size" && "$actual_sha256" == "$expected_sha256" ]]; then
      echo "Already verified: $(basename "$destination")"
      return 0
    fi

    # A previous interrupted transfer may have used the final filename. Keep it
    # as a resumable .part rather than allowing ComfyUI to see a corrupt weight.
    if [[ -e "$partial" ]]; then
      mv -f -- "$destination" "${destination}.unverified.$(date +%s)"
    else
      mv -- "$destination" "$partial"
    fi
  fi

  echo "Downloading: $(basename "$destination")"
  wget --continue --show-progress --progress=dot:giga \
    --timeout=30 --tries=20 --waitretry=5 \
    "$url" -O "$partial"

  actual_size="$(stat -c '%s' "$partial")"
  actual_sha256="$(sha256sum "$partial" | awk '{print $1}')"
  if [[ "$actual_size" != "$expected_size" || "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Verification failed for $(basename "$destination"). Keeping $partial for inspection/resume." >&2
    echo "Expected: size=$expected_size sha256=$expected_sha256" >&2
    echo "Actual:   size=$actual_size sha256=$actual_sha256" >&2
    return 1
  fi

  mv -f -- "$partial" "$destination"
  echo "Verified: $(basename "$destination")"
}

# Pinned revisions and hashes, current on 2026-08-17.
# DiT: molbal/MiniMax-H3-GGUF (maintainer of the installed ComfyUI-GGUF fork).
# Text encoder: the H3 Q2_K GGUF conversion matching this repository's workflow.
# VAE: official Comfy-Org repackaging. Turbo LoRA: its upstream author.
download_one \
  "$MIRROR_BASE/molbal/MiniMax-H3-GGUF/resolve/b45874371c61c49bf04096602f5527aac71b360b/minimax_h3_fl2va_pruned_fp8_Q4_0.gguf" \
  "$MODEL_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_Q4_0.gguf" \
  11377542880 \
  50891b806d6d700f4f20931791ca42a083dd9148609838268ccdc782bf899c1c

download_one \
  "$MIRROR_BASE/realrebelai/MiniMax-H3_GGUFs/resolve/daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7/qwen3vl-32B-MiniMax-H3-Q2_K.gguf" \
  "$MODEL_DIR/text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf" \
  8487968160 \
  5bbc11d0b3ef197c98df2ce8f05de8fbb8eb5917cd91c33d0b59f93759b34914

download_one \
  "$MIRROR_BASE/Comfy-Org/MiniMax-H3/resolve/cec22ac7545ee166df6af79fda42bd41558f8558/vae/minimax_h3_video_vae_fp16.safetensors" \
  "$MODEL_DIR/vae/minimax_h3_video_vae_fp16.safetensors" \
  5207808496 \
  7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522

download_one \
  "$MIRROR_BASE/Comfy-Org/MiniMax-H3/resolve/cec22ac7545ee166df6af79fda42bd41558f8558/vae/minimax_h3_audio_vae_fp32.safetensors" \
  "$MODEL_DIR/vae/minimax_h3_audio_vae_fp32.safetensors" \
  605254808 \
  8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48

download_one \
  "$MIRROR_BASE/larryvrh/MiniMax-H3-Turbo-Lora/resolve/43a74557ac3f6539db8e0f2a959d03feb7a81480/minimax_h3_turbo_v4_step600_ema.safetensors" \
  "$MODEL_DIR/loras/minimax_h3_turbo_v4_step600_ema.safetensors" \
  779849816 \
  5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3

echo "MiniMax-H3 V100 model set is complete and verified."
