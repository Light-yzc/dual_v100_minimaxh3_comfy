# Repository Guidelines

## Project Structure & Module Organization

- `custom_nodes/DualV100/` contains the production ComfyUI nodes, persistent two-rank NCCL runtime, tensor-parallel backbone, VAE/Qwen model-parallel paths, and V100-specific kernels.
- `custom_nodes/NoHostMMap/` provides header-only GGUF and safetensors readers. Preserve its bounded-memory behavior.
- `scripts/` contains installation, service management, smoke tests, audits, and benchmarks. Python checks generally use `benchmark_*.py`, `test_*.py`, or `compare_*.py`; shell smoke tests use `smoke_*.sh`.
- `workflows/` stores ComfyUI API/UI JSON workflows. Names should identify the route, resolution, frame count, and step count.
- `docs/` records measured baselines and design decisions; `patches/` holds pinned upstream compatibility patches. Treat `results/` as generated output.

Edit this repository first. Do not treat `/home/regen/minimax-h3/ComfyUI/custom_nodes` as source; `scripts/setup_ubuntu.sh` performs deployment synchronization.

## Build, Test, and Development Commands

```bash
INSTALL_ROOT=$HOME/minimax-h3 ./scripts/setup_ubuntu.sh
./scripts/start_comfyui.sh start
./scripts/start_comfyui.sh logs
./scripts/start_comfyui.sh stop
```

The setup script installs pinned dependencies, applies patches, copies custom nodes, and runs syntax checks. Service commands run ComfyUI under the repository's guarded configuration.

```bash
/home/regen/minimax-h3/.venv/bin/python -m py_compile custom_nodes/DualV100/*.py custom_nodes/NoHostMMap/*.py
INSTALL_ROOT=$HOME/minimax-h3 ./scripts/benchmark_h3_tp_comm.sh
./scripts/smoke_clipproj_v100.sh
```

Run syntax checks for every Python change. Use the communication gate for NCCL/runtime changes and a matching smoke or benchmark script for behavioral changes.

## Coding Style & Naming Conventions

Use four-space Python indentation, module docstrings, type hints for public interfaces, and `snake_case` functions/files. Classes use `PascalCase`; environment variables use `H3_UPPER_SNAKE_CASE`. Shell scripts should use `set -euo pipefail`. Keep comments focused on non-obvious memory, numerical, or synchronization constraints.

Never reintroduce full model mmap, unbounded host copies, silent numerical fallback, or unmatched collective order. New experimental paths must be opt-in and fail closed.

## Testing Guidelines

There is no single unit-test suite; validation is hardware- and workflow-driven. Record exact commands, model revisions, output metrics, GPU allocated/reserved peaks, host RSS, and mmap status. Compare target-size end-to-end workflows, not only microbenchmarks. Do not commit model files, caches, generated videos, or large benchmark outputs.

## Commit & Pull Request Guidelines

Git history is currently too small to establish a convention. Use concise imperative subjects, optionally scoped, such as `runtime: add rank1 cache clear`. Pull requests should explain the route changed, list verification commands and hardware/software versions, link relevant docs/issues, and include result JSON plus screenshots or MP4 samples for visible workflow changes.
