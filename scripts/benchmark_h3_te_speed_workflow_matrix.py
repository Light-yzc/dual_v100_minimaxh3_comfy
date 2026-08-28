#!/usr/bin/env python3
"""Run a real ComfyUI TE-Speed block calibration matrix.

The script writes small API workflows, then (only with ``--run``) submits
them sequentially to an already-running ComfyUI.  The resident TP runtime is
therefore reused between configurations.  It saves one full latent and one
latent per ``tail x mcs`` candidate, then computes video/audio metrics against
the full latent.  No hidden activation is written.

Prepare only:

  python scripts/benchmark_h3_te_speed_workflow_matrix.py \
    --base-workflow workflows/te-speed-tp-smoke-448x256-2step.json

Run against a resident service:

  python scripts/benchmark_h3_te_speed_workflow_matrix.py \
    --base-workflow workflows/te-speed-tp-smoke-448x256-2step.json \
    --steps 4 --run
"""

from __future__ import annotations

import argparse
import copy
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_h3_latents.py"
DEFAULT_LATENT_DIR = Path(
    "/home/regen/minimax-h3/ComfyUI/output/benchmarks/h3_tp_e2e/"
    "te_speed_block_matrix"
)


def request_json(url: str, payload=None, *, timeout: float = 300.0):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-workflow", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "results/te_speed_block_matrix_workflows",
    )
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="directory where H3_TP_RESULTS_DIR writes forward_*.json",
    )
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--tails", type=int, nargs="+", default=[12, 42])
    parser.add_argument("--mcs-values", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--control-values",
        type=float,
        nargs="+",
        default=[0.02, 0.03, 0.05, 0.12],
        help="normalized sigma-delta thresholds to sweep",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="optionally replace every BasicScheduler step count in the base",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--compare", action="store_true", help="compare existing latents")
    return parser.parse_args()


def find_node(workflow: dict[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (node_id, node)
        for node_id, node in workflow.items()
        if node.get("class_type") == class_type
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {class_type} node, found {len(matches)}"
        )
    return matches[0]


def configure_workflow(
    base: dict[str, Any],
    *,
    enabled: bool,
    tail_blocks: int,
    mcs: int,
    control_value: float,
    filename: Path,
    steps: int | None,
) -> dict[str, Any]:
    workflow = copy.deepcopy(base)
    _te_id, te_node = find_node(workflow, "TESpeedMiniMaxH3TP")
    inputs = te_node.setdefault("inputs", {})
    inputs.update(
        {
            "enabled": bool(enabled),
            "tail_blocks": int(tail_blocks),
            "mcs": int(mcs),
            "processing_control_value": float(control_value),
            "collect_block_stats": True,
        }
    )
    if steps is not None:
        scheduler_nodes = [
            node
            for node in workflow.values()
            if node.get("class_type") == "BasicScheduler"
        ]
        if not scheduler_nodes:
            raise ValueError("--steps requested but workflow has no BasicScheduler")
        for scheduler in scheduler_nodes:
            scheduler.setdefault("inputs", {})["steps"] = int(steps)
    _save_id, save_node = find_node(workflow, "SaveMiniMaxH3Latent")
    save_node.setdefault("inputs", {})["filename"] = str(filename)
    return workflow


def write_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = json.loads(args.base_workflow.read_text(encoding="utf-8"))
    # Validate before writing any generated files.
    find_node(base, "TESpeedMiniMaxH3TP")
    find_node(base, "SaveMiniMaxH3Latent")
    for tail in args.tails:
        if not 1 <= int(tail) <= 49:
            raise ValueError(f"tail must be in [1, 49], got {tail}")
    for mcs in args.mcs_values:
        if not 1 <= int(mcs) <= 10:
            raise ValueError(f"mcs must be in [1, 10], got {mcs}")
    for control in args.control_values:
        if float(control) <= 0.0:
            raise ValueError(f"control value must be positive, got {control}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.latent_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    full_latent = args.latent_dir / "full.pt"
    full_workflow = args.work_dir / "full.json"
    full_workflow.write_text(
        json.dumps(
            configure_workflow(
                base,
                enabled=False,
                tail_blocks=12,
                mcs=1,
                control_value=0.0,
                filename=full_latent,
                steps=args.steps,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    entries.append(
        {
            "label": "full",
            "enabled": False,
            "tail_blocks": 12,
            "boundary_block": 38,
            "mcs": 1,
            "control_value": 0.0,
            "workflow": str(full_workflow),
            "latent": str(full_latent),
        }
    )
    for tail in args.tails:
        for mcs in args.mcs_values:
            for control in args.control_values:
                control_label = format(float(control), ".6g").replace(".", "p")
                label = f"tail{int(tail)}_mcs{int(mcs)}_c{control_label}"
                workflow_path = args.work_dir / f"{label}.json"
                latent_path = args.latent_dir / f"{label}.pt"
                workflow_path.write_text(
                    json.dumps(
                        configure_workflow(
                            base,
                            enabled=True,
                            tail_blocks=int(tail),
                            mcs=int(mcs),
                            control_value=float(control),
                            filename=latent_path,
                            steps=args.steps,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                entries.append(
                    {
                        "label": label,
                        "enabled": True,
                        "tail_blocks": int(tail),
                        "boundary_block": 50 - int(tail),
                        "mcs": int(mcs),
                        "control_value": float(control),
                        "workflow": str(workflow_path),
                        "latent": str(latent_path),
                    }
                )
    manifest = {
        "base_workflow": str(args.base_workflow),
        "server": args.server,
        "steps_override": args.steps,
        "tails": [int(item) for item in args.tails],
        "mcs_values": [int(item) for item in args.mcs_values],
        "control_values": [float(item) for item in args.control_values],
        "resident_model_reused": True,
        "quality_gate_applied": False,
        "entries": entries,
    }
    (args.work_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return entries


def submit_and_wait(
    workflow: Path,
    *,
    server: str,
    timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = request_json(
        f"{server.rstrip('/')}/prompt",
        {"prompt": json.loads(workflow.read_text(encoding="utf-8"))},
        timeout=request_timeout,
    )
    prompt_id = result["prompt_id"]
    while True:
        elapsed = time.perf_counter() - started
        if elapsed > timeout:
            raise TimeoutError(f"{workflow} exceeded {timeout}s")
        try:
            history = request_json(
                f"{server.rstrip('/')}/history/{prompt_id}",
                timeout=request_timeout,
            )
        except (TimeoutError, socket.timeout, urllib.error.URLError):
            continue
        if prompt_id not in history:
            time.sleep(2.0)
            continue
        item = history[prompt_id]
        status = item.get("status", {})
        summary = {
            "prompt_id": prompt_id,
            "workflow": str(workflow),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "status": status,
            "output_nodes": sorted(item.get("outputs", {}).keys()),
        }
        if status.get("status_str") == "error" or status.get("completed") is False:
            raise RuntimeError(json.dumps(summary, ensure_ascii=False))
        return summary


def summarize_forward_files(paths: list[Path]) -> dict[str, Any]:
    """Reduce scalar TP reports for one submitted workflow."""

    steps = []
    for path in sorted(paths, key=lambda item: (item.stat().st_mtime_ns, item.name)):
        report = json.loads(path.read_text(encoding="utf-8"))
        rank0 = report["rank0"]
        rank1 = report["rank1"]
        te_speed = report["te_speed"]
        executed = int(rank0["blocks_executed"])
        steps.append(
            {
                "file": path.name,
                "step": te_speed.get("step"),
                "mode": te_speed.get("mode"),
                "sigma_delta": te_speed.get("sigma_delta"),
                "max_rank_ms": float(report["max_rank_total_ms"]),
                "executed_blocks": executed,
                "skipped_blocks": 50 - executed,
                "cache_bytes_rank0": int(rank0.get("te_speed_cache_bytes", 0)),
                "cache_bytes_rank1": int(rank1.get("te_speed_cache_bytes", 0)),
                "rank0_peak_mib": float(rank0.get("peak_allocated_mib", 0.0)),
                "rank1_peak_mib": float(rank1.get("peak_allocated_mib", 0.0)),
                "rank0_rss_mib": float(
                    rank0.get("process_memory", {}).get("rss_mib", 0.0)
                ),
                "rank1_rss_mib": float(
                    rank1.get("process_memory", {}).get("rss_mib", 0.0)
                ),
                "finite": bool(report.get("finite", False)),
            }
        )
    return {
        "steps": steps,
        "dit_total_ms": sum(item["max_rank_ms"] for item in steps),
        "executed_blocks": sum(item["executed_blocks"] for item in steps),
        "skipped_blocks": sum(item["skipped_blocks"] for item in steps),
        "cache_hits": sum(item["mode"] == "cache" for item in steps),
        "max_cache_bytes_rank0": max(
            (item["cache_bytes_rank0"] for item in steps), default=0
        ),
        "max_cache_bytes_rank1": max(
            (item["cache_bytes_rank1"] for item in steps), default=0
        ),
        "rank0_peak_mib": max(
            (item["rank0_peak_mib"] for item in steps), default=0.0
        ),
        "rank1_peak_mib": max(
            (item["rank1_peak_mib"] for item in steps), default=0.0
        ),
        "finite": all(item["finite"] for item in steps),
    }


def compare_latents(entries: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    full = Path(entries[0]["latent"])
    if not full.is_file():
        raise FileNotFoundError(full)
    comparisons = []
    for entry in entries[1:]:
        candidate = Path(entry["latent"])
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        output = work_dir / f"{entry['label']}_vs_full.json"
        subprocess.run(
            [sys.executable, str(COMPARE_SCRIPT), str(full), str(candidate), "--output", str(output)],
            check=True,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        entry["comparison"] = report
        entry_report = {"label": entry["label"], "comparison": report}
        comparisons.append(entry_report)
    (work_dir / "comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparisons


def main() -> None:
    args = parse_args()
    entries = write_matrix(args)
    run_summaries = []
    if args.run:
        for entry in entries:
            before = {path.resolve() for path in args.results_dir.glob("forward_*.json")}
            summary = submit_and_wait(
                Path(entry["workflow"]),
                server=args.server,
                timeout=args.timeout,
                request_timeout=args.request_timeout,
            )
            after = {path.resolve() for path in args.results_dir.glob("forward_*.json")}
            forward_files = sorted(
                after - before,
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
            if not forward_files:
                raise RuntimeError(
                    f"{entry['label']} completed without a new TP forward report"
                )
            entry["forward_files"] = [str(path) for path in forward_files]
            entry["metrics"] = summarize_forward_files(forward_files)
            entry["run"] = summary
            run_summaries.append(summary)
            (args.work_dir / "run_manifest.json").write_text(
                json.dumps(
                    {"entries": entries, "run_summaries": run_summaries},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.compare:
        comparisons = compare_latents(entries, args.work_dir)
    else:
        comparisons = []
    full_ms = None
    if entries and "metrics" in entries[0]:
        full_ms = float(entries[0]["metrics"]["dit_total_ms"])
    for entry in entries:
        metrics = entry.get("metrics")
        if full_ms is not None and metrics is not None:
            candidate_ms = float(metrics["dit_total_ms"])
            metrics["dit_speedup_vs_full"] = (
                full_ms / candidate_ms if candidate_ms > 0.0 else None
            )
    matrix_summary = {
        "base_workflow": str(args.base_workflow),
        "resident_model_reused": True,
        "results_dir": str(args.results_dir),
        "entries": entries,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "matrix_summary.json").write_text(
        json.dumps(matrix_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "work_dir": str(args.work_dir),
                "entries": len(entries),
                "ran": bool(args.run),
                "compared": len(comparisons),
                "resident_model_reused": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
