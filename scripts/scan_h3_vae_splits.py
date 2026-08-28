"""Scan H3_VAE_DIT_SPLIT x H3_VAE_DECODE_SPLIT against a real workflow.

Why a scan and not a formula: the only numbers that decide whether a layout
fits are whole-card, and they are not the sum of two torch allocators.  GPU0
carries the rank-0 DiT shard, its activations, the CUDA context, the NCCL
buffers and the cuBLAS workspace; GPU1 carries an entirely separate rank-1
process.  Back-of-envelope models of that were wrong by ~4.8 GiB when checked
against nvidia-smi, so each combination is measured end to end.

Each combination restarts the service, because the split is resolved when the
VAE loads.  One request per combination, whole-card telemetry sampled for the
whole run, plus the per-step DiT timings the runtime already prints.

Usage:
    scripts/scan_h3_vae_splits.py \
        --workflow workflows/H3-V100-12-int8-ref2v-720p-243f-4step.json \
        --dit-splits 2,8,18 --decode-splits 12,24 \
        --output results/vae_split_scan_720p.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = os.environ.get("H3_PYTHON", "/home/regen/minimax-h3/.venv/bin/python")
UNIT = os.environ.get("H3_SYSTEMD_UNIT", "minimax-h3-comfy")
PORT = os.environ.get("COMFY_PORT", "8188")
TOTAL_MIB = 16384


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def service_active() -> bool:
    return run(["systemctl", "--user", "is-active", "--quiet",
                f"{UNIT}.service"]).returncode == 0


def wait_ready(timeout: int = 240) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = run(["curl", "-s", "-m", "2",
                     f"http://127.0.0.1:{PORT}/system_stats"])
        if probe.returncode == 0 and probe.stdout.strip():
            return True
        time.sleep 
    return False


def restart(env_overrides: dict[str, str]) -> bool:
    run(["systemctl", "--user", "stop", f"{UNIT}.service"])
    time.sleep(4)
    env = os.environ.copy()
    env.update(env_overrides)
    # start_comfyui.sh owns the defaults; overrides ride in through the
    # environment and are forwarded by the isolated launcher's pass-through.
    proc = run([str(REPO / "scripts" / "start_comfyui.sh"), "start"], env=env)
    if proc.returncode != 0:
        print(f"    启动失败: {proc.stderr.strip()[:300]}", flush=True)
        return False
    return wait_ready()


def journal_since(mark: str) -> str:
    return run(["journalctl", "--user", "-u", f"{UNIT}.service",
                "--since", mark, "--no-pager", "-o", "cat"]).stdout


def parse_forwards(text: str) -> list[dict]:
    out = []
    for m in re.finditer(
        r"forward#(\d+)\s+S=(\d+)\s+([\d.]+)s\s+NCCL=([\d.]+)s\s+"
        r"peak=\((\d+),(\d+)\)", text
    ):
        out.append({
            "index": int(m.group(1)), "sequence": int(m.group(2)),
            "seconds": float(m.group(3)), "nccl_seconds": float(m.group(4)),
            "peak_rank0_mib": int(m.group(5)), "peak_rank1_mib": int(m.group(6)),
        })
    return out


def parse_layout_moves(text: str) -> list[dict]:
    out = []
    for m in re.finditer(
        r"\[H3 VAE MP\] (decode|sampling): layout (\d+)/(\d+) "
        r"\((\d+) blocks over NVLink in ([\d.]+)s\)", text
    ):
        out.append({"reason": m.group(1), "split": int(m.group(2)),
                    "blocks": int(m.group(4)), "seconds": float(m.group(5))})
    return out


def csv_peaks(path: Path) -> dict:
    peaks: dict[str, dict] = {}
    if not path.is_file():
        return peaks
    with path.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 8 or not row[0].strip().isdigit():
                continue
            dev = f"cuda:{row[0].strip()}"
            entry = peaks.setdefault(dev, {"peak_mib": 0, "max_temp": 0,
                                           "min_sm": 0, "max_sm": 0})
            try:
                used, temp, sm = float(row[2]), float(row[5]), float(row[6])
            except ValueError:
                continue
            entry["peak_mib"] = max(entry["peak_mib"], used)
            entry["max_temp"] = max(entry["max_temp"], temp)
            entry["max_sm"] = max(entry["max_sm"], sm)
            entry["min_sm"] = sm if entry["min_sm"] == 0 else min(entry["min_sm"], sm)
    for entry in peaks.values():
        entry["free_mib"] = TOTAL_MIB - entry["peak_mib"]
    return peaks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--dit-splits", default="2,8,18")
    ap.add_argument("--decode-splits", default="12,24")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    wf = Path(args.workflow)
    if not wf.is_file():
        raise SystemExit(f"工作流不存在: {wf}")

    dit_splits = [int(x) for x in args.dit_splits.split(",") if x.strip()]
    decode_splits = [int(x) for x in args.decode_splits.split(",") if x.strip()]

    results = []
    for dit in dit_splits:
        for dec in decode_splits:
            if dec < dit:
                continue
            case = f"dit{dit}_dec{dec}"
            print(f"\n=== dit_split={dit} decode_split={dec} ===", flush=True)
            mark = time.strftime("%Y-%m-%d %H:%M:%S")
            if not restart({"H3_VAE_DIT_SPLIT": str(dit),
                            "H3_VAE_DECODE_SPLIT": str(dec)}):
                results.append({"dit_split": dit, "decode_split": dec,
                                "error": "服务启动失败"})
                continue

            csv_path = REPO / "results" / f"e2e_smi_{case}.csv"
            started = time.time()
            proc = run([
                str(REPO / "scripts" / "sample_gpu_during_run.sh"), case, "--",
                PYTHON, str(REPO / "scripts" / "submit_workflow.py"), str(wf),
                "--wait", "--timeout", str(args.timeout),
                "--output", str(REPO / "results" / f"{case}.json"),
            ])
            elapsed = time.time() - started
            text = journal_since(mark)
            forwards = parse_forwards(text)
            entry = {
                "dit_split": dit, "decode_split": dec,
                "wall_seconds": round(elapsed, 1),
                "submit_ok": proc.returncode == 0,
                "oom": "OutOfMemoryError" in text,
                "forwards": forwards,
                "layout_moves": parse_layout_moves(text),
                "whole_card": csv_peaks(csv_path),
            }
            if forwards:
                steady = forwards[1:] or forwards
                entry["dit_step_seconds"] = round(
                    sum(f["seconds"] for f in steady) / len(steady), 3)
                entry["dit_nccl_seconds"] = round(
                    sum(f["nccl_seconds"] for f in steady) / len(steady), 3)
                entry["sequence"] = forwards[0]["sequence"]
            wc = entry["whole_card"]
            if wc:
                entry["min_free_mib"] = round(
                    min(v["free_mib"] for v in wc.values()), 0)
            status = "OOM" if entry["oom"] else ("OK" if entry["submit_ok"] else "FAIL")
            print(f"    {status}  wall={entry['wall_seconds']}s  "
                  f"step={entry.get('dit_step_seconds', '-')}s  "
                  f"min_free={entry.get('min_free_mib', '-')} MiB", flush=True)
            results.append(entry)

    print("\n" + "=" * 82, flush=True)
    print(f"{'dit':>4} {'dec':>4} {'状态':>6} {'wall_s':>8} {'step_s':>8} "
          f"{'nccl_s':>7} {'GPU0峰':>8} {'GPU1峰':>8} {'最小余量':>9}", flush=True)
    print("-" * 82, flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['dit_split']:>4} {r['decode_split']:>4} {'启动失败':>6}", flush=True)
            continue
        wc = r.get("whole_card", {})
        g0 = wc.get("cuda:0", {}).get("peak_mib", 0)
        g1 = wc.get("cuda:1", {}).get("peak_mib", 0)
        st = "OOM" if r["oom"] else ("OK" if r["submit_ok"] else "FAIL")
        print(f"{r['dit_split']:>4} {r['decode_split']:>4} {st:>6} "
              f"{r['wall_seconds']:>8.1f} {r.get('dit_step_seconds', 0):>8.3f} "
              f"{r.get('dit_nccl_seconds', 0):>7.3f} {g0:>8.0f} {g1:>8.0f} "
              f"{r.get('min_free_mib', 0):>9.0f}", flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"workflow": str(wf), "cases": results}, indent=2, ensure_ascii=False) + "\n")
        print(f"\n已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
