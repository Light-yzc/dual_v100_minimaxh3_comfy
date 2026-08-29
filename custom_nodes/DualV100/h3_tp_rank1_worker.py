#!/usr/bin/env python3
"""Long-lived NCCL rank 1 process for the ComfyUI H3 TP runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


THIS_DIR = Path(__file__).resolve().parent
COMFY_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(COMFY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--weight-format",
        default="auto",
        choices=(
            "auto",
            "q4",
            "q4_0",
            "gguf",
            "int8",
            "int8_convrot",
            "convrot",
            "safetensors",
        ),
    )
    parser.add_argument("--lora", required=True)
    parser.add_argument("--egrid", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--staging-mib", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--qwen-model", default="")
    parser.add_argument("--qwen-staging-mib", type=int, default=4)
    parser.add_argument("--qwen-residency", choices=("evict", "partial", "full"), default="evict")
    parser.add_argument("--qwen-keep-layers", type=int, default=0)
    parser.add_argument("--qwen-cache-dequantized", action="store_true")
    return parser.parse_args()


def emit(kind: str, payload) -> None:
    print(
        "H3TP:" + json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")

    # Import only after selecting cuda:1.  This file is executed directly, so
    # DualV100's ComfyUI node-registration __init__ is deliberately bypassed.
    import h3_q4_cache
    import h3_tp_backbone as tp
    qwen32 = None

    dist.init_process_group(
        "nccl",
        init_method=args.init_method,
        rank=1,
        world_size=2,
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=device,
    )

    def progress(stage: str, current: int, total: int) -> None:
        if current == 1 or current == total or current % 10 == 0:
            print(
                f"[H3TP rank1] loading {stage} {current}/{total}",
                file=sys.stderr,
                flush=True,
            )

    backbone = None
    if args.qwen_model:
        import h3_qwen32_q2_tp as qwen32

    qwen_backbone = None

    def tree_broadcast(spec):
        if spec is None:
            return None
        kind = spec.get("kind")
        if kind == "tensor":
            dtype = getattr(torch, spec["dtype"])
            value = torch.empty(tuple(int(v) for v in spec["shape"]), device=device, dtype=dtype)
            dist.broadcast(value, src=0)
            return value
        values = [tree_broadcast(item) for item in spec.get("items", [])]
        return tuple(values) if kind == "tuple" else values

    def ensure_qwen():
        nonlocal qwen_backbone
        if not args.qwen_model:
            raise RuntimeError("Qwen32 Q2 command received without --qwen-model")
        if qwen_backbone is None:
            qwen_backbone = qwen32.Qwen32Q2TPBackbone(
                args.qwen_model,
                rank=1,
                world_size=2,
                device=device,
                # Keep rank 1 bit-for-bit aligned with rank 0 and ComfyUI's
                # FP32 Qwen conditioning path.  Do not silently downgrade
                # one rank to FP16 when a request is large.
                dtype=torch.float32,
                staging_mib=args.qwen_staging_mib,
                residency=args.qwen_residency,
                keep_layers=args.qwen_keep_layers,
                cache_dequantized=args.qwen_cache_dequantized,
            )
        return qwen_backbone
    te_cache = tp.H3TPResidualCache("cpu")
    group_cache = h3_q4_cache.GroupResidualCache("cpu")
    emit("ready", {"rank": 1, "device": str(device), "h3_ready": False})

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        command = json.loads(raw_line)
        name = command.get("cmd")
        if name == "shutdown":
            emit("shutdown", {"ok": True})
            break
        if name == "h3_init":
            if backbone is None:
                backbone = tp.H3TPBackbone(
                    rank=1,
                    device=device,
                    model_path=args.model,
                    weight_format=args.weight_format,
                    lora_path=args.lora,
                    egrid_path=args.egrid,
                    lora_strength=args.strength,
                    staging_bytes=args.staging_mib << 20,
                    chunk_rows=args.chunk_rows,
                    progress=progress,
                )
            emit("h3_ready", backbone.load_stats)
            continue
        if name == "release_cache":
            # Return this rank's free allocator blocks to the driver without
            # unloading the DiT shard.  After a long forward the caching
            # allocator holds several GiB of empty segments that the driver
            # still counts as used, so a later Qwen dequantisation on this card
            # fails even though the shard itself only needs ~7.5 GiB.  The
            # shard stays resident: reloading it costs ~34 s.
            #
            # Deliberately not collective: the caller may invoke it at any
            # point between forwards, and pairing it with a barrier would make
            # a routine cleanup able to deadlock the pipeline.
            before_reserved = torch.cuda.memory_reserved(device)
            before_allocated = torch.cuda.memory_allocated(device)
            # Drop this request's leftovers first.  ``empty_cache`` only returns
            # *free* segments, so a retained snapshot would keep its segment
            # allocated and the release would under-deliver by ~800 MiB at 720p.
            transient = (
                backbone.release_transient_state() if backbone is not None else None
            )
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            emit(
                "release_cache",
                {
                    "rank": 1,
                    "reserved_mib_before": before_reserved / tp.MIB,
                    "reserved_mib_after": torch.cuda.memory_reserved(device) / tp.MIB,
                    "allocated_mib": before_allocated / tp.MIB,
                    "allocated_mib_after": torch.cuda.memory_allocated(device) / tp.MIB,
                    "transient": transient,
                    "h3_resident": backbone is not None,
                },
            )
            continue
        if name == "qwen_forward":
            qwen = ensure_qwen()
            hidden = torch.empty(
                tuple(int(v) for v in command["hidden_shape"]),
                device=device,
                dtype=getattr(torch, command.get("hidden_dtype", "float32")),
            )
            dist.broadcast(hidden, src=0)
            tree = command.get("tree", {})
            attention_mask = tree_broadcast(tree.get("attention_mask"))
            freqs_cis = tree_broadcast(tree.get("freqs_cis"))
            deepstack_embeds = tree_broadcast(tree.get("deepstack_embeds"))
            visual_pos_masks = tree_broadcast(tree.get("visual_pos_masks"))
            output = qwen.forward_hidden(
                hidden,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                gather=lambda value, **kw: qwen32.all_gather_output_rows(
                    value, rank=1, world_size=2, label=kw.get("label")
                ),
                deepstack_embeds=deepstack_embeds,
                visual_pos_masks=visual_pos_masks,
            )
            # Both ranks must complete the matching barrier before either can
            # report a local finite-check failure.  Raising beforehand leaves
            # rank 0 blocked forever in its corresponding barrier.
            dist.barrier()
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("Qwen32 Q2 rank1 output produced NaN/Inf")
            emit("qwen_forward", qwen.stats())
            del hidden, output
            continue
        if name == "qwen_clear":
            if qwen_backbone is not None:
                qwen_backbone.clear()
            te_cache.clear()
            group_cache.clear()
            dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            emit("qwen_clear", {"rank": 1, "stats": None if qwen_backbone is None else qwen_backbone.stats()})
            continue
        if name == "qwen_trim":
            qwen = ensure_qwen()
            keep_layers = command.get("keep_layers", [])
            qwen.trim(keep_layers)
            dist.barrier()
            emit("qwen_trim", qwen.stats())
            continue
        if name == "qwen_stats":
            emit("qwen_stats", None if qwen_backbone is None else qwen_backbone.stats())
            continue
        if name != "forward":
            raise ValueError(f"unsupported H3 TP rank1 command: {name!r}")
        if backbone is None:
            raise RuntimeError("H3 forward received before h3_init")

        cache_device = str(command.get("cache_device", "cpu")).lower()
        h3_q4_cache.normalize_q4_format(command.get("cache_format"))
        is_group = command.get("mode") == "group"
        if is_group:
            group_config = command.get("group_cache") or {}
            if bool(group_config.get("clear_cache", False)):
                group_cache.clear()
        else:
            if group_cache.entries:
                group_cache.clear()
            if te_cache.set_policy(cache_device) or bool(command.get("clear_cache", False)):
                te_cache.clear()

        residual, t_emb, rope = tp.allocate_and_receive_rank1(
            command["residual_shape"],
            command["t_emb_shape"],
            command["rope_shape"],
            device,
        )
        residual, metrics = backbone.forward(
            residual,
            t_emb,
            command["segments"],
            rope,
            profile=bool(command.get("profile", False)),
            stage_profile=bool(command.get("stage_profile", False)),
            start_block=int(command.get("start_block", 0)),
            end_block=int(command.get("end_block", tp.LAYERS)),
            snapshot_at=command.get("snapshot_at"),
            collect_block_stats=bool(command.get("collect_block_stats", False)),
            stat_ranges=command.get("stat_ranges"),
            reset_block_stats=bool(command.get("reset_block_stats", False)),
            group_cache=group_cache,
            group_config=command.get("group_cache"),
        )
        tail_residual_stats = None
        if command.get("mode") == "group":
            backbone.clear_snapshot()
        elif command.get("mode", "full") == "full":
            if bool(command.get("capture_cache", False)) or bool(
                command.get("collect_anchor_stats", False)
            ):
                tail_residual_stats = te_cache.store(
                    residual,
                    backbone.take_snapshot(),
                    stat_ranges=command.get("stat_ranges"),
                    collect_stats=bool(command.get("collect_block_stats", False)),
                    retain=bool(command.get("capture_cache", False)),
                )
            else:
                backbone.clear_snapshot()
        elif command.get("mode") == "cache":
            te_cache.add_to(
                residual,
                measure=bool(command.get("collect_block_stats", False)),
            )
        else:
            raise ValueError(f"unsupported H3 TP TE-Speed mode: {command.get('mode')!r}")
        output_report = tp.tensor_scalar_stats(
            residual,
            command.get("stat_ranges")
            if bool(command.get("collect_block_stats", False))
            else None,
        )
        output_stats = output_report["overall"]
        metrics["output_rms"] = output_stats["rms"]
        metrics["output_max_abs"] = output_stats["max_abs"]
        metrics["finite"] = output_stats["finite"]
        metrics["process_memory"] = tp.process_memory_stats()
        if bool(command.get("collect_block_stats", False)):
            block_stats = metrics.get("block_stats") or {}
            if not is_group:
                block_stats["tail_residual"] = (
                    tail_residual_stats
                    if tail_residual_stats is not None
                    else te_cache.anchor_stats
                )
                block_stats["cache_operation"] = dict(te_cache.last_operation)
            block_stats["final_output_after_cache_add"] = output_report
            metrics["block_stats"] = block_stats
        metrics["te_speed_mode"] = command.get("mode", "full")
        metrics["te_speed_block_range"] = [
            int(command.get("start_block", 0)),
            int(command.get("end_block", tp.LAYERS)),
        ]
        metrics["te_speed_cache_bytes"] = te_cache.bytes
        metrics["te_speed_cache_device"] = te_cache.policy
        metrics["te_speed_generation_id"] = command.get("generation_id")
        metrics["te_speed_step"] = command.get("step")
        metrics["te_speed_sigma_raw"] = command.get("sigma_raw")
        metrics["te_speed_sigma_normalized"] = command.get("sigma_normalized")
        metrics["te_speed_sigma_delta"] = command.get("sigma_delta")
        metrics["te_speed_boundary_block"] = command.get("boundary_block")
        metrics["te_speed_tail_blocks"] = command.get("tail_blocks")
        metrics["group_cache_mode"] = is_group
        metrics["group_cache_bytes"] = group_cache.bytes
        if is_group:
            metrics["group_cache"] = metrics.get("group_cache") or {
                "format": h3_q4_cache.Q4_FORMAT,
                "cache": group_cache.summary(),
            }
        # Drop this rank's activations *before* reporting memory.  rank1 never
        # returns its residual to rank0 (only metrics travel back), so once the
        # stats above are computed the FP32 hidden stream is pure garbage: at
        # S=68261 that is ~800 MiB of the number this report used to attribute
        # to steady-state retention.  Whatever the caches genuinely hold is
        # still counted, because ``te_cache``/``group_cache`` keep their own
        # references and are unaffected by dropping these local names.
        #
        # Only the local names are dropped here.  ``release_transient_state``
        # deliberately does *not* run per forward: it would clear the
        # cross-step modulation-row cache every denoise step and pin
        # ``modulation_rows_cached`` to false.  Snapshot/cache teardown belongs
        # to the post-sample ``release_cache`` command instead.
        del residual, t_emb, rope
        metrics["allocated_mib"] = torch.cuda.memory_allocated(device) / tp.MIB
        metrics["reserved_mib"] = torch.cuda.memory_reserved(device) / tp.MIB
        metrics["peak_allocated_mib"] = max(
            float(metrics["peak_allocated_mib"]),
            torch.cuda.max_memory_allocated(device) / tp.MIB,
        )
        if not metrics["finite"]:
            raise RuntimeError("H3 TP rank1 TE-Speed output produced NaN/Inf")
        emit("forward", metrics)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        emit(
            "error",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
