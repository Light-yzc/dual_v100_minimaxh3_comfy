#!/usr/bin/env python3
"""CPU-only contract tests for the decoupled Qwen32 layer-MP backend."""

from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DUAL = ROOT / "custom_nodes" / "DualV100"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DUAL) not in sys.path:
    sys.path.insert(0, str(DUAL))


def _load_modules():
    package_name = "custom_nodes.DualV100"
    package = types.ModuleType(package_name)
    package.__path__ = [str(DUAL)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    qwen = importlib.import_module(f"{package_name}.h3_qwen32_q2_tp")
    mp = importlib.import_module(f"{package_name}.h3_qwen32_q2_mp")
    runtime = importlib.import_module(f"{package_name}.h3_tp_runtime")
    return qwen, mp, runtime


QWEN, MP, SHARED_RUNTIME = _load_modules()


def _synthetic_layout(layer_count: int = 6):
    """Build a tiny all-F32 layout without touching a model file."""

    qtype = QWEN.GGMLQuantizationType.F32
    tensors = []
    language_layers = {}
    offset = 0
    for layer in range(layer_count):
        entries = []
        # Shapes need not be Qwen-sized for planner tests; all geometry is
        # still validated by TensorSpec and uses exact F32 row strides.
        matrices = {
            "q_proj": (8, 4),
            "k_proj": (4, 4),
            "v_proj": (4, 4),
            "o_proj": (4, 8),
            "gate_proj": (12, 4),
            "up_proj": (12, 4),
            "down_proj": (4, 12),
        }
        for role, shape in matrices.items():
            size = shape[0] * shape[1] * 4
            spec = QWEN.TensorSpec(
                name=f"model.layers.{layer}.self_attn.{role}.weight"
                if role in {"q_proj", "k_proj", "v_proj", "o_proj"}
                else f"model.layers.{layer}.mlp.{role}.weight",
                qtype=qtype,
                shape=shape,
                data_offset=offset,
                n_bytes=size,
                row_bytes=shape[1] * 4,
                block_elements=1,
                block_bytes=4,
                gguf_shape=tuple(reversed(shape)),
            )
            offset += size
            entries.append(spec)
            tensors.append(spec)
        for role, shape in {
            "input_layernorm": (4,),
            "post_attention_layernorm": (4,),
            "q_norm": (1,),
            "k_norm": (1,),
        }.items():
            size = shape[0] * 4
            spec = QWEN.TensorSpec(
                name=(
                    f"model.layers.{layer}.self_attn.{role}.weight"
                    if role in {"q_norm", "k_norm"}
                    else f"model.layers.{layer}.{role}.weight"
                ),
                qtype=qtype,
                shape=shape,
                data_offset=offset,
                n_bytes=size,
                row_bytes=size,
                block_elements=1,
                block_bytes=4,
                gguf_shape=tuple(reversed(shape)),
            )
            offset += size
            entries.append(spec)
            tensors.append(spec)
        language_layers[layer] = tuple(entries)
    return QWEN.GGUFLayout(
        path="/tmp/synthetic-qwen32.gguf",
        file_size=offset,
        data_offset=0,
        header_prefix_bytes=0,
        tensors=tuple(tensors),
        qtype_counts={"F32": len(tensors)},
        language_layers=language_layers,
    )


class _Reader:
    def stats(self):
        return {"closed": False, "staging_bytes": 0}

    def close(self):
        return None

    def read_tensor(self, spec, *, device="cpu", **_kwargs):
        # F32 zero payload is sufficient to exercise the bounded prefetch
        # ownership/attach path without creating a model-sized fixture.
        return torch.zeros(int(spec.n_bytes), dtype=torch.uint8, device=device)


class MPContractTest(unittest.TestCase):
    def test_shared_runtime_dispatches_mp_without_qwen_nccl(self):
        """MP selection must bypass the output-row TP process protocol."""

        class FakeMPBackend:
            def __init__(self):
                self.last_qwen_profile = {"mode": "mp", "finite": True}
                self.calls = 0

            def qwen_forward(self, hidden, **_kwargs):
                self.calls += 1
                return hidden + 1

        with tempfile.TemporaryDirectory() as results_dir:
            config = SHARED_RUNTIME.RuntimeConfig(
                model_path="/tmp/h3.gguf",
                lora_path="/tmp/h3.safetensors",
                egrid_path="/tmp/egrid.pt",
                results_dir=results_dir,
                qwen_model_path="/tmp/qwen32.gguf",
                qwen_mode="mp",
            )
            runtime = SHARED_RUNTIME.H3TPRuntime(config)
            backend = FakeMPBackend()
            runtime._ensure_qwen_mp_runtime = mock.Mock(return_value=backend)
            runtime.ensure_process_started = mock.Mock(
                side_effect=AssertionError("MP must not start Qwen NCCL")
            )
            hidden = torch.zeros((1, 1, QWEN.QWEN32_HIDDEN_SIZE))
            output = runtime.qwen_forward(hidden)

        self.assertEqual(backend.calls, 1)
        self.assertTrue(torch.equal(output, hidden + 1))
        self.assertEqual(runtime.last_qwen_profile, backend.last_qwen_profile)
        runtime._ensure_qwen_mp_runtime.assert_called_once_with()

    def test_split_plan_is_contiguous_and_balanced(self):
        layout = _synthetic_layout()
        plan = MP.plan_layer_split(
            layout,
            devices=("cpu", "cpu"),
            split="auto",
            baseline_bytes=(300, 0),
        )
        self.assertEqual(plan.layer_count, 6)
        self.assertEqual(plan.first_layers[-1] + 1, plan.second_layers[0])
        self.assertGreaterEqual(plan.split, 1)
        self.assertLess(plan.split, 6)
        self.assertEqual(plan.owner_index(0), 0)
        self.assertEqual(plan.owner_index(5), 1)

    def test_explicit_split_and_memory_report(self):
        layout = _synthetic_layout(4)
        plan = MP.plan_layer_split(
            layout,
            devices=("cpu", "cpu"),
            split=1,
            residency="full",
            dtype=torch.float16,
            capacity_bytes=(10_000, 10_000),
        )
        self.assertEqual(plan.split, 1)
        self.assertTrue(all(item is True for item in plan.fits_capacity))
        self.assertEqual(plan.as_dict()["split"], 1)
        self.assertEqual(len(plan.as_dict()["layer_costs"]), 4)

    def test_dense_cache_is_reflected_in_peak_estimate(self):
        layout = _synthetic_layout(2)
        compressed = MP.plan_layer_split(
            layout,
            devices=("cpu", "cpu"),
            split=1,
            residency="full",
            cache_dequantized=False,
        )
        dense = MP.plan_layer_split(
            layout,
            devices=("cpu", "cpu"),
            split=1,
            residency="full",
            cache_dequantized=True,
        )
        self.assertGreater(
            dense.estimated_peak_bytes[0], compressed.estimated_peak_bytes[0]
        )

    def test_full_descriptor_covers_matrix(self):
        layout = _synthetic_layout(1)
        spec = QWEN.language_matrix_specs(layout)[0]["q_proj"]
        descriptor = MP._full_descriptor(spec)
        self.assertEqual(descriptor.first_output_row, 0)
        self.assertEqual(descriptor.output_row_count, spec.shape[0])
        self.assertEqual(descriptor.n_bytes, spec.n_bytes)
        self.assertEqual(descriptor.data_offset, spec.data_offset)

    def test_tree_move_keeps_structure_on_cpu(self):
        value = (torch.ones(2), [torch.zeros(1), None], {"x": torch.arange(3)})
        moved = MP._move_tree(value, torch.device("cpu"))
        self.assertIsInstance(moved, tuple)
        self.assertEqual(tuple(moved[0].shape), (2,))
        self.assertIsNone(moved[1][1])
        self.assertTrue(torch.equal(moved[2]["x"], torch.arange(3)))

    def test_factory_keeps_tp_explicit(self):
        runtime = MP.create_qwen32_backend(
            "mp",
            model_path=None,
            devices=("cpu", "cpu"),
            check_peer_access=False,
        )
        self.assertEqual(runtime.mode, "mp")
        runtime.close()
        with self.assertRaisesRegex(RuntimeError, "output-row TP"):
            MP.create_qwen32_backend("tp", devices=("cpu", "cpu"))

    def test_mode_environment_is_mp_only_when_explicitly_changed(self):
        with mock.patch.dict("os.environ", {MP.QWEN32_MODE_ENV: "mp"}, clear=False):
            self.assertEqual(MP.resolve_qwen32_mode(), "mp")
        with mock.patch.dict("os.environ", {MP.QWEN32_MODE_ENV: "tp"}, clear=False):
            self.assertEqual(MP.resolve_qwen32_mode(), "tp")
        with self.assertRaises(ValueError):
            MP.resolve_qwen32_mode("unknown")

    def test_runtime_handle_is_duck_compatible_without_starting_model(self):
        runtime = MP.Qwen32Q2LayerMPRuntime(
            devices=("cpu", "cpu"),
            check_peer_access=False,
        )
        handle = MP.Qwen32Q2MPRuntimeHandle(runtime)
        self.assertIs(handle.runtime, runtime)
        self.assertIsNone(handle.qwen_clip())
        runtime.close()

    def test_backbone_starts_meta_only_without_payload_reads(self):
        layout = _synthetic_layout(2)
        backbone = MP.Qwen32Q2LayerMPBackbone(
            layout,
            devices=("cpu", "cpu"),
            layer_split=1,
            reader=_Reader(),
            check_peer_access=False,
        )
        try:
            stats = backbone.stats()
            self.assertEqual(stats["state"], "META_ONLY")
            self.assertEqual(stats["loaded_layers"], [])
            self.assertEqual(stats["layer_split"]["split"], 1)
        finally:
            backbone.close()

    def test_backbone_can_fail_closed_on_reported_capacity(self):
        layout = _synthetic_layout(2)
        # Patch the planner only for this constructor call so the test does
        # not depend on whatever CUDA memory another process currently uses.
        original = MP.plan_layer_split

        def tiny_capacity(*args, **kwargs):
            kwargs["capacity_bytes"] = (1, 1)
            return original(*args, **kwargs)

        with mock.patch.object(MP, "plan_layer_split", tiny_capacity):
            with self.assertRaises(MemoryError):
                MP.Qwen32Q2LayerMPBackbone(
                    layout,
                    devices=("cpu", "cpu"),
                    reader=_Reader(),
                    check_peer_access=False,
                )

    def test_backbone_loop_evicts_layers_and_applies_deepstack(self):
        layout = _synthetic_layout(2)
        backbone = MP.Qwen32Q2LayerMPBackbone(
            layout,
            devices=("cpu", "cpu"),
            layer_split=1,
            reader=_Reader(),
            check_peer_access=False,
        )

        class FakeBlock:
            def __init__(self, layer):
                self.layer = layer
                self.device = torch.device("cpu")
                self.cleared = False

            def __call__(self, value, **_kwargs):
                return value + 1

            def clear(self):
                self.cleared = True

            def stats(self):
                return {"layer": self.layer, "resident_bytes": 0}

            @property
            def resident_bytes(self):
                return 0

        made = []

        def fake_load(layer):
            block = FakeBlock(layer)
            made.append(block)
            return block

        backbone.load_layer = fake_load  # type: ignore[method-assign]
        hidden = torch.zeros((1, 2, QWEN.QWEN32_HIDDEN_SIZE), dtype=torch.float32)
        visual_mask = torch.zeros((1, 2), dtype=torch.bool)
        visual_mask[:, 1] = True
        deepstack = [
            torch.ones((1, QWEN.QWEN32_HIDDEN_SIZE)),
            torch.ones((1, QWEN.QWEN32_HIDDEN_SIZE)),
        ]
        try:
            output = backbone.forward_hidden(
                hidden,
                deepstack_embeds=deepstack,
                visual_pos_masks=visual_mask,
            )
            self.assertEqual(tuple(output.shape), tuple(hidden.shape))
            self.assertTrue(torch.equal(output[:, 0], torch.full_like(output[:, 0], 2)))
            self.assertTrue(torch.equal(output[:, 1], torch.full_like(output[:, 1], 4)))
            self.assertEqual(backbone.stats()["loaded_layers"], [])
            self.assertTrue(all(block.cleared for block in made))
        finally:
            backbone.close()

    def test_prefetch_is_opt_in_and_disabled_for_resident_routes(self):
        layout = _synthetic_layout(2)
        evict = MP.Qwen32Q2LayerMPBackbone(
            layout,
            devices=("cpu", "cpu"),
            layer_split=1,
            reader=_Reader(),
            check_peer_access=False,
            prefetch=True,
            prefetch_max_mib=1,
        )
        try:
            self.assertTrue(evict.stats()["prefetch"]["enabled"])
            self.assertEqual(evict.stats()["prefetch"]["max_bytes"], 1 << 20)
        finally:
            evict.close()

        resident = MP.Qwen32Q2LayerMPBackbone(
            layout,
            devices=("cpu", "cpu"),
            layer_split=1,
            residency="full",
            reader=_Reader(),
            check_peer_access=False,
            prefetch=True,
        )
        try:
            self.assertFalse(resident.stats()["prefetch"]["enabled"])
        finally:
            resident.close()

    def test_prefetch_capacity_is_included_in_fail_closed_gate(self):
        layout = _synthetic_layout(2)
        # The synthetic full layer is about 1 KiB: ordinary evict fits this
        # small capacity, while the extra one-layer overlap does not.
        with self.assertRaises(MemoryError):
            MP.Qwen32Q2LayerMPBackbone(
                layout,
                devices=("cpu", "cpu"),
                layer_split=1,
                reader=_Reader(),
                check_peer_access=False,
                prefetch=True,
                prefetch_max_mib=1,
                capacity_bytes=(1500, 1500),
            )

    def test_prefetch_reads_and_attaches_one_complete_layer(self):
        layout = _synthetic_layout(2)
        backbone = MP.Qwen32Q2LayerMPBackbone(
            layout,
            devices=("cpu", "cpu"),
            layer_split=1,
            reader=_Reader(),
            check_peer_access=False,
            prefetch=True,
            prefetch_max_mib=1,
        )
        try:
            prefetcher = backbone._prefetcher
            self.assertIsNotNone(prefetcher)
            assert prefetcher is not None
            self.assertTrue(prefetcher.submit(0))
            staged = prefetcher.consume(0)
            self.assertIsNotNone(staged)
            assert staged is not None
            self.assertEqual(set(staged.raw), set(QWEN.MATRIX_ROLES))
            self.assertEqual(set(staged.norms), set(QWEN.NORM_ROLES))
            backbone._pending_prefetched_layer = staged
            block = backbone.load_layer(0)
            backbone._pending_prefetched_layer = None
            self.assertEqual(block.compressed_bytes, sum(
                int(spec.n_bytes)
                for spec in backbone.layer_specs[0].values()
            ))
            self.assertEqual(prefetcher.stats()["errors"], 0)
        finally:
            backbone._pending_prefetched_layer = None
            backbone.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
