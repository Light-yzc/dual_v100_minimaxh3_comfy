#!/usr/bin/env python3
"""CPU contract tests for the opt-in Qwen32 TP integration.

These tests deliberately avoid starting NCCL or loading the 8 GiB checkpoint.
They catch the two classes of regressions that previously made a real request
look plausible while producing unrelated conditioning: Comfy attention output
layout and CLIP-compatible request/clear/weight lifecycle.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DUAL_DIR = REPO_ROOT / "custom_nodes" / "DualV100"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DUAL_DIR) not in sys.path:
    sys.path.insert(0, str(DUAL_DIR))


def _load_modules():
    """Import DualV100 modules without executing ComfyUI node discovery."""

    package_name = "custom_nodes.DualV100"
    package = types.ModuleType(package_name)
    package.__path__ = [str(DUAL_DIR)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    qwen = importlib.import_module(f"{package_name}.h3_qwen32_q2_tp")
    node = importlib.import_module(f"{package_name}.h3_qwen32_tp_node")
    runtime = importlib.import_module(f"{package_name}.h3_tp_runtime")
    return qwen, node, runtime


QWEN, NODE, RUNTIME = _load_modules()


class _FakeRuntime:
    def __init__(self):
        self.clear_calls: list[bool] = []

    def qwen_clear(self, *, notify_vae=True):
        self.clear_calls.append(bool(notify_vae))
        return {"vae_notified": bool(notify_vae)}


def _fake_clip(runtime: _FakeRuntime):
    clip = NODE._Qwen32TPClip.__new__(NODE._Qwen32TPClip)
    clip.runtime = runtime
    clip.runtime_handle = None
    clip.qwen_path = "/tmp/test-qwen32.gguf"
    clip.execution_device = torch.device("cpu")
    clip.compute_dtype = torch.float32
    clip._closed = False
    clip._encode_lock = threading.RLock()
    clip._forward_attempted = False
    clip._cache_entries = 4
    clip._cache = OrderedDict()
    clip._cache_hits = 0
    clip._cache_misses = 0
    clip._vision = None
    clip._options = {}
    return clip


class Qwen32TPContractTest(unittest.TestCase):
    def test_comfy_attention_preserves_head_layout(self):
        """The Comfy wrapper must receive skip_output_reshape=True."""

        q = torch.arange(1 * 4 * 3 * 2, dtype=torch.float32).reshape(1, 4, 3, 2)
        calls = {}

        def selector(_device, **_kwargs):
            def attention(query, key, value, heads, **kwargs):
                calls.update(kwargs)
                self.assertEqual(heads, 4)
                self.assertEqual(tuple(query.shape), (1, 4, 3, 2))
                return query + key + value

            return attention

        fake_comfy = types.ModuleType("comfy")
        fake_comfy.__path__ = []  # type: ignore[attr-defined]
        fake_ldm = types.ModuleType("comfy.ldm")
        fake_ldm.__path__ = []  # type: ignore[attr-defined]
        fake_modules = types.ModuleType("comfy.ldm.modules")
        fake_modules.__path__ = []  # type: ignore[attr-defined]
        fake_attention = types.ModuleType("comfy.ldm.modules.attention")
        fake_attention.optimized_attention_for_device = selector
        fake_comfy.ldm = fake_ldm
        fake_ldm.modules = fake_modules
        fake_modules.attention = fake_attention
        with mock.patch.dict(
            sys.modules,
            {
                "comfy": fake_comfy,
                "comfy.ldm": fake_ldm,
                "comfy.ldm.modules": fake_modules,
                "comfy.ldm.modules.attention": fake_attention,
            },
        ):
            output = QWEN._attention(q, q, q, None)
        self.assertEqual(tuple(output.shape), tuple(q.shape))
        self.assertTrue(calls.get("skip_reshape"))
        self.assertTrue(calls.get("skip_output_reshape"))

    def test_successful_encode_clears_once_after_all_sections(self):
        runtime = _FakeRuntime()
        clip = _fake_clip(runtime)

        def process(section):
            clip._forward_attempted = True
            size = len(section)
            hidden = torch.ones((1, size, 5120), dtype=torch.float32)
            binary = torch.ones((1, size), dtype=torch.long)
            tags = torch.ones((size,), dtype=torch.long)
            return hidden, binary, tags, None, torch.ones(size)

        clip._process_one = process
        tokens = {"qwen3vl_32b": [[(101, 1.0)], [(102, 1.0)]]}
        result = clip.encode_from_tokens_scheduled(tokens)
        self.assertEqual(tuple(result[0][0].shape), (1, 2, 5120))
        self.assertEqual(runtime.clear_calls, [True])
        self.assertIsNone(result[0][1]["pooled_output"])

    def test_failed_encode_clears_without_unlocking_vae(self):
        runtime = _FakeRuntime()
        clip = _fake_clip(runtime)

        def process(_section):
            clip._forward_attempted = True
            raise ValueError("synthetic Qwen failure")

        clip._process_one = process
        with self.assertRaisesRegex(ValueError, "synthetic Qwen failure"):
            clip.encode_from_tokens_scheduled(
                {"qwen3vl_32b": [[(101, 1.0)]]}
            )
        self.assertEqual(runtime.clear_calls, [False])

    def test_text_weight_interpolation_uses_empty_baseline(self):
        runtime = _FakeRuntime()
        clip = _fake_clip(runtime)
        calls = []

        def process(section):
            clip._forward_attempted = True
            calls.append([item[0] for item in section])
            value = float(section[0][0])
            hidden = torch.full((1, len(section), 5120), value)
            binary = torch.ones((1, len(section)), dtype=torch.long)
            tags = torch.ones((len(section),), dtype=torch.long)
            weights = torch.tensor([item[1] for item in section])
            return hidden, binary, tags, None, weights

        clip._process_one = process
        result = clip.encode_from_tokens_scheduled(
            {"qwen3vl_32b": [[(7, 2.0), (8, 1.0)]]}
        )
        # The empty baseline starts with PAD (151643), so the first weighted
        # position is (7 - PAD) * 2 + PAD.
        expected = (7.0 - NODE._PAD_TOKEN) * 2.0 + NODE._PAD_TOKEN
        self.assertAlmostEqual(float(result[0][0][0, 0, 0]), expected)
        self.assertEqual(calls, [[7, 8], [NODE._PAD_TOKEN, NODE._PAD_TOKEN]])
        self.assertEqual(runtime.clear_calls, [True])

    def test_runtime_handle_forwards_named_qwen_configuration(self):
        class Runtime:
            def __init__(self):
                self.calls = []

            def configure_qwen(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        runtime = Runtime()
        handle = NODE.H3TPRuntimeHandle(
            runtime,
            qwen_model_path="/tmp/model.gguf",
            qwen_staging_mib=4,
        )
        handle.configure_qwen(
            staging_mib=8,
            residency="evict",
            keep_layers=0,
            cache_dequantized=False,
        )
        self.assertEqual(runtime.calls[0][0], ("/tmp/model.gguf",))
        self.assertEqual(runtime.calls[0][1]["staging_mib"], 8)
        self.assertEqual(runtime.calls[0][1]["residency"], "evict")


if __name__ == "__main__":
    unittest.main(verbosity=2)
