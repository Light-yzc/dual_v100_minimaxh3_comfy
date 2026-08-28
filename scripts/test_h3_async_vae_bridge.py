#!/usr/bin/env python3
"""CPU-only lifecycle tests for the H3 asynchronous VAE bridge."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "custom_nodes" / "DualV100"


def _load_bridge():
    package_name = f"h3_bridge_test_{uuid.uuid4().hex}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MODULE_DIR)]
    sys.modules[package_name] = package

    async_name = f"{package_name}.h3_async_vae"
    async_spec = importlib.util.spec_from_file_location(
        async_name, MODULE_DIR / "h3_async_vae.py"
    )
    if async_spec is None or async_spec.loader is None:
        raise RuntimeError("cannot load h3_async_vae.py")
    async_module = importlib.util.module_from_spec(async_spec)
    sys.modules[async_name] = async_module
    async_spec.loader.exec_module(async_module)

    bridge_name = f"{package_name}.h3_async_vae_bridge"
    bridge_spec = importlib.util.spec_from_file_location(
        bridge_name, MODULE_DIR / "h3_async_vae_bridge.py"
    )
    if bridge_spec is None or bridge_spec.loader is None:
        raise RuntimeError("cannot load h3_async_vae_bridge.py")
    bridge = importlib.util.module_from_spec(bridge_spec)
    sys.modules[bridge_name] = bridge
    bridge_spec.loader.exec_module(bridge)
    return bridge


class FakeHandle:
    def __init__(self, events: list[object]):
        self.events = events
        self.devices = (torch.device("cpu:0"), torch.device("cpu:1"))
        self.facade = object()

    def prepare_for_qwen(self, *, keep_encoder=True):
        self.events.append(("prepare", keep_encoder))

    def mark_dit_ready(self):
        self.events.append("qwen_cleared")

    def mark_denoising(self, *, start_prefetch=True):
        self.events.append(("prefetch", start_prefetch))
        return True

    def finalize_tail(self):
        self.events.append("finalize")
        return self.facade

    def cancel(self):
        self.events.append("cancel")

    def release_decoder(self, *, keep_encoder=True):
        self.events.append(("release", keep_encoder))

    def stats(self):
        return {"state": "fake", "events": len(self.events)}


class AsyncVAEBridgeTest(unittest.TestCase):
    def setUp(self):
        self.bridge = _load_bridge()
        self.events: list[object] = []
        self.fail_sample = False

        test = self

        class FakeKSAMPLER:
            def sample(self, value=None):
                test.events.append("sample_enter")
                if test.fail_sample:
                    raise RuntimeError("injected sampler failure")
                test.events.append("sample_return")
                return value

        comfy = types.ModuleType("comfy")
        samplers = types.ModuleType("comfy.samplers")
        samplers.KSAMPLER = FakeKSAMPLER
        comfy.samplers = samplers
        self.modules = mock.patch.dict(
            sys.modules, {"comfy": comfy, "comfy.samplers": samplers}
        )
        self.modules.start()
        self.sampler_class = FakeKSAMPLER

    def tearDown(self):
        self.bridge.clear_active_async_vae()
        self.modules.stop()

    def test_hook_is_idempotent_and_obeys_qwen_barrier(self):
        handle = FakeHandle(self.events)
        self.bridge.register_active_async_vae(handle)
        self.assertTrue(self.bridge.install_turbo_sampler_hook())
        self.assertFalse(self.bridge.install_turbo_sampler_hook())

        sampler = self.sampler_class()
        self.assertEqual(sampler.sample("first"), "first")
        self.assertEqual(self.events, ["sample_enter", "sample_return"])

        self.assertTrue(self.bridge.notify_qwen_cleared())
        self.assertEqual(sampler.sample("second"), "second")
        self.assertEqual(
            self.events,
            [
                "sample_enter",
                "sample_return",
                "qwen_cleared",
                ("prefetch", True),
                "sample_enter",
                "sample_return",
                "finalize",
            ],
        )
        stats = self.bridge.active_async_vae_stats()
        self.assertTrue(stats["active"])
        self.assertFalse(stats["qwen_cleared"])
        self.assertFalse(stats["dit_active"])

    def test_sampler_failure_cancels_and_clears_partial_handle(self):
        handle = FakeHandle(self.events)
        self.bridge.register_active_async_vae(handle)
        self.bridge.install_turbo_sampler_hook()
        self.bridge.notify_qwen_cleared()
        self.fail_sample = True

        with self.assertRaisesRegex(RuntimeError, "injected sampler failure"):
            self.sampler_class().sample("unused")

        self.assertEqual(
            self.events,
            [
                "qwen_cleared",
                ("prefetch", True),
                "sample_enter",
                "cancel",
                ("release", False),
            ],
        )
        self.assertFalse(self.bridge.active_async_vae_stats()["active"])

    def test_factory_is_opt_in_and_rejects_non_fp16(self):
        path = "/models/minimax_h3_video_vae.safetensors"
        facade = types.SimpleNamespace(async_handle=FakeHandle(self.events))
        fp16_specs = {
            "decoder.weight": types.SimpleNamespace(dtype=torch.float16)
        }
        metadata = {"metadata": {"minimax_h3_video_vae": "{}"}}

        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
            self.bridge, "inspect_safetensors"
        ) as inspect, mock.patch.object(
            self.bridge, "load_h3_video_vae_async", return_value=facade
        ) as loader:
            os.environ.pop("H3_ASYNC_VAE_LOAD", None)
            self.assertIsNone(self.bridge.maybe_load_async_vae_facade(path))
            inspect.assert_not_called()
            loader.assert_not_called()

        with mock.patch.dict(os.environ, {"H3_ASYNC_VAE_LOAD": "1"}), mock.patch.object(
            self.bridge,
            "inspect_safetensors",
            return_value=(
                {"decoder.weight": types.SimpleNamespace(dtype=torch.float32)},
                metadata,
            ),
        ), mock.patch.object(
            self.bridge, "load_h3_video_vae_async", return_value=facade
        ) as loader:
            with self.assertRaisesRegex(ValueError, "all-FP16"):
                self.bridge.maybe_load_async_vae_facade(path)
            loader.assert_not_called()

        with mock.patch.dict(os.environ, {"H3_ASYNC_VAE_LOAD": "1"}), mock.patch.object(
            self.bridge,
            "inspect_safetensors",
            return_value=(fp16_specs, metadata),
        ), mock.patch.object(
            self.bridge, "load_h3_video_vae_async", return_value=facade
        ):
            self.assertIs(self.bridge.maybe_load_async_vae_facade(path), facade)
            self.assertTrue(self.bridge.active_async_vae_stats()["active"])

    def test_explicit_lifecycle_helpers(self):
        handle = FakeHandle(self.events)
        self.bridge.register_active_async_vae(handle)
        self.assertTrue(self.bridge.prepare_active_vae_for_qwen())
        self.assertTrue(self.bridge.notify_qwen_cleared())
        self.assertTrue(self.bridge.notify_dit_start())
        self.assertIs(self.bridge.finalize_active_vae_after_dit(), handle.facade)
        self.assertEqual(
            self.events,
            [
                ("prepare", True),
                "qwen_cleared",
                ("prefetch", True),
                "finalize",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
