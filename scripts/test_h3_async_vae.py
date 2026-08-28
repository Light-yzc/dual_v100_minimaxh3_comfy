#!/usr/bin/env python3
"""CPU-only tests for the bounded H3 asynchronous VAE loader."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_async_vae.py"
SPEC = importlib.util.spec_from_file_location("h3_async_vae_test_target", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
ASYNC_VAE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ASYNC_VAE
SPEC.loader.exec_module(ASYNC_VAE)


def _write_safetensors(path: Path, values: dict[str, torch.Tensor]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    dtype_names = {
        torch.float16: "F16",
        torch.float32: "F32",
        torch.int8: "I8",
        torch.uint8: "U8",
    }
    for name, value in values.items():
        value = value.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype_names[value.dtype],
            "shape": list(value.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.weight = nn.Parameter(
            torch.empty((2,), dtype=torch.float32, device="meta"),
            requires_grad=False,
        )
        self.decoder = nn.Module()
        self.decoder.weight = nn.Parameter(
            torch.empty((4,), dtype=torch.float32, device="meta"),
            requires_grad=False,
        )
        self.decoder.bias = nn.Parameter(
            torch.empty((2,), dtype=torch.float32, device="meta"),
            requires_grad=False,
        )
        self.register_buffer(
            "latents_mean", torch.empty((1,), dtype=torch.float32, device="meta")
        )
        self.register_buffer(
            "latents_std", torch.empty((1,), dtype=torch.float32, device="meta")
        )


def _owner(name, devices, _split):
    return devices[1] if name.startswith("decoder.") else devices[0]


def _make_handle(path: Path, **kwargs):
    return ASYNC_VAE.AsyncVAEHandle(
        path,
        (torch.device("cpu:0"), torch.device("cpu:1")),
        split=1,
        staging_bytes=7,
        model_factory=TinyVAE,
        constants_installer=lambda _model, _device: None,
        model_finalizer=lambda model, _devices, _split: model,
        owner_resolver=_owner,
        **kwargs,
    )


class AsyncVAETest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3_async_vae_")
        self.path = Path(self.temp.name) / "tiny.safetensors"
        self.values = {
            "encoder.weight": torch.tensor([1.0, 2.0]),
            "latents_mean": torch.tensor([3.0]),
            "latents_std": torch.tensor([4.0]),
            "decoder.weight": torch.tensor([5.0, 6.0, 7.0, 8.0]),
            "decoder.bias": torch.tensor([9.0, 10.0]),
        }
        _write_safetensors(self.path, self.values)

    def tearDown(self):
        self.temp.cleanup()

    def test_header_only_and_memory_ledger(self):
        specs, metadata = ASYNC_VAE.inspect_safetensors(self.path)
        self.assertEqual(metadata["tensor_count"], len(self.values))
        self.assertFalse(metadata["host_mmap"])
        self.assertEqual(sum(spec.n_bytes for spec in specs.values()), 40)
        maps = Path("/proc/self/maps").read_text(errors="replace")
        self.assertNotIn(str(self.path), maps)

        ledger = ASYNC_VAE.CUDAMemoryLedger(
            safety_bytes=250,
            probe=lambda _device: (100, 200, 300, 1000),
        )
        snapshot = ledger.snapshot(torch.device("cpu"))
        self.assertEqual(snapshot.available, 50)
        self.assertTrue(snapshot.can_allocate(50))
        self.assertFalse(snapshot.can_allocate(51))

    def test_encoder_cap_and_finalize_tail(self):
        handle = _make_handle(self.path, prefetch_limits=(16, 16))
        self.assertEqual(handle.state, ASYNC_VAE.AsyncVAEState.META_ONLY)

        handle.ensure_encoder_ready()
        self.assertEqual(handle.state, ASYNC_VAE.AsyncVAEState.ENCODER_READY)
        self.assertTrue(torch.equal(handle.model.encoder.weight, self.values["encoder.weight"]))
        self.assertTrue(handle.model.decoder.weight.is_meta)

        self.assertTrue(handle.begin_prefetch())
        deadline = time.monotonic() + 5.0
        while handle.state == ASYNC_VAE.AsyncVAEState.PREFETCHING:
            if time.monotonic() > deadline:
                self.fail("prefetch worker did not finish")
            time.sleep(0.01)
        self.assertEqual(handle.state, ASYNC_VAE.AsyncVAEState.CAPPED)
        stats = handle.stats()
        self.assertEqual(stats["resident_bytes"]["cpu:1"], 16)
        self.assertEqual(stats["deferred_bytes"]["cpu:1"], 8)

        handle.finalize_tail()
        self.assertEqual(handle.state, ASYNC_VAE.AsyncVAEState.READY)
        self.assertTrue(torch.equal(handle.model.decoder.weight, self.values["decoder.weight"]))
        self.assertTrue(torch.equal(handle.model.decoder.bias, self.values["decoder.bias"]))
        stats = handle.stats()
        self.assertEqual(stats["deferred_bytes"], {"cpu:0": 0, "cpu:1": 0})

        ready_model = handle.model
        handle.release_decoder()
        self.assertEqual(handle.state, ASYNC_VAE.AsyncVAEState.ENCODER_READY)
        self.assertIsNot(handle.model, ready_model)
        self.assertTrue(torch.equal(handle.model.encoder.weight, self.values["encoder.weight"]))
        self.assertTrue(handle.model.decoder.weight.is_meta)
        stats = handle.stats()
        self.assertEqual(stats["resident_bytes"]["cpu:0"], 16)
        self.assertEqual(stats["resident_bytes"]["cpu:1"], 0)

        handle.configure_prefetch_limits((None, None))
        handle.begin_prefetch()
        handle.finalize_tail()
        self.assertEqual(handle.state, ASYNC_VAE.AsyncVAEState.READY)
        self.assertEqual(handle.stats()["deferred_bytes"], {"cpu:0": 0, "cpu:1": 0})

    def test_background_failure_discards_partial_then_falls_back(self):
        base_reader = ASYNC_VAE.BoundedSafeTensorReader
        calls = {"factory": 0, "reads": 0}

        class FailSecondPhaseReader(base_reader):
            def __init__(self, *args, fail=False, **kwargs):
                self.fail = fail
                super().__init__(*args, **kwargs)

            def read(self, spec, device):
                calls["reads"] += 1
                if self.fail:
                    self.fail = False
                    raise OSError("injected bounded-reader failure")
                return super().read(spec, device)

        def reader_factory(*args, **kwargs):
            calls["factory"] += 1
            return FailSecondPhaseReader(
                *args, fail=(calls["factory"] == 2), **kwargs
            )

        handle = _make_handle(
            self.path,
            prefetch_limits=(None, None),
            reader_factory=reader_factory,
        )
        handle.ensure_encoder_ready()
        partial_model = handle.model
        with self.assertLogs(level="WARNING") as captured:
            handle.begin_prefetch()
            deadline = time.monotonic() + 5.0
            while handle.state == ASYNC_VAE.AsyncVAEState.PREFETCHING:
                if time.monotonic() > deadline:
                    self.fail("failing prefetch worker did not finish")
                time.sleep(0.01)
            self.assertEqual(handle.state, ASYNC_VAE.AsyncVAEState.FAILED)
            self.assertIsNot(handle.model, partial_model)
            handle.await_ready()
        self.assertTrue(
            any("synchronous no-mmap fallback" in line for line in captured.output)
        )
        self.assertEqual(handle.state, ASYNC_VAE.AsyncVAEState.READY)
        self.assertEqual(handle.stats()["fallback_count"], 1)
        self.assertTrue(torch.equal(handle.model.decoder.bias, self.values["decoder.bias"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
