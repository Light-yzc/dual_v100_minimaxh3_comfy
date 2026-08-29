"""Capped asynchronous no-mmap loader for the two-GPU H3 video VAE.

The handle deliberately separates checkpoint metadata from CUDA residency.
Reference images can use the encoder synchronously while the much larger
decoder remains file-backed.  After Qwen has released its CUDA payload, a
bounded worker loads decoder tensors directly to their permanent 24/12 owner
on low-priority streams.  The final tail is materialised only after the DiT
barrier, immediately before decode.

This module is importable without ComfyUI and without CUDA.  ComfyUI model
construction is deferred until a production handle is created.
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import math
import os
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


MIB = 1 << 20
GIB = 1 << 30
DEFAULT_SAFETY_BYTES = 1 * GIB
DEFAULT_STAGING_BYTES = 4 * MIB
MAX_HEADER_BYTES = 64 * MIB
FP16_TWO_REFERENCE_PREFETCH_MIB = (1962, 1787)
_TRUE = {"1", "true", "yes", "on"}

_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "C64": torch.complex64,
}


class AsyncVAEState(str, Enum):
    """Observable lifecycle of an :class:`AsyncVAEHandle`."""

    META_ONLY = "meta_only"
    # Stage names from the Qwen32B lifecycle are retained as explicit states
    # for runtime telemetry.  The more granular loader states below describe
    # work occurring inside a stage.
    ENCODING = "encoding"
    DIT_READY = "dit_ready"
    DENOISING = "denoising"
    VAE_FINALIZE = "vae_finalize"
    DECODING = "decoding"
    ENCODER_LOADING = "encoder_loading"
    ENCODER_READY = "encoder_ready"
    PREFETCHING = "prefetching"
    PREFETCHED = "prefetched"
    CAPPED = "capped"
    FINALIZING = "finalizing"
    FALLBACK_LOADING = "fallback_loading"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SafeTensorSpec:
    """Validated location and geometry of one safetensors value."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    data_offset: int
    n_bytes: int


@dataclass(frozen=True)
class MemorySnapshot:
    """Allocator and driver view used for one prefetch admission decision."""

    allocated: int
    reserved: int
    driver_free: int
    total: int
    safety: int

    @property
    def available(self) -> int:
        allocator_free = max(0, self.total - self.allocated)
        reserve_free = max(0, self.total - self.reserved)
        raw = min(allocator_free, reserve_free, self.driver_free)
        return max(0, raw - self.safety)

    def can_allocate(self, n_bytes: int) -> bool:
        return n_bytes >= 0 and n_bytes <= self.available


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE


def _resolve_no_mmap_path(path: os.PathLike[str] | str) -> str:
    """Resolve symlinks and the deployment's bounded ``IntxLNK`` files."""

    try:
        from custom_nodes.NoHostMMap.safetensors import resolve_no_host_path
    except ImportError:
        return os.path.realpath(os.fspath(path))
    return str(resolve_no_host_path(path))


def inspect_safetensors(
    path: os.PathLike[str] | str,
) -> tuple[dict[str, SafeTensorSpec], dict[str, object]]:
    """Read and validate only a safetensors header; never mmap its payload."""

    resolved = _resolve_no_mmap_path(path)
    with open(resolved, "rb", buffering=0) as handle:
        file_size = os.fstat(handle.fileno()).st_size
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"incomplete safetensors header: {resolved}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > MAX_HEADER_BYTES or header_size > max(0, file_size - 8):
            raise ValueError(
                f"invalid safetensors header size {header_size}: {resolved}"
            )
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"incomplete safetensors header: {resolved}")

    header = json.loads(raw_header.decode("utf-8"))
    base_offset = 8 + header_size
    specs: dict[str, SafeTensorSpec] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = _SAFETENSORS_DTYPES[info["dtype"]]
            shape = tuple(int(value) for value in info["shape"])
            start, stop = (int(value) for value in info["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed safetensors entry {name!r}: {resolved}") from exc
        if start < 0 or stop < start or base_offset + stop > file_size:
            raise ValueError(f"invalid payload range for {name!r}: {resolved}")
        elements = math.prod(shape)
        expected = elements * torch.empty((), dtype=dtype).element_size()
        if stop - start != expected:
            raise ValueError(
                f"safetensors byte geometry mismatch for {name!r}: "
                f"{stop-start} != {expected}"
            )
        specs[name] = SafeTensorSpec(
            name=name,
            dtype=dtype,
            shape=shape,
            data_offset=base_offset + start,
            n_bytes=stop - start,
        )

    metadata = {
        "path": resolved,
        "file_size": file_size,
        "header_bytes": header_size,
        "tensor_count": len(specs),
        "metadata": header.get("__metadata__", {}),
        "host_mmap": False,
    }
    return specs, metadata


class CUDAMemoryLedger:
    """Conservative admission gate using allocator and CUDA-driver counters."""

    def __init__(
        self,
        safety_bytes: int = DEFAULT_SAFETY_BYTES,
        probe: Callable[[torch.device], tuple[int, int, int, int]] | None = None,
    ):
        if safety_bytes < 0:
            raise ValueError("safety_bytes must be non-negative")
        self.safety_bytes = int(safety_bytes)
        self._probe = probe or self._cuda_probe

    @staticmethod
    def _cuda_probe(device: torch.device) -> tuple[int, int, int, int]:
        if device.type != "cuda":
            maximum = 1 << 62
            return 0, 0, maximum, maximum
        with torch.cuda.device(device):
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
            driver_free, total = torch.cuda.mem_get_info(device)
        return allocated, reserved, int(driver_free), int(total)

    def snapshot(self, device: torch.device) -> MemorySnapshot:
        allocated, reserved, driver_free, total = self._probe(device)
        return MemorySnapshot(
            allocated=int(allocated),
            reserved=int(reserved),
            driver_free=int(driver_free),
            total=int(total),
            safety=self.safety_bytes,
        )

    def can_allocate(self, device: torch.device, n_bytes: int) -> bool:
        return self.snapshot(device).can_allocate(int(n_bytes))


class _CopyToken:
    """CPU/CUDA-neutral completion token."""

    def wait(self) -> None:
        return None


class _CudaCopyToken(_CopyToken):
    def __init__(self, event: torch.cuda.Event):
        self.event = event

    def wait(self) -> None:
        self.event.synchronize()


class _StagingSlot:
    def __init__(self, size: int, *, pin_memory: bool):
        self.buffer = torch.empty(
            (size,), dtype=torch.uint8, device="cpu", pin_memory=pin_memory
        )
        self.token: _CopyToken | None = None

    def acquire(self) -> torch.Tensor:
        if self.token is not None:
            self.token.wait()
            self.token = None
        return self.buffer


class BoundedSafeTensorReader:
    """Ordinary-read loader with a fixed 4--8 MiB ring per target device."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        devices: Sequence[torch.device],
        streams: Mapping[torch.device, torch.cuda.Stream | None],
        *,
        staging_bytes: int = DEFAULT_STAGING_BYTES,
        ring_slots: int = 2,
    ):
        if staging_bytes <= 0 or ring_slots <= 0:
            raise ValueError("staging_bytes and ring_slots must be positive")
        self.path = _resolve_no_mmap_path(path)
        self.file = open(self.path, "rb", buffering=0)
        self.streams = dict(streams)
        slot_bytes = max(1, staging_bytes // ring_slots)
        self._rings: dict[torch.device, list[_StagingSlot]] = {}
        self._next_slot: dict[torch.device, int] = {}
        for device in devices:
            pin = device.type == "cuda"
            self._rings[device] = [
                _StagingSlot(slot_bytes, pin_memory=pin) for _ in range(ring_slots)
            ]
            self._next_slot[device] = 0
        self._advise(0, 0, getattr(os, "POSIX_FADV_SEQUENTIAL", None))

    def _advise(self, offset: int, size: int, advice: int | None) -> None:
        if advice is None or not hasattr(os, "posix_fadvise"):
            return
        try:
            os.posix_fadvise(self.file.fileno(), offset, size, advice)
        except OSError:
            pass

    def _read_into(self, offset: int, target: torch.Tensor) -> None:
        size = int(target.numel())
        if size == 0:
            return
        view_type = ctypes.c_ubyte * size
        view = memoryview(view_type.from_address(target.data_ptr()))
        try:
            self.file.seek(offset)
            done = 0
            while done < size:
                count = self.file.readinto(view[done:])
                if count is None or count <= 0:
                    raise OSError(f"short read from {self.path} at {offset + done}")
                done += count
        finally:
            view.release()

    def _slot(self, device: torch.device) -> _StagingSlot:
        slots = self._rings[device]
        index = self._next_slot[device]
        self._next_slot[device] = (index + 1) % len(slots)
        return slots[index]

    def read(self, spec: SafeTensorSpec, device: torch.device) -> tuple[torch.Tensor, _CopyToken]:
        """Allocate once on the final owner and stream exact file bytes into it."""

        stream = self.streams[device]
        if device.type == "cuda":
            assert stream is not None
            with torch.cuda.device(device), torch.cuda.stream(stream):
                destination = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            destination.record_stream(stream)
        else:
            destination = torch.empty(spec.shape, dtype=spec.dtype, device=device)

        flat = destination.view(torch.uint8).reshape(-1)
        copied = 0
        last_token: _CopyToken = _CopyToken()
        while copied < spec.n_bytes:
            slot = self._slot(device)
            staging = slot.acquire()
            count = min(staging.numel(), spec.n_bytes - copied)
            source = staging[:count]
            self._read_into(spec.data_offset + copied, source)
            if stream is None:
                flat[copied : copied + count].copy_(source)
                token = _CopyToken()
            else:
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    flat[copied : copied + count].copy_(source, non_blocking=True)
                    event = torch.cuda.Event(blocking=False)
                    event.record(stream)
                token = _CudaCopyToken(event)
            slot.token = token
            last_token = token
            self._advise(
                spec.data_offset + copied,
                count,
                getattr(os, "POSIX_FADV_DONTNEED", None),
            )
            copied += count
        return destination, last_token

    def drain(self) -> None:
        for slots in self._rings.values():
            for slot in slots:
                slot.acquire()

    def close(self) -> None:
        try:
            self.drain()
        finally:
            self.file.close()

    def __enter__(self) -> "BoundedSafeTensorReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def h3_vae_owner(name: str, devices: Sequence[torch.device], split: int) -> torch.device:
    """Return the permanent owner used by the existing H3 MP decoder."""

    first, second = devices
    if name.startswith("decoder.transformer_blocks."):
        block_index = int(name.split(".")[2])
        return first if block_index < split else second
    if name.startswith("decoder.norm_out.") or name.startswith("decoder.proj_out."):
        return second
    return first


def _is_encoder_value(name: str) -> bool:
    return name.startswith(("encoder.", "quant_conv.")) or name in {
        "latents_mean",
        "latents_std",
    }


def _default_model_factory() -> nn.Module:
    import comfy.ops
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    with torch.device("meta"):
        return MiniMaxH3VideoVAE(operations=comfy.ops.disable_weight_init).eval()


def _resolve_parent(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _default_tensor_setter(model: nn.Module, name: str, value: torch.Tensor) -> None:
    parent, leaf = _resolve_parent(model, name)
    if leaf in parent._parameters:
        previous = parent._parameters[leaf]
        requires_grad = bool(previous is not None and previous.requires_grad)
        parent._parameters[leaf] = nn.Parameter(value, requires_grad=requires_grad)
        return
    if leaf in parent._buffers:
        parent._buffers[leaf] = value
        return
    raise KeyError(f"checkpoint value does not resolve to a parameter/buffer: {name}")


def _get_tensor(model: nn.Module, name: str) -> torch.Tensor:
    parent, leaf = _resolve_parent(model, name)
    value = getattr(parent, leaf)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"model value is not a tensor: {name}")
    return value


def _install_h3_constants(model: nn.Module, first: torch.device) -> None:
    from comfy.ldm.minimax import vae as vae_module

    model.pixel_mean = torch.tensor(
        vae_module.IMAGENET_MEAN, device=first, dtype=torch.float32
    ).view(1, 3, 1, 1, 1)
    model.pixel_std = torch.tensor(
        vae_module.IMAGENET_STD, device=first, dtype=torch.float32
    ).view(1, 3, 1, 1, 1)
    pos_embed = model.decoder.pos_embed
    dim = int(2 * pos_embed.n_dim * pos_embed.inv_freq.numel())
    inv_freq = 1 / 100.0 ** torch.arange(
        0, 1, 2 * pos_embed.n_dim / dim, device=first, dtype=torch.float32
    )
    pos_embed.inv_freq = inv_freq


def _finalize_h3_model(
    model: nn.Module, devices: Sequence[torch.device], split: int
) -> nn.Module:
    try:
        from .h3_model_parallel import H3ParallelViTDecoder
    except ImportError:
        from custom_nodes.DualV100.h3_model_parallel import H3ParallelViTDecoder

    decoder = model.decoder
    model.decoder = H3ParallelViTDecoder(decoder, devices[0], devices[1], split)
    return model.eval()


def _make_streams(
    devices: Sequence[torch.device],
) -> dict[torch.device, torch.cuda.Stream | None]:
    result: dict[torch.device, torch.cuda.Stream | None] = {}
    for device in devices:
        if device.type != "cuda":
            result[device] = None
            continue
        with torch.cuda.device(device):
            # CUDA priority 0 is the low/default end of the range on V100;
            # negative priorities are reserved for latency-sensitive compute.
            # PyTorch 2.8 does not expose get_stream_priority_range().
            result[device] = torch.cuda.Stream(device=device, priority=0)
    return result


class AsyncVAEHandle:
    """Own one partial/full VAE and coordinate capped background residency."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        devices: Sequence[torch.device | str],
        *,
        split: int = 24,
        safety_bytes: int = DEFAULT_SAFETY_BYTES,
        staging_bytes: int = DEFAULT_STAGING_BYTES,
        prefetch_limits: Sequence[int | None] | None = None,
        model_factory: Callable[[], nn.Module] | None = None,
        tensor_setter: Callable[[nn.Module, str, torch.Tensor], None] | None = None,
        owner_resolver: Callable[[str, Sequence[torch.device], int], torch.device] | None = None,
        model_finalizer: Callable[
            [nn.Module, Sequence[torch.device], int], nn.Module
        ] | None = None,
        constants_installer: Callable[[nn.Module, torch.device], None] | None = None,
        ledger: CUDAMemoryLedger | None = None,
        reader_factory: Callable[..., BoundedSafeTensorReader] = BoundedSafeTensorReader,
        value_loader: Callable[
            [BoundedSafeTensorReader, SafeTensorSpec, torch.device],
            tuple[torch.Tensor, _CopyToken],
        ] | None = None,
        synchronous_fallback: Callable[[], nn.Module] | None = None,
    ):
        if len(devices) != 2:
            raise ValueError("H3 async VAE requires exactly two owner devices")
        self.path = _resolve_no_mmap_path(path)
        self.devices = tuple(torch.device(device) for device in devices)
        self.split = max(1, int(split))
        self.staging_bytes = int(staging_bytes)
        if self.staging_bytes <= 0:
            raise ValueError("staging_bytes must be positive")
        if prefetch_limits is None:
            self.prefetch_limits = (None, None)
        else:
            if len(prefetch_limits) != 2:
                raise ValueError("prefetch_limits must contain two values")
            self.prefetch_limits = tuple(
                None if value is None else max(0, int(value))
                for value in prefetch_limits
            )

        self.specs, self.header_metadata = inspect_safetensors(self.path)
        self._model_factory = model_factory or _default_model_factory
        self._tensor_setter = tensor_setter or _default_tensor_setter
        self._owner_resolver = owner_resolver or h3_vae_owner
        self._model_finalizer = model_finalizer or _finalize_h3_model
        self._constants_installer = constants_installer or _install_h3_constants
        self._ledger = ledger or CUDAMemoryLedger(safety_bytes)
        self._reader_factory = reader_factory
        self._value_loader = value_loader
        self._synchronous_fallback = synchronous_fallback
        self._streams = _make_streams(self.devices)
        self._model = self._new_model()
        self._facade: H3AsyncVAE | None = None

        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._cancel = threading.Event()
        self._prefetch_thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._state = AsyncVAEState.META_ONLY
        self._loaded: set[str] = set()
        self._tokens: dict[str, _CopyToken] = {}
        self._fallback_count = 0
        self._prefetch_started_at: float | None = None
        self._prefetch_finished_at: float | None = None
        self._finalize_started_at: float | None = None
        self._finalize_finished_at: float | None = None

        self._requested = {device: 0 for device in self.devices}
        self._resident = {device: 0 for device in self.devices}
        self._read_bytes = {device: 0 for device in self.devices}
        for spec in self.specs.values():
            owner = self._owner(spec.name)
            self._requested[owner] += spec.n_bytes

    def _new_model(self) -> nn.Module:
        model = self._model_factory()
        self._constants_installer(model, self.devices[0])
        return model

    def _owner(self, name: str) -> torch.device:
        return self._owner_resolver(name, self.devices, self.split)

    @property
    def state(self) -> AsyncVAEState:
        with self._lock:
            return self._state

    @property
    def model(self) -> nn.Module:
        with self._lock:
            return self._model

    @property
    def vae(self) -> "H3AsyncVAE":
        with self._lock:
            if self._facade is None:
                self._facade = H3AsyncVAE(self)
            return self._facade

    def _set_state(self, state: AsyncVAEState) -> None:
        with self._lock:
            self._state = state

    def _ordered_names(self, names: Iterable[str]) -> list[str]:
        selected = set(names)
        # Keep encoder prerequisites first, then preserve checkpoint offset order.
        return sorted(
            selected,
            key=lambda name: (
                0 if _is_encoder_value(name) else 1,
                self.specs[name].data_offset,
            ),
        )

    def _admit_prefetch(self, spec: SafeTensorSpec, owner: torch.device) -> bool:
        index = self.devices.index(owner)
        limit = self.prefetch_limits[index]
        with self._lock:
            resident = self._resident[owner]
        if limit is not None and resident + spec.n_bytes > limit:
            return False
        return self._ledger.can_allocate(owner, spec.n_bytes)

    def configure_prefetch_limits(
        self, limits: Sequence[int | None]
    ) -> None:
        """Select the per-request profile before starting DiT prefetch."""

        if len(limits) != 2:
            raise ValueError("prefetch limits must contain two values")
        values = tuple(
            None if value is None else max(0, int(value)) for value in limits
        )
        with self._lock:
            if self._state in {
                AsyncVAEState.PREFETCHING,
                AsyncVAEState.PREFETCHED,
                AsyncVAEState.CAPPED,
                AsyncVAEState.FINALIZING,
                AsyncVAEState.VAE_FINALIZE,
                AsyncVAEState.FALLBACK_LOADING,
            }:
                raise RuntimeError("cannot change VAE limits during an active load")
            self.prefetch_limits = values

    def _load_names(self, names: Iterable[str], *, capped: bool) -> set[torch.device]:
        blocked: set[torch.device] = set()
        ordered = self._ordered_names(names)
        with self._reader_factory(
            self.path,
            self.devices,
            self._streams,
            staging_bytes=self.staging_bytes,
        ) as reader:
            for name in ordered:
                if self._cancel.is_set():
                    raise RuntimeError("async VAE load cancelled")
                with self._lock:
                    if name in self._loaded:
                        continue
                spec = self.specs[name]
                owner = self._owner(name)
                if capped and (
                    owner in blocked or not self._admit_prefetch(spec, owner)
                ):
                    blocked.add(owner)
                    continue
                if self._value_loader is None:
                    value, token = reader.read(spec, owner)
                else:
                    # INT8-ConvRot can use the same residency state machine by
                    # supplying a loader that pairs qdata/scale/quant markers
                    # into a QuantizedTensor.  The default FP16 path remains a
                    # direct ordinary safetensors copy.
                    value, token = self._value_loader(reader, spec, owner)
                self._tensor_setter(self._model, name, value)
                with self._lock:
                    self._loaded.add(name)
                    self._tokens[name] = token
                    self._resident[owner] += spec.n_bytes
                    self._read_bytes[owner] += spec.n_bytes
        return blocked

    def _wait_loaded(self) -> None:
        with self._lock:
            tokens = tuple(self._tokens.values())
        for token in tokens:
            token.wait()

    def ensure_encoder_ready(self) -> None:
        """Synchronously load only values required by reference-image encode."""

        if self.state == AsyncVAEState.READY:
            # A subsequent request starts with Qwen/reference encoding.  Do
            # not leave the previous request's decoder resident beside Qwen.
            self.release_decoder(keep_encoder=True)
            return
        if self.state == AsyncVAEState.ENCODER_READY:
            return
        with self._operation_lock:
            if self.state in {AsyncVAEState.ENCODER_READY, AsyncVAEState.READY}:
                return
            if self.state not in {AsyncVAEState.META_ONLY}:
                raise RuntimeError(
                    f"VAE encoder must run before DiT prefetch; state={self.state.value}"
                )
            self._set_state(AsyncVAEState.ENCODER_LOADING)
            try:
                names = [name for name in self.specs if _is_encoder_value(name)]
                self._load_names(names, capped=False)
                self._wait_loaded()
                self._set_state(AsyncVAEState.ENCODER_READY)
            except BaseException as exc:
                self._failure = exc
                self._set_state(AsyncVAEState.FAILED)
                self._run_fallback()

    def begin_prefetch(self) -> bool:
        """Start the bounded worker after the Qwen-clear barrier.

        Returns ``True`` only when this call created the worker.  Repeated calls
        are harmless and never create a second loader thread.
        """

        with self._lock:
            if self._state in {
                AsyncVAEState.PREFETCHING,
                AsyncVAEState.PREFETCHED,
                AsyncVAEState.CAPPED,
                AsyncVAEState.FINALIZING,
                AsyncVAEState.FALLBACK_LOADING,
                AsyncVAEState.READY,
            }:
                return False
            if self._state not in {
                AsyncVAEState.META_ONLY,
                AsyncVAEState.ENCODER_READY,
                AsyncVAEState.DIT_READY,
                AsyncVAEState.DENOISING,
            }:
                raise RuntimeError(f"cannot prefetch VAE from state {self._state.value}")
            self._state = AsyncVAEState.PREFETCHING
            self._prefetch_started_at = time.monotonic()
            self._cancel.clear()
            thread = threading.Thread(
                target=self._prefetch_main,
                name="h3-async-vae",
                daemon=True,
            )
            self._prefetch_thread = thread
            thread.start()
            return True

    def _prefetch_main(self) -> None:
        try:
            with self._lock:
                remaining = set(self.specs).difference(self._loaded)
            blocked = self._load_names(remaining, capped=True)
            with self._lock:
                deferred = len(self.specs) - len(self._loaded)
                self._prefetch_finished_at = time.monotonic()
                self._state = (
                    AsyncVAEState.CAPPED
                    if deferred or blocked
                    else AsyncVAEState.PREFETCHED
                )
        except BaseException as exc:
            logging.exception("[H3 async VAE] background prefetch failed")
            with self._lock:
                self._failure = exc
                self._prefetch_finished_at = time.monotonic()
                self._state = AsyncVAEState.FAILED
            # Do not leave a half-materialised model consuming the safety
            # margin for the remaining DiT steps.  Rebuild metadata now; the
            # synchronous payload fallback itself still waits for finalize.
            self._discard_partial()

    def _join_prefetch(self) -> None:
        with self._lock:
            thread = self._prefetch_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def finalize_tail(self) -> "H3AsyncVAE":
        """After the last DiT forward, wait submitted copies and fill the tail."""

        if self.state == AsyncVAEState.READY:
            return self.vae
        if self.state in {AsyncVAEState.META_ONLY, AsyncVAEState.ENCODER_READY}:
            self.begin_prefetch()
        self._join_prefetch()
        with self._operation_lock:
            if self.state == AsyncVAEState.READY:
                return self.vae
            if self._failure is not None or self.state == AsyncVAEState.FAILED:
                self._run_fallback()
                return self.vae

            self._set_state(AsyncVAEState.VAE_FINALIZE)
            self._finalize_started_at = time.monotonic()
            try:
                # Copies already submitted during DiT finish before new tail IO.
                self._wait_loaded()
                with self._lock:
                    remaining = set(self.specs).difference(self._loaded)
                self._load_names(remaining, capped=False)
                self._wait_loaded()
                self._model = self._model_finalizer(
                    self._model, self.devices, self.split
                )
                self._finalize_finished_at = time.monotonic()
                self._set_state(AsyncVAEState.READY)
            except BaseException as exc:
                logging.exception("[H3 async VAE] tail finalize failed; using fallback")
                self._failure = exc
                self._set_state(AsyncVAEState.FAILED)
                self._run_fallback()
        return self.vae

    def await_ready(self) -> "H3AsyncVAE":
        """Decode gate: finalize any deferred tensors and return the facade."""

        return self.finalize_tail()

    def mark_dit_ready(self) -> None:
        """Record the Qwen-clear barrier before DiT starts."""

        with self._lock:
            if self._state in {AsyncVAEState.ENCODER_READY, AsyncVAEState.META_ONLY}:
                self._state = AsyncVAEState.DIT_READY

    def mark_denoising(self, *, start_prefetch: bool = True) -> bool:
        """Record DiT execution and optionally launch capped VAE prefetch."""

        with self._lock:
            if self._state in {AsyncVAEState.ENCODER_READY, AsyncVAEState.META_ONLY}:
                self._state = AsyncVAEState.DENOISING
            elif self._state == AsyncVAEState.DIT_READY:
                self._state = AsyncVAEState.DENOISING
        return self.begin_prefetch() if start_prefetch else False

    def after_qwen_clear(self) -> bool:
        """Barrier callback: Qwen payload is gone, so DiT overlap may begin."""

        self.mark_dit_ready()
        return self.begin_prefetch()

    on_dit_start = after_qwen_clear

    # Lifecycle aliases keep integration call sites descriptive without
    # coupling them to a particular sampler callback name.
    start_prefetch = begin_prefetch
    on_last_dit_forward = finalize_tail

    def cancel(self) -> None:
        self._cancel.set()
        self._join_prefetch()
        self._wait_loaded()
        self._set_state(AsyncVAEState.CANCELLED)

    def release_decoder(self, *, keep_encoder: bool = True) -> None:
        """Release CUDA decoder payload before the next Qwen encode.

        The completed decoder contains the overwhelming majority of VAE
        storage.  Rebuilding a meta-only module tree and reusing the already
        loaded encoder tensors avoids disk IO for another reference encode,
        while ``empty_cache`` makes the released decoder driver-visible before
        Qwen starts allocating its compressed shards.
        """

        with self._operation_lock:
            if self.state in {
                AsyncVAEState.PREFETCHING,
                AsyncVAEState.FINALIZING,
                AsyncVAEState.VAE_FINALIZE,
            }:
                raise RuntimeError("cannot release VAE decoder while loading is active")
            self._join_prefetch()
            self._wait_loaded()
            for device in self.devices:
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.synchronize(device)

            old_model = self._model
            new_model = self._new_model()
            kept: set[str] = set()
            if keep_encoder:
                with self._lock:
                    candidates = [
                        name
                        for name in self._loaded
                        if _is_encoder_value(name)
                    ]
                for name in candidates:
                    self._tensor_setter(new_model, name, _get_tensor(old_model, name))
                    kept.add(name)

            with self._lock:
                self._model = new_model
                self._loaded = kept
                self._tokens = {name: _CopyToken() for name in kept}
                self._resident = {device: 0 for device in self.devices}
                for name in kept:
                    self._resident[self._owner(name)] += self.specs[name].n_bytes
                self._failure = None
                self._prefetch_thread = None
                self._prefetch_started_at = None
                self._prefetch_finished_at = None
                self._finalize_started_at = None
                self._finalize_finished_at = None
                self._state = (
                    AsyncVAEState.ENCODER_READY
                    if kept
                    else AsyncVAEState.META_ONLY
                )
            del old_model
            gc.collect()
            for device in self.devices:
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()

    prepare_for_qwen = release_decoder

    def _discard_partial(self) -> None:
        self._cancel.set()
        self._join_prefetch()
        try:
            self._wait_loaded()
        except Exception:
            pass
        with self._lock:
            self._model = None  # type: ignore[assignment]
            self._loaded.clear()
            self._tokens.clear()
            self._resident = {device: 0 for device in self.devices}
        gc.collect()
        for device in self.devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        with self._lock:
            self._model = self._new_model()
        self._cancel.clear()

    def _run_fallback(self) -> None:
        """Drop the partial model and perform one fresh synchronous no-mmap load."""

        self._set_state(AsyncVAEState.FALLBACK_LOADING)
        original = self._failure
        try:
            self._discard_partial()
            if self._synchronous_fallback is not None:
                self._model = self._synchronous_fallback()
                with self._lock:
                    self._loaded = set(self.specs)
                    self._tokens = {
                        name: _CopyToken() for name in self.specs
                    }
                    self._resident = dict(self._requested)
                    for device in self.devices:
                        self._read_bytes[device] += self._requested[device]
            else:
                self._load_names(self.specs, capped=False)
                self._wait_loaded()
                self._model = self._model_finalizer(
                    self._model, self.devices, self.split
                )
            self._fallback_count += 1
            self._failure = None
            self._finalize_finished_at = time.monotonic()
            self._set_state(AsyncVAEState.READY)
        except BaseException as fallback_exc:
            self._failure = fallback_exc
            self._set_state(AsyncVAEState.FAILED)
            raise RuntimeError(
                "H3 async VAE load and synchronous no-mmap fallback both failed"
            ) from fallback_exc
        finally:
            if original is not None:
                logging.warning(
                    "[H3 async VAE] discarded partial handle after %s and ran "
                    "synchronous no-mmap fallback",
                    original,
                )

    def stats(self) -> dict[str, object]:
        with self._lock:
            requested = {
                str(device): int(self._requested[device]) for device in self.devices
            }
            resident = {
                str(device): int(self._resident[device]) for device in self.devices
            }
            read_bytes = {
                str(device): int(self._read_bytes[device]) for device in self.devices
            }
            deferred = {
                str(device): int(self._requested[device] - self._resident[device])
                for device in self.devices
            }
            failure = None if self._failure is None else repr(self._failure)
            state = self._state.value
            loaded_tensors = len(self._loaded)
        ledger = {}
        for device in self.devices:
            try:
                snapshot = self._ledger.snapshot(device)
                ledger[str(device)] = {
                    "allocated_bytes": snapshot.allocated,
                    "reserved_bytes": snapshot.reserved,
                    "driver_free_bytes": snapshot.driver_free,
                    "total_bytes": snapshot.total,
                    "available_after_safety_bytes": snapshot.available,
                }
            except Exception as exc:
                ledger[str(device)] = {"error": repr(exc)}
        prefetch_seconds = None
        if self._prefetch_started_at is not None and self._prefetch_finished_at is not None:
            prefetch_seconds = self._prefetch_finished_at - self._prefetch_started_at
        finalize_seconds = None
        if self._finalize_started_at is not None and self._finalize_finished_at is not None:
            finalize_seconds = self._finalize_finished_at - self._finalize_started_at
        return {
            "state": state,
            "path": self.path,
            "split": self.split,
            "host_mmap": False,
            "safety_bytes": self._ledger.safety_bytes,
            "staging_bytes_per_device": self.staging_bytes,
            "prefetch_limits": [
                None if value is None else int(value) for value in self.prefetch_limits
            ],
            "requested_bytes": requested,
            "resident_bytes": resident,
            "payload_bytes_read": read_bytes,
            "deferred_bytes": deferred,
            "memory_ledger": ledger,
            "loaded_tensors": loaded_tensors,
            "requested_tensors": len(self.specs),
            "prefetch_seconds": prefetch_seconds,
            "finalize_seconds": finalize_seconds,
            "fallback_count": self._fallback_count,
            "failure": failure,
        }


class _AsyncVAEPatcher:
    """Patcher-shaped diagnostics object that never moves the MP model tree."""

    def __init__(self, handle: AsyncVAEHandle):
        self.handle = handle
        self.load_device = handle.devices[0]
        self.offload_device = handle.devices[0]
        self._h3_parallel_resident = True

    @property
    def model(self) -> nn.Module:
        return self.handle.model

    def is_dynamic(self) -> bool:
        return False

    def model_size(self) -> int:
        return sum(self.handle._requested.values())

    def loaded_size(self) -> int:
        return sum(self.handle._resident.values())

    def current_loaded_device(self) -> torch.device:
        return self.load_device


class H3AsyncVAE:
    """ComfyUI-compatible facade over an encoder-first asynchronous handle."""

    def __init__(self, handle: AsyncVAEHandle):
        self.async_handle = handle
        self.device = handle.devices[0]
        self.parallel_devices = handle.devices
        self.vae_dtype = torch.float16
        self.latent_channels = 24
        self.latent_dim = 3
        self.output_channels = 3
        self.pad_channel_value = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: image
        self.crop_input = True
        self.handles_tiling = True
        self.comfy_has_chunked_io = True
        self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
        self.downscale_ratio = (
            lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1,
            16,
            16,
        )
        self.upscale_index_formula = (4, 16, 16)
        self.downscale_index_formula = (4, 16, 16)
        self.path = handle.path
        self.size = sum(handle._requested.values())
        self.patcher = _AsyncVAEPatcher(handle)

    @property
    def first_stage_model(self) -> nn.Module:
        return self.async_handle.model

    @property
    def output_device(self) -> torch.device:
        raw = os.environ.get("H3_VAE_OUTPUT_DEVICE", "cpu").strip().lower()
        if raw in {"", "cpu", "host", "auto"}:
            return torch.device("cpu")
        try:
            target = torch.device(raw)
        except RuntimeError as exc:
            raise ValueError(
                "H3_VAE_OUTPUT_DEVICE must be cpu, auto, or one of the MP CUDA devices"
            ) from exc
        if target not in self.parallel_devices:
            raise ValueError(
                "H3_VAE_OUTPUT_DEVICE must be cpu/auto or one of "
                f"{self.parallel_devices[0]}, {self.parallel_devices[1]}; got {raw!r}"
            )
        return target

    def throw_exception_if_invalid(self) -> None:
        if self.async_handle.state in {AsyncVAEState.FAILED, AsyncVAEState.CANCELLED}:
            raise RuntimeError(f"H3 async VAE is {self.async_handle.state.value}")

    def model_size(self) -> int:
        return self.size

    def is_dynamic(self) -> bool:
        return False

    def vae_output_dtype(self) -> torch.dtype:
        import comfy.model_management as mm

        return mm.intermediate_dtype()

    def decode_output_dtype(self) -> torch.dtype:
        """Choose the host video canvas dtype without a second full copy.

        ComfyUI's default intermediate dtype is FP32.  A 720p/243-frame
        ``IMAGE`` canvas is then about 2.56 GiB, and allocating it while the
        service is under a 7 GiB cgroup high limit can put the process into
        direct reclaim for minutes.  The decoder writes each finalized tile
        with ``copy_`` (which safely casts), so an FP16 canvas is sufficient
        for the 8-bit video output and cuts the live host buffer in half.
        Keep an explicit escape hatch for numerical A/B checks.
        """
        requested = os.environ.get("H3_ASYNC_VAE_OUTPUT_DTYPE", "fp16").strip().lower()
        if requested in {"fp16", "float16", "half"}:
            return torch.float16
        if requested in {"fp32", "float32", "full"}:
            return torch.float32
        raise ValueError(
            "H3_ASYNC_VAE_OUTPUT_DTYPE must be fp16 or fp32, "
            f"got {requested!r}"
        )

    def spacial_compression_decode(self) -> int:
        return 16

    def spacial_compression_encode(self) -> int:
        return 16

    def temporal_compression_decode(self) -> int:
        return 4

    def vae_encode_crop_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.crop_input:
            dims = pixels.shape[1:-1]
            for index, dim in enumerate(dims):
                target = (dim // 16) * 16
                offset = (dim - target) // 2
                if target != dim:
                    pixels = pixels.narrow(index + 1, offset, target)
        if pixels.shape[-1] > self.output_channels:
            pixels = pixels[..., : self.output_channels]
        return pixels

    @torch.no_grad()
    def encode(self, pixel_samples: torch.Tensor) -> torch.Tensor:
        self.async_handle.ensure_encoder_ready()
        self.async_handle._set_state(AsyncVAEState.ENCODING)
        try:
            pixels = self.vae_encode_crop_pixels(pixel_samples).movedim(-1, 1)
            if pixels.ndim < 5:
                pixels = pixels.unsqueeze(2)
            pixels = self.process_input(pixels).to(
                device=self.device, dtype=self.vae_dtype
            )
            latent = self.first_stage_model.encode(pixels, device=self.device)
            return latent.to(
                device=self.output_device, dtype=self.vae_output_dtype(), copy=True
            )
        finally:
            self.async_handle._set_state(AsyncVAEState.ENCODER_READY)

    @torch.no_grad()
    def decode(self, samples_in: torch.Tensor, **kwargs) -> torch.Tensor:
        self.async_handle.await_ready()
        if samples_in.ndim != 5:
            raise ValueError(f"H3 video VAE expects [B,C,T,H,W], got {samples_in.shape}")
        self.async_handle._set_state(AsyncVAEState.DECODING)
        try:
            samples = samples_in.to(device=self.device, dtype=self.vae_dtype)
            shape = self.first_stage_model.decode_output_shape(samples.shape)
            output = torch.empty(
                shape,
                device=self.output_device,
                dtype=self.decode_output_dtype(),
            )
            self.first_stage_model.decode(samples, output_buffer=output, **kwargs)
            # ``output`` is the complete host video canvas.  Return it directly
            # instead of forcing ComfyUI's intermediate dtype: when that dtype
            # differs, a full conversion canvas would be live at the same time
            # and put the process straight back into cgroup reclaim.  IMAGE
            # consumers accept FP16, and the dtype is explicit/diagnosable
            # through H3_ASYNC_VAE_OUTPUT_DTYPE (fp16 by default, fp32 for A/B
            # checks).
            return output.movedim(1, -1)
        finally:
            self.async_handle._set_state(AsyncVAEState.READY)

    def encode_tiled(self, pixel_samples: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encode(pixel_samples)

    def decode_tiled(self, samples_in: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.decode(samples_in, **kwargs)

    def begin_prefetch(self) -> bool:
        return self.async_handle.begin_prefetch()

    def after_qwen_clear(self) -> bool:
        return self.async_handle.after_qwen_clear()

    on_dit_start = after_qwen_clear

    def finalize_tail(self) -> "H3AsyncVAE":
        self.async_handle.finalize_tail()
        return self

    def await_ready(self) -> "H3AsyncVAE":
        self.async_handle.await_ready()
        return self

    def prepare_for_qwen(self, *, keep_encoder: bool = True) -> None:
        self.async_handle.prepare_for_qwen(keep_encoder=keep_encoder)

    def async_stats(self) -> dict[str, object]:
        return self.async_handle.stats()


_HANDLE_CACHE: dict[tuple[str, str, str, int], AsyncVAEHandle] = {}
_HANDLE_CACHE_LOCK = threading.Lock()


def _production_devices(
    devices: Sequence[torch.device | str] | None,
) -> tuple[torch.device, torch.device] | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None
    values = devices or ("cuda:0", "cuda:1")
    if len(values) != 2:
        raise ValueError("H3 async VAE requires two CUDA devices")
    pair = tuple(torch.device(value) for value in values)
    if any(device.type != "cuda" for device in pair) or pair[0] == pair[1]:
        raise ValueError("H3 async VAE production devices must be distinct CUDA devices")
    if not torch.cuda.can_device_access_peer(pair[0].index, pair[1].index):
        raise RuntimeError("H3 async VAE requires CUDA peer access from rank0 to rank1")
    if not torch.cuda.can_device_access_peer(pair[1].index, pair[0].index):
        raise RuntimeError("H3 async VAE requires CUDA peer access from rank1 to rank0")
    return pair  # type: ignore[return-value]


def _env_prefetch_limits() -> tuple[int | None, int | None]:
    default = ",".join(str(value) for value in FP16_TWO_REFERENCE_PREFETCH_MIB)
    raw = os.environ.get("H3_ASYNC_VAE_PREFETCH_MIB", default).strip()
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("H3_ASYNC_VAE_PREFETCH_MIB must contain two values")
    result: list[int | None] = []
    for part in parts:
        result.append(None if part.lower() in {"", "none", "auto"} else int(part) * MIB)
    return result[0], result[1]


def get_or_create_async_vae_handle(
    path: os.PathLike[str] | str,
    devices: Sequence[torch.device | str] | None = None,
    *,
    split: int | None = None,
    prefetch_limits: Sequence[int | None] | None = None,
) -> AsyncVAEHandle | None:
    """Return the process-local production handle, or ``None`` without CUDA."""

    pair = _production_devices(devices)
    if pair is None:
        return None
    if prefetch_limits is None:
        prefetch_limits = _env_prefetch_limits()
    limits = tuple(prefetch_limits)
    resolved = _resolve_no_mmap_path(path)

    # Inspect the header before choosing a split: the two checkpoint routes want
    # different defaults.  This is a header-only read of a few hundred KiB, not
    # payload.  The handle re-reads it in its constructor; keeping that
    # signature unchanged is worth more than saving one header parse.
    #
    # Imported here, not at module scope: the adapter imports this module for
    # its reader/spec/token types, so a top-level import would be circular.
    try:
        from . import h3_async_vae_int8 as int8_async
    except ImportError:
        from custom_nodes.DualV100 import h3_async_vae_int8 as int8_async

    specs, _metadata = inspect_safetensors(resolved)
    is_int8 = int8_async.is_int8_convrot_vae(specs)

    if split is None:
        # A deferred decoder is materialised after the sampler has returned and
        # is never rebalanced, so it has exactly one layout for its whole life
        # and that layout should be the decode-optimal one.  Measured at 720p,
        # 28 tiles/chunk: split=18 decodes in 3540 ms vs 4357 ms at 24 and
        # 4066 ms at 14, and 18 also balances resident bytes best
        # (2102/2222 MiB vs 2488/1838).  The FP16 route keeps its historical 24
        # because that path does rebalance between stages.
        default_split = 18 if is_int8 else 24
        raw_split = (
            os.environ.get("H3_VAE_SPLIT")
            or os.environ.get("H3_VAE_DECODE_SPLIT")
            or str(default_split)
        ).strip().lower()
        split = (
            default_split
            if raw_split in {"", "auto", "balanced", "default"}
            else int(raw_split)
        )
    split = min(int(split), 35)

    key = (resolved, str(pair[0]), str(pair[1]), int(split))
    with _HANDLE_CACHE_LOCK:
        cached = _HANDLE_CACHE.get(key)
        if cached is not None and cached.state not in {
            AsyncVAEState.FAILED,
            AsyncVAEState.CANCELLED,
        }:
            cached.configure_prefetch_limits(limits)
            return cached
        safety_bytes = int(os.environ.get("H3_ASYNC_VAE_SAFETY_MIB", "1024")) * MIB
        staging_mib = int(os.environ.get("H3_ASYNC_VAE_STAGING_MIB", "4"))
        if not 4 <= staging_mib <= 8:
            raise ValueError("H3_ASYNC_VAE_STAGING_MIB must be between 4 and 8")
        extra: dict[str, object] = {}
        if is_int8:
            # The INT8-ConvRot checkpoint cannot use the FP16 defaults: its
            # weights are I8 qdata paired with F32 scales and a comfy_quant
            # marker, and they need mixed_precision_ops rather than
            # disable_weight_init to host a QuantizedTensor.
            extra = int8_async.build_int8_async_kwargs(specs, pair)
            logging.info(
                "[H3 async VAE] INT8-ConvRot checkpoint detected; decoder "
                "deferred past the DiT peak, single layout split=%d",
                int(split),
            )
        handle = AsyncVAEHandle(
            resolved,
            pair,
            split=int(split),
            safety_bytes=safety_bytes,
            staging_bytes=staging_mib * MIB,
            prefetch_limits=limits,
            **extra,
        )
        _HANDLE_CACHE[key] = handle
        return handle


def load_h3_video_vae_async(
    path: os.PathLike[str] | str,
    devices: Sequence[torch.device | str] | None = None,
    *,
    split: int | None = None,
    prefetch_limits: Sequence[int | None] | None = None,
) -> H3AsyncVAE | None:
    """Opt-in loader entry point used by the Qwen32B lifecycle integration."""

    if not _enabled("H3_ASYNC_VAE_LOAD", False):
        return None
    handle = get_or_create_async_vae_handle(
        path, devices, split=split, prefetch_limits=prefetch_limits
    )
    return None if handle is None else handle.vae


__all__ = [
    "AsyncVAEHandle",
    "AsyncVAEState",
    "BoundedSafeTensorReader",
    "CUDAMemoryLedger",
    "DEFAULT_SAFETY_BYTES",
    "DEFAULT_STAGING_BYTES",
    "FP16_TWO_REFERENCE_PREFETCH_MIB",
    "H3AsyncVAE",
    "MemorySnapshot",
    "SafeTensorSpec",
    "get_or_create_async_vae_handle",
    "h3_vae_owner",
    "inspect_safetensors",
    "load_h3_video_vae_async",
]
