"""GGUF metadata reader which never maps the model data section.

``gguf.GGUFReader`` currently creates a numpy.memmap for the complete file.
That is a poor fit for a host with little RAM: the mapping is virtual, but the
loader and the kernel page cache can still create a large resident set.  H3
only needs GGUF metadata and tensor offsets while constructing a DynamicVRAM
state dict, so this reader parses a bounded header prefix and leaves tensor
payloads on disk.

The returned ``ReaderTensor.data`` is intentionally ``None``.  The patched
ComfyUI-GGUF loader creates a meta tensor of the correct storage shape and
attaches a ``TensorFileSlice`` to it.  AIMDO then reads the exact range into
GPU memory when the layer is used.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

from gguf.gguf_reader import (
    GGUFReader as _GGUFReader,
    GGML_QUANT_SIZES,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGMLQuantizationType,
    GGUFValueType,
    GGUFEndian,
    READER_SUPPORTED_VERSIONS,
    ReaderField,
    ReaderTensor,
    quant_shape_to_byte_shape,
)


class NoMmapGGUFReader(_GGUFReader):
    """Parse GGUF header/metadata from ordinary bytes, never ``np.memmap``."""

    alignment: int = GGUF_DEFAULT_ALIGNMENT

    def __init__(self, path: os.PathLike[str] | str, mode: Literal["r"] = "r"):
        if mode != "r":
            raise ValueError("NoMmapGGUFReader is read-only")

        path = os.fspath(path)
        file_size = os.path.getsize(path)
        # GGUF headers are normally a few MB even when the payload is many GB.
        # Grow only when the parser proves that the prefix is incomplete, and
        # never allow malformed metadata to make us read an entire model.
        prefix_size = min(file_size, 1 << 20)
        max_prefix = min(file_size, 256 << 20)
        last_error = None

        while True:
            with open(path, "rb") as handle:
                raw = handle.read(prefix_size)
            reader = object.__new__(type(self))
            try:
                reader._parse_header_prefix(raw, file_size)
            except IndexError as error:
                # gguf's parser assumes that ``data`` is a complete memmap and
                # indexes the result of a bounded slice directly.  With a
                # prefix-only reader an incomplete string/array therefore
                # appears as IndexError instead of a clean "need more bytes"
                # signal.  Retry with a larger *header* prefix only; never fall
                # back to GGUFReader/np.memmap and never read the payload.
                last_error = error
                if prefix_size >= max_prefix:
                    raise RuntimeError(
                        "GGUF header could not be parsed within the safe "
                        f"{max_prefix // (1 << 20)} MiB limit: {path}"
                    ) from error
                prefix_size = min(max_prefix, prefix_size * 2)
                continue
            if reader.data_offset <= len(raw):
                self.__dict__.update(reader.__dict__)
                self._header_bytes = raw
                self._payload_path = path
                return

            last_error = RuntimeError(
                f"GGUF header extends past prefix ({reader.data_offset} > {len(raw)})"
            )
            if prefix_size >= max_prefix:
                raise RuntimeError(
                    f"GGUF header is larger than the safe {max_prefix // (1 << 20)} MiB limit: {path}"
                ) from last_error
            prefix_size = min(max_prefix, prefix_size * 2)

    def _parse_header_prefix(self, raw: bytes, file_size: int) -> None:
        # This mirrors GGUFReader.__init__, but self.data is only the bounded
        # header prefix and _build_tensors below never touches payload bytes.
        self.data = np.frombuffer(raw, dtype=np.uint8)
        offs = 0

        if self._get(offs, np.uint32, override_order="<")[0] != GGUF_MAGIC:
            raise ValueError("GGUF magic invalid")
        offs += 4

        temp_version = self._get(offs, np.uint32)
        if temp_version[0] & 65535 == 0:
            self.byte_order = "S"
            temp_version = temp_version.view(temp_version.dtype.newbyteorder(self.byte_order))
        version = temp_version[0]
        if version not in READER_SUPPORTED_VERSIONS:
            raise ValueError(f"Sorry, file appears to be version {version} which we cannot handle")
        if np.little_endian:
            host_endian = GGUFEndian.LITTLE
            swapped_endian = GGUFEndian.BIG
        else:
            host_endian = GGUFEndian.BIG
            swapped_endian = GGUFEndian.LITTLE
        self.endianess = swapped_endian if self.byte_order == "S" else host_endian
        self.fields = {}
        self.tensors = []
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        offs += self._push_field(
            ReaderField(offs, "GGUF.version", [temp_version], [0], [GGUFValueType.UINT32])
        )

        temp_counts = self._get(offs, np.uint64, 2)
        offs += self._push_field(
            ReaderField(offs, "GGUF.tensor_count", [temp_counts[:1]], [0], [GGUFValueType.UINT64])
        )
        offs += self._push_field(
            ReaderField(offs, "GGUF.kv_count", [temp_counts[1:]], [0], [GGUFValueType.UINT64])
        )
        tensor_count, kv_count = (int(x) for x in temp_counts)
        offs = self._build_fields(offs, kv_count)
        offs, tensor_fields = self._build_tensor_info(offs, tensor_count)

        new_align = self.fields.get("general.alignment")
        if new_align is not None:
            if new_align.types != [GGUFValueType.UINT32]:
                raise ValueError("Bad type in general.alignment field")
            self.alignment = int(new_align.parts[-1][0])
            if self.alignment == 0 or (self.alignment & (self.alignment - 1)) != 0:
                raise ValueError("Invalid alignment: must be a non-zero power of two")
        padding = offs % self.alignment
        if padding:
            offs += self.alignment - padding
        self.data_offset = offs
        if self.data_offset > file_size:
            raise ValueError("GGUF header points beyond end of file")
        self._build_tensors(offs, tensor_fields)

    def _build_tensors(self, start_offs: int, fields: list[ReaderField]) -> None:
        tensors = []
        tensor_names = set()
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            tensor_name = str(bytes(name_data), encoding="utf-8")
            if tensor_name in tensor_names:
                raise ValueError(f"Found duplicated tensor with name {tensor_name}")
            tensor_names.add(tensor_name)

            ggml_type = GGMLQuantizationType(int(raw_dtype[0]))
            n_elems = int(np.prod(dims))
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            data_offset = int(start_offs + offset_tensor[0])
            tensors.append(
                ReaderTensor(
                    name=tensor_name,
                    tensor_type=ggml_type,
                    shape=dims,
                    n_elements=n_elems,
                    n_bytes=n_bytes,
                    data_offset=data_offset,
                    data=None,
                    field=field,
                )
            )
        self.tensors = tensors


def meta_tensor_for_gguf(tensor, torch):
    """Return a zero-RAM meta tensor with the GGUF tensor's storage geometry."""

    qtype = tensor.tensor_type
    shape = tuple(int(value) for value in reversed(tensor.shape))
    scalar_dtypes = {
        GGMLQuantizationType.F32: torch.float32,
        GGMLQuantizationType.F16: torch.float16,
        GGMLQuantizationType.BF16: torch.bfloat16,
        GGMLQuantizationType.F64: torch.float64,
        GGMLQuantizationType.I8: torch.int8,
        GGMLQuantizationType.I16: torch.int16,
        GGMLQuantizationType.I32: torch.int32,
        GGMLQuantizationType.I64: torch.int64,
    }
    dtype = scalar_dtypes.get(qtype)
    if dtype is not None:
        result = torch.empty(shape, dtype=dtype, device="meta")
    else:
        storage_shape = quant_shape_to_byte_shape(shape, qtype)
        result = torch.empty(storage_shape, dtype=torch.uint8, device="meta")

    expected = int(result.numel() * result.element_size())
    if expected != int(tensor.n_bytes):
        raise ValueError(
            f"GGUF tensor {tensor.name!r} storage mismatch: meta={expected}, file={tensor.n_bytes}"
        )
    return result
