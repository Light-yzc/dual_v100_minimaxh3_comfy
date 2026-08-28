# Adaptive Group Cache Q4_0 implementation notes

This file records implementation details and measured results. The research
requirements remain unchanged in `docs/TP_SPEED_FUTHUR.MD`.

## Implemented route

- The persistent two-rank H3 backbone remains standard GGML Q4_0 plus the
  existing Turbo LoRA shards.
- Whole-tail TE-Speed stores its one tail residual as standard GGML Q4_0.
- Adaptive Group Cache stores, for every group, both `previous_input` and
  `residual` as standard GGML Q4_0.
- The default cache policy is CPU. Q4 bytes are transferred and dequantized in
  bounded row chunks, so there is no complete dequantized cache tensor and no
  persistent cache VRAM allocation.
- H3's later FP32 residual groups can exceed the finite range of Q4_0's FP16
  block scale. The cache wrapper applies a tensor-wide power-of-two pre-scale
  when needed and stores only the integer restore exponent beside the raw
  payload. The payload remains standard GGML Q4_0; decoding restores the scale
  per bounded chunk without retaining an FP32 cache copy.
- No cache tensor is mmap-backed or written to disk.
- The rank-0 decision is broadcast before each group; rank 1 validates its
  independently computed feature error and cache state before entering the
  next TP collective sequence.

The default 50-block partition is:

```text
[0, 8)   always compute
[8, 18)  group 0
[18, 28) group 1
[28, 38) group 2
[38, 50) group 3
```

## Feature metric and threshold

The decision compares:

```text
Q4_0(current group input) vs Q4_0(previous group input)
```

It does not compare live FP32 against a Q4 reference. The mixed FP32/Q4
comparison has a reconstruction-error floor even when the underlying tensor is
unchanged; that would make a small threshold unusable. Q4-vs-Q4 gives exactly
zero for an unchanged input.

The default metric is `relative_l1` and the initial conservative threshold is
`0.005`, reduced from the earlier `0.01` draft. This is only a starting point;
the research matrix must still sweep `0.005, 0.01, 0.02, ...` rather than claim
one default is optimal.

## Memory geometry

Standard Q4_0 stores 32 values in 18 bytes. Relative to an FP32 hidden tensor:

```text
compression = 128 / 18 = 7.1111x
```

For the current 1 MP estimate (`sequence=37746`, `hidden=5376`), per rank:

```text
one FP32 hidden tensor                  811,689,984 bytes
one Q4_0 hidden tensor                 114,143,904 bytes
whole-tail residual cache              114,143,904 bytes
4 groups x (previous_input + residual) 913,151,232 bytes
```

The 4-group value is CPU memory with the default policy. It replaces eight
FP32 persistent tensors that would require about 6.05 GiB per rank. A FULL
group still needs one transient FP32 input snapshot to construct the exact
residual before Q4 quantization; it is released immediately and is not part of
the persistent cache.

When a group refreshes, stale Q4 input/residual buffers are released before
allocating the replacement residual. This avoids keeping old input, old
residual, current input and new residual together at 1 MP.

## Offline correctness test

Run:

```bash
/home/regen/minimax-h3/.venv/bin/python \
  scripts/test_h3_q4_cache.py \
  --output results/h3_q4_cache_offline_20260827.json
```

Current result:

- encoded bytes equal `gguf.quantize(..., Q4_0)` byte for byte;
- dequantized output vs the gguf reference: `max_abs=0`;
- bounded chunked residual add vs the reference: `max_abs=0`;
- identical Q4-vs-Q4 feature error is exactly zero for relative L1, relative
  L2 and cosine distance;
- default group boundaries are exactly `[8,18), [18,28), [28,38), [38,50)`;
- computed cache bytes equal the Q4_0 geometry exactly.

Result JSON:

`results/h3_q4_cache_offline_20260827.json`

## Runtime safety

- `cache_format` is normalized and only `Q4_0`/`ggml_q4_0` is accepted.
- `benchmark_ground_truth` has a hidden-size guard before it is allowed to
  clone and secretly run a true group.
- scalar block statistics and the oracle are disabled by default for normal
  performance runs.
- Whole-tail TE-Speed and Adaptive Group Cache remain mutually exclusive.
- Any rank/state disagreement fails before the next block-range collective.

## Hardware validation status

Hardware measurements must be appended after deploying the source copy and
running the guarded 448x256 TP smoke. Do not infer final quality or a Pareto
advantage from the CPU-only format test.
