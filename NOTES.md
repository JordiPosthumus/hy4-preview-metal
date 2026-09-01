# Hy4-preview STQ1_0 on M3 Ultra — work log

## Status: READY TO LOAD (weights downloaded+verified, runtime built+validated, model NOT loaded yet)

## What's here
- `llama.cpp/` — pinned @ 0cea36222 + AngelSlim hyv4/STQ1_0 patches + custom Metal kernels
  (exportable patch: `patches-applied/0003-custom-metal-stq1_0-kernels.patch`)
- `weights/Hy4-preview-STQ1_0.gguf` — 229.4GB, size verified against HF manifest
- `bench/` — standalone Metal kernel A/B harness (runtime shader compile, polled w/ timeouts)
- `test_stq1_0_metal` — CPU-vs-Metal correctness (3 batch paths)
- `RUNBOOK.sh` — morning launch commands

## Custom Metal kernels (none existed upstream or in the AngelSlim patch)
STQ1_0 = 3:4-sparse symmetric ternary, 42B/256 weights (1.3125 bpw):
qs[32] (4-bit code/group of 4) + sign[8] (table select) + fp16 d.
Weight w -> group g=(w/64)*16 + w%16, lane p=(w%64)/16; codebook 32 entries.
Implemented: mul_mv (decode), mul_mv_ext (BS2-8), mul_mm (prefill), mul_mm_id,
mul_mv_id (MoE), get_rows, cpy. Host: pipeline cases in ggml-metal-device.cpp,
mul_mv_ext admission in ggml-metal-ops.cpp.

## Validation
- All 3 batch paths match float reference: 0.0000 / 0.0001 / 0.0232 max abs err
  (CPU int8 path: 0.63-0.91 — Metal is MORE accurate)
- Prefill (mul_mm BS=32): 22 GB/s vs q1_0's 24 — pattern parity

## Performance findings (standalone A/B bench, 16384x16384 matvec)
- v0 element-offset mapping:            45-52 GB/s
- v2 group-aligned + table, NR0=4/NSG=4: 85-94 GB/s  <-- PORTED to ggml (2.08x)
- v3 arithmetic codebook decode:         58-64 GB/s  (loses to table gather on Apple GPU)
- v4 packed-decode 4-blocks-in-flight:   84 GB/s (no better than v2 by itself)
- llama.cpp q1_0 (ternary cousin):       119 GB/s — family ceiling reference
- Q8_0 (mature int kernel):              396 GB/s
- Gotcha found: 42B block stride => uint loads misaligned on odd blocks (42%2=2 mod 4);
  must use 2-byte-aligned ushort loads. This was the v4 "wrong results" bug.
- Benchmarks are noisy ±20%: other the resident model servers models share the GPU.

## Production-shape scheduling follow-up (2026-08-31)
- GGUF metadata confirms every STQ1_0 tensor is a 6144×2048×256 expert stack.
- At one exact 6144×2048 expert matrix, GPU timestamps across seven paired runs:
  NR0=4/NSG=4 median 55 GB/s; NR0=2/NSG=8 median 61 GB/s (+10.9%).
- A second cold-cache sweep combined NR0=2/NSG=16 with a 32-entry `char4` codebook.
  Across six alternating-order pairs it beat 2×8 every time, median +7.8%; the
  cleaned final harness measured 95 vs 88 GB/s.
- Max absolute error improved slightly from 1.60e-5 to 1.46e-5 against float64.
- Production Metal configuration and exportable patch now use NR0=2/NSG=16 plus
  the vector-valued codebook.
- Rebuilt embedded Metal library passed CPU-vs-Metal paths at BS=1/8/32:
  max abs error 0.0000 / 0.0001 / 0.0232.
- Sweep used 2.1 MB of synthetic compressed weights and at most a bounded 128 MB
  cache-thrash buffer; full model was not loaded and end-to-end throughput was not re-measured.

## Four-block decode follow-up (2026-08-31)
- Combined the compact `char4` decode with an 8-lane block slice: each SIMD group
  now advances four blocks per iteration instead of two, while keeping NR0=2/NSG=16.
- Forced-unroll A/Bs matching production style beat the prior two-block kernel in
  all six alternating-order cold-cache pairs; median pairwise gain was +12.7%.
- Absolute bandwidth varied with other GPU work (roughly 54-104 GB/s for the prior
  kernel and 57-104 GB/s for the new one), so only the within-pair gain is claimed.
- Max absolute error changed only by float reduction order: 1.46e-5 to 1.84e-5 in
  the standalone float64 comparison. The rebuilt ggml library again passed BS=1/8/32
  at 0.0000 / 0.0001 / 0.0232 max error.
- Rejected: a 24 KiB threadgroup activation cache (86 GB/s vs 96) and a native
  `float4` codebook (56-70 GB/s vs 96-101). Neither is present in production.
- The model was not loaded; the test allocation remained a 2.1 MB matrix plus the
  optional bounded 128 MB cache-thrash buffer.

## y-column hoist follow-up (2026-09-01)
- The four-block charfull decode rebuilt its eight per-group float4 y-columns inside
  the NR0=2 row loop even though they depend only on the block. Hoisting them out
  (one construction per block iteration) plus splitting the accumulator into two
  independent chains gave a real but modest gain: ~4-5% at the 6144xN expert
  aggregate, ~3% at 16384x16384, neutral at a single 6144x2048 expert (same-process
  paired A/B). Correctness unchanged (0.0000/0.0001/0.0232 at BS 1/8/32).
- **Decode is near its practical ceiling.** Diagnostic kernels decompose the cost:
  at 16384x16384, "y + dot only" (no decode) reaches ~272 GB/s while the full
  four-block kernel is ~250-270 after the hoist — i.e. the scalar float dot, not the
  codebook gather, is the binding constraint, and the kernel is essentially there.
  Higher NR0 (4/8) fails even after the hoist via register spilling (50-150 GB/s),
  shared-memory y (v11) gives nothing, and the arithmetic decode is still slower
  than the char4 table gather on Apple (81 vs 200+ GB/s, confirming the earlier
  rejection). No int8/int16 simdgroup_matrix element type exists on this Metal
  compiler (float/half/bfloat only), so there is no hardware integer-MMA lever for
  the scalar decode; the batch-1 matvec has no weight reuse to amortize either.
  The remaining big-kernel lever is the simdgroup_half8x8 mm/prefill path, which is
  currently dequantize-to-shared bound at ~22 GB/s (q1_0 parity).

## Estimated real-model decode
Active ~23GB/token (experts ~5GB at STQ/IQ2 + shared expert ~14GB + attn + F32 lm_head).
Experts at 94 GB/s -> est. ~9-10 tok/s (was ~6 with v0). For reference, the other MoE
models resident on this class of machine run roughly 12-15 tok/s, and a smaller
MTP-speculating model reaches ~27-31 tok/s.

## Process notes (avoid freezes)
- Never block on GPU: poll command buffer status with hard timeout; use completed
  command-buffer GPU timestamps for kernel timing (see bench/bench.mm)
- Long runs: nohup + poll log, never foreground
- Test data gotcha: random bytes as half -> ~3% inf/nan; use finite scales everywhere
