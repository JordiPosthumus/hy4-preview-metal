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
- v4 q4_K-style 4-blocks-in-flight:      84 GB/s (no better than v2)
- llama.cpp q1_0 (ternary cousin):       119 GB/s — family ceiling reference
- Q8_0 (mature int kernel):              396 GB/s
- Gotcha found: 42B block stride => uint loads misaligned on odd blocks (42%2=2 mod 4);
  must use 2-byte-aligned ushort loads. This was the v4 "wrong results" bug.
- Benchmarks are noisy ±20%: other the resident model servers models share the GPU.

## Estimated real-model decode
Active ~23GB/token (experts ~5GB at STQ/IQ2 + shared expert ~14GB + attn + F32 lm_head).
Experts at 94 GB/s -> est. ~9-10 tok/s (was ~6 with v0). Compare: other resident MoE models on this class of machine ~12-15,
a smaller MTP-speculating model ~27-31.

## Process notes (avoid freezes)
- Never block on GPU: poll command buffer status with hard timeout (see bench/bench.mm)
- Long runs: nohup + poll log, never foreground
- Test data gotcha: random bytes as half -> ~3% inf/nan; use finite scales everywhere
