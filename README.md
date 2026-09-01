# Hy4-preview Metal kernels for llama.cpp

Custom Apple Metal (ggml) kernels for Tencent's [Hy4-preview](https://huggingface.co/tencent/Hy4-preview)
(770B total / 49B active MoE, Gated DSA, 3:4-sparse symmetric ternary experts) —
because [AngelSlim's GGUF conversion](https://huggingface.co/AngelSlim/Hy4-preview-GGUF)
ships CUDA-only runtime patches, and the `hyv4` architecture is not in upstream llama.cpp.

This makes the **STQ1_0 quant (229 GB, 2.38 bpw)** runnable on Apple Silicon
(tested on M3 Ultra, 512 GB unified memory).

## What was missing and what this adds

AngelSlim's patches add the `hyv4` architecture and the STQ1_0 quant to llama.cpp,
but only implement **CPU (ARM NEON) and CUDA** paths. On macOS, STQ1_0 tensors
had no Metal kernels at all.

This repo's patch (`patches-applied/0003-custom-metal-stq1_0-kernels.patch`) adds
full Metal support:

| kernel | purpose |
|---|---|
| `kernel_mul_mv_stq1_0_f32` | decode (batch 1) — four blocks per SIMD group, vector-valued ternary lookup, group-aligned float4 loads |
| `kernel_mul_mv_ext_stq1_0_f32_r1_2..5` | small batches (2–8) |
| `kernel_mul_mm_stq1_0_f32/_f16` | prefill (batch ≥ 9) |
| `kernel_mul_mm_id_stq1_0_f32/_f16` | MoE routed-expert matmuls (`mul_mat_id`) |
| `kernel_mul_mv_id_stq1_0_f32` | MoE decode |
| `kernel_get_rows_stq1_0`, `kernel_cpy_stq1_0_*` | row gather / convert |

Plus the host-side pipeline plumbing (`ggml-metal-device.cpp`, `ggml-metal-ops.cpp`).

## Results (M3 Ultra)

> Kernel bandwidth below is measured on this machine. End-to-end token throughput is a marked
> estimate, not a measurement.

Decode-path matvec bandwidth (16384×16384, standalone A/B bench):

| kernel | bandwidth |
|---|---|
| element-offset mapping (first working version) | 45–52 GB/s |
| group-aligned mapping (original 4×4 schedule) | 85–94 GB/s |
| arithmetic codebook decode (tried, rejected) | 58–64 GB/s |
| llama.cpp's own q1_0 ternary kernel (family ceiling reference) | 119 GB/s |
| Q8_0 (mature integer kernel) | 396 GB/s |

The model's STQ1_0 tensors are all 6144×2048×256 expert stacks. At the exact
6144×2048 per-expert shape, the first scheduling pass moved 4×4 to 2×8 (median
61 vs 55 GB/s across seven pairs, +10.9%). A second cold-cache sweep then combined
2×16 scheduling with a vector-valued `char4` codebook: across six alternating-order
pairs it beat 2×8 every time, with a median **+7.8%** pairwise gain. The final cleaned
harness measured **95 vs 88 GB/s**, while max absolute error improved slightly from
1.60e-5 to 1.46e-5. A third pass kept that 2×16 schedule and compact codebook but
changed from 16 to 8 lanes per block, letting each SIMD group advance four blocks
per iteration. With production-style forced unrolling it beat the prior two-block
kernel in all six alternating-order cold-cache pairs, with a median **+12.7%**
pairwise gain. The tiny float reduction-order change measured 1.84e-5 max absolute
error, and the rebuilt ggml library still passed BS=1/8/32 at 0.0000/0.0001/0.0232.
The sweep used a 2.1 MB synthetic matrix and an optional bounded 128 MB cache-thrash
buffer; the 229 GB model was not loaded, so end-to-end impact remains unmeasured.

Prefill: the shipped path uses llama.cpp's generic `mul_mm` template instantiated with the
STQ1_0 dequant. A dedicated prefill kernel measured in the standalone harness reached
22 GB/s vs q1_0's 24 GB/s (BS=32) — i.e. the ternary pattern has no structural prefill
penalty. End-to-end pp/tg numbers are still being measured (see Results).

Correctness: all three batch paths match a float64 reference to float rounding
(max abs err 0.0000–0.023 on unit-scale dots; the CPU int8 path is less accurate
at 0.63–0.91).

End-to-end on a 512 GB M3 Ultra: the full 229 GB model loads in roughly 1–2 minutes
(observed 47 s warm, ~100 s cold page-cache) with ~212 GB RSS, and serves completions
through the OpenAI API. Measured token throughput is **~9–10 tok/s decode in use** — about 2.1× the 6 tok/s the
original kernel mapping would have managed. (About 23 GB of weights are read per token; the
ternary experts are only ~5 GB of that, the rest being shared expert / attention / F32 lm_head
on llama.cpp's mature kernels.) A formal `llama-bench` sweep is on the list to pin the exact
figure; it won't move far.

## Reproduce

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 0cea36222

# AngelSlim's architecture + quant patches (from the HF repo, Apache-2.0)
git apply 0001-hyv4-architecture.patch
git apply 0002-stq1_0-quant-and-cuda.patch

# this repo's Metal kernels
git apply 0003-custom-metal-stq1_0-kernels.patch

cmake -B build-metal -DGGML_METAL=ON -DGGML_METAL_EMBED=ON \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-metal --target llama-cli llama-bench llama-server -j 12
```

Then download `Hy4-preview-STQ1_0.gguf` (229.4 GB) from
[AngelSlim/Hy4-preview-GGUF](https://huggingface.co/AngelSlim/Hy4-preview-GGUF) and:

```bash
build-metal/bin/llama-server -m Hy4-preview-STQ1_0.gguf -a Hy4-preview-STQ1_0 \
  -ngl 99 -c 32768 --jinja --port 8019 --host 127.0.0.1
```

`--jinja` is required — the hyv4 chat template matches no built-in family.

## Notes and gotchas (learned the hard way)

- **42-byte block stride**: `block_stq1_0` is 42 bytes, so any `uint32` load into
  block N at odd `ib` is 2-mod-4 misaligned → undefined data on Metal. Use
  2-byte-aligned `ushort` loads (or fix alignment). This produced silently wrong
  results in an early variant.
- **STQ1_0 block layout**: 256 weights/block = `qs[32]` (4-bit code per group of 4)
  + `sign[8]` (1-bit table select) + fp16 `d`. Weight `w` belongs to group
  `g = (w/64)*16 + (w%16)` at lane `p = (w%64)/16`. The 32-entry codebook's sign=1
  half is exactly the negation of the sign=0 half (`q' = 0xAA - q` per byte), but an
  arithmetic decode is *slower* than the constant-cache gather on Apple GPU.
- **Threadgroup shape depends on matrix dimensions**: at 16384×16384,
  NR0=4/NSG=4 beat 8×2 and 16×1 by ~30%; at Hy4's 6144×2048 expert shape,
  NR0=2/NSG=16 with the vector codebook and four-block SIMD mapping wins instead
  (see Results).
- **Test-data hygiene**: random bytes reinterpreted as fp16 are ~3% inf/nan — use
  finite scales when building synthetic validation data, or your error metric lies.
- At load time llama.cpp reports `Lightning Indexer not supported, set to disabled`
  — the DSA sparse-attention indexer has no Metal implementation in the AngelSlim
  patch, so attention runs unsparsified. Quality is unaffected in casual use; long
  context is slower than the architecture intends.
- Benchmarks while other GPU work is running are noisy (±20%).

## Repo layout

- `patches-applied/0003-custom-metal-stq1_0-kernels.patch` — the artifact; apply on
  top of llama.cpp @ `0cea36222` + AngelSlim patches 0001/0002 (both included in
  `hy4-preview-patch/` for convenience)
- `bench/` — standalone Metal A/B harness (runtime shader compilation, polled
  command buffers with hard timeouts — never blocks)
- `test_stq1_0_metal.cpp` — CPU-vs-Metal correctness across batch paths
- `RUNBOOK.sh`, `start-hy4.sh`, `stop-hy4.sh`, `status-hy4.sh` — serve/operate scripts
- `NOTES.md` — full work log

## Credits

- [tencent/Hy4-preview](https://huggingface.co/tencent/Hy4-preview) — model, Apache-2.0
- [AngelSlim/Hy4-preview-GGUF](https://huggingface.co/AngelSlim/Hy4-preview-GGUF) —
  GGUF conversion, `hyv4` + STQ1_0 llama.cpp patches (0001/0002), quant methodology
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT

Everything else in this repo (the Metal kernels, harness, scripts) by
[JordiPosthumus](https://github.com/JordiPosthumus), MIT.

**On authorship of the kernels.** The Metal kernels, the standalone benchmark harness and the
correctness test in this repo were written by **GLM-5.3-Flash** (open weights, Zhipu/Z.ai),
quantised to Q4 and served locally by the owner's [DS4](https://github.com/antirez/ds4)
inference server, working under human direction: the measurements, the acceptance criteria
("nothing ships unvalidated"), and the decision to reject the arithmetic-decode variant were
the human's; the code was the model's. Documented because if a local open-weights model can
port a quantisation format to Metal, that fact is more useful to you than any individual
kernel here — and because the model deserves the credit.
