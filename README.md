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
| `kernel_mul_mv_stq1_0_f32` | decode (batch 1) — four blocks per SIMD group, direct exact-3:4 dot, group-aligned float4 loads |
| `kernel_mul_mv_ext_stq1_0_f32_r1_2..5` | small batches (2–8) |
| `kernel_mul_mm_stq1_0_f32/_f16` | prefill (batch ≥ 9) |
| `kernel_mul_mm_id_stq1_0_f32/_f16` | MoE routed-expert matmuls (`mul_mat_id`) |
| `kernel_mul_mv_id_stq1_0_f32` | MoE decode |
| `kernel_get_rows_stq1_0`, `kernel_cpy_stq1_0_*` | row gather / convert |

Plus the host-side pipeline plumbing (`ggml-metal-device.cpp`, `ggml-metal-ops.cpp`).

## Results (M3 Ultra)

> Kernel bandwidth below comes from bounded standalone measurements. Full-model
> throughput is measured from completed live server turns and is labelled separately.

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

A second decode pass changed only the reduction schedule: the four ternary
components now accumulate in independent FMA chains before a balanced final sum.
A later benchmark audit found two stale trailing entries had shifted the dispatch
metadata for newly appended v18/v19 variants, so the earlier 18.6%/22.5% claims are
withdrawn. With every variant name statically matched to its real NR0/NSG geometry,
the component-wise kernel still wins: at 6144x2048 its paired GPU-time median is
**4.4% lower** than the prior hoisted-dot kernel (AB 2.2%, BA 8.1%; 15/20 wins),
with max absolute errors of 1.69e-5 and 1.84e-5 respectively.

A fourth decode pass uses STQ1_0's exact 3:4 structure directly. The code's high
bits identify the zero lane, its low bits identify two sign flips, and the separate
sign bit negates the group. Computing that three-term signed dot avoids constructing
a decoded `float4` and multiplying the known zero lane. In a cache-realistic routed
wrapper with eight distinct 6144x2048 expert matrices (about 17 MB of compressed
weights), six balanced ABBA runs cut median best graph time from **2.150 ms to
1.664 ms (22.6%)**; every candidate run beat every baseline run. The median of run
means improved 15.6%, and maximum absolute error stayed below 9e-6. This is an
isolated STQ1_0 routed-kernel result, not a token-throughput claim; the full model
was not loaded or mapped.

Small-batch scheduling is now specialized only for Hy4's exact 6144x2048 STQ1_0
experts. Batches 4, 5, 7, and 8 use one SIMD group and a four-lane row reduction;
batches 2, 3, and 6 retain the prior geometry. Randomized same-process comparisons
measured candidate/baseline paired time medians of 0.9594, 0.9254, 0.9602, and
0.9407 respectively (about **4.1%, 7.5%, 4.0%, and 5.9% faster**), with 18/20,
20/20, 19/20, and 19/20 wins. The largest baseline/candidate output difference was
8.392e-5. Exact-shape synthetic tensors were used; the full model was not loaded.

Prefill: the legacy Metal `mul_mm` and routed `mul_mm_id` paths now use a cooperative
STQ1_0 loader. Two threads per A row split the ternary groups and keep adjacent p lanes,
halving codebook gathers per 32-weight tile; native `half2` codebook views avoid the
remaining char conversion. On the exact 6144x2048 expert shape, final bounded A/Bs against
the prior library won every pair: best time fell 18.5-19.3% at batch 32, 19.9-21.9% at
batch 64, and 16.7-25.9% at batch 128. One-expert routed matmul improved 7.6-13.7% with
identical reference error. The model was not loaded for these measurements.

Correctness: all three batch paths match a float64 reference to float rounding
(max abs err 0.0000–0.023 on unit-scale dots; the CPU int8 path is less accurate
at 0.63–0.91).

End-to-end on a 512 GB M3 Ultra: the full 229 GB model loaded in **80.77 s**
for the latest observed run (earlier warm-cache run: 47.40 s), occupied **214.1 GB
RSS**, and served completions through the OpenAI API. A completed 10,808-token
live prompt measured **70.89 prompt tok/s** followed by 159 decoded tokens at
**5.27 tok/s**. An earlier nearly identical-length 10,801-token turn on the initial
Metal implementation measured 66.55 prompt tok/s and 4.80 decode tok/s, making
the observational deltas +6.5% and +9.8%. These are not controlled A/B results:
output length, cache state, thermals, and machine contention differed.

The much smaller end-to-end gain than the standalone expert-matvec gain is
important: STQ1_0 expert decoding is only one part of each token. Shared-expert,
attention/DSA, and F32 output-head work now dominate the remaining time. A
following prefix-cache-reuse turn evaluated 1,455 new prompt tokens at 50.07
tok/s and decoded 89 tokens at 5.03 tok/s at the longer context. The complete
configuration, raw timing lines, cache behavior, contention caveats, and a rough
Amdahl analysis are in the [live-server record](results/live-server-2026-09-01.md).
A controlled `llama-bench` sweep remains useful, but must not be run alongside the
resident server because it would load a second copy of the 229 GB model.

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
- **Exploit sparsity at the dot, not by reconstructing weights**: each STQ1_0 code
  has exactly one zero and three signed unit lanes. A direct signed three-term dot
  is faster than the `char4 -> float4 -> dot` path at cache-realistic expert
  working sets. A switch-based form and two/eight accumulator variants did not
  survive balanced production-wrapper testing and are not shipped.
- **Threadgroup shape depends on matrix dimensions**: at 16384×16384,
  NR0=4/NSG=4 beat 8×2 and 16×1 by ~30%; at Hy4's 6144×2048 expert shape,
  NR0=2/NSG=16 with the direct 3:4 dot and four-block SIMD mapping wins instead
  (see Results).
- **Test-data hygiene**: random bytes reinterpreted as fp16 are ~3% inf/nan — use
  finite scales when building synthetic validation data, or your error metric lies.
- At load time llama.cpp reports `Lightning Indexer not supported, set to disabled`
  — the DSA sparse-attention indexer has no Metal implementation in the AngelSlim
  patch, so attention runs unsparsified. Quality is unaffected in casual use; long
  context is slower than the architecture intends.
- **Contention discipline**: same-process candidates share the Metal queue,
  buffers, and library; warmups are symmetric; AB and BA orders are exactly
  balanced and deterministically shuffled; timings use completed command-buffer
  GPU timestamps. Aggregate and order-split paired ratios are reported, and any
  candidate whose sign changes with order or background load is rejected.

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
