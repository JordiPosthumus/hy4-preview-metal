# Live Hy4 server throughput — 2026-09-01

This is an observational measurement of completed real server turns, not a
controlled benchmark. No request was injected for this record: the existing
server log and process were observed passively while the model was already in
use. Prompts and generated content are neither copied nor described here.

## Runtime identity

| field | observed value |
|---|---|
| machine | Mac15,14; Apple M3 Ultra; 32 logical CPUs; 512 GiB unified memory |
| server start | 2026-09-01 22:25:25 -0300 |
| model load time | 80.773 s |
| server flags | `-ngl 99 -c 32768 --jinja --port 8019 --host 127.0.0.1` |
| slots/context | 4 slots; 32,768 tokens per slot; unified KV |
| model | Hy4-preview STQ1_0; 769,907,408,797 parameters; 229.4 GB file |
| loaded RSS | 214.1 GB |
| Metal source commit | `175fe67e6ee7d53e446fc1a0678da060bffda93f` |
| repository commit before this record | `73c475b3d56302cb51f66bd0da9cc7a20694e0e8` |
| loaded Metal dylib | inode `263825307`; built 2026-09-01 22:06:24 -0300 |
| Metal dylib SHA-256 | `3129c69d0f4df3c4bd38c1f3c0839074dcb8580f269f3eea0ae009bc4d4d83af` |
| post-turn load snapshot | 4.61 / 4.15 / 4.45; 54% system memory free |

The process started after the accepted Metal library was built, and `lsof`
reported the same dylib inode as the hashed file above. This verifies that the
live server used the accepted kernel rather than a stale pre-rebuild library.
The server also reported `Lightning Indexer not supported, set to disabled`, so
the intended DSA sparse-attention path was not active.

## Completed turns

| run | prompt tokens actually evaluated | prompt tok/s | decoded tokens | decode tok/s | total time | notes |
|---|---:|---:|---:|---:|---:|---|
| initial Metal server, 2026-08-30 | 10,801 | 66.55 | 282 | 4.80 | 221.007 s | Nearly identical prompt length; loaded dylib hash was not captured |
| accepted kernels, 2026-09-01 | 10,808 | **70.89** | 159 | **5.27** | 182.627 s | Full prompt evaluation; 158 graphs reused during decode |
| accepted kernels, cached follow-up | 1,455 | 50.07 | 89 | 5.03 | 46.743 s | Existing slot selected at 0.883 prefix similarity; final context 12,509 tokens |

The comparable full-prompt turns differ by only seven prompt tokens. Observed
throughput changed by **+6.52% for prompt evaluation** and **+9.79% for decode**.
This comparison spans the initial working Metal implementation through all
accepted kernel passes; it is not the isolated effect of the final small-batch
change.

### Full-prompt checkpoints

| cumulative prompt tokens | earlier tok/s | accepted tok/s | observed delta |
|---:|---:|---:|---:|
| 2,048 | 95.32 | 98.53 | +3.4% |
| 4,096 | 85.92 | 90.27 | +5.1% |
| 6,144 | 77.96 | 83.62 | +7.3% |
| 8,192 | 72.54 | 78.02 | +7.6% |
| 10,240 | 68.23 | 72.93 | +6.9% |
| final, 10.8k | 66.55 | 70.89 | +6.5% |

### Raw completion timing lines

```text
prompt eval time = 152467.86 ms / 10808 tokens (14.11 ms per token, 70.89 tokens per second)
       eval time =  30158.79 ms /   159 tokens (189.68 ms per token, 5.27 tokens per second)
      total time = 182626.65 ms / 10967 tokens

prompt eval time =  29058.18 ms /  1455 tokens (19.97 ms per token, 50.07 tokens per second)
       eval time =  17684.81 ms /    89 tokens (198.71 ms per token, 5.03 tokens per second)
      total time =  46742.99 ms /  1544 tokens
```

## What this teaches us

1. **The optimized kernels transfer to real use, but kernel bandwidth is not
   token throughput.** The long full-prompt turn is faster in both phases, while
   the decode improvement is far smaller than the roughly 2.09x standalone
   expert-matvec bandwidth increase from the original mapping.
2. **Amdahl's law points away from more work on only this kernel.** If the
   standalone 94/45 GB/s ratio and the 5.27/4.80 end-to-end ratio are treated as
   comparable, the implied original time share of the optimized expert matvec is
   about 17%. This is only a directional estimate because the live runs were not
   controlled. Shared-expert, attention/DSA, and F32 output-head work are the
   higher-leverage next targets.
3. **Longer context is visible in decode speed.** The cached follow-up decoded at
   5.03 tok/s versus 5.27 tok/s for the preceding turn, a 4.55% reduction at the
   longer context. The disabled Lightning Indexer is a plausible contributor,
   but this pair alone does not establish causality.
4. **Prefix reuse helps latency even when incremental prompt throughput falls.**
   The follow-up evaluated only 1,455 new prompt tokens rather than processing the
   entire 12.5k-token slot again.
5. **This traffic does not validate the new batch-4/5/7/8 scheduler.** Sequential
   single-slot generation predominantly exercises batch-1 decode. That scheduler
   needs a separate concurrent-request or microbatch measurement.

## Contention and interpretation rules

- The system was in normal use and load was not controlled. The load-average and
  memory snapshot above was taken after the turns, not continuously during them.
- Prompt content, output length, cache state, thermal state, and other GPU work
  can differ between live turns. Accordingly, this record reports observational
  deltas and does not promote them to causal kernel A/B claims.
- The accepted standalone kernel results remain the controlled evidence: shared
  process, shared Metal queue/library/buffers, symmetric warmups, deterministic
  balanced AB/BA order, and command-buffer GPU timestamps.
- `/metrics` returned HTTP 501 because the current production server was not
  launched with `--metrics`. The running configuration was deliberately not
  changed or restarted just to improve this record.

## Passive reproduction

These commands inspect an already-running server and do not load another model:

```bash
./status-hy4.sh
tail -F hy4.log | rg 'prompt processing|n_decoded|prompt eval time|eval time|total time|release'
```

Do not run `bench-hy4.sh` while `llama-server` is resident: `llama-bench` would
load a second copy of the model. The script intentionally refuses that condition.
