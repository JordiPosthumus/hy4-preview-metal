#!/bin/zsh
# Hy4-preview STQ1_0 — morning test runbook (M3 Ultra, 512GB)
# Model is NOT loaded by this script's helpers below; each command is run manually.
set -euo pipefail

RUNTIME=~/Hy4/llama.cpp/build-metal/bin
MODEL=~/Hy4/weights/Hy4-preview-STQ1_0.gguf

echo "== 0. Preconditions"
echo "   - Stop/switch away the other resident GPU models first (your other models' stop scripts)."
echo "   - Free memory target: ~240GB wired (weights 229GB + KV + buffers)."
echo "   - Weights are on local disk (required: mmap page faults over NFS are brutal)."

echo "== 1. Quick smoke (loads full weights, ~1-3 min from page cache/disk)"
echo "   \$ ${RUNTIME}/llama-cli -m ${MODEL} -ngl 99 -c 8192 --temp 0 -n 128 --no-warmup --jinja -st -f ~/Hy4/prompt.txt"

echo "== 2. Throughput benchmark"
echo "   \$ ${RUNTIME}/llama-bench -m ${MODEL} -ngl 99 -p 512 -n 128 -r 3"
echo "   Expected ballpark: tg ~8-13 tok/s (active ~23GB/token, experts at ~94 GB/s), pp several hundred tok/s."

echo "== 3. Chat mode (interactive)"
echo "   \$ ${RUNTIME}/llama-cli -m ${MODEL} -ngl 99 -c 16384 --temp 0.7 --jinja -cnv"

echo "== 4. Server (the resident model servers-style API on port 8019)"
echo "   \$ ${RUNTIME}/llama-server -m ${MODEL} -ngl 99 -c 32768 --jinja --port 8019"

echo "== Notes"
echo "   - --jinja is REQUIRED (hyv4 chat template matches no built-in family)."
echo "   - If OOM: lower -ngl (e.g. -ngl 90) or -c; wired limit is 488GB, macOS needs headroom."
echo "   - First token after load is slow (page-in); judge speed after warmup."
