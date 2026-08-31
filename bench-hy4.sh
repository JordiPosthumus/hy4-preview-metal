#!/usr/bin/env zsh
# bench-hy4.sh — measure end-to-end throughput and write repo-ready results.
#
# Why this exists: the README carries a *derived estimate* for token throughput.
# This script replaces the estimate with a measurement. It does the job safely:
# llama-bench loads the 229 GB model itself, so it must not race another resident
# model, and it must never hang a shared machine — everything runs in the
# background and is polled with a hard deadline.
#
# Usage:
#   ~/Hy4/bench-hy4.sh --yes            # full run (long: expect tens of minutes)
#   ~/Hy4/bench-hy4.sh --yes --quick    # short pp/n matrix, fewer reps
#
# Refuses to run without --yes, and refuses outright if memory looks insufficient.

set -uo pipefail

HY4_HOME="${HOME}/Hy4"
MODEL="${HY4_HOME}/weights/Hy4-preview-STQ1_0.gguf"
BIN="${HY4_HOME}/llama.cpp/build-metal/bin/llama-bench"
RESULTS_DIR="${HY4_HOME}/results"
MIN_FREE_GB=245          # 229 GB weights + KV + headroom
MAX_MINUTES=240          # hard deadline; we kill rather than hang

QUICK=0
CONFIRM=0
for a in "$@"; do
  case "$a" in
    --yes)   CONFIRM=1 ;;
    --quick) QUICK=1 ;;
    *) print -u2 -- "unknown arg: $a"; exit 2 ;;
  esac
done

err() { print -u2 -- "✗ $*" ; }
ok()  { print -- "✓ $*"; }

(( CONFIRM )) || { err "This loads ~229 GB onto the GPU. Re-run with --yes when the machine is yours to use."; exit 1; }

# ---- pre-flight -------------------------------------------------------------
[[ -x "$BIN"   ]] || { err "llama-bench not built: $BIN"; exit 1; }
[[ -f "$MODEL" ]] || { err "weights missing: $MODEL"; exit 1; }
mkdir -p "$RESULTS_DIR"

if pgrep -x llama-server >/dev/null 2>&1; then
  err "a llama-server is running — stop it first (llama-bench loads the model itself and would double-pin memory)"
  exit 1
fi

total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
pct=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2); print $2}')
free_gb=""
free_gb=$(awk -v p="${pct:-0}" -v t="${total_bytes}" 'BEGIN{printf "%.0f", p*t/100/1073741824}')
if [[ -z "${free_gb}" || "${free_gb}" == "0" ]]; then
  free_gb=$(vm_stat | awk '/page size of/{ps=$8*1024} /^Pages free|^Pages speculative/{if($3){s+=$3}} END{printf "%.0f", s*ps/1073741824}')
fi
if (( ${free_gb:-0} < MIN_FREE_GB )); then
  err "only ~${free_gb} GB apparently free; need >= ${MIN_FREE_GB} GB. Stop the other resident models first."
  err "top RSS consumers:"
  ps -arcuxo rss,comm | awk 'NR>1{printf "   %7.1f GB  %s\n", $1/1048576, $2}' | head -5
  exit 1
fi
ok "pre-flight: ~${free_gb} GB free, model + binary present"

if (( QUICK )); then
  BENCH_ARGS=(-p 512 -n 128 -r 2)
else
  # separate pp and tg sweeps: -pg would pair them instead
  BENCH_ARGS=(-p 512,2048,8192 -n 128,512 -r 3)
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RAW="${RESULTS_DIR}/llama-bench-${STAMP}.raw.txt"
LOG="${RESULTS_DIR}/llama-bench-${STAMP}.log"
MD="${RESULTS_DIR}/llama-bench-${STAMP}.md"

print -- "→ llama-bench -ngl 99 ${BENCH_ARGS[*]}  (hard cap ${MAX_MINUTES} min)"
print -- "  raw → ${RAW}"

# ---- run in background, poll with a hard deadline ---------------------------
"$BIN" -m "$MODEL" -ngl 99 -o md ${BENCH_ARGS[@]} > "$RAW" 2> "$LOG" &
pid=$!
start=$SECONDS
while kill -0 "$pid" 2>/dev/null; do
  elapsed=$(( SECONDS - start ))
  if (( elapsed > MAX_MINUTES * 60 )); then
    err "deadline ${MAX_MINUTES} min reached — killing llama-bench (partial output kept in $RAW)"
    kill -TERM "$pid" 2>/dev/null; sleep 5; kill -KILL "$pid" 2>/dev/null
    exit 124
  fi
  rss=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f", $1/1048576}')
  (( elapsed % 30 < 15 )) && print -- "   … ${elapsed}s elapsed, RSS ~${rss:-?} GB"
  sleep 15
done
wait "$pid"; rc=$?
if (( rc != 0 )); then
  err "llama-bench exited ${rc}. Last log lines:"
  tail -15 "$LOG" | sed 's/^/   /'
  exit "$rc"
fi
ok "run complete in ${SECONDS-start}s"

# ---- render repo-ready markdown --------------------------------------------
{
  echo "## llama-bench — Hy4-preview STQ1_0, M3 Ultra 512 GB"
  echo
  echo "- date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- llama.cpp: $(cd "${HY4_HOME}/llama.cpp" && git rev-parse --short HEAD) (with hyv4 + STQ1_0 + custom Metal kernels)"
  echo "- flags: \`-ngl 99 --load-mode mmap ${BENCH_ARGS[*]}\`"
  echo
  echo '```'
  cat "$RAW"
  echo '```'
} > "$MD"

ok "results → ${MD}"
print --
print -- "---- paste-ready summary ----"
grep -E "^\||model|pp|tg" "$RAW" | head -20

print --
print -- "Next: put the tg/pp numbers into README.md 'Results' (replacing the estimate),"
print -- "and into ~/jordisblog/drafts/ternary-kernel.md + its EDITORIAL checklist."
