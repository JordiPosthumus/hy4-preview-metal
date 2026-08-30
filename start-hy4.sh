#!/bin/zsh
# Start Hy4-preview (STQ1_0, 229 GB) via llama-server on port 8019.
# Mirrors the ~/the resident model servers start-script conventions: PID file, port guard, readiness poll.
# Reporting: pre-flight memory check, live load progress, exit-reason forensics.
set -euo pipefail

readonly SCRIPT_NAME=${0:t}
readonly RUNTIME_DIR=$HOME/Hy4
readonly BIN="$RUNTIME_DIR/llama.cpp/build-metal/bin/llama-server"
readonly MODEL="$RUNTIME_DIR/weights/Hy4-preview-STQ1_0.gguf"
readonly ALIAS="Hy4-preview-STQ1_0"
readonly PORT=8019
readonly PID_FILE="$RUNTIME_DIR/hy4.pid"
readonly LOG_FILE="$RUNTIME_DIR/hy4.log"
readonly CTX=32768

# memory budget: weights (229.4 GB) + KV/compute headroom + macOS
readonly NEED_GB=245
readonly OS_HEADROOM_GB=12

fail() {
  print -u2 -- "$SCRIPT_NAME: $*"
  exit 1
}

gb_human() {
  awk -v kb="$1" 'BEGIN{printf "%.1f GB", kb/1048576}'
}

# ---- pre-flight: is there enough memory headroom for a 229 GB model? --------
check_memory() {
  local phys_kb used_kb avail_gb
  phys_kb=$(( $(sysctl -n hw.memsize) / 1024 ))

  # sum RSS of processes holding > 1 GB (the resident models dominate this)
  used_kb=$(ps -axo rss= | awk '$1 > 1048576 {s += $1} END {print s+0}')
  local avail_kb=$(( phys_kb - used_kb ))
  avail_gb=$(( avail_kb / 1048576 ))

  print -- "Memory pre-flight: phys $(gb_human $phys_kb) | resident-in-big-processes $(gb_human $used_kb) | available ~${avail_gb} GB"

  if (( avail_gb < NEED_GB )); then
    print -u2 -- ""
    print -u2 -- "NOT ENOUGH FREE MEMORY: Hy4 needs ~${NEED_GB} GB (weights 229.4 GB + KV + buffers),"
    print -u2 -- "but only ~${avail_gb} GB appears available. Top residents right now:"
    ps -axo rss=,comm= | sort -rn | head -5 | while read -r kb cmd; do
      print -u2 -- "  $(gb_human $kb)  $cmd"
    done
    print -u2 -- ""
    print -u2 -- "Stop or switch away the other resident GPU models first (your other models' stop scripts), then retry."
    print -u2 -- "Refusing to start: last attempt died from an apparent memory-pressure SIGKILL."
    exit 1
  fi
}

[[ -x "$BIN" ]] || fail "llama-server binary missing or not executable: $BIN"
[[ -s "$MODEL" ]] || fail "model weights missing: $MODEL"
command -v curl >/dev/null || fail "curl is required"
command -v lsof >/dev/null || fail "lsof is required"

if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(<"$PID_FILE")
  if [[ "$existing_pid" == <-> ]] && kill -0 "$existing_pid" 2>/dev/null; then
    if lsof -nP -a -p "$existing_pid" -iTCP:"$PORT" -sTCP:LISTEN \
      >/dev/null 2>&1; then
      print -- "Hy4-preview is already running (PID $existing_pid)."
      print -- "Log: $LOG_FILE"
      exit 0
    fi
    fail "PID file points to another live process ($existing_pid); refusing to overwrite it"
  fi
  rm -f "$PID_FILE"
fi

listener_pid=$(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
[[ -z "$listener_pid" ]] || fail "port $PORT is already in use by PID $listener_pid"

check_memory

print -- "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Hy4-preview (STQ1_0) on port $PORT" >> "$LOG_FILE"

nohup "$BIN" \
  -m "$MODEL" \
  -a "$ALIAS" \
  -ngl 99 \
  -c "$CTX" \
  --jinja \
  --port "$PORT" \
  --host 127.0.0.1 \
  >> "$LOG_FILE" 2>&1 < /dev/null &
server_pid=$!
print -r -- "$server_pid" > "$PID_FILE"

sleep 1
if ! kill -0 "$server_pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  print -u2 -- "llama-server exited during startup. Recent log output:"
  tail -n 50 "$LOG_FILE" >&2
  exit 1
fi

print -- "Started Hy4-preview (PID $server_pid). Loading 229 GB — progress every 10 s (ctrl-c here is safe; the server keeps loading)."

# /health returns 200 {"status":"ok"} once the model is fully loaded; 503 while loading.
typeset -i attempt=0
while (( attempt < 900 )); do
  attempt=$((attempt+1))

  if ! kill -0 "$server_pid" 2>/dev/null; then
    # ---- died mid-load: forensics --------------------------------------
    rm -f "$PID_FILE"
    local_exit=0
    wait "$server_pid" 2>/dev/null || local_exit=$?
    print -u2 -- ""
    print -u2 -- "llama-server DIED while loading (elapsed ~$((attempt)) s)."
    if (( local_exit == 137 )); then
      print -u2 -- "Exit status 137 = SIGKILL. On macOS this is almost always memory pressure:"
      print -u2 -- "the 229 GB model did not fit alongside whatever else is resident."
      print -u2 -- "Recent memory-pressure kill records (if any):"
      log show --last 5m --style compact \
        --predicate 'eventMessage CONTAINS[c] "memorystatus" OR eventMessage CONTAINS[c] "jetsam" OR eventMessage CONTAINS[c] "highwater"' \
        2>/dev/null | tail -n 8 | sed 's/^/  /' || print -u2 -- "  (none found in system log)"
    elif (( local_exit == 134 )); then
      print -u2 -- "Exit status 134 = SIGABRT (assertion failure) — see log excerpt below."
    else
      print -u2 -- "Exit status: $local_exit"
    fi
    print -u2 -- "Last 30 log lines:"
    tail -n 30 "$LOG_FILE" >&2
    exit 1
  fi

  # ---- live progress: elapsed, RSS, latest log line -------------------
  if (( attempt % 10 == 0 )); then
    rss_kb=$(ps -o rss= -p "$server_pid" 2>/dev/null | tr -d ' ' || echo 0)
    last_log=$(tail -n 1 "$LOG_FILE" | cut -c1-100)
    print -- "  [$((attempt)) s] resident $(gb_human ${rss_kb:-0}) | $last_log"
  fi

  health=$(curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  if print -r -- "$health" | grep -q '"ok"'; then
    print -- ""
    print -- "Hy4-preview is ready at http://127.0.0.1:$PORT (PID $server_pid, ctx $CTX, alias $ALIAS)."
    print -- "Log: $LOG_FILE"
    print -- "Quick test:"
    print -- "  curl http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\"
    print -- "    -d '{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words\"}]}'"
    exit 0
  fi
  sleep 1
done

print -- "llama-server is still running but /health was not ok after 900 seconds."
print -- "Watch progress with: tail -f $LOG_FILE"
exit 0
