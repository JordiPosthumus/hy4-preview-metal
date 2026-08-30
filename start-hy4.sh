#!/bin/zsh
# Start Hy4-preview (STQ1_0, 229 GB) via llama-server on port 8019.
# Mirrors the ~/the resident model servers start-script conventions: PID file, port guard, readiness poll.
set -euo pipefail

readonly SCRIPT_NAME=${0:t}
readonly RUNTIME_DIR=$HOME/Hy4
readonly BIN="$RUNTIME_DIR/llama.cpp/build-metal/bin/llama-server"
readonly MODEL="$RUNTIME_DIR/weights/Hy4-preview-STQ1_0.gguf"
readonly PORT=8019
readonly PID_FILE="$RUNTIME_DIR/hy4.pid"
readonly LOG_FILE="$RUNTIME_DIR/hy4.log"
readonly CTX=32768

fail() {
  print -u2 -- "$SCRIPT_NAME: $*"
  exit 1
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

print -- "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Hy4-preview (STQ1_0) on port $PORT" >> "$LOG_FILE"

nohup "$BIN" \
  -m "$MODEL" \
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

print -- "Started Hy4-preview (PID $server_pid). Waiting for the 229 GB model to load..."
# /health returns 200 {"status":"ok"} once the model is fully loaded; 503 while loading.
for _attempt in {1..900}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    print -u2 -- "llama-server exited while loading. Recent log output:"
    tail -n 50 "$LOG_FILE" >&2
    exit 1
  fi

  health=$(curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  if print -r -- "$health" | grep -q '"ok"'; then
    print -- "Hy4-preview is ready at http://127.0.0.1:$PORT (PID $server_pid, ctx $CTX)."
    print -- "Log: $LOG_FILE"
    print -- "Quick test: curl http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words\"}]}'"
    exit 0
  fi
  sleep 1
done

print -- "llama-server is still running but /health was not ok after 900 seconds."
print -- "Watch progress with: tail -f $LOG_FILE"
exit 0
