#!/bin/zsh
# Gracefully stop Hy4-preview (llama-server on port 8019).
set -euo pipefail

readonly SCRIPT_NAME=${0:t}
readonly RUNTIME_DIR="${HOME}/Hy4"
readonly PORT=8019
readonly PID_FILE="$RUNTIME_DIR/hy4.pid"
readonly LOG_FILE="$RUNTIME_DIR/hy4.log"

fail() {
  print -u2 -- "$SCRIPT_NAME: $*"
  exit 1
}

if [[ ! -f "$PID_FILE" ]]; then
  print -- "Hy4-preview is not running under these scripts (no PID file)."
  exit 0
fi

server_pid=$(<"$PID_FILE")
[[ "$server_pid" == <-> ]] || fail "invalid PID file: $PID_FILE"

if ! kill -0 "$server_pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  print -- "Hy4-preview was already stopped; removed the stale PID file."
  exit 0
fi

if ! lsof -nP -a -p "$server_pid" -iTCP:"$PORT" -sTCP:LISTEN \
  >/dev/null 2>&1; then
  fail "PID $server_pid is live but is not the Hy4 listener on port $PORT; refusing to signal it"
fi

print -- "Gracefully stopping Hy4-preview (PID $server_pid)..."
kill -TERM "$server_pid"

for _attempt in {1..180}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    print -- "Hy4-preview stopped cleanly."
    print -- "Log: $LOG_FILE"
    exit 0
  fi
  sleep 1
done

print -u2 -- "llama-server is still running after 180 seconds. It was not hard-killed."
print -u2 -- "Inspect the log before taking further action: tail -n 100 $LOG_FILE"
exit 1
