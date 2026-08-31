#!/bin/zsh
# Status for Hy4-preview (llama-server on port 8019): process, health, memory, quick self-test.
set -euo pipefail

readonly SCRIPT_NAME=${0:t}
readonly RUNTIME_DIR="${HOME}/Hy4"
readonly PORT=8019
readonly PID_FILE="$RUNTIME_DIR/hy4.pid"
readonly LOG_FILE="$RUNTIME_DIR/hy4.log"
readonly MODEL="$RUNTIME_DIR/weights/Hy4-preview-STQ1_0.gguf"

command -v curl >/dev/null || { print -u2 -- "$SCRIPT_NAME: curl is required"; exit 1; }

print -- "== Hy4-preview (STQ1_0, llama-server :$PORT)"

if [[ -f "$PID_FILE" ]]; then
  server_pid=$(<"$PID_FILE")
  if [[ "$server_pid" == <-> ]] && kill -0 "$server_pid" 2>/dev/null; then
    print -- "process:   running (PID $server_pid)"
    if lsof -nP -a -p "$server_pid" -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      print -- "listener:  port $PORT"
    else
      print -- "listener:  NOT listening on port $PORT (still loading?)"
    fi
    # memory footprint (RSS grows as mmap pages are touched)
    ps -o rss= -p "$server_pid" | read rss_kb
    printf 'memory:    %.1f GB RSS\n' $(echo "$rss_kb" | awk '{print $1/1048576}')
  else
    print -- "process:   NOT running (stale PID file: $server_pid)"
  fi
else
  print -- "process:   not running (no PID file)"
fi

health=$(curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
if [[ -n "$health" ]]; then
  print -- "health:    $health"
else
  print -- "health:    no response (server down or still loading)"
fi

if [[ -s "$LOG_FILE" ]]; then
  print -- "log tail:  $LOG_FILE"
  tail -n 3 "$LOG_FILE" | sed 's/^/   /'
fi

if [[ -s "$MODEL" ]]; then
  printf 'weights:   %.1f GB on disk\n' $(stat -f%z "$MODEL" | awk '{print $1/1e9}')
fi
