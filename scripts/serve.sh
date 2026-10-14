#!/usr/bin/env bash
# Start the genphish-guard API + console.
#   ./scripts/serve.sh            foreground, reload on edit
#   ./scripts/serve.sh --bg       background, pid in .server.pid
#   ./scripts/serve.sh --stop     stop the background server
set -euo pipefail

cd "$(dirname "$0")/.."
PIDFILE=".server.pid"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

stop() {
  if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "stopped"
  else
    echo "not running"
  fi
}

case "${1:-}" in
  --stop)
    stop
    ;;
  --bg)
    stop >/dev/null 2>&1 || true
    nohup python -m uvicorn api.main:app --host "$HOST" --port "$PORT" \
      > server.log 2>&1 &
    echo $! > "$PIDFILE"
    echo "serving on http://localhost:$PORT  (pid $(cat "$PIDFILE"), log: server.log)"
    ;;
  *)
    exec python -m uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
    ;;
esac
