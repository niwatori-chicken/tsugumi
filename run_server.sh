#!/bin/bash
# Run server.py and bring it back after a device hang.
#   ./run_server.sh [server.py args ...]       (e.g. --port 8010 --max-seq 262144)
# When the server exits abnormally (75 = device timeout detected, 1 = backstop / crash), the device is reset
# (tt-smi -r 0 twice, waiting after each) and the server is started again. Gives up after MAX_RESTARTS restarts
# within RESTART_WINDOW seconds. Ctrl-C / SIGTERM stops the server and this script.
#   TT_SMI (default: tt-smi on PATH, else ~/venv-tt-smi/bin/tt-smi), MAX_RESTARTS (3), RESTART_WINDOW (600)
cd "$(dirname "$0")"
TT_SMI=${TT_SMI:-$(command -v tt-smi || echo ~/venv-tt-smi/bin/tt-smi)}
MAX_RESTARTS=${MAX_RESTARTS:-3}
RESTART_WINDOW=${RESTART_WINDOW:-600}
restarts=()
child=
trap 'echo "[run_server] stopping"; [ -n "$child" ] && kill -TERM $child 2>/dev/null; wait $child 2>/dev/null; exit 0' INT TERM

reset_device() {
  for i in 1 2; do
    echo "[run_server] $(date '+%F %T') tt-smi -r 0 ($i/2)"
    "$TT_SMI" -r 0 > /dev/null 2>&1 || echo "[run_server] tt-smi failed (rc=$?)"
    sleep 15
  done
}

while true; do
  python server.py "$@" &
  child=$!
  wait $child
  rc=$?
  child=
  case $rc in
    0|130|143) echo "[run_server] server exited (rc=$rc)"; exit $rc ;;
  esac
  now=$(date +%s)
  recent=()
  for t in "${restarts[@]}"; do [ $((now - t)) -lt $RESTART_WINDOW ] && recent+=("$t"); done
  restarts=("${recent[@]}" "$now")
  echo "[run_server] $(date '+%F %T') server exited abnormally (rc=$rc), restart ${#restarts[@]}/$MAX_RESTARTS"
  if [ ${#restarts[@]} -gt $MAX_RESTARTS ]; then
    echo "[run_server] too many restarts within ${RESTART_WINDOW}s, giving up"
    exit $rc
  fi
  reset_device
done
