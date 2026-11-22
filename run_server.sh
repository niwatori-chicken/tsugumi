#!/bin/bash
# Run server.py and bring it back after a device hang.
#   ./run_server.sh [server.py args ...]       (e.g. --port 8010 --max-seq 262144)
# When the server exits abnormally (75 = device timeout detected, 1 = backstop / crash), the device is reset
# (tt-smi -r 0 twice, waiting after each) and the server is started again. Gives up after MAX_RESTARTS restarts
# within RESTART_WINDOW seconds. Ctrl-C / SIGTERM stops the server and this script.
#   TT_SMI (default: tt-smi on PATH, else ~/.tenstorrent-venv/bin/tt-smi from tt-installer), MAX_RESTARTS (3), RESTART_WINDOW (600),
#   TT_LOGGER_LEVEL (tt-metal, default warn here), LOGURU_LEVEL (ttnn Python, default WARNING), TSUGUMI_LOG_LEVEL (server, INFO)
cd "$(dirname "$0")"
export TT_LOGGER_LEVEL=${TT_LOGGER_LEVEL:-warn}  # tt-metal (spdlog level names: warn, not warning)
export LOGURU_LEVEL=${LOGURU_LEVEL:-WARNING}  # ttnn (Python)
say() { echo "$(date '+%F %T.%3N') $1 [run_server] $2"; }
export TT_SMI=${TT_SMI:-$(command -v tt-smi || echo ~/.tenstorrent-venv/bin/tt-smi)}  # also used by the server for /health board telemetry
MAX_RESTARTS=${MAX_RESTARTS:-3}
RESTART_WINDOW=${RESTART_WINDOW:-600}
restarts=()
child=
trap 'say INFO stopping; [ -n "$child" ] && kill -TERM $child 2>/dev/null; wait $child 2>/dev/null; exit 0' INT TERM

reset_device() {
  for i in 1 2; do
    say INFO "tt-smi -r 0 ($i/2)"
    "$TT_SMI" -r 0 > /dev/null 2>&1 || say ERROR "tt-smi failed (rc=$?)"
    sleep 15
  done
}

while true; do
  say INFO "starting: python server.py $*"
  python server.py "$@" &
  child=$!
  wait $child
  rc=$?
  child=
  case $rc in
    0|130|143) say INFO "server exited (rc=$rc)"; exit $rc ;;
  esac
  now=$(date +%s)
  recent=()
  for t in "${restarts[@]}"; do [ $((now - t)) -lt $RESTART_WINDOW ] && recent+=("$t"); done
  restarts=("${recent[@]}" "$now")
  say WARN "server exited abnormally (rc=$rc, 75 = device hang), restart ${#restarts[@]}/$MAX_RESTARTS"
  if [ ${#restarts[@]} -gt $MAX_RESTARTS ]; then
    say ERROR "too many restarts within ${RESTART_WINDOW}s, giving up"
    exit $rc
  fi
  reset_device
done
