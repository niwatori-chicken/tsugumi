#!/bin/bash
# Server-level hang test (todo 1c): run_server.sh with the test-only /debug/hang endpoint.
#   1. a normal request works   2. /debug/hang -> error response, /health 503, process exits 75
#   3. run_server.sh resets the device and restarts   4. a normal request works again
#   scripts/test_server_hang.sh [port]      log: /mnt/HDD/eval/server_hang.log
cd "$(dirname "$0")/.."
PORT=${1:-8012}
LOG=/mnt/HDD/eval/server_hang.log
U=http://127.0.0.1:$PORT
ask() {
  curl -s -m 120 $U/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"1+1は？数字だけ。"}],"max_tokens":20}' |
    python -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'] if 'choices' in d else d)"
}
wait_ok() {
  for i in $(seq 1 120); do
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' $U/health)" = 200 ] && return 0; sleep 5
  done
  return 1
}
TSUMI_DEBUG_HANG=1 TSUMI_HANG_BACKSTOP=${BACKSTOP:-300} ./run_server.sh --port $PORT --max-seq 16384 > $LOG 2>&1 &
wrapper=$!
trap 'kill -TERM $wrapper 2>/dev/null; wait $wrapper 2>/dev/null' EXIT
t0=$(date +%s)
wait_ok || { echo "FAIL: server did not start"; exit 1; }
echo "1. up in $(( $(date +%s) - t0 ))s, answer: $(ask)"
t1=$(date +%s)
echo "2. /debug/hang -> $(curl -s -m 200 -w ' (HTTP %{http_code})' -X POST $U/debug/hang) after $(( $(date +%s) - t1 ))s"
echo "   /health -> $(curl -s -m 5 -w ' (HTTP %{http_code})' $U/health)"
echo "   chat -> $(curl -s -m 5 -o /dev/null -w 'HTTP %{http_code}' $U/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hi"}]}')"
wait_ok || { echo "FAIL: server did not come back"; grep run_server $LOG; exit 1; }
echo "3. back after $(( $(date +%s) - t1 ))s"
grep -E "\[run_server\]|device error" $LOG | sed 's/^/   /'
echo "4. answer: $(ask)"
