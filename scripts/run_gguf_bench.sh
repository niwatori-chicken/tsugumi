#!/bin/bash
# Same benchmark prompts on a llama.cpp GGUF (CPU) through llama-server (todo 2: reference point for the engine).
#   scripts/run_gguf_bench.sh QUANT TASK [bench_quality args ...]     e.g. Q4_K_M jcqa
# Results: /mnt/HDD/eval/quality/gguf_<QUANT>/<TASK>.jsonl. Builds llama-server first if needed.
# Do not run while the Tenstorrent device is in use: a saturated CPU has broken device opens before.
cd "$(dirname "$0")/.."
Q=$1; TASK=$2; shift 2
LC=/mnt/HDD/src/llama.cpp
GGUF=/mnt/HDD/models/gguf/Qwen3.8-27B-$Q.gguf
OUT=/mnt/HDD/eval/quality/gguf_$Q; mkdir -p $OUT
PORT=8090
[ -x $LC/build/bin/llama-server ] || cmake --build $LC/build --target llama-server -j 16 > $OUT/build.log 2>&1
$LC/build/bin/llama-server -m $GGUF --jinja -c 8192 -np 1 -t 16 --port $PORT --host 127.0.0.1 > $OUT/server_$TASK.log 2>&1 &
SP=$!
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  kill -0 $SP 2>/dev/null || { echo "llama-server died"; exit 1; }
  sleep 5
done
python scripts/bench_quality.py $TASK $OUT/$TASK.jsonl --api http://127.0.0.1:$PORT "$@" > $OUT/$TASK.log 2>&1
grep -E '^\[' $OUT/$TASK.log | tail -1
kill $SP; wait $SP 2>/dev/null
