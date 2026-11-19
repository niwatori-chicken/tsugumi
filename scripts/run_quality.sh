#!/bin/bash
# Overnight quality benchmarks for the quantization variants (todo 2). Results: /mnt/HDD/eval/quality/<variant>/
#   scripts/run_quality.sh [variant ...]        (default: bfp4 attn8 mixed)
#   GPQA=1 scripts/run_quality.sh bfp4 mixed   (GPQA Diamond, thinking on; needs /mnt/HDD/eval/bench/gpqa/gpqa_diamond.csv)
# Each task is a separate process (device opened per task); a failed task does not stop the rest. Re-running resumes.
cd "$(dirname "$0")/.."
OUT=/mnt/HDD/eval/quality
EP=/mnt/HDD/eval/venv-evalplus/bin/evalplus.evaluate
declare -A ARGS=(
  [bfp4]="--prec bfp4 --cache bfp4"
  [mixed]="--prec mixed --cache mixed"
  [attn8]="--prec bfp4 --set gdn_in=bfp8 gdn_out=bfp8 attn_qkvg=bfp8 attn_o=bfp8 --cache combo"
)
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(bfp4 attn8 mixed)
for v in "${VARIANTS[@]}"; do
  d=$OUT/$v; mkdir -p $d
  if [ "${GPQA:-0}" = "1" ]; then
    TASKS=("gpqa --n ${GPQA_N:-50} --think --max-new 12000")
  else
    TASKS=("jcqa" "jsquad --n 500" "jemhop" "mgsm" "humaneval")
  fi
  for t in "${TASKS[@]}"; do
    name=${t%% *}
    echo "=== $(date '+%F %T') $v $name" | tee -a $OUT/run.log
    timeout 6h python scripts/bench_quality.py $name $d/$name.jsonl ${t#$name} ${ARGS[$v]} >> $d/$name.log 2>&1
    echo "    rc=$? $(grep -E '^\[' $d/$name.log | tail -1)" | tee -a $OUT/run.log
    if [ "$name" = humaneval ] && [ -f $d/humaneval_samples.jsonl ]; then
      (cd $d && timeout 1h $EP --dataset humaneval --samples humaneval_samples.jsonl > evalplus.log 2>&1)
      echo "    evalplus: $(grep -E 'pass@1' $d/evalplus.log | tr '\n' ' ')" | tee -a $OUT/run.log
    fi
  done
done
echo "=== $(date '+%F %T') done" | tee -a $OUT/run.log
