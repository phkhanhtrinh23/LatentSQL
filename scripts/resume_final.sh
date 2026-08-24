#!/usr/bin/env bash
# Resume the 90-question eval after the session restart killed it.
# 4 baselines already have preds; only chase_sql + the 3 student checkpoints remain.
set -u
OUT=outputs/pilot3; LOGS=$OUT/logs; N=90
export PYTORCH_ALLOC_CONF=expandable_segments:True LATENTSQL_GPU_FRACTION=0.5 LATENTSQL_CPU_THREADS=6
PY=${PY:-python3}
stage(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOGS/final.log"; }

stage "resume 1/4 baseline chase_sql"
[ -f "$OUT/big_eval_chase/reports.json" ] || \
  $PY scripts/04_eval.py --config configs/pilot_mix3_ctrl.yaml --methods chase_sql \
      --model Qwen/Qwen2.5-Coder-3B-Instruct --limit $N --max-new-tokens 2048 \
      --out "$OUT/big_eval_chase" 2>&1 | tee "$LOGS/B_eval_chase.log"

i=2
for ARM in sft grpo_dense grpo_brev; do
  stage "resume $i/4 student:$ARM"
  [ -f "$OUT/big_eval_$ARM/reports.json" ] || \
    $PY scripts/04_eval.py --config configs/pilot_mix3_ctrl.yaml --methods student \
        --student "$OUT/$ARM" --limit $N --max-new-tokens 2048 \
        --out "$OUT/big_eval_$ARM" 2>&1 | tee "$LOGS/B_eval_$ARM.log"
  i=$((i+1))
done
stage "resume done"
