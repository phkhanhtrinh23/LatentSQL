#!/usr/bin/env bash
# Phase A: GRPO with the brevity term, from run 3's SFT checkpoint and traces,
#          so the ONLY difference from run 3's dense arm is w_brevity.
# Phase B: re-evaluate everything on 120 dev questions (was 24) for real CIs.
set -u
OUT=outputs/pilot3
LOGS=$OUT/logs
N_EVAL=${N_EVAL:-120}
export PYTORCH_ALLOC_CONF=expandable_segments:True LATENTSQL_GPU_FRACTION=0.5 LATENTSQL_CPU_THREADS=6
PY=${PY:-python3}
stage() { echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOGS/final.log"; }

stage "A/2 GRPO arm=brev (dense + w_brevity=0.5)"
if [ ! -f "$OUT/grpo_brev/adapter_config.json" ]; then
  timeout 3300 $PY scripts/03_grpo.py --config configs/pilot_mix3_brev.yaml \
      --data "$OUT/traces/sft.jsonl" --out "$OUT/grpo_brev" \
      2>&1 | tee "$LOGS/03_grpo_brev.log"
fi

stage "B/2 eval on $N_EVAL questions - baselines"
$PY scripts/04_eval.py --config configs/pilot_mix3_ctrl.yaml \
    --methods single,din_sql,dail_sql,chess,chase_sql \
    --model Qwen/Qwen2.5-Coder-3B-Instruct \
    --limit "$N_EVAL" --max-new-tokens 2048 \
    --out "$OUT/big_eval_3b" 2>&1 | tee "$LOGS/B_eval_3b.log"

for ARM in sft grpo_dense grpo_brev; do
  stage "B/2 eval on $N_EVAL questions - student:$ARM"
  $PY scripts/04_eval.py --config configs/pilot_mix3_ctrl.yaml --methods student \
      --student "$OUT/$ARM" --limit "$N_EVAL" --max-new-tokens 2048 \
      --out "$OUT/big_eval_$ARM" 2>&1 | tee "$LOGS/B_eval_$ARM.log"
done

stage "done - big_eval_* dirs hold the $N_EVAL-question results"
