#!/usr/bin/env bash
# Run 3: does FINER-SQL-style dense partial credit fix GRPO's degenerate groups?
#
# Corpus is 1/3 BIRD + 2/3 Spider. The teacher and SFT run ONCE; GRPO then runs
# twice from the same SFT checkpoint, differing only in w_atomic / w_memory, so
# the reward is the single changed variable.
#
#   bash scripts/run_ablation3.sh
set -u
OUT=outputs/pilot3
CTRL=configs/pilot_mix3_ctrl.yaml
DENSE=configs/pilot_mix3_dense.yaml
LOGS=$OUT/logs; TRACES=$OUT/traces
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
stage() { echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOGS/run3.log"; }

stage "1/6 teacher: 1/3 BIRD + 2/3 Spider (7B, 4-bit, four-system panel)"
if [ ! -f "$TRACES/sft.jsonl" ]; then
  $PY scripts/01_build_traces.py --config "$CTRL" --dataset mix \
      --limit "${N_TRACE:-90}" --out "$TRACES" 2>&1 | tee "$LOGS/01_traces.log"
fi
N=$(wc -l < "$TRACES/sft.jsonl" 2>/dev/null || echo 0)
stage "traces kept: $N"
[ "$N" -lt 8 ] && { echo "FATAL: $N traces is too few" | tee -a "$LOGS/run3.log"; exit 1; }

stage "2/6 SFT (shared by both arms)"
if [ ! -f "$OUT/sft/adapter_config.json" ]; then
  $PY scripts/02_sft.py --config "$CTRL" --data "$TRACES/sft.jsonl" \
      2>&1 | tee "$LOGS/02_sft.log"
fi

# --- the ablation: same SFT start, same data, only the reward differs -------
for ARM in ctrl dense; do
  CFG=$CTRL; [ "$ARM" = dense ] && CFG=$DENSE
  stage "3/6 GRPO arm=$ARM  ($([ "$ARM" = dense ] && echo 'w_atomic=0.5 w_memory=0.3' || echo 'binary reward'))"
  if [ ! -f "$OUT/grpo_$ARM/adapter_config.json" ]; then
    # generous timeout: run 2 was cut at 15/23 steps, which leaves the anneal
    # stranded mid-curriculum and is worse than not running GRPO at all
    timeout "${GRPO_TIMEOUT:-3300}" $PY scripts/03_grpo.py --config "$CFG" \
        --data "$TRACES/sft.jsonl" --out "$OUT/grpo_$ARM" \
        2>&1 | tee "$LOGS/03_grpo_$ARM.log"
    echo "  arm=$ARM exit: $?" | tee -a "$LOGS/run3.log"
  fi
done

stage "4/6 eval: the four published systems on the same 3B"
$PY scripts/04_eval.py --config "$CTRL" \
    --methods single,din_sql,dail_sql,chess,chase_sql \
    --model Qwen/Qwen2.5-Coder-3B-Instruct \
    --limit "${N_EVAL:-24}" --max-new-tokens 2048 \
    --out "$OUT/eval_3b" 2>&1 | tee "$LOGS/04_eval_3b.log"

stage "5/6 eval: SFT-only student (the shared starting point)"
$PY scripts/04_eval.py --config "$CTRL" --methods student --student "$OUT/sft" \
    --limit "${N_EVAL:-24}" --max-new-tokens 2048 \
    --out "$OUT/eval_sftonly" 2>&1 | tee "$LOGS/05_eval_sftonly.log"

stage "6/6 eval: both GRPO arms"
for ARM in ctrl dense; do
  D=$OUT/grpo_$ARM
  [ -f "$D/adapter_config.json" ] || { echo "  skip $ARM (no adapter)"; continue; }
  $PY scripts/04_eval.py --config "$CTRL" --methods student --student "$D" \
      --limit "${N_EVAL:-24}" --max-new-tokens 2048 \
      --out "$OUT/eval_$ARM" 2>&1 | tee "$LOGS/06_eval_$ARM.log"
done

stage "done - compare $OUT/eval_ctrl vs $OUT/eval_dense"
