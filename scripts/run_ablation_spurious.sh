#!/usr/bin/env bash
# Does the spurious-correctness term (latentsql/db/spurious.py, w_spurious)
# help GRPO beyond dense partial credit alone?
#
# Extends scripts/run_ablation3.sh with a third arm: teacher and SFT run
# ONCE and are shared by all three; GRPO then runs three times from the same
# SFT checkpoint, differing only in the reward -
#   ctrl:     binary reward only
#   dense:    + w_atomic/w_memory (FINER-SQL dense partial credit)
#   spurious: + w_atomic/w_memory AND w_spurious (this session's new term)
# so "does spurious help" is read directly off dense vs spurious, holding
# everything else (including the dense terms themselves) fixed.
#
#   bash scripts/run_ablation_spurious.sh
set -u
OUT=outputs/pilot3
CTRL=configs/pilot_mix3_ctrl.yaml
DENSE=configs/pilot_mix3_dense.yaml
SPURIOUS=configs/pilot_mix3_spurious.yaml
LOGS=$OUT/logs; TRACES=$OUT/traces
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
stage() { echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOGS/run_spurious.log"; }

stage "1/6 teacher: 1/3 BIRD + 2/3 Spider (7B, 4-bit, four-system panel)"
if [ ! -f "$TRACES/sft.jsonl" ]; then
  $PY scripts/01_build_traces.py --config "$CTRL" --dataset mix \
      --limit "${N_TRACE:-90}" --out "$TRACES" 2>&1 | tee "$LOGS/01_traces.log"
fi
N=$(wc -l < "$TRACES/sft.jsonl" 2>/dev/null || echo 0)
stage "traces kept: $N"
[ "$N" -lt 8 ] && { echo "FATAL: $N traces is too few" | tee -a "$LOGS/run_spurious.log"; exit 1; }

stage "2/6 SFT (shared by all three arms)"
if [ ! -f "$OUT/sft/adapter_config.json" ]; then
  $PY scripts/02_sft.py --config "$CTRL" --data "$TRACES/sft.jsonl" \
      2>&1 | tee "$LOGS/02_sft.log"
fi

# --- the ablation: same SFT start, same data, only the reward differs -------
for ARM in ctrl dense spurious; do
  case "$ARM" in
    ctrl) CFG=$CTRL; DESC="binary reward" ;;
    dense) CFG=$DENSE; DESC="w_atomic=0.5 w_memory=0.3" ;;
    spurious) CFG=$SPURIOUS; DESC="w_atomic=0.5 w_memory=0.3 w_spurious=0.5" ;;
  esac
  stage "3/6 GRPO arm=$ARM  ($DESC)"
  if [ ! -f "$OUT/grpo_$ARM/adapter_config.json" ]; then
    timeout "${GRPO_TIMEOUT:-3300}" $PY scripts/03_grpo.py --config "$CFG" \
        --data "$TRACES/sft.jsonl" --out "$OUT/grpo_$ARM" \
        2>&1 | tee "$LOGS/03_grpo_$ARM.log"
    echo "  arm=$ARM exit: $?" | tee -a "$LOGS/run_spurious.log"
  fi
done

stage "4/6 eval: SFT-only student (the shared starting point)"
$PY scripts/04_eval.py --config "$CTRL" --methods student --student "$OUT/sft" \
    --limit "${N_EVAL:-24}" --max-new-tokens 2048 \
    --out "$OUT/eval_sftonly" 2>&1 | tee "$LOGS/04_eval_sftonly.log"

stage "5/6 eval: all three GRPO arms"
for ARM in ctrl dense spurious; do
  D=$OUT/grpo_$ARM
  [ -f "$D/adapter_config.json" ] || { echo "  skip $ARM (no adapter)"; continue; }
  $PY scripts/04_eval.py --config "$CTRL" --methods student --student "$D" \
      --limit "${N_EVAL:-24}" --max-new-tokens 2048 \
      --out "$OUT/eval_$ARM" 2>&1 | tee "$LOGS/05_eval_$ARM.log"
done

stage "6/6 done - compare $OUT/eval_dense vs $OUT/eval_spurious (same SFT start, only w_spurious differs)"
