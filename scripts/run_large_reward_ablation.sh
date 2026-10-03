#!/usr/bin/env bash
set -uo pipefail

CFG=configs/large_reward_ablation.yaml
DATA=outputs/latent_sft_1000/traces/sft.jsonl
SFT=outputs/latent_sft_1000/sft_7b_bf16
OUT=outputs/large_reward_ablation
LOGS=$OUT/logs
PY=${PY:-python3}
TRAIN_LIMIT=${TRAIN_LIMIT:-192}
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True

stage() { echo "=== [$(date --iso-8601=seconds)] $* ===" | tee -a "$LOGS/run.log"; }

for ARM in binary dense spurious; do
  ARM_OUT=$OUT/grpo_$ARM
  EXTRA=()
  case "$ARM" in
    binary) EXTRA+=(--w-atomic 0 --w-memory 0 --w-spurious 0) ;;
    dense) EXTRA+=(--w-atomic 0.5 --w-memory 0.3 --w-spurious 0) ;;
    spurious) EXTRA+=(--w-atomic 0.5 --w-memory 0.3 --w-spurious 0.5 \
                     --n-stress-instances 3 --cleanup-stress-cache) ;;
  esac
  stage "GRPO $ARM: $TRAIN_LIMIT/677 traces, shared 7B SFT initialization"
  "$PY" scripts/03_grpo.py --config "$CFG" --data "$DATA" --sft "$SFT" \
    --out "$ARM_OUT" --limit "$TRAIN_LIMIT" --max-new-tokens 512 \
    --no-spec-decode "${EXTRA[@]}" 2>&1 | tee "$LOGS/grpo_$ARM.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    stage "FAILED $ARM exit=$status"
    exit "$status"
  fi
done

for ARM in binary dense spurious; do
  stage "ordinary EX evaluation: $ARM"
  "$PY" scripts/04_eval.py --config "$CFG" --methods student \
    --student "$OUT/grpo_$ARM" --limit 90 --max-new-tokens 1024 \
    --out "$OUT/eval_$ARM" 2>&1 | tee "$LOGS/eval_$ARM.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    stage "FAILED eval $ARM exit=$status"
    exit "$status"
  fi
done

stage "training and ordinary EX evaluation complete"
