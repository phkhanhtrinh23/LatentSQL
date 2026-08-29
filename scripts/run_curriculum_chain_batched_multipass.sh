#!/usr/bin/env bash
# Wraps run_curriculum_chain_batched.sh to run N_PASSES full monotonic
# curriculum ramps (0 -> full depth) back to back over the SAME corpus,
# instead of just one - a practical middle ground toward COCONUT's own
# per-stage training budget (they hold each discrete depth stage for 3
# full epochs; we don't have discrete stages, but repeating the whole
# ramp N_PASSES times gives each depth level N_PASSES x more exposure
# than a single pass, which only touched it for a handful of batches).
#
# Checkpoint AND AdamW momentum both carry across passes (WIPE=0 from
# pass 2 onward) - only the curriculum's own ramp position resets to
# batch 0 each pass, not the model or optimizer state.
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

N_PASSES=${1:-5}
N=${2:-813}
BATCH_SIZE=${3:-16}
BASE=${4:-outputs/latent_sft_big/sft}
N_ROLES=${5:-8}
DATA=${6:-outputs/latent_sft_1000/traces/traces.jsonl}
CONFIG=${7:-configs/latent_sft_1000.yaml}
EVAL_EVERY=${8:-10}
EVAL_N=${9:-30}
LR=${10:-1e-4}
LATENT_STEPS=${11:-12}
RUNDIR=${12:-outputs/latent_sft_1000/curriculum_chain_batched}
OUT="$RUNDIR/latent_sft"
MPLOG="${RUNDIR}_multipass.log"
: > "$MPLOG"

CUR="$BASE"
for p in $(seq 1 "$N_PASSES"); do
  wipe=1
  [ "$p" -gt 1 ] && wipe=0
  echo "=== [$(date +%H:%M:%S)] PASS $p/$N_PASSES starting from $CUR (wipe=$wipe) ===" | tee -a "$MPLOG"
  bash scripts/run_curriculum_chain_batched.sh "$N" "$BATCH_SIZE" 0 "$CUR" "$N_ROLES" \
    "$DATA" "$CONFIG" "$EVAL_EVERY" "$EVAL_N" "$LR" "$wipe" "$LATENT_STEPS" "$RUNDIR"
  code=$?
  echo "=== [$(date +%H:%M:%S)] PASS $p/$N_PASSES done, exit code $code ===" | tee -a "$MPLOG"
  if [ $code -ne 0 ]; then
    echo "!!! pass $p failed - stopping multipass run" | tee -a "$MPLOG"
    exit 1
  fi
  CUR="$OUT"
done
echo "=== multipass done: $N_PASSES passes folded into $OUT ===" | tee -a "$MPLOG"
