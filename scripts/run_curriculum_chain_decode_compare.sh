#!/usr/bin/env bash
# Same one-record-per-fresh-subprocess chain as run_curriculum_chain.sh (see
# its header comment for why: in-process runs of this same curriculum left
# PyTorch/CUDA state corrupted after any backward-pass OOM, so every later
# step in that process failed too - a fresh process per record has no such
# history to corrupt).
#
# This variant adds --decode-compare: a "decode-and-compare" auxiliary loss
# on latent roles that still have real teacher text available. Right now
# nothing forces a latent slot to correspond to what that debate agent would
# actually have said - a latent role's role_turns[idx] is computed but
# thrown away, only its bare tag is used to seed the m latent steps. This
# reuses each latent step's own already-computed logits (no extra teacher
# forward pass, no extra vocab matmul - see LatentReasoner._latent_steps'
# docstring) to cross-entropy against the first `latent_steps` tokens of
# that role's real turn, added to the main gold-SQL loss with weight
# --aux-weight (default 0.5, overridable via the 5th positional arg here).
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

N=${1:-282}                                    # total records in this ramp
START=${2:-0}
BASE=${3:-outputs/latent_sft_big/sft}
N_ROLES=${4:-8}                                 # non-consensus roles
AUX_WEIGHT=${5:-0.5}
OUT=outputs/latent_sft_big/curriculum_chain_dc/latent_sft
LOG=outputs/latent_sft_big/curriculum_chain_dc.log
[ "$START" -eq 0 ] && : > "$LOG"

CUR="$BASE"
for i in $(seq "$START" $((N - 1))); do
  # round(i/(N-1) * N_ROLES), matching train_latent_sft's own frac formula
  n_latent_roles=$(python3 -c "print(round($i/max(1,$N-1)*$N_ROLES))")
  echo "=== [$(date +%H:%M:%S)] record $i/$((N-1)), n_latent_roles=$n_latent_roles, from $CUR ===" | tee -a "$LOG"
  LATENTSQL_GPU_FRACTION=0.72 LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 scripts/02c_latent_sft.py \
    --config configs/latent_sft_big.yaml \
    --data outputs/latent_sft_big/traces/traces.jsonl \
    --sft "$CUR" \
    --out "$OUT" \
    --curriculum --latent-steps 12 \
    --roles din_sql,dail_sql,chess,chase_sql,din_sql,dail_sql,chess,chase_sql,consensus \
    --max-prompt-chars 3000 \
    --force-n-latent-roles "$n_latent_roles" \
    --decode-compare --aux-weight "$AUX_WEIGHT" \
    --offset "$i" --limit 1 --epochs 1 --grad-accum 1 --save-steps 1 \
    >> "$LOG" 2>&1
  code=$?
  echo "record $i exit code: $code" | tee -a "$LOG"
  if [ $code -ne 0 ]; then
    echo "!!! record $i FAILED (see $LOG) - skipping, next record starts fresh from $CUR (unchanged)" | tee -a "$LOG"
    continue
  fi
  CUR="$OUT"
done
echo "=== chain done: records $START..$((N-1)) folded into $OUT ===" | tee -a "$LOG"
