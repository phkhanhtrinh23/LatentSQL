#!/usr/bin/env bash
# COCONUT-style curriculum latent-SFT, one record per fresh subprocess,
# chaining the checkpoint forward - the curriculum analogue of
# run_latent_sft_chain.sh (see its header comment: this exact in-process
# accumulation/fragmentation OOM is a failure mode this project has already
# hit and solved once before with the same trick).
#
# Three separate in-process attempts at the full 217-282 record curriculum
# corpus all died at a reproducible-but-content-independent point (same
# step, same qid, across reruns, regardless of exception-handling details or
# how conservatively the per-record length was capped): something in
# PyTorch/CUDA's own state gets left corrupted once a backward pass OOMs
# inside a checkpointed region, and every later step in that SAME process
# then OOMs too, at near-identical numbers, no matter what record comes
# next. A fresh process per record has no such history to corrupt.
#
# Curriculum depth can't be computed from a subprocess's own step count
# (always 1, one record per process) - this script computes each record's
# position in the GLOBAL ramp itself and passes it via
# --force-n-latent-roles.
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

N=${1:-282}                                    # total records in this ramp
START=${2:-0}
BASE=${3:-outputs/latent_sft_big/sft}
N_ROLES=${4:-8}                                 # non-consensus roles
OUT=outputs/latent_sft_big/curriculum_chain/latent_sft
LOG=outputs/latent_sft_big/curriculum_chain.log
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
