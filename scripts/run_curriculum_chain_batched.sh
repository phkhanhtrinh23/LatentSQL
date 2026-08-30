#!/usr/bin/env bash
# Curriculum latent-SFT chain, BATCHED: BATCH_SIZE records per fresh
# subprocess (grad_accum = the batch's own size, so all of it folds into
# ONE optimizer.step(), not BATCH_SIZE separate noisy ones), with AdamW
# momentum (exp_avg/exp_avg_sq) persisted across subprocess boundaries via
# --optim-state - so the ONLY thing reset per subprocess is CUDA/checkpoint
# state (the actual corruption bug this whole chain design exists to avoid),
# not the optimizer's own memory of past gradients. This is meant to close
# both gaps identified against COCONUT's own training regime: real batched
# gradients, real momentum continuity.
#
# See run_curriculum_chain.sh for why one-record-per-fresh-process exists at
# all (in-process runs left PyTorch/CUDA state corrupted after any
# checkpointed backward-pass OOM).
#
# With grad_accum spanning the whole batch, ANY one record's OOM aborts the
# WHOLE batch (all-or-nothing backward before optim.step()) - the earlier
# single-record chains lost ~1/3 of records to isolated OOMs, which at
# batch=8 would poison nearly every batch. --max-role-chars keeps most
# records on the always-cheap fully-latent fallback path (a record with no
# usable role-turn text runs latent regardless of curriculum depth - see
# build_prefix's `not role_turns` check) rather than the more expensive
# real-text path only a minority of short-role-turn records now take.
# EVAL_EVERY batches, this stops and runs a smaller-n eval (--methods latent)
# on the checkpoint SO FAR, logging EX + loss trend to EVAL_LOG - "track
# progress closely" instead of only learning the result at the very end.
# Uses the held-out dev split from EVAL_CONFIG's own dataset mix.
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

N=${1:-auto}                                    # total FILTERED records in this ramp
BATCH_SIZE=${2:-16}
START_BATCH=${3:-0}
BASE=${4:-outputs/latent_sft_big/sft}
N_ROLES=${5:-8}                                 # non-consensus roles
DATA=${6:-outputs/latent_sft_1000/traces/traces.jsonl}
CONFIG=${7:-configs/latent_sft_1000.yaml}
EVAL_EVERY=${8:-10}                             # batches between intermediate evals (0 = off)
EVAL_N=${9:-30}                                 # smaller n for speed; final eval uses the full 90
LR=${10:-1e-4}                                  # COCONUT's own value (was 1e-5)
WIPE=${11:-1}                                   # 0 for pass 2+ of a multi-pass run - see
                                                 # run_curriculum_chain_batched_multipass.sh;
                                                 # keeps the log/optimizer-momentum continuous
                                                 # across passes instead of resetting them just
                                                 # because each pass's own curriculum ramp
                                                 # restarts at batch 0.
LATENT_STEPS=${12:-2}                           # COCONUT's own demonstrated max is c_thought=2 x
                                                 # max_latent_stage=3 = 6 total latent tokens ever.
                                                 # Also a hard memory constraint on this card: each
                                                 # latent role is LATENT_STEPS chained, uncheckpointed
                                                 # (HF per-layer checkpointing is off for this whole
                                                 # path - see latent_sft.py) forward calls: at
                                                 # LATENT_STEPS=5, n_latent_roles=4 (20 chained calls)
                                                 # OOM'd on 100% of batches in the 7B bf16 curriculum
                                                 # run; LATENT_STEPS=2 (8 chained calls at the same
                                                 # depth, 16 at full depth=8) fits with margin.
RUNDIR=${13:-outputs/latent_sft_1000/curriculum_chain_batched}
GPU_FRACTION=${14:-0.72}                        # 0.893 ~= 21GB of the 23.508GB card
BASE_MODEL=${15:-Qwen/Qwen2.5-Coder-3B-Instruct}
MAX_PROMPT_CHARS=${16:-3000}
MAX_ROLE_CHARS=${17:-600}
OUT="$RUNDIR/latent_sft"
OPTIM_STATE="$RUNDIR/optim_state.pt"
LOG="${RUNDIR}.log"
EVAL_LOG="${RUNDIR}_eval_progress.log"
EVAL_DIR="${RUNDIR}_eval_progress"
mkdir -p "$RUNDIR"
[ "$START_BATCH" -eq 0 ] && [ "$WIPE" -eq 1 ] && { : > "$LOG"; : > "$EVAL_LOG"; rm -f "$OPTIM_STATE"; }

USABLE_N=$(python3 - "$DATA" "$CONFIG" "$MAX_PROMPT_CHARS" "$MAX_ROLE_CHARS" <<'PY'
import contextlib
import io
import sys

from latentsql.config import load_config
from latentsql.distill.latent_sft import build_records_from_traces
from latentsql.mas.trace import load_traces

data, config, max_prompt, max_role = sys.argv[1:5]
cfg = load_config(config)
max_prompt = None if max_prompt in ("", "none", "None") else int(max_prompt)
max_role = None if max_role in ("", "none", "None") else int(max_role)
with contextlib.redirect_stdout(io.StringIO()):
    records = build_records_from_traces(
        load_traces(data),
        cfg,
        max_prompt_chars=max_prompt,
        max_role_chars=max_role,
    )
print(len(records))
PY
)
case "$N" in
  auto)
    N="$USABLE_N"
    ;;
  *)
    if [ "$N" -gt "$USABLE_N" ]; then
      echo "[curriculum] requested N=$N, but only $USABLE_N records remain after filtering; clamping N=$USABLE_N" | tee -a "$LOG"
      N="$USABLE_N"
    fi
    ;;
esac

N_BATCHES=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "[curriculum] using N=$N filtered records ($USABLE_N available), batch_size=$BATCH_SIZE" | tee -a "$LOG"

CUR="$BASE"
declare -a OK_DEPTH=()
for b in $(seq "$START_BATCH" $((N_BATCHES - 1))); do
  offset=$((b * BATCH_SIZE))
  remaining=$((N - offset))
  limit=$BATCH_SIZE
  [ "$remaining" -lt "$BATCH_SIZE" ] && limit=$remaining
  [ "$limit" -le 0 ] && break
  # round(b/(N_BATCHES-1) * N_ROLES), matching train_latent_sft's own frac formula
  n_latent_roles=$(python3 -c "print(round($b/max(1,$N_BATCHES-1)*$N_ROLES))")
  echo "=== [$(date +%H:%M:%S)] batch $b/$((N_BATCHES-1)) (records $offset..$((offset+limit-1))), n_latent_roles=$n_latent_roles, from $CUR ===" | tee -a "$LOG"
  LATENTSQL_GPU_FRACTION=$GPU_FRACTION LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 scripts/02c_latent_sft.py \
    --config "$CONFIG" \
    --data "$DATA" \
    --sft "$CUR" \
    --out "$OUT" \
    --curriculum --latent-steps "$LATENT_STEPS" \
    --roles din_sql,dail_sql,chess,chase_sql,din_sql,dail_sql,chess,chase_sql,consensus \
    --max-prompt-chars "$MAX_PROMPT_CHARS" --max-role-chars "$MAX_ROLE_CHARS" \
    --force-n-latent-roles "$n_latent_roles" \
    --optim-state "$OPTIM_STATE" \
    --supervise-text --text-weight 1.0 --lr "$LR" \
    --offset "$offset" --limit "$limit" --epochs 1 --grad-accum "$limit" --save-steps "$limit" \
    >> "$LOG" 2>&1
  code=$?
  echo "batch $b exit code: $code" | tee -a "$LOG"
  if [ $code -ne 0 ]; then
    echo "!!! batch $b FAILED (see $LOG) - skipping, next batch starts fresh from $CUR (unchanged)" | tee -a "$LOG"
    continue
  fi
  OK_DEPTH[$n_latent_roles]=$(( ${OK_DEPTH[$n_latent_roles]:-0} + 1 ))
  CUR="$OUT"

  if [ "$EVAL_EVERY" -gt 0 ] && [ $(( (b + 1) % EVAL_EVERY )) -eq 0 ]; then
    echo "=== [$(date +%H:%M:%S)] intermediate eval after batch $b (n=$EVAL_N) ===" | tee -a "$LOG" "$EVAL_LOG"
    LATENTSQL_GPU_FRACTION=$GPU_FRACTION LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
    python3 scripts/04_eval.py \
      --config "$CONFIG" \
      --methods latent \
      --student "$CUR" \
      --base "$BASE_MODEL" \
      --limit "$EVAL_N" \
      --out "${EVAL_DIR}/batch_${b}" \
      >> "$EVAL_LOG" 2>&1
    eval_code=$?
    if [ $eval_code -eq 0 ]; then
      grep "^latent " "$EVAL_LOG" | tail -1 | sed "s/^/batch $b: /" | tee -a "$LOG"
    else
      echo "!!! intermediate eval after batch $b FAILED (exit $eval_code, see $EVAL_LOG) - training continues" | tee -a "$LOG"
    fi
  fi
done
echo "=== chain done: batches $START_BATCH..$((N_BATCHES-1)) folded into $OUT ===" | tee -a "$LOG"
if [ "${OK_DEPTH[$N_ROLES]:-0}" -eq 0 ]; then
  echo "!!! no successful full-depth batches (n_latent_roles=$N_ROLES); checkpoint is not valid for full latent eval" | tee -a "$LOG"
  exit 2
fi
