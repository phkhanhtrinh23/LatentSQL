#!/usr/bin/env bash
# COCONUT-faithful STAGED curriculum: hold n_latent_roles FIXED at each
# discrete stage for EPOCHS_PER_STAGE full passes over the WHOLE corpus,
# THEN advance to the next stage - matching the actual schedule in
# COCONUT's own reference repo (facebookresearch/coconut,
# args/gsm_coconut.yaml: epochs_per_stage=3, max_latent_stage=3), i.e. a
# hold-then-advance schedule over discrete stages.
#
# This is deliberately different from run_curriculum_chain_batched.sh's
# CONTINUOUS ramp (n_latent_roles = round(batch_index / n_batches *
# n_roles), so depth changes every batch within a single pass over the
# corpus) - that script's own multipass wrapper repeats the whole
# continuous ramp N_PASSES times as "a practical middle ground" toward
# this, not a reproduction of it (see its own header comment). This
# script is the real thing: for EACH discrete n_latent_roles value
# (0..N_ROLES inclusive), the WHOLE corpus is walked EPOCHS_PER_STAGE
# times with that depth held constant, before moving to the next depth.
#
# Uses the strict no-KV COCONUT training objective in 02c_latent_sft.py:
# examples without full teacher role text are dropped, prompt/latent tokens
# are masked, and the remaining role text plus consensus answer is trained
# with normal teacher-forced next-token loss. One fresh subprocess per batch
# keeps a failed CUDA backward from corrupting a long-running process.
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

N=${1:-auto}                                    # strict usable corpus size, or an explicit cap
BATCH_SIZE=${2:-16}
N_ROLES=${3:-8}                                 # non-consensus roles; stages run 0..N_ROLES inclusive
EPOCHS_PER_STAGE=${4:-3}                        # COCONUT's own gsm_coconut.yaml value
BASE=${5:-outputs/latent_sft_1000/sft_7b_bf16}
DATA=${6:-outputs/latent_sft_1000/traces/traces.jsonl}
CONFIG=${7:-configs/latent_sft_1000_7b_bf16.yaml}
EVAL_N=${8:-30}                                 # eval runs once per COMPLETED stage, not per batch count
LR=${9:-1e-4}
LATENT_STEPS=${10:-2}                           # validated safe across full depth 0..8 this session -
                                                 # see comment above; do not raise without re-testing
RUNDIR=${11:-outputs/latent_sft_1000/coconut_staged}
GPU_FRACTION=${12:-0.96}
BASE_MODEL=${13:-Qwen/Qwen2.5-Coder-7B-Instruct}
MAX_PROMPT_CHARS=${14:-2000}
MAX_ROLE_CHARS=${15:-800}
START_STAGE=${16:-1}                            # stage 0 is ordinary CoT/SFT; use BASE for it
START_EPOCH=${17:-1}                            # 1-indexed epoch within START_STAGE to resume at
MAX_TOTAL_TOKENS=${18:-1200}                    # preflight prompt+all-role-turn token cap;
                                                 # records above this are dropped, not truncated

OUT="$RUNDIR/latent_sft"
OPTIM_STATE="$RUNDIR/optim_state.pt"
LOG="${RUNDIR}.log"
EVAL_LOG="${RUNDIR}_eval_progress.log"
EVAL_DIR="${RUNDIR}_eval_progress"
mkdir -p "$RUNDIR"

USABLE_N=$(python3 - "$DATA" "$CONFIG" "$MAX_PROMPT_CHARS" "$MAX_ROLE_CHARS" "$BASE_MODEL" "$MAX_TOTAL_TOKENS" <<'PY'
import sys
import contextlib
import io
from latentsql.config import load_config
from latentsql.mas.trace import load_traces
from latentsql.distill.latent_sft import build_records_from_traces
from latentsql.llm.hf_local import resolve_model_path
from transformers import AutoTokenizer

data, config, max_prompt, max_role, base_model, max_total = sys.argv[1:7]
cfg = load_config(config)
base_path = resolve_model_path(base_model, hub_dir=cfg.paths.hf_hub, local_only=False)
tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
with contextlib.redirect_stdout(io.StringIO()):
    records = build_records_from_traces(
        load_traces(data), cfg,
        max_prompt_chars=int(max_prompt) if max_prompt else None,
        max_role_chars=int(max_role) if max_role else None,
        tokenizer=tok,
        max_total_tokens=int(max_total) if max_total else None,
    )
print(sum(1 for r in records if r.get("role_turns") is not None))
PY
)
if [ "$N" = "auto" ]; then
  N="$USABLE_N"
elif [ "$N" -gt "$USABLE_N" ]; then
  echo "[coconut] requested N=$N, but only $USABLE_N strict role-turn records are usable; clamping N=$USABLE_N" | tee -a "$LOG"
  N="$USABLE_N"
fi
if [ "$N" -le 0 ]; then
  echo "!!! no strict role-turn records available for COCONUT staged training" | tee -a "$LOG"
  exit 1
fi
N_BATCHES=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "[coconut] using N=$N strict role-turn records ($USABLE_N available), batch_size=$BATCH_SIZE" | tee -a "$LOG"

FRESH_START=0
[ "$START_STAGE" -le 1 ] && [ "$START_EPOCH" -eq 1 ] && FRESH_START=1
if [ "$FRESH_START" -eq 1 ]; then
  : > "$LOG"; : > "$EVAL_LOG"; rm -f "$OPTIM_STATE"
  CUR="$BASE"
else
  CUR="$OUT"   # resuming: pick up whatever this RUNDIR already trained
fi

for stage in $(seq "$START_STAGE" "$N_ROLES"); do
  ep_start=1
  [ "$stage" -eq "$START_STAGE" ] && ep_start="$START_EPOCH"
  if [ "$ep_start" -eq 1 ]; then
    # COCONUT resets optimizer state when switching stages. Keep momentum
    # across subprocess batches inside a stage, but not across stage changes.
    rm -f "$OPTIM_STATE"
  fi
  for epoch in $(seq "$ep_start" "$EPOCHS_PER_STAGE"); do
    echo "=== [$(date +%H:%M:%S)] STAGE $stage/$N_ROLES epoch $epoch/$EPOCHS_PER_STAGE (n_latent_roles=$stage FIXED, from $CUR) ===" | tee -a "$LOG"
    for b in $(seq 0 $((N_BATCHES - 1))); do
      offset=$((b * BATCH_SIZE))
      remaining=$((N - offset))
      limit=$BATCH_SIZE
      [ "$remaining" -lt "$BATCH_SIZE" ] && limit=$remaining
      [ "$limit" -le 0 ] && break
      LATENTSQL_GPU_FRACTION=$GPU_FRACTION LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
      python3 scripts/02c_latent_sft.py \
        --config "$CONFIG" \
        --data "$DATA" \
        --sft "$CUR" \
        --out "$OUT" \
        --curriculum --latent-steps "$LATENT_STEPS" \
        --roles din_sql,dail_sql,chess,chase_sql,din_sql,dail_sql,chess,chase_sql,consensus \
        --max-prompt-chars "$MAX_PROMPT_CHARS" --max-role-chars "$MAX_ROLE_CHARS" \
        --max-total-tokens "$MAX_TOTAL_TOKENS" \
        --force-n-latent-roles "$stage" \
        --optim-state "$OPTIM_STATE" \
        --coconut-objective --require-role-turns --text-weight 1.0 --lr "$LR" \
        --offset "$offset" --limit "$limit" --epochs 1 --grad-accum "$limit" --save-steps "$limit" \
        >> "$LOG" 2>&1
      code=$?
      echo "stage $stage epoch $epoch batch $b exit code: $code" | tee -a "$LOG"
      if [ $code -ne 0 ]; then
        echo "!!! stage $stage epoch $epoch batch $b FAILED - stopping staged COCONUT run" | tee -a "$LOG"
        exit $code
      fi
      CUR="$OUT"
    done
    echo "=== [$(date +%H:%M:%S)] stage $stage epoch $epoch/$EPOCHS_PER_STAGE done, from $CUR ===" | tee -a "$LOG"
  done

  echo "=== [$(date +%H:%M:%S)] intermediate eval after stage $stage (n=$EVAL_N) ===" | tee -a "$LOG" "$EVAL_LOG"
  LATENTSQL_GPU_FRACTION=$GPU_FRACTION LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 scripts/04_eval.py \
    --config "$CONFIG" \
    --methods latent \
    --student "$CUR" \
    --base "$BASE_MODEL" \
    --latent-steps "$LATENT_STEPS" \
    --latent-roles "$stage" \
    --limit "$EVAL_N" \
    --out "${EVAL_DIR}/stage_${stage}" \
    >> "$EVAL_LOG" 2>&1
  eval_code=$?
  if [ $eval_code -eq 0 ]; then
    grep "^latent " "$EVAL_LOG" | tail -1 | sed "s/^/stage $stage: /" | tee -a "$LOG"
  else
    echo "!!! eval after stage $stage FAILED (exit $eval_code, see $EVAL_LOG) - training continues" | tee -a "$LOG"
  fi
  # Preserve the adapter actually evaluated at this stage. The previous
  # runner reused one directory, making it impossible to rerun stage-4/5
  # evaluation after correcting the evaluator.
  STAGE_OUT="$RUNDIR/checkpoints/stage_${stage}"
  mkdir -p "$STAGE_OUT"
  cp -a "$OUT/." "$STAGE_OUT/"
  echo "saved stage $stage checkpoint to $STAGE_OUT" | tee -a "$LOG"
done
echo "=== coconut-staged done: stages $START_STAGE..$N_ROLES folded into $OUT ===" | tee -a "$LOG"
