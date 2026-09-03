#!/usr/bin/env bash
# CHESS-step latent curriculum, SQL-direct.
#
# Stages hide CHESS-style pipeline steps as latent vectors:
#   information_retriever -> schema_linker -> query_classifier ->
#   candidate_generator -> query_fixer -> unit_tester -> consensus SQL
#
# No remaining agent debate text is generated or supervised. At eval time the
# model runs N latent CHESS step markers, then jumps straight to <|consensus|>
# and decodes SQL.
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

N=${1:-94}
BATCH_SIZE=${2:-16}
N_STEPS=${3:-6}
EPOCHS_PER_STAGE=${4:-3}
BASE=${5:-outputs/latent_sft_1000/sft_7b_bf16}
DATA=${6:-outputs/latent_sft_1000/traces/traces.jsonl}
CONFIG=${7:-configs/latent_sft_1000_7b_bf16.yaml}
EVAL_N=${8:-30}
LR=${9:-5e-5}
LATENT_STEPS=${10:-2}
RUNDIR=${11:-outputs/latent_sft_1000/chess_coconut_sql_direct}
GPU_FRACTION=${12:-0.96}
BASE_MODEL=${13:-Qwen/Qwen2.5-Coder-7B-Instruct}
MAX_PROMPT_CHARS=${14:-2000}
MAX_ROLE_CHARS=${15:-1600}
START_STAGE=${16:-1}
START_EPOCH=${17:-1}
MAX_TOTAL_TOKENS=${18:-1400}
LATE_LR=${19:-2e-5}
SQL_READY_ROLE=${20:-4}
SQL_READY_WEIGHT=${21:-0.5}
SQL_READY_TOKENS=${22:-16}

# Each training batch and each eval is a fresh process, and the runner launches
# the next one the moment the previous shell command returns. Process exit does
# not mean the CUDA context is torn down: a predecessor can still hold several
# GB for a second or two while it releases. The successor then prefills into
# whatever is left of the 0.96 budget and dies with OOM -- observed on the
# stage-4 restart, where batch 1 hit "Process <pid> has 5.91 GiB memory in use"
# and aborted a run whose own peak was only 18.5 GB. Wait for the card to
# actually drain before handing it to the next process.
gpu_settle() {
  local want=${LATENTSQL_GPU_SETTLE_MB:-1500}
  local used=""
  local tries=0
  while [ "$tries" -lt 60 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && return 0
    [ "$used" -lt "$want" ] && return 0
    sleep 2
    tries=$((tries + 1))
  done
  echo "[chess_coconut] warning: GPU still holding ${used}MiB after 120s; launching anyway" | tee -a "$LOG"
}

ROLES="information_retriever,schema_linker,query_classifier,candidate_generator,query_fixer,unit_tester,consensus"
OUT="$RUNDIR/latent_sft"
OPTIM_STATE="$RUNDIR/optim_state.pt"
LOG="${RUNDIR}.log"
EVAL_LOG="${RUNDIR}_eval_progress.log"
EVAL_DIR="${RUNDIR}_eval_progress"
mkdir -p "$RUNDIR"

AVAILABLE_N=$(python3 - "$DATA" "$CONFIG" "$MAX_PROMPT_CHARS" "$MAX_ROLE_CHARS" "$BASE_MODEL" "$MAX_TOTAL_TOKENS" <<'PY'
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
cfg.latent.roles = [
    "information_retriever", "schema_linker", "query_classifier",
    "candidate_generator", "query_fixer", "unit_tester", "consensus",
]
base_path = resolve_model_path(base_model, hub_dir=cfg.paths.hf_hub, local_only=False)
tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
with contextlib.redirect_stdout(io.StringIO()):
    records = build_records_from_traces(
        load_traces(data), cfg,
        max_prompt_chars=int(max_prompt) if max_prompt else None,
        max_role_chars=int(max_role) if max_role else None,
        tokenizer=tok,
        max_total_tokens=int(max_total) if max_total else None,
        chain_format="chess_steps",
    )
print(sum(1 for r in records if r.get("gold_sql")))
PY
)
if [ "$N" = "auto" ]; then
  N="$AVAILABLE_N"
elif [ "$N" -gt "$AVAILABLE_N" ]; then
  echo "[chess_coconut] requested N=$N, but only $AVAILABLE_N records are usable; clamping" | tee -a "$LOG"
  N="$AVAILABLE_N"
fi
if [ "$N" -le 0 ]; then
  echo "!!! no records available for CHESS-step SQL-direct training" | tee -a "$LOG"
  exit 1
fi
N_BATCHES=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))

FRESH_START=0
[ "$START_STAGE" -le 1 ] && [ "$START_EPOCH" -eq 1 ] && FRESH_START=1
if [ "$FRESH_START" -eq 1 ]; then
  : > "$LOG"; : > "$EVAL_LOG"; rm -f "$OPTIM_STATE"
  CUR="$BASE"
else
  CUR="$OUT"
fi

echo "[chess_coconut] using N=$N records ($AVAILABLE_N available), batch_size=$BATCH_SIZE, roles=$ROLES" | tee -a "$LOG"

for stage in $(seq "$START_STAGE" "$N_STEPS"); do
  ep_start=1
  [ "$stage" -eq "$START_STAGE" ] && ep_start="$START_EPOCH"
  if [ "$ep_start" -eq 1 ]; then
    rm -f "$OPTIM_STATE"
  fi
  for epoch in $(seq "$ep_start" "$EPOCHS_PER_STAGE"); do
    EFFECTIVE_LR="$LR"
    if [ "$stage" -ge 4 ]; then
      EFFECTIVE_LR="$LATE_LR"
    fi
    SQL_READY_ARGS=()
    if [ -n "$SQL_READY_ROLE" ] && [ "$SQL_READY_WEIGHT" != "0" ] && [ "$SQL_READY_WEIGHT" != "0.0" ]; then
      SQL_READY_ARGS=(
        --sql-ready-role "$SQL_READY_ROLE"
        --sql-ready-weight "$SQL_READY_WEIGHT"
        --sql-ready-tokens "$SQL_READY_TOKENS"
      )
    fi
    echo "=== [$(date +%H:%M:%S)] CHESS STAGE $stage/$N_STEPS epoch $epoch/$EPOCHS_PER_STAGE (latent_roles=$stage, lr=$EFFECTIVE_LR, from $CUR) ===" | tee -a "$LOG"
    for b in $(seq 0 $((N_BATCHES - 1))); do
      offset=$((b * BATCH_SIZE))
      remaining=$((N - offset))
      limit=$BATCH_SIZE
      [ "$remaining" -lt "$BATCH_SIZE" ] && limit=$remaining
      [ "$limit" -le 0 ] && break
      # Retry once on failure: the two failure modes seen at a batch boundary
      # (predecessor's CUDA context not yet released -> OOM, or
      # torch.cuda.is_available() transiently False -> LATENTSQL_REQUIRE_CUDA
      # abort) are both transient and both clear within seconds.
      code=1
      for attempt in 1 2; do
      gpu_settle
      LATENTSQL_GPU_FRACTION=$GPU_FRACTION LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
      LATENTSQL_REQUIRE_CUDA=1 \
      python3 scripts/02c_latent_sft.py \
        --config "$CONFIG" \
        --data "$DATA" \
        --sft "$CUR" \
        --out "$OUT" \
        --curriculum --latent-steps "$LATENT_STEPS" \
        --roles "$ROLES" \
        --chain-format chess_steps \
        --max-prompt-chars "$MAX_PROMPT_CHARS" --max-role-chars "$MAX_ROLE_CHARS" \
        --max-total-tokens "$MAX_TOTAL_TOKENS" \
        --force-n-latent-roles "$stage" \
        --optim-state "$OPTIM_STATE" \
        --coconut-objective --lr "$EFFECTIVE_LR" \
        "${SQL_READY_ARGS[@]}" \
        --offset "$offset" --limit "$limit" --epochs 1 --grad-accum "$limit" --save-steps "$limit" \
        >> "$LOG" 2>&1
      code=$?
      [ $code -eq 0 ] && break
      echo "stage $stage epoch $epoch batch $b attempt $attempt failed (exit $code)" | tee -a "$LOG"
      [ $attempt -lt 2 ] && sleep 20
      done
      echo "stage $stage epoch $epoch batch $b exit code: $code" | tee -a "$LOG"
      if [ $code -ne 0 ]; then
        echo "!!! stage $stage epoch $epoch batch $b FAILED - stopping CHESS COCONUT run" | tee -a "$LOG"
        exit $code
      fi
      CUR="$OUT"
    done
  done

  echo "=== [$(date +%H:%M:%S)] CHESS eval after stage $stage (n=$EVAL_N) ===" | tee -a "$LOG" "$EVAL_LOG"
  gpu_settle
  LATENTSQL_GPU_FRACTION=$GPU_FRACTION LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  LATENTSQL_REQUIRE_CUDA=1 \
  python3 scripts/04_eval.py \
    --config "$CONFIG" \
    --methods latent \
    --student "$CUR" \
    --base "$BASE_MODEL" \
    --latent-steps "$LATENT_STEPS" \
    --latent-role-list "$ROLES" \
    --latent-roles "$stage" \
    --limit "$EVAL_N" \
    --out "${EVAL_DIR}/stage_${stage}" \
    >> "$EVAL_LOG" 2>&1
  eval_code=$?
  if [ $eval_code -eq 0 ]; then
    grep "^latent " "$EVAL_LOG" | tail -1 | sed "s/^/stage $stage: /" | tee -a "$LOG"
  else
    echo "!!! eval after stage $stage FAILED (exit $eval_code, see $EVAL_LOG)" | tee -a "$LOG"
  fi
  STAGE_OUT="$RUNDIR/checkpoints/stage_${stage}"
  mkdir -p "$STAGE_OUT"
  cp -a "$OUT/." "$STAGE_OUT/"
  echo "saved stage $stage checkpoint to $STAGE_OUT" | tee -a "$LOG"
done

echo "=== CHESS COCONUT SQL-direct done: $OUT ===" | tee -a "$LOG"
