#!/usr/bin/env bash
# End-to-end pilot: 7B teacher -> 3B student, BIRD+Spider combined, all local.
#
# Each stage is a separate process, so exactly one model is resident on the GPU
# at any moment. Every stage appends to outputs/pilot/logs/ and the whole run is
# resumable at stage granularity - rerun the script and finished stages are
# skipped if their outputs exist.
set -u
CFG=${CFG:-configs/pilot_mix.yaml}
OUT=${OUT:-outputs/pilot}
LOGS=$OUT/logs
TRACES=$OUT/traces
mkdir -p "$LOGS"

PY=${PY:-python3}
stamp() { date +%H:%M:%S; }
stage() { echo "=== [$(stamp)] $* ===" | tee -a "$LOGS/pilot.log"; }

stage "1/5 teacher: build traces (7B, 4-bit, four-system panel)"
if [ ! -f "$TRACES/sft.jsonl" ]; then
  $PY scripts/01_build_traces.py --config "$CFG" --dataset mix \
      --limit "${N_TRACE:-90}" --out "$TRACES" 2>&1 | tee "$LOGS/01_traces.log"
else
  echo "  skip: $TRACES/sft.jsonl exists" | tee -a "$LOGS/pilot.log"
fi

N_SFT=$(wc -l < "$TRACES/sft.jsonl" 2>/dev/null || echo 0)
stage "traces kept: $N_SFT"
if [ "$N_SFT" -lt 4 ]; then
  echo "FATAL: only $N_SFT traces survived the filter - too few to train on." \
    | tee -a "$LOGS/pilot.log"
  exit 1
fi

stage "2/5 student: SFT (3B, QLoRA 4-bit)"
if [ ! -f "$OUT/sft/adapter_config.json" ]; then
  $PY scripts/02_sft.py --config "$CFG" --data "$TRACES/sft.jsonl" \
      2>&1 | tee "$LOGS/02_sft.log"
else
  echo "  skip: $OUT/sft exists" | tee -a "$LOGS/pilot.log"
fi

stage "3/5 student: GRPO internalisation"
if [ ! -f "$OUT/grpo/adapter_config.json" ]; then
  timeout "${GRPO_TIMEOUT:-1500}" $PY scripts/03_grpo.py --config "$CFG" \
      --data "$TRACES/sft.jsonl" 2>&1 | tee "$LOGS/03_grpo.log"
  echo "  grpo exit: $?" | tee -a "$LOGS/pilot.log"
else
  echo "  skip: $OUT/grpo exists" | tee -a "$LOGS/pilot.log"
fi

# --- evaluation: baselines and student are separate processes so that only one
# --- model is on the GPU at a time, and both run at 3B for a fair comparison
STUDENT_DIR=$OUT/grpo
[ -f "$STUDENT_DIR/adapter_config.json" ] || STUDENT_DIR=$OUT/sft

# eval draws from the BIRD/Spider *dev* splits (mix splits follow the caller),
# so nothing scored here was ever seen by the teacher
stage "4/5 eval: DIN/DAIL/CHESS/CHASE on the SAME 3B as the student"
$PY scripts/04_eval.py --config "$CFG" \
    --methods single,din_sql,dail_sql,chess,chase_sql \
    --model Qwen/Qwen2.5-Coder-3B-Instruct \
    --limit "${N_EVAL:-24}" --out "$OUT/eval_3b" 2>&1 | tee "$LOGS/04_eval_baselines_3b.log"

stage "5/5 eval: the internalised student (+ the 7B teacher for reference)"
$PY scripts/04_eval.py --config "$CFG" --methods student \
    --student "$STUDENT_DIR" \
    --limit "${N_EVAL:-24}" --out "$OUT/eval_student" 2>&1 | tee "$LOGS/05_eval_student.log"

stage "done - logs in $LOGS, per-sample records in $TRACES/teacher_samples.jsonl"
