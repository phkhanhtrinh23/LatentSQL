#!/usr/bin/env bash
# Train the corrected two-role latent student once, then compare two GRPO arms
# from the exact same checkpoint: binary versus spurious+dense rewards.
set -euo pipefail

cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

PY=${PY:-python3}
CFG=${CFG:-configs/latent_sft_1000_7b_bf16.yaml}
ROOT=${ROOT:-outputs/latent_sft_1000/chess_two_role_ex1}
PREPARED=${PREPARED:-$ROOT/prepared_records.jsonl}
BASE_SFT=${BASE_SFT:-outputs/latent_sft_1000/sft_7b_bf16}
LATENT_SFT=$ROOT/latent_sft
LOGS=$ROOT/logs
TRAIN_LIMIT=${TRAIN_LIMIT:-192}
EVAL_LIMIT=${EVAL_LIMIT:-90}
GRPO_EPOCHS=${GRPO_EPOCHS:-1}

mkdir -p "$ROOT" "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True

stage() {
  echo "=== [$(date --iso-8601=seconds)] $* ===" | tee -a "$LOGS/run.log"
}

stage "validating corrected two-role EX=1 corpus"
"$PY" - "$PREPARED" <<'PY'
import json
import os
import sys
from collections import Counter

path = sys.argv[1]
if not os.path.isfile(path) or os.path.getsize(path) == 0:
    raise SystemExit(f"prepared corpus is missing or empty: {path}")
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
if not rows:
    raise SystemExit("prepared corpus contains no records")
labels = Counter()
with_negatives = 0
for i, row in enumerate(rows, 1):
    turns = row.get("role_turns") or []
    if len(turns) != 2:
        raise SystemExit(f"record {i}: expected exactly 2 role turns, got {len(turns)}")
    joined = "\n".join(str(x) for x in turns)
    if "Outcome: valid" not in str(turns[1]) or "EX=1" not in str(turns[1]):
        raise SystemExit(f"record {i}: candidate turn lacks valid EX=1 supervision")
    if "query_classifier" in joined or "query_fixer" in joined or "unit_tester" in joined:
        raise SystemExit(f"record {i}: forbidden planner/revision/verification role found")
    if str(row.get("prompt", "")).count("Retrieved values:") > 1:
        raise SystemExit(f"record {i}: retrieval was inserted more than once")
    found = False
    for label in ("syntax_error", "runtime_error", "empty_result", "result_mismatch"):
        n = joined.count(label)
        labels[label] += n
        found = found or n > 0
    with_negatives += int(found)
print(
    f"validated {len(rows)} records: exactly two roles, final EX=1; "
    f"offline-negative coverage={with_negatives}/{len(rows)}; labels={dict(labels)}"
)
PY

if [ ! -f "$LATENT_SFT/adapter_config.json" ]; then
  stage "two-stage latent SFT on the full corrected corpus"
  bash scripts/run_chess_coconut_sql_direct.sh \
    auto 16 2 1 \
    "$BASE_SFT" "$PREPARED" "$CFG" \
    30 5e-5 2 "$ROOT" 0.96 \
    Qwen/Qwen2.5-Coder-7B-Instruct \
    4000 2400 1 1 1800 2e-5
else
  stage "latent SFT checkpoint already complete; reusing $LATENT_SFT"
fi

for ARM in binary spurious; do
  ARM_OUT=$ROOT/grpo_$ARM
  EXTRA=()
  case "$ARM" in
    binary)
      EXTRA+=(--w-atomic 0 --w-memory 0 --w-spurious 0)
      ;;
    spurious)
      # This arm includes the dense partial-credit terms for wrong answers.
      EXTRA+=(--w-atomic 0.5 --w-memory 0.3 --w-spurious 0.5 \
              --n-stress-instances 3 --cleanup-stress-cache)
      ;;
  esac
  if [ ! -f "$ARM_OUT/adapter_config.json" ]; then
    stage "GRPO $ARM: $TRAIN_LIMIT examples from shared latent SFT"
    "$PY" scripts/03_grpo.py \
      --config "$CFG" \
      --data outputs/latent_sft_1000/traces/sft.jsonl \
      --from-examples mix \
      --sft "$LATENT_SFT" \
      --out "$ARM_OUT" \
      --limit "$TRAIN_LIMIT" \
      --epochs "$GRPO_EPOCHS" \
      --latent-steps 2 \
      --latent-role-list schema_linker,candidate_generator,consensus \
      --no-spec-decode \
      --max-new-tokens 200 \
      --max-prompt-chars 4000 \
      "${EXTRA[@]}" \
      2>&1 | tee "$LOGS/grpo_$ARM.log"
  else
    stage "GRPO $ARM checkpoint already complete; reusing $ARM_OUT"
  fi
done

for ARM in binary spurious; do
  EVAL_OUT=$ROOT/eval_$ARM
  stage "ordinary EX evaluation: $ARM ($EVAL_LIMIT examples)"
  "$PY" scripts/04_eval.py \
    --config "$CFG" \
    --methods latent \
    --student "$ROOT/grpo_$ARM" \
    --base Qwen/Qwen2.5-Coder-7B-Instruct \
    --student-quantization 4bit \
    --latent-steps 2 \
    --latent-role-list schema_linker,candidate_generator,consensus \
    --limit "$EVAL_LIMIT" \
    --max-new-tokens 400 \
    --out "$EVAL_OUT" \
    2>&1 | tee "$LOGS/eval_$ARM.log"
done

stage "true EX audit on shared deterministic stress databases"
"$PY" scripts/05_true_ex.py \
  --pred "binary=$ROOT/eval_binary/preds_latent.jsonl" \
  --pred "spurious=$ROOT/eval_spurious/preds_latent.jsonl" \
  --instances 5 \
  --cache-dir "$ROOT/cache/true_ex" \
  --out "$ROOT/true_ex" \
  2>&1 | tee "$LOGS/true_ex.log"

stage "binary/spurious training and evaluation complete"
