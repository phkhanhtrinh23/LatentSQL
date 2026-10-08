#!/usr/bin/env bash
# Baselines for the two-role binary/spurious comparison, on the same 90 eval
# examples:
#   sft     - the shared latent SFT checkpoint, no GRPO
#   stdadv  - binary GRPO with original (r-mean)/std advantages (pre-Dr. GRPO)
# The SFT eval runs on the GPU alongside stdadv GRPO training; then stdadv is
# evaluated and one true-EX audit covers sft, stdadv, binary and spurious.
set -euo pipefail

cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

PY=${PY:-python3}
CFG=${CFG:-configs/latent_sft_1000_7b_bf16.yaml}
ROOT=${ROOT:-outputs/latent_sft_1000/chess_two_role_ex1}
LOGS=$ROOT/logs
TRAIN_LIMIT=${TRAIN_LIMIT:-192}
EVAL_LIMIT=${EVAL_LIMIT:-90}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATENTSQL_REQUIRE_CUDA=1

stage() {
  echo "=== [$(date --iso-8601=seconds)] $* ===" | tee -a "$LOGS/baselines.log"
}

evaluate() {  # evaluate <name> <student dir>
  LATENTSQL_GPU_FRACTION=${EVAL_GPU_FRACTION:-0.3} "$PY" scripts/04_eval.py \
    --config "$CFG" \
    --methods latent \
    --student "$2" \
    --base Qwen/Qwen2.5-Coder-7B-Instruct \
    --student-quantization 4bit \
    --latent-steps 2 \
    --latent-role-list schema_linker,candidate_generator,consensus \
    --limit "$EVAL_LIMIT" \
    --max-new-tokens 400 \
    --out "$ROOT/eval_$1" \
    > "$LOGS/eval_$1.log" 2>&1
}

stage "eval sft (background) + GRPO stdadv"
evaluate sft "$ROOT/latent_sft" &
EVAL_PID=$!

if [ ! -f "$ROOT/grpo_stdadv/grpo_result.json" ]; then
  rm -rf "$ROOT/grpo_stdadv"
  LATENTSQL_GPU_FRACTION=${GRPO_GPU_FRACTION:-0.6} "$PY" scripts/03_grpo.py \
    --config "$CFG" \
    --data outputs/latent_sft_1000/traces/sft.jsonl \
    --from-examples mix \
    --sft "$ROOT/latent_sft" \
    --out "$ROOT/grpo_stdadv" \
    --limit "$TRAIN_LIMIT" \
    --epochs 1 \
    --latent-steps 2 \
    --latent-role-list schema_linker,candidate_generator,consensus \
    --no-spec-decode \
    --max-new-tokens 200 \
    --max-prompt-chars 4000 \
    --w-atomic 0 --w-memory 0 --w-spurious 0 \
    --std-adv \
    > "$LOGS/grpo_stdadv.log" 2>&1
fi

wait "$EVAL_PID"
stage "eval sft done; eval stdadv"
evaluate stdadv "$ROOT/grpo_stdadv"

stage "true EX audit: sft, stdadv, binary, spurious"
"$PY" scripts/05_true_ex.py \
  --pred "sft=$ROOT/eval_sft/preds_latent.jsonl" \
  --pred "stdadv=$ROOT/eval_stdadv/preds_latent.jsonl" \
  --pred "binary=$ROOT/eval_binary/preds_latent.jsonl" \
  --pred "spurious=$ROOT/eval_spurious/preds_latent.jsonl" \
  --instances 5 \
  --cache-dir "$ROOT/cache/true_ex" \
  --out "$ROOT/true_ex_all" \
  > "$LOGS/true_ex_all.log" 2>&1

stage "baselines complete"
