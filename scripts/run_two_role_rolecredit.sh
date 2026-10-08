#!/usr/bin/env bash
# Per-role credit arm: same rewards and data as the two-role "spurious" arm
# (scripts/run_two_role_binary_spurious.sh), plus --role-credit, which also
# samples SQL decoded right after the schema linker and rewards it. Then the
# same 90-example eval and a true-EX audit against the earlier arms.
set -euo pipefail

cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

PY=${PY:-python3}
CFG=${CFG:-configs/latent_sft_1000_7b_bf16.yaml}
ROOT=${ROOT:-outputs/latent_sft_1000/chess_two_role_ex1}
LOGS=$ROOT/logs
OUT=$ROOT/grpo_rolecredit
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATENTSQL_REQUIRE_CUDA=1

stage() {
  echo "=== [$(date --iso-8601=seconds)] $* ===" | tee -a "$LOGS/rolecredit.log"
}

# grpo_result.json is written only when training finishes; a periodic adapter
# save alone does not mean the arm is complete.
if [ ! -f "$OUT/grpo_result.json" ]; then
  rm -rf "$OUT"
  stage "GRPO rolecredit: 192 examples from shared latent SFT"
  LATENTSQL_GPU_FRACTION=0.7 "$PY" scripts/03_grpo.py \
    --config "$CFG" \
    --data outputs/latent_sft_1000/traces/sft.jsonl \
    --from-examples mix \
    --sft "$ROOT/latent_sft" \
    --out "$OUT" \
    --limit 192 \
    --epochs 1 \
    --latent-steps 2 \
    --latent-role-list schema_linker,candidate_generator,consensus \
    --no-spec-decode \
    --max-new-tokens 200 \
    --max-prompt-chars 4000 \
    --w-atomic 0.5 --w-memory 0.3 --w-spurious 0.5 \
    --n-stress-instances 3 --cleanup-stress-cache \
    --role-credit \
    > "$LOGS/grpo_rolecredit.log" 2>&1
fi

stage "eval rolecredit (90 examples)"
LATENTSQL_GPU_FRACTION=0.5 "$PY" scripts/04_eval.py \
  --config "$CFG" \
  --methods latent \
  --student "$OUT" \
  --base Qwen/Qwen2.5-Coder-7B-Instruct \
  --student-quantization 4bit \
  --latent-steps 2 \
  --latent-role-list schema_linker,candidate_generator,consensus \
  --limit 90 \
  --max-new-tokens 400 \
  --out "$ROOT/eval_rolecredit" \
  > "$LOGS/eval_rolecredit.log" 2>&1

stage "true EX audit: sft, stdadv, binary, spurious, rolecredit"
"$PY" scripts/05_true_ex.py \
  --pred "sft=$ROOT/eval_sft/preds_latent.jsonl" \
  --pred "stdadv=$ROOT/eval_stdadv/preds_latent.jsonl" \
  --pred "binary=$ROOT/eval_binary/preds_latent.jsonl" \
  --pred "spurious=$ROOT/eval_spurious/preds_latent.jsonl" \
  --pred "rolecredit=$ROOT/eval_rolecredit/preds_latent.jsonl" \
  --instances 5 \
  --cache-dir "$ROOT/cache/true_ex" \
  --out "$ROOT/true_ex_rolecredit" \
  > "$LOGS/true_ex_rolecredit.log" 2>&1

stage "rolecredit complete"
