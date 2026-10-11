#!/usr/bin/env bash
# Larger r3 run: Dr. GRPO with the paper reward (Eq. 13) on 1000 training
# questions, then every checkpoint on 300 dev questions (150 BIRD + 150 Spider)
# and a true-EX audit. The 90-question results remain as they were.
set -euo pipefail
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

PY=${PY:-python3}
CFG=${CFG:-configs/latent_sft_1000_7b_bf16.yaml}
R3=outputs/latent_sft_1000/chess_three_role_r3
TWO=outputs/latent_sft_1000/chess_two_role_ex1
OUT=$R3/scale
LOGS=$OUT/logs
N_TRAIN=${N_TRAIN:-1000}
N_TEST=${N_TEST:-300}
R3_ROLES=schema_linker,candidate_generator,reviser,consensus
TWO_ROLES=schema_linker,candidate_generator,consensus
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATENTSQL_REQUIRE_CUDA=1

stage() {
  echo "=== [$(date --iso-8601=seconds)] $* ===" | tee -a "$LOGS/run.log"
}

GRPO=$OUT/grpo_paper_$N_TRAIN
if [ ! -f "$GRPO/grpo_result.json" ]; then
  rm -rf "$GRPO"
  stage "Dr. GRPO, paper reward, $N_TRAIN examples"
  LATENTSQL_GPU_FRACTION=0.7 "$PY" scripts/03_grpo.py \
    --config "$CFG" \
    --data outputs/latent_sft_1000/traces/sft.jsonl \
    --from-examples mix \
    --sft "$R3/latent_sft" \
    --out "$GRPO" \
    --limit "$N_TRAIN" \
    --epochs 1 \
    --latent-steps 2 \
    --latent-role-list "$R3_ROLES" \
    --no-spec-decode \
    --max-new-tokens 200 \
    --max-prompt-chars 4000 \
    --reward-mode paper \
    --w-memory 0 \
    --n-stress-instances 5 --cleanup-stress-cache \
    > "$LOGS/grpo_paper_$N_TRAIN.log" 2>&1
fi

evaluate() {  # name dir roles
  [ -f "$OUT/eval_$1/preds_latent.jsonl" ] && return 0
  stage "eval $1 ($N_TEST examples)"
  LATENTSQL_GPU_FRACTION=0.5 "$PY" scripts/04_eval.py \
    --config "$CFG" \
    --methods latent \
    --student "$2" \
    --base Qwen/Qwen2.5-Coder-7B-Instruct \
    --student-quantization 4bit \
    --latent-steps 2 \
    --latent-role-list "$3" \
    --limit "$N_TEST" \
    --max-new-tokens 400 \
    --out "$OUT/eval_$1" \
    > "$LOGS/eval_$1.log" 2>&1
}

evaluate r3_grpo_paper_$N_TRAIN "$GRPO" "$R3_ROLES"
evaluate r3_grpo_paper_192 "$R3/grpo_paper" "$R3_ROLES"
evaluate r3_sft "$R3/latent_sft" "$R3_ROLES"
evaluate 2role_sft "$TWO/latent_sft" "$TWO_ROLES"
evaluate 2role_spurious "$TWO/grpo_spurious" "$TWO_ROLES"
evaluate 2role_rolecredit "$TWO/grpo_rolecredit" "$TWO_ROLES"

stage "true EX audit ($N_TEST examples)"
PREDS=()
for NAME in 2role_sft 2role_spurious 2role_rolecredit r3_sft r3_grpo_paper_192 r3_grpo_paper_$N_TRAIN; do
  PREDS+=(--pred "$NAME=$OUT/eval_$NAME/preds_latent.jsonl")
done
"$PY" scripts/05_true_ex.py "${PREDS[@]}" \
  --instances 5 \
  --cache-dir "$TWO/cache/true_ex" \
  --out "$OUT/true_ex" \
  > "$LOGS/true_ex.log" 2>&1

stage "r3 scale run complete"
