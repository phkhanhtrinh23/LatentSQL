#!/usr/bin/env bash
# Three-role student trained on the repair-rich r3 corpus (reviser_repair):
# repairs drawn from every teacher agent, debate references removed, repaired
# records written twice (~48% of records). Same curriculum and eval as
# run_three_role_r3.sh / run_r3_scale.sh, so the only change is the corpus.
# Prepare the corpus first:
#   scripts/02d_prepare_latent_records.py --chain-format reviser_repair --repeat-fixes 2 ...
set -euo pipefail
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

PY=${PY:-python3}
CFG=${CFG:-configs/latent_sft_1000_7b_bf16.yaml}
ROOT=${ROOT:-outputs/latent_sft_1000/chess_three_role_repair}
SCALE=outputs/latent_sft_1000/chess_three_role_r3/scale
TWO=outputs/latent_sft_1000/chess_two_role_ex1
PREPARED=$ROOT/prepared_records.jsonl
BASE_SFT=${BASE_SFT:-outputs/latent_sft_1000/sft_7b_bf16}
LATENT_SFT=$ROOT/latent_sft
LOGS=$ROOT/logs
ROLE_LIST=schema_linker,candidate_generator,reviser,consensus
N_TEST=${N_TEST:-300}
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATENTSQL_REQUIRE_CUDA=1

stage() {
  echo "=== [$(date --iso-8601=seconds)] $* ===" | tee -a "$LOGS/run.log"
}

if [ ! -f "$ROOT/checkpoints/stage_3/adapter_config.json" ]; then
  stage "three-stage latent SFT on the repair corpus"
  ROLES=$ROLE_LIST CHAIN_FORMAT=reviser_repair bash scripts/run_chess_coconut_sql_direct.sh \
    auto 16 3 1 \
    "$BASE_SFT" "$PREPARED" "$CFG" \
    30 5e-5 2 "$ROOT" 0.96 \
    Qwen/Qwen2.5-Coder-7B-Instruct \
    4000 2400 1 1 1800 2e-5
else
  stage "latent SFT already complete; reusing $LATENT_SFT"
fi

if [ ! -f "$ROOT/verbalize_sft/verbalize.json" ]; then
  stage "verbalize latent thoughts (90 examples)"
  LATENTSQL_GPU_FRACTION=0.5 "$PY" scripts/08_latent_trace.py \
    --config "$CFG" --student "$LATENT_SFT" --base Qwen/Qwen2.5-Coder-7B-Instruct \
    --latent-steps 2 --roles "$ROLE_LIST" \
    --limit 90 --mode verbalize --text-tokens 160 --out "$ROOT/verbalize_sft" \
    > "$LOGS/verbalize_sft.log" 2>&1
fi

if [ ! -f "$ROOT/eval_sft/preds_latent.jsonl" ]; then
  stage "eval repair sft ($N_TEST examples)"
  LATENTSQL_GPU_FRACTION=0.5 "$PY" scripts/04_eval.py \
    --config "$CFG" \
    --methods latent \
    --student "$LATENT_SFT" \
    --base Qwen/Qwen2.5-Coder-7B-Instruct \
    --student-quantization 4bit \
    --latent-steps 2 \
    --latent-role-list "$ROLE_LIST" \
    --limit "$N_TEST" \
    --max-new-tokens 400 \
    --out "$ROOT/eval_sft" \
    > "$LOGS/eval_sft.log" 2>&1
fi

stage "true EX audit ($N_TEST examples)"
PREDS=()
for NAME in 2role_sft 2role_spurious 2role_rolecredit r3_sft r3_grpo_paper_192 r3_grpo_paper_1000; do
  PREDS+=(--pred "$NAME=$SCALE/eval_$NAME/preds_latent.jsonl")
done
"$PY" scripts/05_true_ex.py "${PREDS[@]}" \
  --pred "r3_repair_sft=$ROOT/eval_sft/preds_latent.jsonl" \
  --instances 5 \
  --cache-dir "$TWO/cache/true_ex" \
  --out "$ROOT/true_ex" \
  > "$LOGS/true_ex.log" 2>&1

stage "three-role repair complete"
