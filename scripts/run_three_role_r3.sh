#!/usr/bin/env bash
# Three-role latent student with the reviser r3 (main_latentsql.pdf Sec. 4.1):
#   schema_linker -> candidate_generator (draft) -> reviser (execute, keep/fix)
# Teacher traces are kept when the reviser's final SQL has EX = 1, so repaired
# drafts are included. Then a 3-stage COCONUT curriculum, Dr. GRPO with the
# paper's reward (Eq. 13), the same 90-example eval, and a true-EX audit
# against the two-role checkpoints.
set -euo pipefail

cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

PY=${PY:-python3}
CFG=${CFG:-configs/latent_sft_1000_7b_bf16.yaml}
ROOT=${ROOT:-outputs/latent_sft_1000/chess_three_role_r3}
TWO=${TWO:-outputs/latent_sft_1000/chess_two_role_ex1}
PREPARED=$ROOT/prepared_records.jsonl
BASE_SFT=${BASE_SFT:-outputs/latent_sft_1000/sft_7b_bf16}
LATENT_SFT=$ROOT/latent_sft
LOGS=$ROOT/logs
ROLE_LIST=schema_linker,candidate_generator,reviser,consensus
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATENTSQL_REQUIRE_CUDA=1

stage() {
  echo "=== [$(date --iso-8601=seconds)] $* ===" | tee -a "$LOGS/run.log"
}

stage "validating three-role r3 corpus"
"$PY" - "$PREPARED" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if not rows:
    raise SystemExit("prepared corpus is empty")
fixed = 0
for i, row in enumerate(rows, 1):
    turns = row.get("role_turns") or []
    if len(turns) != 3:
        raise SystemExit(f"record {i}: expected 3 role turns, got {len(turns)}")
    if not turns[2].startswith("<|agent reviser|>") or "Execute the draft once." not in turns[2]:
        raise SystemExit(f"record {i}: third turn is not an executing reviser")
    fixed += "Diagnosis and fix" in turns[2]
print(f"validated {len(rows)} records: {fixed} repaired drafts, {len(rows) - fixed} kept drafts")
PY

if [ ! -f "$LATENT_SFT/adapter_config.json" ] || [ ! -f "$ROOT/checkpoints/stage_3/adapter_config.json" ]; then
  stage "three-stage latent SFT on the r3 corpus"
  ROLES=$ROLE_LIST CHAIN_FORMAT=chess_reviser bash scripts/run_chess_coconut_sql_direct.sh \
    auto 16 3 1 \
    "$BASE_SFT" "$PREPARED" "$CFG" \
    30 5e-5 2 "$ROOT" 0.96 \
    Qwen/Qwen2.5-Coder-7B-Instruct \
    4000 2400 1 1 1800 2e-5
else
  stage "latent SFT already complete; reusing $LATENT_SFT"
fi

OUT=$ROOT/grpo_paper
if [ ! -f "$OUT/grpo_result.json" ]; then
  rm -rf "$OUT"
  stage "Dr. GRPO, paper reward (Eq. 13), 192 examples"
  LATENTSQL_GPU_FRACTION=0.7 "$PY" scripts/03_grpo.py \
    --config "$CFG" \
    --data outputs/latent_sft_1000/traces/sft.jsonl \
    --from-examples mix \
    --sft "$LATENT_SFT" \
    --out "$OUT" \
    --limit 192 \
    --epochs 1 \
    --latent-steps 2 \
    --latent-role-list "$ROLE_LIST" \
    --no-spec-decode \
    --max-new-tokens 200 \
    --max-prompt-chars 4000 \
    --reward-mode paper \
    --w-memory 0 \
    --n-stress-instances 5 --cleanup-stress-cache \
    > "$LOGS/grpo_paper.log" 2>&1
fi

for ARM in sft:$LATENT_SFT grpo:$OUT; do
  NAME=${ARM%%:*}; DIR=${ARM#*:}
  stage "eval r3 $NAME (90 examples)"
  LATENTSQL_GPU_FRACTION=0.5 "$PY" scripts/04_eval.py \
    --config "$CFG" \
    --methods latent \
    --student "$DIR" \
    --base Qwen/Qwen2.5-Coder-7B-Instruct \
    --student-quantization 4bit \
    --latent-steps 2 \
    --latent-role-list "$ROLE_LIST" \
    --limit 90 \
    --max-new-tokens 400 \
    --out "$ROOT/eval_$NAME" \
    > "$LOGS/eval_$NAME.log" 2>&1
done

stage "true EX audit: two-role vs three-role r3"
"$PY" scripts/05_true_ex.py \
  --pred "2role_sft=$TWO/eval_sft/preds_latent.jsonl" \
  --pred "2role_spurious=$TWO/eval_spurious/preds_latent.jsonl" \
  --pred "2role_rolecredit=$TWO/eval_rolecredit/preds_latent.jsonl" \
  --pred "r3_sft=$ROOT/eval_sft/preds_latent.jsonl" \
  --pred "r3_grpo_paper=$ROOT/eval_grpo/preds_latent.jsonl" \
  --instances 5 \
  --cache-dir "$TWO/cache/true_ex" \
  --out "$ROOT/true_ex" \
  > "$LOGS/true_ex.log" 2>&1

stage "three-role r3 complete"
