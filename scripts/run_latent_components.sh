#!/usr/bin/env bash
# main_latentsql.pdf Sec. 4.2-4.3 on the two-role student (schema_linker ->
# candidate_generator), same corpus and curriculum as chess_two_role_ex1:
#
#   Component 1  r1 aligner: vocab mixture vs schema mixture vs hard token vs
#                raw hidden state, plus schema recall@k
#   Component 2  schema mixture + execution-property heads on r2's states
#   Component 3  Dr. GRPO, paper reward, from the best SFT arm: fixed stress
#                instances vs counterexample-guided pools (CEGIS)
#
# Every SFT arm (the vocab baseline included, retrained) adds the probe loss
# L_probe (Eq. 16, lambda = 0.5, kappa = 16), so the arms differ only in the
# component under test. Every arm: 300-question eval (150 BIRD + 150 Spider, same questions as
# chess_three_role_r3/scale), then one true-EX audit across all of them.
set -euo pipefail
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql

PY=${PY:-python3}
CFG=${CFG:-configs/latent_sft_1000_7b_bf16.yaml}
ROOT=${ROOT:-outputs/latent_sft_1000/latent_components}
DATA=outputs/latent_sft_1000/schema_latent/prepared_records.jsonl
BASE_SFT=outputs/latent_sft_1000/sft_7b_bf16
TWO=outputs/latent_sft_1000/chess_two_role_ex1
SCALE=outputs/latent_sft_1000/chess_three_role_r3/scale
ROLES=schema_linker,candidate_generator,consensus
BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
N_TEST=${N_TEST:-300}
N_GRPO=${N_GRPO:-1000}
LOGS=$ROOT/logs
PROBE_ARGS="--probe-weight 0.5 --probe-tokens 16"
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATENTSQL_REQUIRE_CUDA=1

stage() {
  echo "=== [$(date --iso-8601=seconds)] $* ===" | tee -a "$LOGS/run.log"
}

evaluate() {  # name adapter_dir
  [ -f "$ROOT/$1/eval/preds_latent.jsonl" ] && return 0
  stage "eval $1 ($N_TEST examples)"
  LATENTSQL_GPU_FRACTION=0.5 "$PY" scripts/04_eval.py \
    --config "$CFG" --methods latent --student "$2" --base "$BASE_MODEL" \
    --student-quantization 4bit --latent-steps 2 --latent-role-list "$ROLES" \
    --limit "$N_TEST" --max-new-tokens 400 --out "$ROOT/$1/eval" \
    > "$LOGS/eval_$1.log" 2>&1
}

recall() {  # name adapter_dir
  [ -f "$ROOT/$1/schema_recall.json" ] && return 0
  stage "schema recall $1"
  mkdir -p "$ROOT/$1"
  LATENTSQL_GPU_FRACTION=0.5 "$PY" scripts/09_schema_recall.py \
    --config "$CFG" --student "$2" --base "$BASE_MODEL" --roles "$ROLES" \
    --limit "$N_TEST" --out "$ROOT/$1/schema_recall.json" \
    > "$LOGS/recall_$1.log" 2>&1
}

train_arm() {  # name "extra 02c args"
  local RUN=$ROOT/$1
  if [ ! -f "$RUN/checkpoints/stage_2/adapter_config.json" ] && [ ! -f "$RUN/sft_done" ]; then
    stage "latent SFT arm $1 ($2)"
    mkdir -p "$RUN"
    EXTRA_SFT_ARGS="$2 $PROBE_ARGS" bash scripts/run_chess_coconut_sql_direct.sh \
      auto 16 2 1 "$BASE_SFT" "$DATA" "$CFG" 10 5e-5 2 "$RUN" 0.96 "$BASE_MODEL" \
      4000 2400 1 1 1800 \
      > "$LOGS/sft_$1.log" 2>&1
    touch "$RUN/sft_done"
  fi
  evaluate "$1" "$RUN/latent_sft"
  recall "$1" "$RUN/latent_sft"
  # resume-only state: intermediate stages and optimizer moments
  rm -rf "${RUN:?}/checkpoints" "${RUN:?}/optim_state.pt" "${RUN:?}/optim_state.pt.extras"
}

# ---- Components 1 and 2: SFT arms -----------------------------------------
train_arm vocab       "--align-r1 vocab"
train_arm schema      "--align-r1 schema"
train_arm schema_exec "--align-r1 schema --exec-heads"
train_arm hard        "--align-r1 hard"
train_arm identity    "--align-r1 identity"

if [ ! -f "$ROOT/schema_exec/exec_probe.json" ]; then
  stage "execution-property probe (schema_exec)"
  LATENTSQL_GPU_FRACTION=0.5 "$PY" scripts/10_exec_probe.py \
    --config "$CFG" --student "$ROOT/schema_exec/latent_sft" --base "$BASE_MODEL" \
    --roles "$ROLES" --limit "$N_TEST" \
    --preds "$ROOT/schema_exec/eval/preds_latent.jsonl" \
    --out "$ROOT/schema_exec/exec_probe.json" > "$LOGS/exec_probe.log" 2>&1
fi

# ---- Component 3: GRPO from the best SFT arm --------------------------------
BEST=$("$PY" - "$ROOT" "$SCALE" <<'PY'
import json, sys
root, scale = sys.argv[1], sys.argv[2]
def ex(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return sum(bool(r.get("correct")) for r in rows) / max(1, len(rows))
arms = {}
for a in ("vocab", "schema", "schema_exec", "hard", "identity"):
    arms[a] = f"{root}/{a}/eval/preds_latent.jsonl"
scores = {a: ex(p) for a, p in arms.items()}
best = max(scores, key=lambda a: (scores[a], a == "vocab"))
print(best)
print(json.dumps(scores, indent=4), file=sys.stderr)
PY
)
BEST_DIR=$ROOT/$BEST/latent_sft
stage "best SFT arm: $BEST ($BEST_DIR)"

for MODE in static cegis; do
  OUT=$ROOT/grpo_${MODE}
  if [ ! -f "$OUT/grpo_result.json" ]; then
    rm -rf "$OUT"
    stage "Dr. GRPO paper reward, $MODE pools, from $BEST"
    EXTRA=()
    [ "$MODE" = "cegis" ] && EXTRA=(--cegis --cegis-k-max 0 --cegis-n-delta 3
      --verieql-python /home/trinh/miniconda3/envs/verieql/bin/python
      --verieql-dir /mnt/win_d/tools/VeriEQL --verieql-bound 2 --verieql-timeout 60)
    LATENTSQL_GPU_FRACTION=0.7 "$PY" scripts/03_grpo.py \
      --config "$CFG" --data outputs/latent_sft_1000/traces/sft.jsonl --from-examples mix \
      --sft "$BEST_DIR" --out "$OUT" --limit "$N_GRPO" --epochs 1 \
      --latent-steps 2 --latent-role-list "$ROLES" --no-spec-decode \
      --max-new-tokens 200 --max-prompt-chars 4000 \
      --reward-mode paper --w-memory 0 --n-stress-instances 5 --cleanup-stress-cache \
      "${EXTRA[@]}" > "$LOGS/grpo_$MODE.log" 2>&1
  fi
  evaluate "grpo_$MODE" "$OUT"
done

# ---- true EX across every arm -----------------------------------------------------
stage "true EX audit ($N_TEST examples)"
# previous two-role models (no L_probe) for reference
PREDS=(--pred "old_2role_sft=$SCALE/eval_2role_sft/preds_latent.jsonl"
       --pred "old_2role_rolecredit=$SCALE/eval_2role_rolecredit/preds_latent.jsonl")
for A in vocab schema schema_exec hard identity grpo_static grpo_cegis; do
  PREDS+=(--pred "$A=$ROOT/$A/eval/preds_latent.jsonl")
done
"$PY" scripts/05_true_ex.py "${PREDS[@]}" --instances 5 \
  --cache-dir "$TWO/cache/true_ex" --out "$ROOT/true_ex" > "$LOGS/true_ex.log" 2>&1

stage "latent components complete"
