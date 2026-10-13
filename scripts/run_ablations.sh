#!/usr/bin/env bash
# Ablations of Table 2 (LatentSQL 8B on BIRD and Spider). Reuses the teacher traces
# of the full model; each variant changes one setting.
#
#   bash scripts/run_ablations.sh configs/qwen3_8b.yaml
#   SEEDS="0 1 2 3 4 5 6 7 8 9" bash scripts/run_ablations.sh configs/qwen3_8b.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/qwen3_8b.yaml}
SEEDS=${SEEDS:-0}
PY=${PY:-python}
EXTRA=${EXTRA:-}
OUT=$($PY -c "import sys; sys.path.insert(0, '.'); from latentsql.config import load_config; print(load_config('$CONFIG').paths.output_dir)")
TRACES=$OUT/teacher/traces.jsonl
EVAL="--set data.eval_benchmarks=[bird,spider]"

VARIANTS=(
  "wo_role_traces|--set curriculum.use_role_traces=false"
  "wo_reviser|--set latent.roles=[schema_selector,candidate_generator]"
  "wo_curriculum|--set curriculum.curriculum=false"
  "wo_probe_loss|--set curriculum.probe_weight=0"
  "wo_schema_alignment|--set latent.schema_alignment=false"
  "wo_fk_propagation|--set latent.schema_beta=0"
  "wo_r_atomic|--set rl.use_atomic=false"
  "wo_r_spur|--set rl.use_spurious=false"
  "wo_counterexample_pool|--set rl.use_counterexamples=false"
)
# RL-only variants start from the full model's curriculum checkpoint
RL_ONLY="wo_r_atomic wo_r_spur wo_counterexample_pool"

[ -f "$TRACES" ] || $PY scripts/teacher_traces.py --config "$CONFIG" --out "$TRACES" $EXTRA

for SEED in $SEEDS; do
  SET="--set curriculum.seed=$SEED --set rl.seed=$SEED --set seed=$SEED"
  FULL=$OUT/seed_$SEED
  for ENTRY in "${VARIANTS[@]}"; do
    NAME=${ENTRY%%|*}; V=${ENTRY#*|}
    RUN=$OUT/ablations/$NAME/seed_$SEED
    echo "=== $NAME (seed $SEED): $V ==="
    if [[ " $RL_ONLY " == *" $NAME "* ]]; then
      INIT=$FULL/latent/final
      [ -d "$INIT" ] || $PY scripts/train_latent.py --config "$CONFIG" --traces "$TRACES" --out "$FULL/latent" $SET $EXTRA
    else
      $PY scripts/train_latent.py --config "$CONFIG" --traces "$TRACES" --out "$RUN/latent" $SET $V $EXTRA
      INIT=$RUN/latent/final
    fi
    $PY scripts/train_rl.py --config "$CONFIG" --init "$INIT" --out "$RUN/rl" $SET $V $EXTRA
    $PY scripts/evaluate.py --config "$CONFIG" --model "$RUN/rl/final" --out "$RUN/eval" $SET $V $EVAL $EXTRA
  done
done

for ENTRY in "${VARIANTS[@]}"; do
  NAME=${ENTRY%%|*}
  $PY scripts/aggregate.py --runs "$OUT/ablations/$NAME"/seed_*/eval --name "$NAME" \
    --out "$OUT/ablations/$NAME/results.md"
done
