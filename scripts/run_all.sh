#!/usr/bin/env bash
# End-to-end LatentSQL: teacher traces -> warm-up + curriculum -> Dr. GRPO -> evaluation,
# repeated over SEEDS, then mean EX +- 95% CI (Table 1).
#
#   bash scripts/run_all.sh configs/qwen3_8b.yaml            # one run (seed 0)
#   SEEDS="0 1 2 3 4 5 6 7 8 9" bash scripts/run_all.sh configs/qwen3_8b.yaml   # 10 runs, as in the paper
#   bash scripts/run_all.sh configs/smoke.yaml               # offline plumbing check
#
# Prerequisites: benchmarks converted with scripts/prepare_data.py (see README),
# OPENAI_API_KEY for the teacher, and rl.verieql_python / rl.verieql_dir set in
# the config (or pass EXTRA="--set rl.verifier=mutation").
# The teacher traces are shared across seeds; each seed trains and evaluates its own student.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/qwen3_8b.yaml}
SEEDS=${SEEDS:-0}
PY=${PY:-python}
EXTRA=${EXTRA:-}
OUT=$($PY -c "import sys; sys.path.insert(0, '.'); from latentsql.config import load_config; print(load_config('$CONFIG').paths.output_dir)")
TRACES=$OUT/teacher/traces.jsonl
mkdir -p "$OUT/logs"

log() { echo "=== [$(date '+%F %T')] $* ===" | tee -a "$OUT/logs/run_all.log"; }

# 0. convert every raw benchmark found under $DATASETS/<name>/ (once)
DATASETS=${DATASETS:-datasets}
DATA_DIR=$($PY -c "import sys; sys.path.insert(0, '.'); from latentsql.config import load_config; print(load_config('$CONFIG').paths.data_dir)")
for B in wikisql bird spider kaggledbqa sparc cosql ehrsql livesqlbench; do
  if [ -d "$DATASETS/$B" ] && [ ! -f "$DATA_DIR/$B/test.jsonl" ]; then
    log "0/4 converting $B"
    GT=(); [ "$B" = livesqlbench ] && [ -n "${LIVESQLBENCH_GT:-}" ] && GT=(--gt "$LIVESQLBENCH_GT")
    $PY scripts/prepare_data.py --benchmark "$B" --raw "$DATASETS/$B" --data-dir "$DATA_DIR" ${GT[@]+"${GT[@]}"}
  fi
done

log "1/4 teacher traces ($CONFIG)"
$PY scripts/teacher_traces.py --config "$CONFIG" --out "$TRACES" $EXTRA 2>&1 | tee -a "$OUT/logs/teacher.log"

for SEED in $SEEDS; do
  RUN=$OUT/seed_$SEED
  SET="--set curriculum.seed=$SEED --set rl.seed=$SEED --set seed=$SEED"
  log "2/4 warm-up + text-to-latent curriculum (seed $SEED)"
  $PY scripts/train_latent.py --config "$CONFIG" --traces "$TRACES" --out "$RUN/latent" $SET $EXTRA \
    2>&1 | tee -a "$OUT/logs/latent_seed_$SEED.log"
  log "3/4 Dr. GRPO (seed $SEED)"
  $PY scripts/train_rl.py --config "$CONFIG" --init "$RUN/latent/final" --out "$RUN/rl" $SET $EXTRA \
    2>&1 | tee -a "$OUT/logs/rl_seed_$SEED.log"
  log "4/4 evaluation (seed $SEED)"
  $PY scripts/evaluate.py --config "$CONFIG" --model "$RUN/rl/final" --out "$RUN/eval" $SET $EXTRA \
    2>&1 | tee -a "$OUT/logs/eval_seed_$SEED.log"
done

log "mean EX +- 95% CI over seeds: $SEEDS"
$PY scripts/aggregate.py --runs "$OUT"/seed_*/eval --out "$OUT/results.md"
