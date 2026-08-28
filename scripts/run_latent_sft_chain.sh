#!/usr/bin/env bash
# Latent SFT, one record per fresh subprocess, chaining the checkpoint forward.
# Each process only ever holds a single step's memory (proven safe, ~6.5-8.4GB)
# and fully releases everything on exit, so there is no cross-step
# accumulation/fragmentation to OOM on - the failure mode every in-process
# attempt hit. Slower (a model reload per record) but bounded and reliable.
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql
N=${1:-8}
START=${2:-0}
BASE=${3:-outputs/imad_boost/sft}
OUT=outputs/latent_sft_probe/latent_sft
LOG=outputs/latent_sft_probe/chain.log
[ "$START" -eq 0 ] && : > "$LOG"

CUR="$BASE"
for i in $(seq "$START" $((N - 1))); do
  echo "=== [$(date +%H:%M:%S)] record $i, from $CUR ===" | tee -a "$LOG"
  LATENTSQL_GPU_FRACTION=0.5 LATENTSQL_CPU_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 scripts/02c_latent_sft.py \
    --config configs/imad_boost.yaml \
    --data outputs/imad_boost/traces/sft.jsonl \
    --sft "$CUR" \
    --out "$OUT" \
    --latent-steps 12 \
    --roles din_sql,dail_sql,chess,chase_sql,din_sql,dail_sql,chess,chase_sql,consensus \
    --max-prompt-chars 3000 --offset "$i" --limit 1 --epochs 1 --grad-accum 1 --save-steps 1 \
    >> "$LOG" 2>&1
  code=$?
  echo "record $i exit code: $code" | tee -a "$LOG"
  if [ $code -ne 0 ]; then
    echo "stopping chain at record $i - see $LOG" | tee -a "$LOG"
    exit 1
  fi
  CUR="$OUT"
done
echo "=== chain done: $N records folded into $OUT ===" | tee -a "$LOG"
