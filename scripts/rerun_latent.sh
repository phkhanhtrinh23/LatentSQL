#!/usr/bin/env bash
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql
CFG=configs/latent_imad.yaml; OUT=outputs/latent_imad; LOGS=$OUT/logs
export PYTORCH_ALLOC_CONF=expandable_segments:True LATENTSQL_GPU_FRACTION=0.5 LATENTSQL_CPU_THREADS=6
stage(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOGS/run.log"; }
stage "RERUN 1/2 GRPO with latent rollouts (checkpointing fix applied)"
timeout 28800 python3 scripts/03_grpo.py --config $CFG --data "$OUT/traces/sft.jsonl" \
  --out "$OUT/grpo" 2>&1 | tee "$LOGS/03_grpo.log"
[ -f "$OUT/grpo/adapter_config.json" ] || { echo "FATAL: GRPO produced no adapter"; exit 1; }
stage "RERUN 2/2 eval the TRAINED latent student"
python3 scripts/04_eval.py --config $CFG --methods latent --student "$OUT/grpo" \
  --limit 90 --max-new-tokens 2048 --out "$OUT/eval_latent" 2>&1 | tee "$LOGS/04_eval_latent.log"
stage "done"
