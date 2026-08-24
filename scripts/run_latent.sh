#!/usr/bin/env bash
# The experiment that was never run: LatentMAS inference on the trained students.
#
# `latent` mode emits the role tag, then thinks for `latent.latent_steps` forward
# passes WITHOUT decoding tokens, carrying each role's KV cache to the next, and
# decodes only after <|consensus|>. If internalisation is possible at all, this
# is where the token count collapses - no retraining, same checkpoints.
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql
OUT=outputs/pilot3; LOGS=$OUT/logs; N=90
export PYTORCH_ALLOC_CONF=expandable_segments:True LATENTSQL_GPU_FRACTION=0.5 LATENTSQL_CPU_THREADS=6
stage(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOGS/latent.log"; }

# wait for the in-flight student evals to release the GPU
while pgrep -f "04_eval.py" >/dev/null; do sleep 30; done
stage "GPU free - starting LatentMAS evaluation"

for ARM in sft grpo_dense grpo_brev; do
  [ -f "$OUT/$ARM/adapter_config.json" ] || { echo "skip $ARM"; continue; }
  stage "latent eval: $ARM  (4 roles x 8 latent steps, decode only the SQL)"
  python3 scripts/04_eval.py --config configs/pilot_mix3_ctrl.yaml --methods latent \
      --student "$OUT/$ARM" --limit $N --max-new-tokens 2048 \
      --out "$OUT/latent_eval_$ARM" 2>&1 | tee "$LOGS/latent_$ARM.log"
done
stage "latent evaluation done"
