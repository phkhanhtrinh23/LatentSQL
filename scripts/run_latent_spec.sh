#!/usr/bin/env bash
# CHESS teacher -> SFT on the full debate -> GRPO with LATENT rollouts.
# The debate runs in continuous space, so there is no token-cutting: w_fmt=0,
# clip inert, reward = execution + atomic + memory.
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql
CFG=configs/latent_spec.yaml; OUT=outputs/latent_spec; LOGS=$OUT/logs
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True LATENTSQL_GPU_FRACTION=0.5 LATENTSQL_CPU_THREADS=6
while pgrep -f "run_latent_imad.sh" >/dev/null; do sleep 60; done
stage(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOGS/run.log"; }

# the teacher is identical to the latent_imad run, so reuse its traces and SFT
# checkpoint rather than paying for them twice - the only variable is speculative
# decoding inside the GRPO rollouts
stage "1/4 reuse traces + SFT from outputs/latent_imad"
mkdir -p "$OUT"
[ -d "$OUT/traces" ] || cp -r outputs/latent_imad/traces "$OUT/traces"
[ -d "$OUT/sft" ]    || cp -r outputs/latent_imad/sft "$OUT/sft"
stage "traces kept: $(wc -l < $OUT/traces/sft.jsonl 2>/dev/null || echo 0)"

stage "2/4 SFT: reused"

stage "3/4 GRPO with LATENT rollouts (no clip, no format reward)"
[ -f "$OUT/grpo/adapter_config.json" ] || timeout 28800 python3 scripts/03_grpo.py --config $CFG \
  --data "$OUT/traces/sft.jsonl" --out "$OUT/grpo" 2>&1 | tee "$LOGS/03_grpo.log"

stage "4/4 eval, 90 questions: SFT baseline, then the latent-trained student"
python3 scripts/04_eval.py --config $CFG --methods student --student "$OUT/sft" \
  --limit 90 --max-new-tokens 2048 --out "$OUT/eval_sft" 2>&1 | tee "$LOGS/04_eval_sft.log"
python3 scripts/04_eval.py --config $CFG --methods latent --student "$OUT/grpo" \
  --limit 90 --max-new-tokens 2048 --out "$OUT/eval_latent" 2>&1 | tee "$LOGS/04_eval_latent.log"
stage "done"
