#!/usr/bin/env bash
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql
CFG=configs/chess_only.yaml; OUT=outputs/chess; LOGS=$OUT/logs
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True LATENTSQL_GPU_FRACTION=0.5 LATENTSQL_CPU_THREADS=6
stage(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOGS/chess.log"; }
while pgrep -f "04_eval.py|01_build_traces.py|03_grpo.py" >/dev/null; do sleep 30; done

stage "1/4 teacher: CHESS only (IR -> selector -> 3 candidates -> unit test)"
[ -f "$OUT/traces/sft.jsonl" ] || python3 scripts/01_build_traces.py --config $CFG \
  --dataset mix --limit 90 --out "$OUT/traces" 2>&1 | tee "$LOGS/01_traces.log"
stage "traces kept: $(wc -l < $OUT/traces/sft.jsonl 2>/dev/null || echo 0)"

stage "2/4 SFT"
[ -f "$OUT/sft/adapter_config.json" ] || python3 scripts/02_sft.py --config $CFG \
  --data "$OUT/traces/sft.jsonl" 2>&1 | tee "$LOGS/02_sft.log"

stage "3/4 GRPO (dense rewards)"
[ -f "$OUT/grpo/adapter_config.json" ] || timeout 28800 python3 scripts/03_grpo.py --config $CFG \
  --data "$OUT/traces/sft.jsonl" --out "$OUT/grpo" 2>&1 | tee "$LOGS/03_grpo.log"

stage "4/4 eval: student + latent, 90 questions"
for M in student; do
  python3 scripts/04_eval.py --config $CFG --methods $M --student "$OUT/grpo" \
    --limit 90 --max-new-tokens 2048 --out "$OUT/eval_$M" 2>&1 | tee "$LOGS/04_eval_$M.log"
done
stage "done"
