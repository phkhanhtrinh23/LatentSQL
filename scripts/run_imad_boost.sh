#!/usr/bin/env bash
# Pure text-mode IMAD + speculative decoding, 4-system panel with rescue.
# Reuses outputs/arm_a/traces/{traces.jsonl,sft.jsonl} (already the 63-record
# rescue-rebuilt corpus, no need to re-run the ~65min teacher stage).
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql
export PYTORCH_ALLOC_CONF=expandable_segments:True LATENTSQL_GPU_FRACTION=0.75 LATENTSQL_CPU_THREADS=6
CFG=configs/imad_boost.yaml
OUT=outputs/imad_boost
L=$OUT/run.log
stage(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$L"; }

mkdir -p "$OUT/traces"
cp -n outputs/arm_a/traces/traces.jsonl outputs/arm_a/traces/sft.jsonl "$OUT/traces/" 2>/dev/null
stage "reused corpus: $(wc -l < "$OUT/traces/sft.jsonl") kept traces"

stage "SFT"
[ -f "$OUT/sft/adapter_config.json" ] || \
  python3 scripts/02_sft.py --config "$CFG" --data "$OUT/traces/sft.jsonl" 2>&1 | tee "$OUT/sft.log"

stage "GRPO (text mode, speculative decoding)"
[ -f "$OUT/grpo/adapter_config.json" ] || \
  timeout 36000 python3 scripts/03_grpo.py --config "$CFG" --data "$OUT/traces/sft.jsonl" \
    --sft "$OUT/sft" --out "$OUT/grpo" 2>&1 | tee "$OUT/grpo.log"

stage "eval (student, 90 questions)"
python3 scripts/04_eval.py --config "$CFG" --methods student --student "$OUT/grpo" \
  --limit 90 --max-new-tokens 3072 --out "$OUT/eval" 2>&1 | tee "$OUT/eval.log"

stage "done"
