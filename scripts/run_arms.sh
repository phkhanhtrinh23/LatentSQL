#!/usr/bin/env bash
# Four-arm factorial. Both teacher corpora already exist, so nothing re-runs the
# teacher; SFT is shared per corpus. Only DPO/GRPO differ between arms.
#
#   A  4-system,  no DPO, dense    <- baseline
#   B  4-system, +DPO,    dense    <- B vs A : does DPO help
#   C  4-system, +DPO,    binary   <- C vs B : dense vs binary reward
#   D  CHESS,    +DPO,    dense    <- D vs B : do many systems help
set -u
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql
export PYTORCH_ALLOC_CONF=expandable_segments:True LATENTSQL_GPU_FRACTION=0.75 LATENTSQL_CPU_THREADS=6
L=outputs/arms.log
stage(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a $L; }
while pgrep -f "rerun_latent.sh|03_grpo.py|04_eval.py" >/dev/null; do sleep 60; done

# corpora: 4-system from pilot3, CHESS from latent_imad
for A in a b c; do mkdir -p outputs/arm_$A; cp -rn outputs/pilot3/traces outputs/arm_$A/ 2>/dev/null; done
mkdir -p outputs/arm_d; cp -rn outputs/latent_imad/traces outputs/arm_d/ 2>/dev/null

# rebuild the SFT corpus under the rescue rule
for A in a b c d; do
  CFG=$(ls configs/arm_${A}_*.yaml)
  python3 - "$CFG" "outputs/arm_$A" <<'PY'
import sys, os, json
from latentsql.config import load_config
from latentsql.mas.trace import load_traces
from latentsql.distill.build_dataset import build_sft_records, save_records
cfg=load_config(sys.argv[1]); out=sys.argv[2]
tr=load_traces(f"{out}/traces/traces.jsonl")
recs,st=build_sft_records(tr,cfg.trace)
for r in recs:
    if not r.db_path:
        m=next((t for t in tr if t.qid==r.qid),None)
        if m: r.db_path=m.meta.get("db_path","")
save_records(recs,f"{out}/traces/sft.jsonl")
print(f"{out}: {st.summary()}")
PY
done

stage "SFT (shared per corpus)"
for A in a d; do
  CFG=$(ls configs/arm_${A}_*.yaml)
  [ -f outputs/arm_$A/sft/adapter_config.json ] || \
    python3 scripts/02_sft.py --config $CFG --data outputs/arm_$A/traces/sft.jsonl 2>&1 | tee outputs/arm_$A/sft.log
done
for A in b c; do cp -rn outputs/arm_a/sft outputs/arm_$A/ 2>/dev/null; done

stage "DPO on arms B, C, D"
for A in b c d; do
  CFG=$(ls configs/arm_${A}_*.yaml)
  [ -f outputs/arm_$A/dpo/adapter_config.json ] || \
    python3 scripts/02b_dpo.py --config $CFG --traces outputs/arm_$A/traces/traces.jsonl \
      --sft outputs/arm_$A/sft --out outputs/arm_$A/dpo 2>&1 | tee outputs/arm_$A/dpo.log
done

stage "GRPO (latent rollouts) on all four arms"
for A in a b c d; do
  CFG=$(ls configs/arm_${A}_*.yaml)
  START=outputs/arm_$A/dpo; [ -d "$START" ] || START=outputs/arm_$A/sft
  [ -f outputs/arm_$A/grpo/adapter_config.json ] || \
    timeout 7200 python3 scripts/03_grpo.py --config $CFG --data outputs/arm_$A/traces/sft.jsonl \
      --sft "$START" --out outputs/arm_$A/grpo 2>&1 | tee outputs/arm_$A/grpo.log
  [ -f outputs/arm_$A/grpo/adapter_config.json ] || echo "WARN arm $A: GRPO produced no adapter" | tee -a $L
done

stage "eval (latent, 90 questions)"
for A in a b c d; do
  CFG=$(ls configs/arm_${A}_*.yaml)
  [ -f outputs/arm_$A/grpo/adapter_config.json ] || continue
  python3 scripts/04_eval.py --config $CFG --methods latent --student outputs/arm_$A/grpo \
    --limit 90 --max-new-tokens 2048 --out outputs/arm_$A/eval 2>&1 | tee outputs/arm_$A/eval.log
done
stage "done"
