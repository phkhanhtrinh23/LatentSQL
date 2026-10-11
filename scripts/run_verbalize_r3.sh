#!/usr/bin/env bash
# Verbalize the r3 student's latent thoughts z1..z6 (plain and reviser cues)
# for the GRPO and SFT checkpoints on the 90 eval questions.
set -euo pipefail
cd /home/trinh/Dropbox/thesis/latent_agent_text2sql
R=outputs/latent_sft_1000/chess_three_role_r3
export LATENTSQL_REQUIRE_CUDA=1 LATENTSQL_GPU_FRACTION=0.5 PYTORCH_ALLOC_CONF=expandable_segments:True
for ARM in grpo:$R/grpo_paper sft:$R/latent_sft; do
  NAME=${ARM%%:*}; DIR=${ARM#*:}
  [ -f "$R/verbalize_$NAME/verbalize.json" ] && continue
  echo "=== [$(date --iso-8601=seconds)] verbalize r3 $NAME ==="
  python3 scripts/08_latent_trace.py --config configs/latent_sft_1000_7b_bf16.yaml \
    --student "$DIR" --base Qwen/Qwen2.5-Coder-7B-Instruct \
    --latent-steps 2 --roles schema_linker,candidate_generator,reviser,consensus \
    --limit 90 --mode verbalize --text-tokens 160 --out "$R/verbalize_$NAME"
done
echo "=== [$(date --iso-8601=seconds)] verbalize r3 done ==="
