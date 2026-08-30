# COCONUT-Style Latent Reasoning Training

This is the current record for the COCONUT/LatentMAS-style Text-to-SQL
experiments in this repository. It documents the implementation, the staged
CHESS-step SQL-direct run, the results saved on disk, and the next code path to
try.

## Objective

The goal is to compress an explicit multi-agent Text-to-SQL debate into a local
student model that runs most of the reasoning as hidden-state latent steps and
emits only the final SQL. The intended win is token efficiency while retaining
execution accuracy.

Current checkpoints:

```text
SFT start: outputs/latent_sft_1000/sft_7b_bf16
stage-6 latent: outputs/latent_sft_1000/chess_coconut_sql_direct/latent_sft
student base: Qwen/Qwen2.5-Coder-7B-Instruct
```

## What Counts As A Latent Thought

A normal generated token is a vocabulary ID. A COCONUT-style latent thought is a
continuous hidden vector fed back as the next input embedding. It is not decoded
as text and it is not supervised as a target token.

For a simplified sequence:

```text
question + schema
role tag
h1
h2
consensus marker
SQL target
```

`h1` and `h2` are hidden-state vectors. Their labels are masked with `-100`.
The visible SQL continuation is trained with ordinary next-token loss.

## CHESS-Step SQL-Direct Curriculum

The current runner is:

```text
scripts/run_chess_coconut_sql_direct.sh
```

This is different from the older debate-role curriculum. Instead of preserving
the full DIN/DAIL/CHESS/CHASE debate transcript, it converts each trace into a
short CHESS-style chain:

```text
information_retriever
schema_linker
query_classifier
candidate_generator
query_fixer
unit_tester
consensus
```

At stage `N`, the first `N` non-consensus roles are replaced by latent
computation. With `latent_steps=2`, stage 6 uses twelve latent vectors:

```text
6 roles * 2 latent steps per role = 12 latent steps
```

The consensus SQL is still decoded as visible text. There is no hidden
"execution accuracy of latent tokens"; latent steps are compute/context, not
text.

## Command Used For The Completed Stage-6 Run

The completed run used the old training hyperparameters:

```bash
bash scripts/run_chess_coconut_sql_direct.sh \
  94 16 6 3 \
  outputs/latent_sft_1000/sft_7b_bf16 \
  outputs/latent_sft_1000/traces/traces.jsonl \
  configs/latent_sft_1000_7b_bf16.yaml \
  30 5e-5 2 \
  outputs/latent_sft_1000/chess_coconut_sql_direct \
  0.96 Qwen/Qwen2.5-Coder-7B-Instruct \
  2000 1600 1 1 1400
```

Important settings:

```text
usable records: 94
batch size: 16
stages: 1 through 6
epochs per stage: 3
eval probe size after each stage: 30
learning rate in this completed run: 5e-5
latent steps per role: 2
max prompt chars: 2000
max role chars: 1600
max total tokens: 1400
GPU fraction: 0.96
```

This completed run did not include the later lower-learning-rate and SQL-ready
auxiliary changes described below.

## Stage Results

Saved probe results:

```text
outputs/latent_sft_1000/chess_coconut_sql_direct_eval_progress/stage_N/results.md
```

| Stage | Latent roles | Latent steps | n | EX | bird EX | spider EX | Avg tokens | Avg output tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 30 | 66.67 | 40.00 | 93.33 | 1462.6 | 62.7 |
| 2 | 2 | 4 | 30 | 66.67 | 33.33 | 100.00 | 1467.7 | 67.8 |
| 3 | 3 | 6 | 30 | 63.33 | 33.33 | 93.33 | 1460.8 | 61.0 |
| 4 | 4 | 8 | 30 | 26.67 | 20.00 | 33.33 | 1525.6 | 125.7 |
| 5 | 5 | 10 | 30 | 56.67 | 26.67 | 86.67 | 1472.3 | 72.4 |
| 6 | 6 | 12 | 30 | 63.33 | 33.33 | 93.33 | 1467.3 | 67.4 |

Stage 4 was the clear failure point. Its main observed failure mode was repeated
`Tables:`-style schema-linking text that hit the 200-token decode cap; 14 of 30
predictions were unparseable. Stage 5 and stage 6 recovered most of the lost
accuracy on the 30-example probe, but the instability is real.

After stage 6 completed, the running shell printed:

```text
scripts/run_chess_coconut_sql_direct.sh: line 158: ole-list: command not found
```

The stage-6 result file had already been written before that error.

## Full Latent Eval After Stage 6

The full held-out latent eval was started after stage 6 with:

```bash
LATENTSQL_GPU_FRACTION=0.96 \
LATENTSQL_CPU_THREADS=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 scripts/04_eval.py \
  --config configs/latent_sft_1000_7b_bf16.yaml \
  --methods latent \
  --student outputs/latent_sft_1000/chess_coconut_sql_direct/latent_sft \
  --base Qwen/Qwen2.5-Coder-7B-Instruct \
  --latent-steps 2 \
  --latent-role-list information_retriever,schema_linker,query_classifier,candidate_generator,query_fixer,unit_tester,consensus \
  --latent-roles 6 \
  --out outputs/latent_sft_1000/chess_coconut_sql_direct_full_latent
```

Saved result:

```text
outputs/latent_sft_1000/chess_coconut_sql_direct_full_latent/results.md
```

| Run | n | EX | bird EX | spider EX | Avg tokens | Avg output tokens | Unparseable |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full latent after stage 6 | 90 | 52.22 | 33.33 | 71.11 | 1638.0 | 67.5 | 4 |

This was a full latent-role eval for the six CHESS-step roles listed above. It
used `--methods latent`, `--latent-roles 6`, and `--latent-steps 2`.

## Token Accounting

`Avg tokens` is:

```text
prompt_tokens + completion_tokens
```

For latent evals, this does not include latent hidden vectors as normal output
tokens. Latent steps are tracked separately in each prediction under
`extra.latent_steps`.

`Avg output tokens` is the number of decoded visible completion tokens. For the
full latent eval this is mostly the consensus SQL wrapper and SQL body.

Example saved prediction metadata:

```json
{
  "extra": {
    "latent_steps": 12,
    "latent_roles": 6,
    "roles": [
      "information_retriever",
      "schema_linker",
      "query_classifier",
      "candidate_generator",
      "query_fixer",
      "unit_tester",
      "consensus"
    ]
  }
}
```

## Comparison With Saved Baselines

Individual held-out evals for standalone `din_sql`, `dail_sql`, `chess`, and
`chase_sql` are not saved in the current output tree. The saved teacher number
below is from trace generation with the four-system panel, not a matched
90-example held-out eval.

| Run | Source file | n | EX | bird EX | spider EX | Avg tokens | Avg output tokens | Token saved vs previous row |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher panel consensus | `outputs/latent_sft_1000/traces/teacher_samples.jsonl` | 900 | 63.22 | 51.00 | 75.39 | 22995.6 | 2251.6 | - |
| Previous SFT student | `outputs/latent_sft_1000/eval_sft_7b_bf16/results.md` | 90 | 56.67 | 31.11 | 82.22 | 2922.8 | 1339.3 | 87.29% |
| Full latent after stage 6 | `outputs/latent_sft_1000/chess_coconut_sql_direct_full_latent/results.md` | 90 | 52.22 | 33.33 | 71.11 | 1638.0 | 67.5 | 43.96% |
| Stage 6 compressed probe | `outputs/latent_sft_1000/chess_coconut_sql_direct_eval_progress/stage_6/results.md` | 30 | 63.33 | 33.33 | 93.33 | 1467.3 | 67.4 | 10.42% |

Useful direct comparisons:

| Comparison | Token saved |
|---|---:|
| Full latent after stage 6 vs previous SFT student | 43.96% |
| Full latent after stage 6 vs teacher panel consensus | 92.88% |
| Stage 6 compressed probe vs previous SFT student | 49.80% |
| Stage 6 compressed probe vs teacher panel consensus | 93.62% |

Formula:

```text
token_saved_percent = (baseline_avg_tokens - current_avg_tokens) / baseline_avg_tokens * 100
```

The full latent model saved about 44% of reported tokens versus the previous
SFT student, but it also lost 4.45 EX points overall and lost 11.11 EX points on
Spider.

## Code Changes For The Next Run

After diagnosing the stage-4 collapse, the code was changed so the next staged
run can use a lower LR from stage 4 onward and an auxiliary SQL-readiness
constraint.

### Lower Late-Stage Learning Rate

`scripts/run_chess_coconut_sql_direct.sh` now accepts:

```text
argument 19: LATE_LR, default 2e-5
```

For `stage >= 4`, the runner uses `LATE_LR` instead of the initial `LR`.

### SQL-Ready Auxiliary Constraint

`scripts/02c_latent_sft.py` now accepts:

```text
--sql-ready-role
--sql-ready-weight
--sql-ready-tokens
```

`latentsql/distill/latent_sft.py` adds an auxiliary probe inside
`_coconut_sequence_loss`. After the configured latent role, it appends
`<|consensus|>` and supervises the first `--sql-ready-tokens` tokens of the SQL
continuation. The intent is to keep a mid-curriculum latent state aligned with
SQL-generating behavior instead of drifting into schema-linking or agent-loop
text.

Runner defaults for the next run:

```text
SQL_READY_ROLE=4
SQL_READY_WEIGHT=0.5
SQL_READY_TOKENS=16
LATE_LR=2e-5
```

The completed stage-6 run did not use this auxiliary loss. Its logs show
`sql_ready_loss: None`, confirming it was pre-patch.

## Checkpoints And Cleanup

Intermediate checkpoint directories that were not needed were moved out of the
workspace to:

```text
/tmp/latent_agent_text2sql_removed_checkpoints_20260830_2005
```

Moved directories:

```text
outputs/latent_sft_1000/chess_coconut_sql_direct/checkpoints/stage_1
outputs/latent_sft_1000/chess_coconut_sql_direct/checkpoints/stage_2
outputs/latent_sft_1000/chess_coconut_sql_direct/checkpoints/stage_3
outputs/latent_sft_1000/chess_coconut_sql_direct/checkpoints/stage_4
outputs/latent_sft_1000/chess_coconut_sql_direct/checkpoints/stage_5
outputs/latent_sft_1000/coconut_staged_budgetfix/checkpoints/stage_1
outputs/latent_sft_1000/coconut_staged_budgetfix/checkpoints/stage_2
outputs/latent_sft_1000/coconut_staged_budgetfix/checkpoints/stage_3
outputs/latent_sft_1000/coconut_staged_budgetfix/checkpoints/stage_4
```

The live adapter, optimizer state, logs, and eval result artifacts were left in
place.

## Interpretation

The strict conclusion is:

1. The COCONUT-style mechanism now works as a real latent-state path rather than
   as special-token text generation.
2. The CHESS-step SQL-direct format greatly reduces decoded output length.
3. The stage-6 full latent run is cheaper than SFT in reported token accounting,
   but its accuracy is lower overall.
4. The 30-example stage probe is optimistic relative to the 90-example full
   eval, especially on Spider.
5. Stage 4 exposed an optimization/representation instability. The next run
   should use the lower late-stage LR and SQL-ready auxiliary loss already wired
   into the code.

Do not claim that this experiment beats the previous SFT student. The correct
claim is that it compresses visible output substantially and partially retains
accuracy, with a clear next experiment for the stage-4 failure mode.

## Reproducing The Next Run

Use the patched defaults, or pass them explicitly:

```bash
bash scripts/run_chess_coconut_sql_direct.sh \
  94 16 6 3 \
  outputs/latent_sft_1000/sft_7b_bf16 \
  outputs/latent_sft_1000/traces/traces.jsonl \
  configs/latent_sft_1000_7b_bf16.yaml \
  30 5e-5 2 \
  outputs/latent_sft_1000/chess_coconut_sql_direct_lr2e5_sqlready \
  0.96 Qwen/Qwen2.5-Coder-7B-Instruct \
  2000 1600 1 1 1400 \
  2e-5 4 0.5 16
```

After stage 6, run the full latent eval:

```bash
LATENTSQL_GPU_FRACTION=0.96 \
LATENTSQL_CPU_THREADS=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 scripts/04_eval.py \
  --config configs/latent_sft_1000_7b_bf16.yaml \
  --methods latent \
  --student outputs/latent_sft_1000/chess_coconut_sql_direct_lr2e5_sqlready/latent_sft \
  --base Qwen/Qwen2.5-Coder-7B-Instruct \
  --latent-steps 2 \
  --latent-role-list information_retriever,schema_linker,query_classifier,candidate_generator,query_fixer,unit_tester,consensus \
  --latent-roles 6 \
  --out outputs/latent_sft_1000/chess_coconut_sql_direct_lr2e5_sqlready_full_latent
```

## Verification Already Run

The patched code was syntax-checked with:

```bash
python3 -m py_compile scripts/02c_latent_sft.py latentsql/distill/latent_sft.py
bash -n scripts/run_chess_coconut_sql_direct.sh
```
