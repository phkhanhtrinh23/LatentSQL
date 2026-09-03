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

Latent GRPO also now enables a correct-rollout brevity tie-breaker:

```text
grpo.reward.w_brevity=0.2
```

Without this term, all-correct rollout groups receive identical reward
(`1.0` correctness + `0.1` executable SQL) and therefore zero GRPO advantage.
The brevity term is awarded only to correct rollouts, so it separates saturated
groups by decoded length without rewarding short wrong answers.

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

```
The equal reward is expected from the current reward design, not a speculative-decoding bug.

What happened in the larger run:

step 1: reward_mean=1.1, r_clip_mean=1.0, reward_std_mean=0.0, adv_nonzero_frac=0.0
step 2: reward_mean=1.1, r_clip_mean=1.0, reward_std_mean=0.0, adv_nonzero_frac=0.0
step 3: reward_mean=0.1633, r_clip_mean=0.0, reward_std_mean=0.0315, adv_nonzero_frac=1.0

Why steps 1-2 had identical rewards:

The latent GRPO reward is mostly binary execution correctness:
      + w_exec_ok * r_exec
      + w_atomic * r_atomic for wrong rollouts
      + w_memory * r_memory for wrong rollouts
      + w_brevity * r_brevity for correct rollouts

Current config has:

w_clip = 1.0
w_exec_ok = 0.1
w_brevity = 0.0
w_fmt = 0.0

So if every rollout in a group is correct and executable, each gets:

1.0 + 0.1 = 1.1

No variance means GRPO advantage is zero:

adv_nonzero_frac = 0.0

So GRPO cannot learn from all-correct groups unless we enable a tie-breaker, e.g. brevity reward.
```
## GRPO Eval After The Brevity Run (2026-08-31)

Checkpoint evaluated:

```text
outputs/latent_sft_1000/chess_coconut_sql_direct_grpo_spec7b_hard50_brevity_cap1200
```

This is the stage-6 CHESS-step SQL-direct student after 50 GRPO optimizer steps
on 50 hard prompts (`max_prompt_chars=1200`, group size 8, brevity reward on,
`w_fmt=0`), rollouts sampled through a k=3 speculative draft. GRPO training
details are in `grpo_result.json` next to the adapter.

Both evals below use the same held-out n=90 mix, the same six latent CHESS roles
plus a decoded consensus, and `--latent-steps 2`.

### Result 1 — greedy decode (the headline number)

```bash
LATENTSQL_GPU_FRACTION=0.96 \
LATENTSQL_CPU_THREADS=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 scripts/04_eval.py \
  --config configs/latent_sft_1000_7b_bf16.yaml \
  --methods latent \
  --student outputs/latent_sft_1000/chess_coconut_sql_direct_grpo_spec7b_hard50_brevity_cap1200 \
  --base Qwen/Qwen2.5-Coder-7B-Instruct \
  --latent-steps 2 \
  --latent-role-list information_retriever,schema_linker,query_classifier,candidate_generator,query_fixer,unit_tester,consensus \
  --latent-roles 6 \
  --out outputs/latent_sft_1000/chess_coconut_sql_direct_grpo_spec7b_hard50_brevity_cap1200_eval
```

| Run | n | EX | bird EX | spider EX | Avg tokens | Avg output tokens | Unparseable |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full latent after stage 6 (SFT, pre-GRPO) | 90 | 52.22 | 33.33 | 71.11 | 1638.0 | 67.5 | 4 |
| **After GRPO brevity cap1200** | 90 | **58.89** | **37.78** | **80.00** | 1632.4 | **61.9** | **0** |
| Δ | | **+6.67** | +4.45 | +8.89 | −5.6 | −5.6 | −4 |

By difficulty: spider-all 80.00 (36/45), bird simple 44.00 (11/25), bird
moderate 33.33 (5/15), bird challenging 20.00 (1/5).

GRPO raised accuracy while output length *fell* 8%, which is what the brevity
reward was shaped to do. The gain is concentrated on Spider (+8.89). Two caveats
worth carrying: the 63.33 figure quoted for the earlier probe was n=30, not
comparable to this n=90; and GRPO's 50 training prompts were drawn from the same
pool under the same `max_prompt_chars` filter, so the split is held out but the
difficulty filtering is shared.

`n_unparseable` went 4 → 0: every one of the 90 generations produced extractable
SQL, so the latent → `<|consensus|>` → SQL format survived GRPO intact.

### Result 2 — speculative decode, untrained 7B as draft

Same checkpoint, same 90 questions, with the *base* (untrained)
Qwen2.5-Coder-7B-Instruct as the speculative draft:

```bash
  ... same as above, plus:
  --spec-decode \
  --spec-draft-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --spec-draft-quantization 4bit \
  --spec-k 3 \
  --out outputs/latent_sft_1000/chess_coconut_sql_direct_grpo_spec7b_hard50_brevity_cap1200_eval_spec
```

| | greedy | speculative (7B draft) |
|---|---:|---:|
| EX (n=90) | **58.89** | 57.78 |
| bird EX | **37.78** | 35.56 |
| spider EX | 80.00 | 80.00 |
| Avg output tokens | 61.94 | 61.79 |
| **Latency / sample** | **2.092 s** | 4.257 s |
| Total wall-clock | **188.3 s** | 383.1 s |
| Peak GPU | **15.85 GB** | 21.85 GB |
| Acceptance length | — | 2.337 |
| Acceptance rate | — | 0.446 |

**Speculative decoding is 2.03x slower here, not faster.** This is the expected
outcome for a same-size draft: a 7B drafting for a 7B pays roughly the target's
own cost per proposed token, so a mean acceptance length of 2.34 cannot buy back
the draft's forward passes (316.6 s of the 383.1 s total was spent inside the
sampler). The 4-bit draft quantization, needed to fit two 7B models on a 24 GB
card, makes the draft slower per token than the bf16 target it is drafting for.
Speculative decoding pays off with a *small* draft; `--spec-draft-model` still
defaults to Qwen2.5-Coder-0.5B-Instruct for that reason.

The latent path also gives speculation less to work with than the `student` path
does: only the decoded SQL is drafted (~62 tokens), never the latent prefix, and
the draft cannot see the latent thoughts at all — it conditions on the plain
text prompt, which is why acceptance sits at 0.446.

### On the 1.11-point EX difference

Greedy verification at `eval.temperature: 0.0` is *distributionally* lossless,
and 86/90 predictions came back byte-identical, 89/90 identical in correctness.
The four that differ are floating-point, not distributional: verification scores
k+1 tokens in one batched forward pass, whereas the greedy path runs one token
per forward, and in bf16 those are not bit-identical. Three of the four diverge
into an equivalent query (a flipped join order, `"` vs `'` quoting) and score the
same; one (`bird_dev_565`) degrades into a different query and flips True →
False. So read 58.89 as the accuracy of this checkpoint and the 1.11 gap as
decode noise at n=90, not as a property of speculative decoding.

### Code Changes This Eval Required

`--spec-decode` previously applied only to `--methods student`; on `latent` it
was accepted and silently ignored (`scripts/04_eval.py` forwarded the sampler
only when `method == "student"`, and the banner said so).

- `latentsql/distill/latent.py` — `LatentReasoner.generate()` takes an optional
  `spec` sampler. The latent prefix is built exactly as on the greedy path, then
  handed to `SpeculativeSampler.generate()` as `target_past` / `target_attn` /
  `target_first_logits`, with the plain text prompt as `draft_prompt_ids`. This
  is the same wiring GRPO's `_sample_group_latent` already used. `LatentRun`
  gained a `spec` field carrying `SpecStats.as_dict()`.
- `latentsql/eval/run_eval.py` — the latent branch passes `spec` through and
  records the per-sample stats under `extra["spec"]`.
- `scripts/04_eval.py` — forwards the sampler for `latent` as well as `student`;
  new `--spec-draft-quantization` (defaults to `sft.quantization`) so the draft
  can be quantized independently of the target, which is what makes a 7B draft
  fit alongside a bf16 7B target on 24 GB; corrected the stale `[spec]` banner.

Draft quality affects only acceptance rate, never correctness, so quantizing the
draft is safe.

## Speculative Draft Sweep — 0.5B / 3B / 7B (2026-08-31)

Same checkpoint, same held-out n=90, same six latent roles. Only the draft
model changes. This supersedes the closing claim of the previous section
("speculative decoding pays off with a *small* draft"), which was an
extrapolation from the 7B result and is **wrong**: measured at three draft
sizes, speculation loses at every one.

| draft | EX | bird | spider | Latency/sample | Total | vs greedy | Acc. length | Acc. rate | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none (greedy) | **58.89** | **37.78** | 80.00 | **2.092 s** | **188 s** | **1.00x** | — | — | **15.85 GB** |
| Coder-0.5B bf16 | 57.78 | 35.56 | 80.00 | 2.989 s | 269 s | 0.70x | 2.26 | 0.418 | 17.27 GB |
| Coder-3B bf16 | 57.78 | 35.56 | 80.00 | 3.392 s | 305 s | 0.62x | 2.37 | 0.458 | 22.28 GB |
| Coder-7B 4bit | 57.78 | 35.56 | 80.00 | 4.257 s | 383 s | 0.49x | 2.34 | 0.446 | 21.84 GB |

Every speculative run scores an identical 57.78 EX with an identical
bird/spider split, and each agrees with greedy on 89/90 questions (86-88/90
byte-identical predictions). The single flipped question is the same bf16
batched-verification noise described in the previous section, not an effect of
draft size.

### Reading The Sweep

Two numbers carry the finding.

1. **Acceptance length is flat across a 14x range of draft size** (2.26 / 2.37 /
   2.34 for 0.5B / 3B / 7B). Draft *quality* is not the limiting factor. A
   bigger draft proposes barely better tokens here, so it buys nothing.
2. **Latency is nearly insensitive to draft size.** Shrinking the draft 14x
   (7B -> 0.5B) improved wall-clock only 1.42x, and still landed 1.43x *slower*
   than plain greedy decoding.

Together these say the cost is dominated by fixed **per-round** overhead, not by
the draft's forward passes - which is why no draft size rescues it.

One structural cause is specific to the latent path: the target enters decoding
with the latent prefix already in its KV cache, but the draft cannot see latent
thoughts, so it re-prefills the full ~1570-token text prompt once per sample,
amortized over only ~62 generated tokens. Speculation needs long generations to
pay off, and the whole point of this student is that its generation is short.

`_copy_cache`'s `copy.deepcopy` was measured and **ruled out** as the bottleneck:
a realistic 28-layer / seq-1600 / 88 MB Qwen-7B cache deep-copies in 0.94 ms, so
the ~54 copies per sample cost ~50 ms against a ~900 ms gap. The leading
remaining hypothesis - not yet measured - is host-device synchronization inside
the accept/reject loop: `float(p_rows[i][0, x])`, `float(q_x)`,
`float(torch.rand(1).item())` and `_sample`'s `.item()` each force a CUDA sync,
running to roughly 400-500 syncs per sample, plus a full 152k-wide softmax
allocated per proposal. Both costs are per round and independent of draft size,
matching the observed insensitivity. Confirming this needs a profiler run.

### Conclusion For This Path

Use greedy decoding for `--methods latent`. Speculative decoding is a wall-clock
regression here at every draft size tested, and it costs 1.4-6.4 GB of extra
GPU. It remains potentially useful on the `student` path, which generates a
whole debate (hundreds to ~1000 tokens) rather than ~62 tokens of SQL - that
case was not measured here.

### Bug Fixed: Draft/Target Vocab Mismatch

The first 3B run crashed in the rejection step:

```text
RuntimeError: The size of tensor a (152064) must match the size of tensor b
(151936) at non-singleton dimension 1
```

Qwen pads each model's embedding matrix independently even though the tokenizer
is byte-identical:

| model | `vocab_size` |
|---|---:|
| Qwen2.5-Coder-7B-Instruct | 152064 |
| Qwen2.5-Coder-3B-Instruct | 151936 |
| Qwen2.5-Coder-1.5B-Instruct | 151936 |
| Qwen2.5-Coder-0.5B-Instruct | 151936 |

Step 3 of `SpeculativeSampler.generate` computes `p - q` elementwise, so a draft
whose padded width differs from the target's dies there. **`--spec-decode`
therefore only ever worked when the draft was the 7B target itself** - which is
why GRPO used `spec7b`, and it means the documented default
(`--spec-draft-model Qwen/Qwen2.5-Coder-0.5B-Instruct`) would have crashed for
anyone who tried it, on the `student` path as well as `latent`.

Fix in `latentsql/distill/speculative.py`: truncate both logit vectors to
`vocab = min(target_width, draft_width)` *before* the softmax, which
renormalises exactly over the retained support. The dropped slots sit above the
real tokenizer vocab (151665 tokens) and no valid token id ever lands in them.
The 7B-draft path is unchanged, since `min(152064, 152064)` is a no-op - the
numbers in the previous section stand.

## Checkpoints And Cleanup — Second Pass (2026-08-31)

A second cleanup followed the same policy as the 2026-08-30 pass: nothing was
deleted, everything was **moved** out of the workspace so it stays recoverable.

```text
/tmp/latent_agent_text2sql_removed_checkpoints_20260831_0146
```

`outputs/` went **4.1 GB -> 1.6 GB** (2.5 GB moved).

Moved:

| Item | Size | Why it was safe |
|---|---:|---|
| `chess_coconut_sql_direct/checkpoints/stage_6` | 319 MB | Byte-identical duplicate of `chess_coconut_sql_direct/latent_sft`; md5 verified equal before moving |
| `chess_coconut_sql_direct/optim_state.pt` | 617 MB | Resume-only optimizer state; that curriculum finished at stage 6 and GRPO has already run on top of it |
| `curriculum_chain_7b_bf16_multipass_fixed/optim_state.pt` | 617 MB | Resume-only optimizer state for a superseded arm |
| `chess_coconut_sql_direct_grpo_brevity_probe` | 319 MB | GRPO smoke probe, 2 optimizer steps / 40 s |
| `chess_coconut_sql_direct_grpo_spec7b_probe` | 319 MB | GRPO smoke probe, 1 optimizer step / 13 s |
| `chess_coconut_sql_direct_grpo_probe_nospec` | 319 MB | GRPO smoke probe, 2 optimizer steps / 20 s |

`chess_coconut_sql_direct/checkpoints/` is now empty and was removed; stage_1
through stage_5 had already been moved out in the 2026-08-30 pass.

The five adapters that remain are the ones worth keeping:

```text
outputs/latent_sft_1000/sft_7b_bf16                                          # SFT baseline
outputs/latent_sft_1000/chess_coconut_sql_direct/latent_sft                  # stage-6 latent student, the 52.22 EX baseline
outputs/latent_sft_1000/chess_coconut_sql_direct_grpo_spec7b_hard50_brevity_cap1200  # current best, 58.89 EX
outputs/latent_sft_1000/curriculum_chain_7b_bf16_multipass_fixed/latent_sft  # earlier documented arm
outputs/latent_sft_1000/coconut_staged/latent_sft                            # earlier documented arm
```

For the two earlier arms only the optimizer state was dropped, never the
weights. `traces/` (13 MB, the training corpus) and all 40 eval-result
directories were left in place - they are ~1 MB each and hold the actual
recorded numbers. In particular
`chess_coconut_sql_direct_grpo_spec7b_probe_eval30/` survives, so the 63.33 EX
(n=30) probe figure cited elsewhere in this document is still backed by its
predictions even though the 319 MB probe adapter it came from is gone.

Two things were deliberately left alone: `0.96.log` in the repo root (a stray
untracked log from an aborted run) and `pdfs/` (papers the README documents as
intentionally uncommitted).

## Root Cause Of The Stage-4 Collapse (2026-09-03)

The earlier section "Stage Results" recorded stage 4 dropping to 26.67 EX
between a 63.33 EX stage 3 and a 56.67 EX stage 5, and attributed it to "an
optimization/representation instability". That attribution was wrong. The
training log says the opposite, and the real mechanism is now identified.

### The training loss shows no instability at all

Per-stage aggregates over
`outputs/latent_sft_1000/chess_coconut_sql_direct/latent_sft_result_steps.jsonl`
(282 logged steps per stage = 94 records x 3 epochs):

| n_latent_roles | loss mean | loss max | probe EX | avg output tokens |
|---:|---:|---:|---:|---:|
| 1 | 0.1332 | 0.9836 | 66.67 | 62.7 |
| 2 | 0.0186 | 0.1907 | 66.67 | 67.8 |
| 3 | 0.0019 | 0.1571 | 63.33 | 61.0 |
| **4** | **0.0008** | **0.1265** | **26.67** | **125.7** |
| 5 | 0.0015 | 0.1875 | 56.67 | 72.4 |
| 6 | 0.0014 | 0.1416 | 63.33 | 67.4 |

Stage 4 has the **lowest mean loss and the lowest max loss of any stage**. There
is no spike, no divergence, no NaN. Whatever broke is invisible to the objective.

### Three mechanisms, in the order they compound

**1. The objective is saturated by stage 3, so stage 4 trains on noise.**

`text_loss` is `null` in all 1908 step records of that run. That is not a logging
gap - it is the format. `_coconut_sequence_loss`
(`latentsql/distill/latent_sft.py:113`) breaks out of the role loop at
`idx >= n_latent_roles` and never adds the remaining roles back as text, so the
supervised sequence is exactly:

```text
chat_prompt  +  [<|role_i|> + latent_steps latent vectors] * n_latent_roles
             +  <|consensus|>  +  gold SQL
```

The only supervised tokens are the SQL - a constant 62.3 target tokens per record
at every stage. With 94 records and 3 epochs per stage, a per-token cross-entropy
of 8e-4 by stage 4 means those 94 SQL strings are memorised. There is no learning
signal left for stage 4 to use.

**2. Adam takes full-size steps into that noise anyway.**

`grad_accum` is set to the batch size (`--grad-accum "$limit"`, so 16), giving one
optimizer step per batch: 6 optimizer steps per epoch, **18 per stage**. The
runner also wipes the momentum at every stage boundary
(`rm -f "$OPTIM_STATE"`, `scripts/run_chess_coconut_sql_direct.sh`, in the
`ep_start -eq 1` branch).

Adam normalises the update by `sqrt(v_hat)`, so a 1e-4 gradient produces the same
~`lr`-sized parameter change as a 1e-1 one, and with freshly reset moments the
first steps of a stage are exactly `lr * sign(g)`. At `lr=5e-5` over 18 steps that
is real LoRA drift, and at stage 4 its direction is set by whichever one or two
of 16 records still have non-negligible loss (the max of 0.1265 against a
background of 1e-4). Stage 4 is where the gradient is purest noise *and* the steps
are still full size.

**3. Nothing anchors "`<|consensus|>` means SQL", so the drift lands on the base
model's transcript prior.**

14 of 30 stage-4 predictions were unparseable, and they all fail the same way -
the model decodes `schema_linker` body text after the consensus marker and loops
it to the 200-token cap:

```text
<|consensus|>Tables:
Tables: visitor, visit, museum
Tables:
Tables: visitor, visit, museum
... (repeated to the cap)
```

`Tables: ...` is the rendering of `trace.schema_links`. The starting adapter
`sft_7b_bf16` was trained on full debate transcripts, so "after a role tag comes
`Tables: ...`" is a strong prior in the weights. Latent training never penalises
it, because *on the training records the model does start SQL correctly* - loss
8e-4. The prior only surfaces off-distribution. This is a pure generalisation
failure that the training objective is structurally incapable of observing, which
is exactly why the loss curve looks perfect while the probe craters.

The doubled output length (125.7 vs 61.0 tokens at stage 3) is the same fact seen
from the other side: those are cap-length loops, not longer queries.

### The fix wired in after that run was a no-op at the stage it targeted

The "SQL-Ready Auxiliary Constraint" section above describes
`SQL_READY_ROLE=4`. In `_coconut_sequence_loss` the probe fires on
`idx + 1 == sql_ready_role` (`latentsql/distill/latent_sft.py:130`), and
`sql_ready_active` additionally requires `n_latent_roles >= sql_ready_role`
(`latentsql/distill/latent_sft.py:506`).

At stage 4 with `sql_ready_role=4`, the probe therefore fires after the **last**
latent role. Its prefix is byte-identical to the main sequence's prefix and its
target is the first 16 tokens of the same gold SQL - so it is a weighted duplicate
of the main loss, contributing nothing new. It only becomes a genuine
mid-curriculum probe at stages 5 and 6, which were not the problem.

### Retrain configuration (in flight)

Resumed from the stage-3 checkpoint - recovered intact from
`/tmp/latent_agent_text2sql_removed_checkpoints_20260830_2005/stage_3`, md5
`5d61d233bccc0e8b29658f7073a24116` - into a **new** run directory
`outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix`, leaving the original
run and the GRPO checkpoint untouched.

| Knob | Original run | Retrain | Targets |
|---|---|---|---|
| lr at stage >= 4 | 5e-5 | **1e-5** | mechanism 2 |
| `SQL_READY_ROLE` | (inactive) | **3** | mechanism 3 |
| `SQL_READY_WEIGHT` | (inactive) | 0.5 | |
| `SQL_READY_TOKENS` | (inactive) | 24 | |

`SQL_READY_ROLE=3` makes the probe a real mid-prefix anchor at every one of
stages 4, 5 and 6: "after 3 latents plus `<|consensus|>`, emit SQL", with a prefix
strictly shorter than the main sequence's.

Honest caveat recorded before the result is known: the first steps of the retrain
log `sql_ready_loss` around 1e-4, i.e. the probe is saturated too. It contributes
almost no gradient, so **the lower learning rate is doing nearly all the work**.
If stage 4 still collapses, the next lever is structural rather than a
hyperparameter - the SQL-direct format supervises 62 tokens and discards every
role body, so no term in the objective can penalise the transcript prior
off-distribution.

### A separate, unrelated failure on the first launch attempt

The first relaunch aborted at stage 4 epoch 1 batch 1 with

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 374.00 MiB.
... this process has 17.47 GiB memory in use. Process 1961847 has 5.91 GiB memory in use.
```

Batch 0 had completed normally (16 steps, exit 0, adapter and optimizer state
written) with a peak of 18.5 GB against a 22.57 GB budget. The abort was a race,
not a capacity problem: every batch is a fresh process and the runner launches the
next one as soon as the previous shell command returns, but process exit does not
mean the CUDA context is released. Two contexts overlapped and 17.47 + 5.91 GB
exceeded the budget.

`scripts/run_chess_coconut_sql_direct.sh` now has a `gpu_settle` helper that polls
`nvidia-smi` until the card drops below `LATENTSQL_GPU_SETTLE_MB` (default 1500)
or 120 s elapses, called before every training batch and every eval. The failed
attempt's log is kept at
`outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix.attempt1_oom.log`; the
adapter was reset to the pristine stage-3 copy before relaunching, so no batch was
applied twice.

### And a second, worse one: a silent CPU downgrade

The next attempt did not crash - it went quiet. After 48 minutes it had logged
zero training steps while burning **1h53m of CPU time**, with 18.3 GB resident in
host RAM and the GPU sitting at 40 MiB and power state P8.

The 7B was training on the CPU. `resolve_device("auto")`
(`latentsql/llm/hf_local.py`) returned `"cpu"` because
`torch.cuda.is_available()` was False at that instant, and with
`sft.quantization: none` there is no `quantization_config`, so the loader simply
called `model.to("cpu")` and everything downstream worked correctly and roughly
500x too slowly. `torch.cuda.is_available()` was True again when tested
minutes later, so the condition was transient - most plausibly the same
context-teardown window as the OOM above.

This failure mode is much worse than the OOM: an OOM aborts the run loudly,
whereas this one would have silently produced a "successful" 6-stage run days
later. Two guards were added:

- `LATENTSQL_REQUIRE_CUDA=1` makes `resolve_device` raise instead of returning
  `"cpu"`. The runner now sets it on every training batch and every eval.
- Each training batch retries **once** after a 20 s pause. Both observed
  boundary failures (context not yet released -> OOM; `is_available()`
  transiently False -> the new abort) clear within seconds, so one retry turns a
  fatal race into a logged hiccup.

Log kept at
`outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix.attempt2_cpu_stall.log`.
The stall never reached step 1 and `--save-steps` is 16, so nothing was written;
the adapter was re-verified against the stage-3 md5 before the third launch.

## Stage-4 Retrain — Recorded Results (2026-09-03)

Stages 4, 5 and 6 were retrained from the stage-3 checkpoint with the two changes
described in the previous section (`lr` 5e-5 -> 1e-5 for stage >= 4,
`SQL_READY_ROLE` 4 -> 3). Run directory:
`outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix`.

### Stage probes (n=30, identical eval invocation to the original run)

| Stage | EX old | EX new | bird new | spider new | unparseable old -> new | hit 200-cap old -> new | out tokens old -> new |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 26.67 | **66.67** | 33.33 | 100.00 | 14 -> **0** | 14 -> **0** | 125.7 -> **61.6** |
| 5 | 56.67 | **66.67** | 33.33 | 100.00 | 3 -> **0** | 2 -> **0** | 72.4 -> **61.2** |
| 6 | 63.33 | 63.33 | 33.33 | 93.33 | 0 -> 0 | 1 -> **0** | 67.4 -> **62.2** |

The stage-4 collapse is gone: +40.00 EX, and the `Tables:` loop failure mode is
eliminated outright - zero unparseable predictions and zero outputs reaching the
200-token decode cap, with output length back to the stage-3 level (61.6 vs 61.0).

Two secondary observations worth recording:

- **The old run's stage-5 "recovery" was never complete.** It still carried 3
  unparseable predictions and an inflated 72.4-token output length. Stages 5 and 6
  were partly spending capacity undoing stage-4 damage rather than learning the
  added latent roles. The retrain shows no such residue.
- **The n=30 probe saturates at stage 6** - both runs land on exactly 63.33. On the
  probe alone the retrain's advantage appears to wash out by the final stage. It
  does not; see below. This is the same probe-vs-full-eval optimism already
  documented for the original run.

### Full held-out latent eval (n=90, `--latent-roles 6 --latent-steps 2`)

```text
outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix_full_latent/results.md
```

| Run | n | EX | bird EX | spider EX | Avg tokens | Avg output tokens | Unparseable | Hit 200-cap | Latency/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage 6, original | 90 | 52.22 | 33.33 | 71.11 | 1638.0 | 67.5 | 4 | 5 | 2.212 s |
| stage 6, **retrained 4-6** | 90 | **57.78** | **37.78** | **77.78** | 1631.7 | **61.2** | **0** | **0** | **2.073 s** |

**+5.56 EX on the full held-out set**, improving on both datasets (bird +4.45,
spider +6.67), with output length down 9.3% and every unparseable prediction
eliminated. Latency drops 6.3% as a direct consequence of not generating
cap-length loops.

### Attribution caveat

Two knobs changed simultaneously, so this is not a clean ablation. The evidence
strongly favours the learning rate as the cause: `sql_ready_loss` logged around
1e-4 throughout the retrain, i.e. the auxiliary probe was as saturated as the main
objective and contributed almost no gradient. The claim supported by the data is
"lowering the late-stage learning rate fixes the stage-4 collapse", **not** "the
SQL-readiness probe fixes it". Separating them would need a third run at
`lr=1e-5` with `SQL_READY_WEIGHT=0`.

### Relationship to the GRPO result

Section "GRPO Eval After The Brevity Run" recorded GRPO lifting the *original*
stage-6 adapter from 52.22 to 58.89 EX (+6.67). The retrained stage-6 adapter
reaches **57.78 EX with no GRPO at all** - essentially the same place, from a
curriculum fix rather than a reinforcement stage. These are independent levers:
GRPO has not yet been run on top of the retrained checkpoint, and its +6.67 was
measured from a base that had a broken stage 4 in its history. Do not add the two
numbers; the correct next experiment is GRPO from the new stage-6 adapter.

### Artifacts

```text
outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix/latent_sft              # retrained stage-6 adapter, 57.78 EX
outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix/checkpoints/stage_{3,4,5,6}
outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix_eval_progress/stage_{4,5,6}
outputs/latent_sft_1000/chess_coconut_sql_direct_s4fix_full_latent            # the n=90 numbers above
```

The original run, its checkpoints and the GRPO adapter were left untouched
throughout - the retrain wrote only to the new `_s4fix` paths.
