# LatentSQL — recorded results

Everything here comes from the JSONL logs under `outputs/`, assembled by
[`scripts/06_report.py`](scripts/06_report.py). Nothing is estimated.

**Read [§6 What these numbers do not show](#6-what-these-numbers-do-not-show)
before quoting anything.** At n = 24 no two methods here are statistically
separable.

---

## 1. Setup

| | |
|---|---|
| Hardware | 1 × RTX 4090 (24 GB), 28-core CPU |
| Teacher | `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4 |
| Student | `Qwen/Qwen2.5-Coder-3B-Instruct`, QLoRA 4-bit, r=32 |
| Baselines | the **same 3B** as the student — a size-matched comparison |
| Teacher panel | DIN-SQL + DAIL-SQL + CHESS + CHASE-SQL, 2 debate rounds |
| Datasets | BIRD + Spider, `/mnt/win_d/data/sft_data_collections` |
| Train split | 81 questions (42 BIRD + 39 Spider) from the **train** splits |
| Eval split | 24 questions (12 BIRD + 12 Spider) from the **dev** splits |
| Leakage | zero qid overlap between train and eval; DAIL shots drawn from train only |

Three runs are recorded. **Run 1** predates the Spider loader fix and its
training corpus is BIRD-only. **Run 2** is the corrected run and is the one to
read for the headline comparison. **Run 3** (§5b) is a controlled ablation of
the dense reward terms on a 1/3 BIRD + 2/3 Spider corpus; its eval set is a
different draw, so its numbers are comparable within run 3 only.

---

## 2. Accuracy

### 2.1 Run 2 — our student vs. the four published systems at 3B

n = 24 (12 BIRD + 12 Spider). Same questions, schema rendering, executor and
scorer for every row. 95% Wilson intervals.

| Method | EX (%) | 95% CI | EM (%) | bird | spider | Tokens | Output tok | Latency | GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single agent (plain 3B) | 45.83 | 27.9–64.9 | 0.00 | 25.0 | 66.7 | 1 585 | 180 | 3.6 s | 2.25 GB |
| DIN-SQL | 54.17 | 35.1–72.1 | 8.33 | 33.3 | 75.0 | 3 186 | 321 | 6.3 s | 2.26 GB |
| **DAIL-SQL** | **62.50** | 42.7–78.8 | 16.67 | 41.7 | 83.3 | 4 697 | 350 | 7.0 s | 2.26 GB |
| CHESS | 41.67 | 24.5–61.2 | 8.33 | 16.7 | 66.7 | 7 253 | 1 079 | 26.4 s | 2.27 GB |
| CHASE-SQL | 50.00 | 31.4–68.6 | 8.33 | 33.3 | 66.7 | 6 744 | 653 | 12.9 s | 2.27 GB |
| Ours, SFT only | 41.67 | 24.5–61.2 | 4.17 | 25.0 | 58.3 | 2 414 | 993 | 34.1 s | 2.69 GB |
| **Ours, SFT + GRPO** | **50.00** | 31.4–68.6 | 4.17 | 25.0 | **75.0** | 2 404 | 982 | 34.4 s | 2.69 GB |

**On EM**: it is near-zero for everything because it penalises any aliasing or
formatting difference even when the query returns identical rows. BIRD dropped
EM for exactly this reason. **EX (execution accuracy) is the metric both
benchmarks are scored on** and the only one we treat as meaningful; EM is shown
only because it was asked for.

### 2.2 The one positive result: GRPO works when its gradient is alive

| | run 1 | run 2 |
|---|---:|---:|
| teacher EX | 28.4% | **63.0%** |
| traces kept | 19/81 (23.5%) | **46/81 (56.8%)** |
| SFT final loss | 0.32 | **0.15** |
| GRPO steps producing a gradient | 4/9 (44%) | **13/15 (87%)** |
| SFT-only student | 54.17% | 41.67% |
| **+ GRPO** | 54.17% (**Δ 0.0**) | 50.00% (**Δ +8.3**) |

**GRPO added +8.3 EX in run 2, entirely on Spider (58.3 → 75.0).** This is the
first evidence in the project that stage 2 does anything, and it appeared
exactly when the gradient stopped being degenerate. In run 1, with 44% of steps
dead, GRPO moved the weights by `||ΔW||/||W|| = 6.3e-04` and changed nothing.

Run 2's *absolute* number is still below run 1's because run 2's **SFT** is
worse (41.67%), despite 2.4× the traces and half the loss — it fits the
transcripts tightly and emits 993 output tokens against run 1's 653.

---

## 3. Cost

### 3.1 Inference, per question

| Compared with | its EX | its tokens | our tokens | our latency |
|---|---:|---:|---:|---:|
| Single agent | 45.83% | 1 585 | 152% | 963% |
| DIN-SQL | 54.17% | 3 186 | **75%** | 543% |
| DAIL-SQL | 62.50% | 4 697 | **51%** | 490% |
| CHESS | 41.67% | 7 253 | **33%** | 130% |
| CHASE-SQL | 50.00% | 6 744 | **36%** | 266% |
| **7B teacher panel** | — | 25 499 (27.3 calls) | **9.4%** (1 call) | — |

**Token savings are real; latency savings are not — yet.** Total tokens are
dominated by *prompt* tokens (the student reads one prompt where DIN-SQL reads
five), but latency is dominated by *output* tokens, generated serially. The
student emits 982 output tokens against DIN-SQL's 321, so it is ~5× slower
despite using 25% fewer total tokens. A student that had genuinely internalised
the debate would emit ~50 output tokens and be the fastest method here. Until
that is demonstrated, quoting the token ratio alone would be misleading.

### 3.2 Training, wall-clock and GPU

| Stage | Work | Time | GPU peak |
|---|---|---:|---:|
| Teacher | 81 questions, 27.3 calls each | 65.0 min | 8.82 GB |
| SFT | 46 traces, 34 steps, 267k tokens | 81 s | 9.36 GB |
| GRPO | 15 steps, 117k tokens generated, 249 SQL executions | 23.7 min | 6.37 GB |
| Eval | 7 methods × 24 questions | ~35 min | 2.69 GB |

Teacher cost splits sharply by benchmark:

| | n | EX | Consensus | Latency | Tokens | Calls |
|---|---:|---:|---:|---:|---:|---:|
| BIRD | 42 | 50.00% | 73.8% | 61.5 s | 29 552 | 30.6 |
| Spider | 39 | **76.92%** | 94.9% | 33.8 s | 21 135 | 23.7 |

Filtering kept **46/81 (56.8%)** — 22 dropped incorrect, 13 no-consensus, 0 for
any other reason. That is **~42 usable traces per GPU-hour**, up from 17 in run 1.

---

## 4. Where our method wins — worked examples

### 4.1 What internalisation (GRPO) fixed

Three questions the SFT-only student got wrong and GRPO corrected (net +2 after
one regression). These are the +8.3 points.

**(a) Missing the final join — BIRD, `superhero`**
> *"List down Ajax's superpowers."*

```sql
-- gold
SELECT T3.power_name FROM superhero T1
  JOIN hero_power T2 ON T1.id = T2.hero_id
  JOIN superpower T3 ON T2.power_id = T3.id
 WHERE T1.superhero_name = 'Ajax'

-- SFT only  ✗  selects two columns, leaks the hero name into the result
SELECT superhero.superhero_name, superpower.power_name FROM superhero ...

-- + GRPO    ✓
SELECT sp.power_name FROM superhero s
  JOIN hero_power hp ON s.id = hp.hero_id
  JOIN superpower sp ON hp.power_id = sp.id
 WHERE s.superhero_name = 'Ajax'
```
The SFT student projected an extra column. GRPO — whose reward executes the
query and compares rows — removed it.

**(b) Ignoring "in alphabetical order" — Spider, `tvshow`**
> *"List the title of all cartoons in alphabetical order."*

```sql
-- gold      SELECT Title FROM Cartoon ORDER BY title
-- SFT only ✗ SELECT Title FROM Cartoon
-- + GRPO   ✓ SELECT Title FROM Cartoon ORDER BY Title ASC
```
Row *order* is scored when the gold query has `ORDER BY`. Only an
execution-based reward can teach this; a string-similarity reward would rate the
two nearly identical.

**(c) Superlative without ordering — Spider, `employee_hire_evaluation`**
> *"Which employee received the most awards in evaluations?"*

```sql
-- SFT only ✗ ... GROUP BY e.Name ORDER BY COUNT(ev.Year_awarded)          -- ascending, no LIMIT
-- + GRPO   ✓ ... GROUP BY e.Name ORDER BY COUNT(ev.Year_awarded) DESC LIMIT 1
```
The classic "most/least" failure: right aggregation, wrong direction, no limit.

### 4.2 Where the student beat a plain 3B

Net +1 against `single` (won 3, lost 2).

**Two-hop join a single agent collapsed — BIRD, `superhero`**
```sql
-- single ✗  stops at hero_power, so power_name does not exist in scope
SELECT T2.power_name FROM superhero T1 JOIN hero_power T2 ON T1.id = T2.hero_id ...
-- ours   ✓  carries the join through to superpower
```

**Spurious join — Spider, `car_1`**
> *"What is the number of cars with over 6 cylinders?"*
```sql
-- single ✗  invents a join on mismatched keys, inflating the count
SELECT COUNT(*) FROM cars_data T1 JOIN car_names T2 ON T1.MakeId = T2.ModelId WHERE T1.Cylinders > 6
-- ours   ✓
SELECT COUNT(*) FROM cars_data WHERE Cylinders > 6
```
Notable because the debate teaches the student to *not* over-complicate — the
panel's execution feedback punishes joins that change row counts.

**Null-aware counting — BIRD, `formula_1`**
> *"From race no. 50 to 100, how many finishers have been disqualified?"*

Gold uses `SUM(IIF(time IS NOT NULL,...))` — "finishers" means a non-null time.
`single` ignored it; ours reproduced the `time IS NOT NULL` condition.

### 4.3 Honest counter-evidence

- The student solved **zero** questions that no other method solved.
- It lost 2 questions to `single` and 2 to DAIL-SQL that it should have won.
- In run 1 it was the only method to emit unparseable output (2 of 24).
- Of 24 questions, 6 were solved by every method and 7 by none — only **11
  discriminate at all**.

---

## 5. Diagnostics that explain the behaviour

### 5.1 GRPO groups die from both ends

`grpo_log_steps.jsonl` records `reward_std_mean` and `adv_nonzero_frac`. A group
whose rollouts all score alike yields zero advantage and **no gradient**:

| run 2 step | R_clip | adv_nonzero_frac | what happened |
|---:|---:|---:|---|
| 8 | 1.00 | **0.00** | every rollout correct *and* identically formatted |
| 9 | 1.00 | **0.00** | saturated on every term |
| 12 | 0.00 | 1.00 | every rollout wrong; rescued only by a weak R_fmt spread (0.05) |

Run 1 died of all-*wrong* groups; run 2 also hits all-*correct* ones. This
exposes a gap: **nothing in the reward distinguishes two correct rollouts of
different lengths** — both score `R_clip = 1` — so GRPO cannot learn "correct
*and* shorter", which is the whole internalisation objective.

### 5.2 The student does not reliably close the debate

| eval budget | EX | hit cap | reached `<\|consensus\|>` |
|---|---:|---:|---:|
| 1 024 tokens | 50.00% | 17/24 | 1/24 |
| 3 072 tokens | **50.00%** | 2/24 | 11/24 |

Identical accuracy. Truncation was a tempting explanation — tested and refuted
(questions with short traces are simply the easier ones). The real finding is
that even with ample room the student emits `<|consensus|>` on only 11 of 24
questions, so the fallback extractor is scraping a mid-debate query about half
the time. A **format-adherence** failure, not a budget one.

---

## 5b. Run 3 — does dense partial credit fix GRPO? (the reward ablation)

Corpus **1/3 BIRD + 2/3 Spider** (30/60 train, 8/16 eval). Teacher and SFT run
**once**; GRPO then runs **twice from the same SFT checkpoint**, differing only
in `w_atomic` / `w_memory`, so the reward is the single changed variable.
`outputs/pilot3/`.

Teacher: 81 questions, 64.2% EX (BIRD 50.0%, Spider 71.7%), **48/81 kept
(59.3%)**, 64 min, GPU peak 8.8 GB. Each GRPO arm: 24 steps, ~40 min, peak
6.4 GB — and for the first time in the project **the anneal ran to completion**
(`w_fmt` 0.96 → 0.05, clip 878 → 400).

### The mechanism worked exactly as designed — and exactly as limited

| group type | | ctrl (binary) | dense (atomic+memory) |
|---|---|---:|---:|
| **all-wrong** | count | 3 | 6 |
| | **dead steps** | **1** | **0** |
| | **mean `reward_std`** | **0.029** | **0.091** (**3.2x**) |
| all-correct | count | 5 | 7 |
| | dead steps | 2 | 3 |
| | mean `reward_std` | 0.052 | 0.054 (**1.0x**) |

**3.2x the gradient on failing groups, zero dead steps there; nothing whatsoever
on saturated groups.** That is precisely the mechanism: the dense terms score
only *wrong* rollouts, so they can spread a failing group and cannot spread a
saturated one. On step 1 — the only true paired comparison, before the arms
diverged — dense turned an *exactly dead* gradient (`std = 0.0000`,
`adv_nonzero_frac = 0`) into a live one (`std = 0.0068`,
`adv_nonzero_frac = 1`).

### Accuracy

| checkpoint | EX | bird | spider | out tok | latency |
|---|---:|---:|---:|---:|---:|
| **SFT-only** (shared start) | **66.67%** | 37.5 | 81.2 | 1 235 | 39.7 s |
| GRPO ctrl (binary) | 58.33% | 25.0 | 75.0 | 1 275 | 41.6 s |
| GRPO dense (atomic+memory) | 62.50% | 37.5 | 75.0 | 1 245 | 42.1 s |

Baselines on the same 24 questions: DAIL-SQL 70.83%, DIN-SQL / CHESS /
CHASE-SQL 62.50%, single agent 50.00%.

**Dense beats binary by +4.17 points**, recovering all of the BIRD loss
(25.0 → 37.5). **But both GRPO arms are worse than the SFT checkpoint they
started from** (−8.33 and −4.17). Dense partial credit reduced the damage GRPO
did; it did not make GRPO helpful.

### Why, and what is actually missing

This run had **7 all-correct groups against 6 all-wrong** — saturation was the
*dominant* failure mode, and dense rewards are structurally incapable of
addressing it. Generated length confirms nothing was internalised: 1 024 → 922
across 24 fully-annealed steps, essentially flat.

The missing piece is the one named in [METHOD.md §8.1b](METHOD.md): **no term
distinguishes a 900-token correct rollout from a 300-token one.** Both score
`R_clip = 1`, so on more than half the steps GRPO has nothing to optimise
toward. Until a brevity term is added among correct rollouts, more GRPO steps
cannot help.

### Caveats

- n = 24; **+4.17 points is a single question**. Directional evidence, not a
  result.
- The arms diverged after step 1 (the first update changes the policy), so only
  step 1 is a matched comparison; everything else is an aggregate over
  different sampled rollouts.
- Run 3's eval set is a different draw from runs 1-2 (8/16 rather than 12/12),
  so numbers are comparable *within* run 3 only. CHESS moved 41.67% → 62.50%
  between draws, which is a fair measure of how much n = 24 swings.

---

## 6. What these numbers do not show

1. **Nothing is statistically separable.** n = 24 gives ~±19-point intervals;
   every method's interval overlaps every other's.
2. **GRPO ran for 15 steps.** The schedule was cut by a timeout at 15 of 23, so
   `w_fmt` stopped at 0.35 instead of reaching 0.05 — the run executed the
   "verbalise more" half of an annealed curriculum and was killed before the
   "stop writing it down" half. A real attempt is ~13.5 h at 97 s/step.
3. **The debate is not internalised.** The student still emits ~982 output
   tokens. The headline claim — same accuracy at a fraction of the *output* —
   remains untested.
4. **The teacher is weak.** At 63% EX it is far below the GPT-4-class systems
   the four papers assume, and the student cannot exceed what it demonstrated.
5. **The corpus is filtered to easy questions.** `require_correct` discards
   every question the teacher failed, so the student never sees a worked trace
   for the hard cases debate should help most.

See [METHOD.md §8](METHOD.md) for the full list of drawbacks and their
mitigations.

---

## 7. Reproducing

```bash
bash scripts/run_pilot.sh                 # run 2 (configs/pilot_mix.yaml)
python scripts/06_report.py --run outputs/pilot2
```

Per-sample logs: `outputs/pilot2/traces/teacher_samples.jsonl`,
`outputs/pilot2/{sft,grpo}/*_log_steps.jsonl`,
`outputs/pilot2/eval*/eval_samples.jsonl`, and every predicted query in
`outputs/pilot2/eval*/preds_<method>.jsonl`.
