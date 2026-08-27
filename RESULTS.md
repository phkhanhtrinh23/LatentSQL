# LatentSQL — recorded results

Everything here comes from the JSONL logs under `outputs/`, assembled by [`scripts/06_report.py`](scripts/06_report.py). Nothing is estimated.

**Read [§6 What these numbers do not show](#6-what-these-numbers-do-not-show) before quoting anything.**

---

# 0. What success looks like — the two tests that matter

Everything in this project reduces to two questions with two *different* reference points. Conflating them is the single easiest way to misread these results, so they are stated first.

## 0.1 Test A — did internalisation actually happen? *(vs. the SFT student)*

> **EX(GRPO) ≈ EX(SFT), while output tokens collapse.**

The reference is **the SFT checkpoint that GRPO started from** — nothing else. This is IMAD's mechanism test, and it is a test of *compression*, not of accuracy.

**GRPO is not supposed to beat SFT.** Stage 2 does not add capability; it removes the need to write the debate down. Stage 1 has already installed whatever the model knows. So:

| outcome | verdict |
|---|---|
| EX holds, output tokens **collapse** | **success — the debate was internalised** |
| EX holds, tokens **flat** | failure — nothing was internalised (**every run so far**) |
| EX **falls**, tokens collapse | failure — compression destroyed the capability |
| EX **rises** | a bonus, not the claim |

The measurable is **output tokens**, not total tokens. Total tokens are dominated by the *prompt*; latency and the internalisation claim both track what the model *generates*.

**Status: never achieved.** Across runs 1-3 with fully annealed schedules, generated length went **1 024 → 922**. Flat. The SFT student emits ~1 242-1 955 output tokens and still writes the entire four-agent transcript, tag by tag (100% of generations emit `<|schema_link|>`, `<|round 1|>`, `<|agent din_sql|>`).

## 0.2 Test B — is the thesis true? *(vs. the explicit MAS at the same size)*

> **EX(student, 3B) ≥ EX(CHESS, 3B) at a fraction of the output tokens.**

The reference is the **explicit multi-agent system running on the same 3B backbone** — *not* the 7B teacher. Beating a 3B baseline with a 3B student is a claim about the **method**; beating a 7B one would be a claim about **model size**, which is a different and much less interesting statement.

Measured target, n = 90 (§2.3):

| reference | EX | output tokens |
|---|---:|---:|
| **CHESS @ 3B — the bar to clear** | **48.89%** | **1 009** |
| ours, SFT-only @ 3B | 52.22% | 1 242 |

So the SFT student already clears CHESS on accuracy (+3.3) — but **loses on output tokens** (1 242 vs 1 009), which is the half of the claim that matters. The thesis needs *both*.

## 0.3 What is explicitly **not** the target

**GRPO > the 7B teacher.** A 3B student does not beat the 7B system that taught it; the teacher is the ceiling, not the bar. Teacher EX was 64.2% (run 3). Any framing that compares our 3B student to the 7B teacher and calls the gap a failure has misunderstood the experiment.

**GRPO > SFT.** See 0.1 — stage 2 trades tokens for nothing, ideally. An accuracy gain there is welcome noise, not evidence.

## 0.4 Scoreboard

| | test | reference | target | status |
|---|---|---|---|---|
| **A** | internalisation | SFT student | EX holds, **output tokens collapse** | ❌ tokens never moved (1 024 → 922) |
| **B** | the thesis | CHESS @ 3B (48.89%) | **≥ 48.89% at fewer output tokens** | ⚠️ accuracy yes (52.22%), tokens no (1 242 vs 1 009) |

**Neither test has been passed.** Test A has never moved; test B is half-passed on the accuracy axis and failing on the cost axis — and the cost axis is the entire contribution.

### Why A has never passed — three causes, all now fixed

1. **The clip started below the trace length.** IMAD anneals `l` 2000 → 500, starting *above* the natural output so early rollouts are unpenalised and pressure builds. We ran `l` **900 → 400** against a ~1 950-token trace, so the clip bit from step 1 and the curriculum ran backwards.
2. **`max_new_tokens` (1 024) was below `l_start`.** Rollouts truncated at the cap, so on **16 of 24 steps every rollout had identical length** — nothing to reward for being shorter.
3. **24 GRPO steps.** IMAD-scale internalisation is hundreds.

The CHESS-only run now in flight uses `l` **2 400 → 500**, `max_new_tokens` **2 560**, and **~192 GRPO steps** (~5.2 h) — the first configuration in this project that can actually test Test A.

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

Three runs are recorded. **Run 1** predates the Spider loader fix and its training corpus is BIRD-only. **Run 2** is the corrected run and is the one to read for the headline comparison. **Run 3** (§5b) is a controlled ablation of the dense reward terms on a 1/3 BIRD + 2/3 Spider corpus; its eval set is a different draw, so its numbers are comparable within run 3 only.

---

## 2. Accuracy

### 2.1 Run 2 — our student vs. the four published systems at 3B

n = 24 (12 BIRD + 12 Spider). Same questions, schema rendering, executor and scorer for every row. 95% Wilson intervals.

| Method | EX (%) | 95% CI | EM (%) | bird | spider | Tokens | Output tok | Latency | GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single agent (plain 3B) | 45.83 | 27.9–64.9 | 0.00 | 25.0 | 66.7 | 1 585 | 180 | 3.6 s | 2.25 GB |
| DIN-SQL | 54.17 | 35.1–72.1 | 8.33 | 33.3 | 75.0 | 3 186 | 321 | 6.3 s | 2.26 GB |
| **DAIL-SQL** | **62.50** | 42.7–78.8 | 16.67 | 41.7 | 83.3 | 4 697 | 350 | 7.0 s | 2.26 GB |
| CHESS | 41.67 | 24.5–61.2 | 8.33 | 16.7 | 66.7 | 7 253 | 1 079 | 26.4 s | 2.27 GB |
| CHASE-SQL | 50.00 | 31.4–68.6 | 8.33 | 33.3 | 66.7 | 6 744 | 653 | 12.9 s | 2.27 GB |
| Ours, SFT only | 41.67 | 24.5–61.2 | 4.17 | 25.0 | 58.3 | 2 414 | 993 | 34.1 s | 2.69 GB |
| **Ours, SFT + GRPO** | **50.00** | 31.4–68.6 | 4.17 | 25.0 | **75.0** | 2 404 | 982 | 34.4 s | 2.69 GB |

**On EM**: it is near-zero for everything because it penalises any aliasing or formatting difference even when the query returns identical rows. BIRD dropped EM for exactly this reason. **EX (execution accuracy) is the metric both benchmarks are scored on** and the only one we treat as meaningful; EM is shown only because it was asked for.

### 2.2 The reference SFT number

**The SFT student scores 41.67% EX** (run 2, 12 BIRD + 12 Spider). That is the figure used as the shared starting point throughout this document, and the number the GRPO variants in §5b are judged against.

Upstream health improved sharply between runs, and this is what governs whether stage 2 can learn anything at all:

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| teacher EX | 28.4% | 63.0% | 64.2% |
| traces kept | 19/81 (23.5%) | 46/81 (56.8%) | **48/81 (59.3%)** |
| SFT final loss | 0.32 | 0.15 | 0.15 |
| GRPO steps producing a gradient | 4/9 (44%) | 13/15 (87%) | **21/24 (88%)** |
| anneal completed | no | no (cut 15/23) | **yes** |

In run 1, with 44% of steps dead, GRPO moved the weights by `||ΔW||/||W|| = 6.3e-04` and changed nothing measurable. From run 2 onward the gradient is alive on ~88% of steps, which is the precondition for §5b's comparison to mean anything.

**On accuracy deltas from GRPO, do not over-read them.** Run 2 measured SFT 41.67% → binary-GRPO 50.00% (+8.33); run 3 measured SFT 66.67% → binary-GRPO 58.33% (−8.33). Both are **two questions**, in opposite directions, with exact McNemar p = 0.625 and p = 0.500. Binary GRPO has **no detectable effect on accuracy at this scale**. What *is* measurable is the per-step gradient behaviour in §5b, which has 24 observations per arm rather than 24 questions.

### 2.3 The n = 90 evaluation — larger sample, real intervals

Runs 1-3 evaluated on n = 24 (±19-point intervals, nothing separable). This is the same protocol on **n = 90** (30 BIRD + 60 Spider, dev splits), 95% Wilson intervals, all methods on the same `Qwen2.5-Coder-3B-Instruct`.

| Method | EX (%) | 95% CI | bird | spider | Total tok | **Output tok** | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DAIL-SQL** | **60.00** | 49.7-69.5 | 30.0 | 75.0 | 4 369 | 326 | 6.0 s |
| CHASE-SQL | 57.78 | 47.5-67.5 | 26.7 | 73.3 | 5 817 | 599 | 13.1 s |
| **ours, SFT-only** | **52.22** | 42.0-62.2 | 23.3 | 66.7 | **2 561** | **1 242** | 40.8 s |
| **CHESS  ← Test B bar** | **48.89** | 38.8-59.0 | 30.0 | 58.3 | 6 622 | **1 009** | 20.2 s |
| DIN-SQL | 45.56 | 35.7-55.8 | 20.0 | 58.3 | 3 026 | 333 | 6.3 s |
| Single agent | 45.56 | 35.7-55.8 | 20.0 | 58.3 | 1 476 | 172 | 3.1 s |

Three things this larger sample establishes that n = 24 could not:

1. **Our student clears CHESS on accuracy but not on output tokens** — the exact split described in §0.2. It uses the **lowest total tokens of any multi-call method** (2 561) and the **highest output tokens on the table** (1 242). Prompt-side cost won, generation-side cost lost.
2. **DIN-SQL buys nothing over a plain single-agent prompt** — identical 45.56% for 2x the tokens. 74 of 90 predictions differ; the errors cancel exactly.
3. **DAIL-SQL leads at the second-lowest token cost.** CHESS spends 1.5x DAIL's tokens for 11 points less accuracy.

### 2.4 The teacher panel does not earn its cost

Measured on run 3's 81 traces: "at least one panellist correct" is the ceiling on trace yield, and cost is LLM calls per question.

| panel | >=1 correct | ~calls |
|---|---:|---:|
| **DAIL-SQL alone** | **81.5%** | **4** |
| CHASE-SQL alone | 69.1% | 11 |
| CHESS alone | 59.3% | 8 |
| DAIL + any one other | 82.7% | 9-15 |
| all four | 82.7% | 28 |

**The four-system panel buys +1.2 points over DAIL-SQL alone for 7x the compute** — one question in 81. The panel is also largely redundant: four panellists produce four *distinct* queries only 6% of the time, and two or fewer 37% of the time.

This is why the current run uses **one teacher system, not four**.

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

**Token savings are real; latency savings are not — yet.** Total tokens are dominated by *prompt* tokens (the student reads one prompt where DIN-SQL reads five), but latency is dominated by *output* tokens, generated serially. The student emits 982 output tokens against DIN-SQL's 321, so it is ~5× slower despite using 25% fewer total tokens. A student that had genuinely internalised the debate would emit ~50 output tokens and be the fastest method here. Until that is demonstrated, quoting the token ratio alone would be misleading.

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

Filtering kept **46/81 (56.8%)** — 22 dropped incorrect, 13 no-consensus, 0 for any other reason. That is **~42 usable traces per GPU-hour**, up from 17 in run 1.

---

## 4. Where our method wins — worked examples

### 4.1 What internalisation (GRPO) fixed

Three questions the SFT-only student got wrong and GRPO corrected (net +2 after one regression). These are the +8.3 points.

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
The SFT student projected an extra column. GRPO — whose reward executes the query and compares rows — removed it.

**(b) Ignoring "in alphabetical order" — Spider, `tvshow`**
> *"List the title of all cartoons in alphabetical order."*

```sql
-- gold      SELECT Title FROM Cartoon ORDER BY title
-- SFT only ✗ SELECT Title FROM Cartoon
-- + GRPO   ✓ SELECT Title FROM Cartoon ORDER BY Title ASC
```
Row *order* is scored when the gold query has `ORDER BY`. Only an execution-based reward can teach this; a string-similarity reward would rate the two nearly identical.

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
Notable because the debate teaches the student to *not* over-complicate — the panel's execution feedback punishes joins that change row counts.

**Null-aware counting — BIRD, `formula_1`**
> *"From race no. 50 to 100, how many finishers have been disqualified?"*

Gold uses `SUM(IIF(time IS NOT NULL,...))` — "finishers" means a non-null time. `single` ignored it; ours reproduced the `time IS NOT NULL` condition.

### 4.3 Honest counter-evidence

- The student solved **zero** questions that no other method solved.
- It lost 2 questions to `single` and 2 to DAIL-SQL that it should have won.
- In run 1 it was the only method to emit unparseable output (2 of 24).
- Of 24 questions, 6 were solved by every method and 7 by none — only **11 discriminate at all**.

---

## 5. Diagnostics that explain the behaviour

### 5.1 GRPO groups die from both ends

`grpo_log_steps.jsonl` records `reward_std_mean` and `adv_nonzero_frac`. A group whose rollouts all score alike yields zero advantage and **no gradient**:

| run 2 step | R_clip | adv_nonzero_frac | what happened |
|---:|---:|---:|---|
| 8 | 1.00 | **0.00** | every rollout correct *and* identically formatted |
| 9 | 1.00 | **0.00** | saturated on every term |
| 12 | 0.00 | 1.00 | every rollout wrong; rescued only by a weak R_fmt spread (0.05) |

Run 1 died of all-*wrong* groups; run 2 also hits all-*correct* ones. This exposes a gap: **nothing in the reward distinguishes two correct rollouts of different lengths** — both score `R_clip = 1` — so GRPO cannot learn "correct *and* shorter", which is the whole internalisation objective.

### 5.2 The student does not reliably close the debate

| eval budget | EX | hit cap | reached `<\|consensus\|>` |
|---|---:|---:|---:|
| 1 024 tokens | 50.00% | 17/24 | 1/24 |
| 3 072 tokens | **50.00%** | 2/24 | 11/24 |

Identical accuracy. Truncation was a tempting explanation — tested and refuted (questions with short traces are simply the easier ones). The real finding is that even with ample room the student emits `<|consensus|>` on only 11 of 24 questions, so the fallback extractor is scraping a mid-debate query about half the time. A **format-adherence** failure, not a budget one.

---

## 5b. Run 3 — does dense partial credit fix GRPO? (the reward ablation)

Corpus **1/3 BIRD + 2/3 Spider** (30/60 train, 8/16 eval). Teacher and SFT run **once**; GRPO then runs **twice from the same SFT checkpoint**, differing only in `w_atomic` / `w_memory`, so the reward is the single changed variable. `outputs/pilot3/`.

Teacher: 81 questions, 64.2% EX (BIRD 50.0%, Spider 71.7%), **48/81 kept (59.3%)**, 64 min, GPU peak 8.8 GB. Each GRPO arm: 24 steps, ~40 min, peak 6.4 GB — and for the first time in the project **the anneal ran to completion** (`w_fmt` 0.96 → 0.05, clip 878 → 400).

### The mechanism worked exactly as designed — and exactly as limited

| group type | | ctrl (binary) | dense (atomic+memory) |
|---|---|---:|---:|
| **all-wrong** | count | 3 | 6 |
| | **dead steps** | **1** | **0** |
| | **mean `reward_std`** | **0.029** | **0.091** (**3.2x**) |
| all-correct | count | 5 | 7 |
| | dead steps | 2 | 3 |
| | mean `reward_std` | 0.052 | 0.054 (**1.0x**) |

**3.2x the gradient on failing groups, zero dead steps there; nothing whatsoever on saturated groups.** That is precisely the mechanism: the dense terms score only *wrong* rollouts, so they can spread a failing group and cannot spread a saturated one. On step 1 — the only true paired comparison, before the arms diverged — dense turned an *exactly dead* gradient (`std = 0.0000`, `adv_nonzero_frac = 0`) into a live one (`std = 0.0068`, `adv_nonzero_frac = 1`).

### Accuracy — dense beats binary

| checkpoint | EX | bird | spider | out tok | latency |
|---|---:|---:|---:|---:|---:|
| GRPO **binary** reward (ctrl) | 58.33% | 25.0 | 75.0 | 1 275 | 41.6 s |
| GRPO **dense** reward (atomic + memory) | **62.50%** | **37.5** | 75.0 | 1 245 | 42.1 s |
| *run-3 SFT starting point* | *66.67%* | *37.5* | *81.2* | *1 235* | *39.7 s* |

Baselines on the same 24 questions: DAIL-SQL 70.83%, DIN-SQL / CHESS / CHASE-SQL 62.50%, single agent 50.00%.

**The dense reward beats the binary reward by +4.17 points, and the gain is exactly where the mechanism predicts it: BIRD, 25.0 → 37.5.** BIRD is the harder benchmark, so it is where all-wrong rollout groups concentrate — and all-wrong groups are precisely what the dense terms rescue. Binary reward gives every rollout in such a group the same score, so the step teaches nothing; dense partial credit ranks them by how *close* they came, and the step teaches something. The per-step numbers above (3.2x gradient, zero dead steps) are the direct evidence for that pathway; the BIRD recovery is its downstream effect.

Note the headline SFT figure used in this document is **41.67%** (§2.2, run 2). Run 3's own SFT checkpoint scored 66.67% on run 3's different eval draw, shown italicised above only as that run's internal starting point — the two are not comparable, and the ablation being made here is **binary vs dense**, both trained from the *same* checkpoint on the *same* data.

### What dense partial credit still cannot do

Dense rewards fix *wrong* rollouts. They cannot touch **saturated** groups — where every rollout is already correct — because a correct rollout is fully credited and earns nothing extra for resembling anything. This run had **7 all-correct groups against 6 all-wrong**, so saturation was the dominant failure mode by count, and the table above shows the dense terms moved it not at all (1.0x).

Generated length confirms nothing was internalised: 1 024 → 922 across 24 fully-annealed steps, essentially flat.

The missing piece is the one named in [METHOD.md §8.1b](METHOD.md): **no term distinguishes a 900-token correct rollout from a 300-token one.** Both score `R_clip = 1`, so on more than half the steps GRPO has nothing to optimise toward. Until a brevity term is added among correct rollouts, more GRPO steps cannot help.

### Caveats

- n = 24, so **+4.17 points is a single question**. The accuracy ordering is consistent with the mechanism but is not on its own statistically powered; the load-bearing evidence is the per-step gradient measurement (24 observations per arm), not the eval delta.
- The arms diverged after step 1 (the first update changes the policy), so only step 1 is a matched comparison; everything else is an aggregate over different sampled rollouts.
- Run 3's eval set is a different draw from runs 1-2 (8/16 rather than 12/12), so numbers are comparable *within* run 3 only. CHESS moved 41.67% → 62.50% between draws, which is a fair measure of how much n = 24 swings.

---

## 5c. Run 4 (live, in progress) — the four-arm factorial after the Option A fix

**Status as of 2026-08-27, arm A complete, arm B in progress: arm A's GRPO stage ran to its 2-hour timeout at 35 of the planned 252 steps and has now handed off to arm B (DPO-initialized). Nothing below is final for the factorial as a whole — no arm has reached eval yet — but arm A's window is now a closed, complete observation, not a mid-flight one, and it is a clean negative result worth recording precisely.**

### The good news: Option A held under real load, for the full 2-hour window

Every previous attempt at GRPO through the latent debate died from OOM within the first few steps (see the saga in §6 and the commit history). Arm A's full 2-hour window ran clean: **35 consecutive optimizer steps, ~7,144 s elapsed, GPU peak 15.52 GB**, still inside the 17.63 GB cap (`LATENTSQL_GPU_FRACTION=0.75`, i.e. 18/24 GB) — the peak rose from ~9.3 GB to ~15.5 GB exactly at the one step that produced a real gradient (step 33, see below), consistent with the backward pass only fully engaging when there is a non-zero loss to propagate. The per-role gradient checkpointing fix (`d4b1b18`) held for the entire capped window, not just a handful of steps. Speculative decoding also ran cleanly start to finish: mean accepted length 2.2-2.8 of a max-3 draft, acceptance rate 40-56%, ~180-210 s per step.

### Arm A's complete result: 34 of 35 steps were exactly dead, and execution correctness never once left zero

**34 of the 35 logged steps have `adv_nonzero_frac: 0.0` and `reward_std_mean: 0.0`.** By the zero-variance proof in [PROOFS.md](PROOFS.md) §1.4, that means the GRPO gradient was the **exact zero vector on 34 of 35 steps** — not small, exactly zero. The single exception, step 33, had `reward_std_mean: 0.0433` and `adv_nonzero_frac: 1.0` — but even there, `r_clip_mean` (execution correctness) was still `0.0`; the entire live signal on that one step came from `r_fmt_mean` (format) spread among the 4 rollouts, not from any rollout actually producing correct SQL. **Across all 35 steps × 4 rollouts = 140 sampled completions, `r_clip_mean` never once left `0.0`.** Not one sampled SQL query, at any point in arm A's GRPO stage, executed to match the gold answer. `r_fmt_mean` fluctuates (0.0-0.19) throughout, so the only thing GRPO ever had to learn from in this window was formatting, never correctness.

**Root cause, verified against the actual training data, not just suspected:** the SFT completion target is the full tagged text transcript — `<|schema_link|>...<|round 1|>...<|agent din_sql|>...` — generated from the *plain* prompt with no latent prefix at all (confirmed by inspecting `outputs/arm_a/traces/sft.jsonl` directly). GRPO's latent sampler (`_sample_group_latent` in `grpo.py`) does something the SFT/DPO stages never once demonstrated: it runs the prompt through 12 latent steps per role across 9 roles (`build_prefix`), then asks the model to emit **only a bare SQL query** continuing from that latent-primed KV cache — no tags, no debate scaffolding, a completion shape the checkpoint has never seen conditioned this way. GRPO is effectively evaluating the student zero-shot on a conditioning distribution SFT and DPO never trained it on, which is a plausible and sufficient explanation for 100% execution failure over the first 85 minutes: there has been no stage in this pipeline yet that teaches the model to produce a good SQL completion *specifically after a latent-only prefix*.

This does not mean the run is broken or should be killed — GRPO's own reward signal is the only mechanism that could ever fix this, given enough steps and at least occasional non-zero-variance groups to learn from — but it means **the eventual four-arm accuracy numbers need to be read against this starting condition**, not assumed to reflect the method working as intended. If `adv_nonzero_frac` stays at 0.0 for the bulk of each arm's capped 2-hour window, the resulting "GRPO" checkpoints will be very close to their DPO/SFT starting points by construction, and any accuracy delta will not be attributable to GRPO having learned anything.

### What's mechanically confirmed working, independent of the above

| component | status | evidence |
|---|---|---|
| Option A checkpointing at scale | ✅ | 35/35 steps, no OOM, full 2h window, GPU peak 15.52 GB (cap 17.63 GB) |
| Speculative decoding, live | ✅ | acceptance rate 40-56%, mean accepted length 2.2-2.8/3, all 35 steps |
| VRAM safety cap | ✅ | never exceeded 15.52 GB of the 17.63 GB (18/24 GB) limit |
| GRPO learning signal (arm A, complete) | ❌ | 34/35 steps exactly dead; `r_clip_mean = 0.0` on all 35 — 0% execution correctness the entire window |

Arm A's resulting `outputs/arm_a/grpo` checkpoint is, by the proof in PROOFS.md §1.4, essentially the SFT checkpoint plus one real update out of 35 — that single update was driven by formatting, not correctness.

### Arm A's eval is now in: 0.00% EX on all 90 held-out questions

All four arms eventually produced a GRPO adapter (each ran its full 2-hour timeout window), and eval had started when the run was terminated (below). **Arm A scored EX = 0.00% on the full n = 90 set (0% BIRD, 0% Spider)** — not a single held-out question was answered correctly. Every one of the 90 generations also hit the `max_new_tokens = 2048` cap.

### Confirmed root cause, and the run has been terminated

The four-arm run was killed and the pipeline was inspected directly rather than left to keep burning GPU hours on a mechanism now understood to be broken. Inspecting arm A's 90 raw completions (`outputs/arm_a/eval/preds_latent.jsonl`) found **only 15 unique output strings across 90 completely different questions and schemas**, and their content is unrelated to any SQL task (multiple-choice quizzes about ages, sums of odd numbers, pedestrians crossing a street). This rules out "45/90 wrong but plausible SQL" and points to something more specific: the model cannot recover per-question information at all once decoding starts.

Code-level check: `LatentReasoner.build_prefix` (what GRPO trains through) and `LatentReasoner.generate` (what eval calls) were compared line by line and found consistent with each other — both run the identical 9-role loop, both insert `<|consensus|>` and begin decoding immediately after it. There is no GRPO-vs-eval code mismatch. The mechanism is instead this: after 96 latent steps (12 per role × 8 debate roles, never decoded to text) the KV-cache holds a trajectory of raw hidden-state feedback that SFT never produced or shaped, because SFT's only training target is the *written* debate from a *plain* prompt (`outputs/arm_a/traces/sft.jsonl`, confirmed directly). The attention patterns that would normally let the model read the schema/question back out of the preceding context have nothing coherent to read once that context is 96 rounds of never-trained continuous vectors, so decoding collapses onto a small set of input-independent, memorized attractor completions rather than failing gracefully per-question.

**Why IMAD's own paper does not hit this:** it never has a latent step — SFT and GRPO both operate in one modality end to end (plain prompt in, written debate text out), so GRPO only ever refines a skill SFT already demonstrated in the identical input/output shape.

**Why LatentMAS does not hit this:** it is training-free by construction, applying latent hidden-state passing to an already-strong frozen pretrained model that was never narrowly fine-tuned on a conflicting flat-text task.

**Why runs 1-3 and run 3's dense-reward ablation (§2, §5b) did not hit this:** confirmed by inspection — `configs/pilot_mix.yaml` and `configs/pilot_mix3_ctrl.yaml` never set `latent_rollouts: true`. GRPO ran in plain text mode there, generating from the exact conditioning SFT trained on. `latent_rollouts: true` was never turned on for a real end-to-end run before the four-arm factorial — this session's combined IMAD+LatentMAS design is the first time this exact regime was exercised, and it is the regime that fails.

**Decision: the IMAD+LatentMAS combination is shelved.** The project reverts to pure text-mode IMAD (matched SFT/GRPO conditioning, the regime that produced every real result in this document), now with speculative decoding used to accelerate rollout generation so GRPO can reach the "hundreds of steps" scale IMAD's own paper needs (§0.4) instead of the 15-35 steps every run so far has been capped at.

---

## 5d. Run 5 — pure text-mode IMAD + speculative decoding (post-pivot, complete)

The first full SFT→GRPO→eval cycle since abandoning the IMAD+LatentMAS combination (§5c). Config `configs/imad_boost.yaml`: text-mode GRPO (`latent_rollouts: false`, matched to SFT's actual training regime - the fix for §5c's root cause), speculative decoding (k=3), the 4-system panel with `rescue_correct_agent: true` (63 kept traces, not 48).

**GRPO**: timed out at its 10h cap after **62 of 252 planned steps** - **34% dead** (21/62), **66% live**, by a wide margin the healthiest gradient behaviour recorded in this project (every prior text-mode run was 44-56% dead; every latent-mode run was 97-100% dead). The anneal only got ~25% of the way through its schedule: `w_fmt` 1.0 → 0.765 (target 0.05), `length_limit` 3600 → 2834 (target 500).

**Eval, n = 90 (30 BIRD + 60 Spider)**: `student mix n=90 EX=53.33% VES=- tok=2535.8 out=1216.3 39.50s [bird: 23.33% (n=30) spider: 68.33% (n=60)]`.

### Reading this against the two tests that matter (§0)

| | EX | output tokens | verdict |
|---|---:|---:|---|
| this run (SFT+GRPO, 62/252 steps) | **53.33%** | 1216.3 | |
| **Test A** reference - SFT-only (run 2, different corpus) | 52.22% | 1242 | tokens essentially flat (1242 → 1216, -2%) - **internalisation still not demonstrated**, consistent with the anneal being only 25% complete |
| **Test B** bar - CHESS @ 3B | 48.89% | 1009 | EX clears the bar (+4.44) but **loses on tokens** (1216 vs 1009) - the exact same half-passed pattern every prior run has shown |
| DAIL-SQL (best baseline) | 60.00% | 326 | still well behind on both axes |

**This is the first non-catastrophic result since the pivot** - EX is a real, plausible number (not 0%, not saturated), and it lands almost exactly where the old SFT-only reference did, which is the expected outcome for a GRPO run that only completed a quarter of its compression curriculum: capability held, compression barely started. Neither test is newly passed by this run; what changed is that the pipeline is demonstrably healthy again (66% live gradient, real EX, real per-example variance) rather than broken (§5c's 0% collapse).

### What would change the picture

The anneal needs roughly 4x the wall-clock already spent to reach its target (`length_limit` → 500, `w_fmt` → 0.05) - the point at which token collapse, if it happens at all, would actually show up.

### The latent-SFT checkpoint, evaluated: a different failure, not a fix

A "latent SFT" checkpoint was built in parallel with this run (57/63 traces, `outputs/latent_sft_probe/latent_sft`, one record per fresh subprocess chained together - see the worked-example discussion in this session) as a candidate fix for §5c's 0% collapse. Evaluated on the same n = 90 latent-mode protocol: **still EX = 0.00%** (0% BIRD, 0% Spider) - but the *shape* of the failure changed completely, and instructively.

`outputs/arm_a/eval/preds_latent.jsonl` (no worked example, §5c): 15 unique outputs across 90 questions, every completion hit the 2048-token cap, content unrelated to any SQL task (quiz questions about ages, pedestrians). `outputs/latent_sft_probe/eval/preds_latent.jsonl` (with the worked example): **average completion length dropped to 38.9 tokens** (well under the 400-token cap - the model now reliably terminates cleanly at `<|consensus|>...```sql...```", instead of degenerating into an endless loop), average latency 4.82s (vs 39.5s for the text-mode Run 5 checkpoint). Format-wise, the worked example clearly worked: the model now knows *how* to close out a latent-primed generation.

But **81 of 90 questions produced the exact same output** (`SELECT DISTINCT T1.name FROM station AS T1 JOIN station AS T2 ON T1.id = T2.id - 1 WHERE T1.name != T2.name`), and the remaining 9 a one-token variant of it - **2 unique completions across 90 completely different questions and schemas.** This is not the OOD collapse from §5c (which produced unrelated garbage); this is textbook catastrophic overfitting. The mechanism is visible in how the checkpoint was built: 57 sequential single-example gradient updates, `grad_accum=1`, `lr=1e-5`, no batching to average out any one example's influence - so whichever example(s) landed late in the chain with a strong gradient dominated the weights, and the model memorized one answer rather than learning to condition on the actual per-question input.

**Net read:** the worked-example fix solved the problem it was built to solve - the model no longer produces incoherent, never-terminating output after a latent-primed prefix - but the *training procedure* used to build this particular checkpoint (57 uncontrolled single-example updates) was too aggressive and overfit hard. The hypothesised fix was mechanical: batch multiple records per optimizer step so gradients average across examples the way ordinary SFT does. That hypothesis was tested directly below, and it was wrong (or at least incomplete).

### Testing the batching fix: three doses, three outcomes, the hypothesis falsified

Same `train_latent_sft` function, same corpus, `grad_accum=4` throughout - only the number of real (post-averaging) optimizer updates differs.

| run | updates | batched? | unique outputs / n | termination | EX |
|---|---:|---|---:|---|---:|
| §above (baseline) | 57 | no (one record per process, immediate update) | 2 / 90 | clean (38.9 tok avg) | 0% |
| under-dosed probe | 4 | yes | 11 / 20 | **broken** - hit the 400-tok cap every time, off-topic content | 0% |
| **scaled probe** | **~45** | **yes** | **1 / 90** | clean (45 tok avg) | 0% |

4 batched updates was too little to learn anything robust - the model reverted to the exact never-terminating garbage failure from the original 0% collapse, just with more superficial answer diversity (11/20 unique, but none coherent). ~45 batched updates (3 epochs over the full 57-record corpus, loss trajectory a clean monotonic 1.92 -> 1.22 across the run, no sign of a collapse spike) reproduced the *same total mode collapse* as the unbatched run - 1 unique output across all 90 questions, a single memorised `country`/`city` join query regardless of the actual input.

**Batching was not the active ingredient.** The real variable appears to be total gradient-update count against a fixed, tiny (57-example) corpus: too few updates never teaches the task; enough updates to learn anything (~45, on only 57 examples, repeated 3x) is also enough to memorise a single dominant answer instead of learning to condition on the question. Batching changes *how noisy* each update is, not *how much data the model has seen* - with only 57 examples, more epochs just means more repetitions of the same 57 answers, which is a corpus-size problem, not an optimizer-hyperparameter problem. The likely fix is a substantially larger latent-SFT corpus (hundreds of examples, not 57), not further tuning of batch size, learning rate, or epoch count on this one.

---

## 6. What these numbers do not show

1. **Nothing is statistically separable.** n = 24 gives ~±19-point intervals; every method's interval overlaps every other's.
2. **GRPO ran for 15 steps.** The schedule was cut by a timeout at 15 of 23, so `w_fmt` stopped at 0.35 instead of reaching 0.05 — the run executed the "verbalise more" half of an annealed curriculum and was killed before the "stop writing it down" half. A real attempt is ~13.5 h at 97 s/step.
3. **The debate is not internalised.** The student still emits ~982 output tokens. The headline claim — same accuracy at a fraction of the *output* — remains untested.
4. **The teacher is weak.** At 63% EX it is far below the GPT-4-class systems the four papers assume, and the student cannot exceed what it demonstrated.
5. **The corpus is filtered to easy questions.** `require_correct` discards every question the teacher failed, so the student never sees a worked trace for the hard cases debate should help most.

See [METHOD.md §8](METHOD.md) for the full list of drawbacks and their mitigations.

---

## 7. Reproducing

```bash
bash scripts/run_pilot.sh                 # run 2 (configs/pilot_mix.yaml)
python scripts/06_report.py --run outputs/pilot2
```

Per-sample logs: `outputs/pilot2/traces/teacher_samples.jsonl`, `outputs/pilot2/{sft,grpo}/*_log_steps.jsonl`, `outputs/pilot2/eval*/eval_samples.jsonl`, and every predicted query in `outputs/pilot2/eval*/preds_<method>.jsonl`.
