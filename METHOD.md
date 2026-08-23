# LatentSQL — the method in detail

How a multi-agent Text-to-SQL system is compressed into a single sub-7B model
that runs the debate in its own activation space and emits only the final query.

This document explains **what the method is and why each piece is there**. For
what has actually been measured, see [README §11c](README.md); for the code
layout, the implementation map is at the end.

---

## 0. The claim, stated precisely

> A single **3B** model, post-trained on the transcripts of a multi-agent
> Text-to-SQL debate, matches or beats the explicit multi-agent system it was
> distilled from, on BIRD and Spider, at a **fraction of the token cost**.

Three things make that claim non-trivial:

1. **It is not ordinary distillation.** Ordinary distillation copies the
   teacher's *answers*. Here the student is trained on the teacher's *entire
   argument* — four systems proposing, criticising and revising — and then
   pressured to stop writing it down.
2. **The comparison is size-matched.** The baselines (DIN-SQL, DAIL-SQL, CHESS,
   CHASE-SQL) are re-run on the **same 3B backbone** as the student, not on the
   7B teacher. Beating a 3B DIN-SQL with a 3B student is a claim about the
   *method*; beating a 7B one would be a claim about model size.
3. **Correctness is executable.** "Right answer" means the query returns the
   same rows as gold when actually run against SQLite — not string similarity.
   This makes the reward in stage 2 a hard, non-gameable signal.

The intellectual debt is to two papers, applied to a domain neither addressed:

- **IMAD** (Yi et al., ACL 2026) — *Internalized Multi-Agent Debate*: the
  two-stage SFT→GRPO recipe with an annealed reward, §§2.2–2.3, and the
  agent-subspace analysis, §3.
- **LatentMAS** (Zou et al., ICML 2026) — *Latent Collaboration in Multi-Agent
  Systems*: training-free latent thoughts and latent working memory, §§3.1–3.2.

---

## 1. The shape of the whole thing

```
   ┌──────────── TEACHER: four published systems argue (expensive, explicit) ────────────┐
   │                                                                                     │
   │  value retrieval ─▶ schema linking ─▶ classify/decompose      ← shared front-end     │
   │                          │                                                          │
   │            ┌─────────────┴──────────── round 1 ───────────────┐                     │
   │            │  DIN-SQL   DAIL-SQL   CHESS   CHASE-SQL          │  each runs its OWN   │
   │            │     │          │        │         │              │  published pipeline  │
   │            │     └──────────┴────────┴─────────┘              │                      │
   │            │        ↕ round 2: each revises after seeing      │                      │
   │            │          the others' SQL *and their rows*        │                      │
   │            └─────────────────────┬───────────────────────────┘                      │
   │                                  ▼  selector (CHESS unit tester)                    │
   └──────────────────────────────────┬──────────────────────────────────────────────────┘
                                      │  tagged debate transcript, filtered on correctness
                                      ▼
        ┌────────────────── STUDENT: one 3B model ──────────────────┐
        │  stage 1  SFT on the FULL transcript          (IMAD §2.2) │
        │  stage 2  GRPO, w_fmt 1.0→0.05, l 2000→500    (IMAD §2.3) │
        │           ⇒ writing the debate down stops paying          │
        │  optional latent roles before decoding    (LatentMAS §3)  │
        └───────────────────────────┬───────────────────────────────┘
                                    ▼
                              just the SQL
```

---

## Part I — The teacher

### 1.1 Why a panel of *systems*, not a panel of prompts

The default panel is three prompt personas (`divide_conquer`, `query_plan`,
`dail_icl`). The configuration this project actually uses puts the **four
published systems themselves** on the panel:

```yaml
mas:
  agents: [din_sql, dail_sql, chess, chase_sql]
  rounds: 2
```

Two reasons, one methodological and one brutally practical:

**Methodological.** CHASE-SQL §3.3 shows that what makes multi-candidate
generation pay is *diversity of reasoning path*, not sampling the same prompt
repeatedly. Four independently-designed systems are about as diverse as
candidate generators get. IMAD §3 adds the complementary point: distinct agent
*personas* are what produce separable agent subspaces after internalisation. If
the panellists are near-duplicates, there is nothing for the student to separate.

**Practical.** The training corpus is filtered on `require_correct`, so **every
question the teacher gets wrong is thrown away**. Trace yield is the binding
constraint on the whole project. Four independent pipelines give four
independent shots at a correct candidate, where three prompt variants of one
pipeline give something closer to three correlated shots. Measured in run 1:
yield was bounded *entirely* by teacher accuracy — of 62 dropped traces, 51 were
dropped as incorrect and 11 for lack of consensus, and **zero** for any other
reason.

The cost is real: each panellist is itself multi-call, so the four-system panel
runs ~27 LLM calls and ~50 s per question against the persona panel's ~14 calls
and ~19 s.

### 1.2 The shared front-end

Run once per question, before the debate, and written into the trace:

| Stage | From | What it does |
|---|---|---|
| Keyword extraction + value retrieval | CHESS §3.1, CHASE §3.2 | MinHash-LSH over character trigrams of every cell value, re-ranked by edit distance. Finds that the question's `"North America"` is literally `region.region_name`. |
| Schema linking | DIN-SQL §4.1 | CoT prompt producing `Tables:` / `Columns:` / `Conditions:`, parsed back onto real identifiers. |
| Schema pruning | CHESS Schema Selector | **Skipped below 40 columns** — CHESS §3.3 reports pruning *hurts* when the schema already fits. Join keys are always kept. |
| Classification + decomposition | DIN-SQL §4.2 | `EASY` / `NON-NESTED` / `NESTED`; nested questions also get sub-questions. |

These stages are **idempotent**. A panellist that is itself a full system will
call schema linking again; the second call is a no-op returning the cached
result. Without that, four panellists pay four times for identical work — it cut
measured teacher tokens ~9% at identical yield.

### 1.3 Round 1 — each system runs its own pipeline

Each panellist executes its published method end to end against the shared
context, and contributes **its winning candidate's reasoning and SQL**:

- **DIN-SQL** — link → classify → class-specific generation → self-correct (§4.4).
- **DAIL-SQL** — `CR^p` code representation, masked-question + query-skeleton
  example selection (§§3.1–3.3), self-consistency over *execution results* across
  3 samples.
- **CHESS** — IR → Schema Selector → 3 candidates from the Candidate Generator,
  each through the revise loop → Unit Tester picks.
- **CHASE-SQL** — value retrieval → three generators (divide-and-conquer,
  query-plan, retrieved-example) → fixer → pairwise selection agent (§3.5).

**Crucially, only the winner's reasoning + SQL enters the trace.** CHESS's
unit-tester loop, CHASE's pairwise tournament, DAIL's retrieval — that machinery
*produced* the answer and is then discarded. It is scaffolding, not curriculum.
Recovering the reasoning text behind a selector's chosen SQL is what
`_pick()` in `baselines.py` exists for; without it the trace would carry a bare
query and there would be nothing to learn.

### 1.4 Round 2 — the actual debate

Every panellist is shown the other three's answers **and the rows those queries
returned**, then revises. This is Du et al. (2023)'s protocol with one
deliberate deviation:

> **Execution feedback is inside the debate.** Round-2 agents see each other's
> *results*, not just their SQL.

This is what makes the transcript worth distilling. A student trained on
"agent A said X, agent B said Y, the answer was Y" learns an arbitrary
preference. A student trained on "agent A said X, X returned 0 rows, agent B
said Y, Y returned 7 rows, the answer was Y" learns *which query shapes fail and
why* — a transferable skill.

The debate exits early if every panellist already agrees and executes cleanly.

### 1.5 Selection

Candidates are first **grouped by execution signature** — queries returning
identical rows form one group — so the selector compares semantically distinct
options rather than cosmetic variants. Then:

- **`unit_test`** (default, CHESS): the LLM writes discriminating assertions and
  votes.
- **`pairwise`** (CHASE §3.5): round-robin comparisons, weighted by group size.
- **`exec_consistency`**: plain self-consistency over results.

A single surviving group is `unanimous` — consensus by construction.

---

## Part II — The trace

### 2.1 The format

This is the exact string the student is trained to emit:

```
<|schema_link|>
Tables: director
Columns: name, director
Values: director.director = 'Wolfgang Reitherman'
<|/schema_link|>
<|classify|> NON-NESTED <|/classify|>
<|round 1|>
<|agent din_sql|>
… reasoning …
```sql
SELECT COUNT(name) FROM director WHERE director = 'Wolfgang Reitherman'
```
<|exec|> movie_count (9,) <|/exec|>
<|/agent|>
<|agent dail_sql|>  … <|/agent|>
<|agent chess|>     … <|/agent|>
<|agent chase_sql|> … <|/agent|>
<|/round|>
<|round 2|>
… each of the four, having seen the others' SQL and rows …
<|/round|>
<|consensus|>
```sql
SELECT COUNT(DISTINCT name) FROM director WHERE director = 'Wolfgang Reitherman'
```
<|/consensus|>
<|endofdebate|>
```

### 2.2 Why the tags carry the method

IMAD §2.1 is explicit that the structure tags are what make internalisation
work: without them "the distinction between agents becomes difficult to make",
and the agent subspaces of §3 come out less separable. The tags do three jobs:

1. **They give `R_fmt` something to measure.** Stage 2's format reward is
   literally "are the required tags present, in order". No tags, no stage-2
   curriculum.
2. **They mark the spans that become agent subspaces.** The steering analysis
   builds contrastive pairs from text between `<|agent i|>` and `<|/agent|>`.
3. **They give the latent layer anchor points.** LatentMAS mode emits the role
   tag, then thinks in activation space — the tag is what selects the subspace.

The required tags, in canonical order, are
`<|schema_link|>` → `<|round 1|>` → `<|consensus|>` → `<|endofdebate|>`.

### 2.3 Filtering — this is where the corpus is made or lost

| Filter | Default | Rationale |
|---|---|---|
| `require_consensus` | on | IMAD §2.1 discards debates that never converged. |
| `require_correct` | on | We have gold SQL and an executor, so we can filter harder than IMAD could on arithmetic. |
| `require_execution` | on | The final query must actually run. |
| `dedup` | on | Identical transcripts add nothing. |
| `max_trace_chars` | 12 000 | Bounds SFT sequence length. |

**Consequence to internalise:** the corpus size is
`n_questions × teacher_accuracy × consensus_rate`. At the run-1 teacher's 28%
accuracy that is ~17 usable traces per GPU-hour. Everything about scaling this
project is downstream of that number.

---

## Part III — Stage 1: SFT (IMAD §2.2)

Train the student on `(prompt, completion)` where the completion is the **entire
tagged transcript** — not just the final SQL.

- **Loss is masked to the completion only.** Prompt tokens contribute nothing.
- **LoRA by default** (r=32, α=64) so a 3B student trains on one consumer card;
  with 4-bit base weights this is QLoRA.
- **`train_on_full_trace: false` is the DebateGPT ablation** — train on the
  consensus SQL alone. This is the single most important ablation in the
  project, because it isolates whether the *transcript* is doing any work or
  whether the answers alone would do.

After stage 1 the student can *reproduce the debate*. It is slower and more
verbose than any baseline. That is expected and is the point: stage 1 installs
the reasoning structure, stage 2 removes the need to write it down.

---

## Part IV — Stage 2: GRPO internalisation (IMAD §2.3)

This is the mechanism the whole project rests on.

### 4.1 The reward

For a rollout `y` on question `x`:

```
r(x, y) = w_fmt · R_fmt(y)  +  w_clip · R(y; l)  +  0.1 · R_exec  +  0.1 · R_order
```

- **`R_fmt`** — fraction of required tags present. Rewards *verbalising* the debate.
- **`R(y; l)`** — **1 if the correct query appears within the first `l` tokens
  of `y`, else 0.** Clipping is in *token* space, using the model's own
  tokenizer, so the schedule is not model-dependent.
- **`R_exec`** — the clipped query parses and runs (small shaping term).
- **`R_order`** — the tags appear in canonical order (small shaping term).

The two shaping terms are deliberately weighted at 0.1 so IMAD's two real terms
dominate.

"Correct" runs SQL. The rollout's clipped text is parsed for the final query,
executed against the actual database, and its rows compared to gold's —
order-sensitively if and only if the gold query has an `ORDER BY`. An
`ExecutionOracle` caches gold results and per-`(db, sql)` verdicts, because GRPO
re-scores the same queries constantly.

### 4.2 The two schedules, and why their *interaction* is the method

Over training:

| | start | end |
|---|---:|---:|
| `w_fmt` | 1.0 | 0.05 |
| `l` (token clip) | 2 000 | 500 |

Read them together:

- **Early**: `w_fmt` is high and `l` is generous. Writing out the full debate is
  rewarded *and* affordable. The student is reinforced for doing what stage 1
  taught it.
- **Late**: `w_fmt` has collapsed to 0.05, so verbalising earns almost nothing —
  and `l` has shrunk to 500 tokens, so a verbalised four-way debate **cannot
  physically fit** before the clip. Any rollout that spends its budget narrating
  gets `R(y; l) = 0`.

The only strategy that still scores is: *do the multi-perspective analysis
internally, then emit the query directly.* That is the entire argument. Neither
schedule alone would work — dropping `w_fmt` alone permits rambling, shrinking
`l` alone punishes a model still being paid to narrate.

### 4.3 GRPO specifics

- **Group-relative advantage**: `A_i = (r_i − mean(r)) / std(r)` within each
  group of `k` rollouts for the same prompt. A degenerate group (all rewards
  equal) yields all-zero advantages — correct, since there is no preference
  signal.
- **One update per group**, so the sampling policy *is* the scored policy: the
  importance ratio is identically 1 and the clipped surrogate reduces to
  `−A·logp`. The clipped form is written out anyway so more inner epochs stay a
  one-line change.
- **KL to the frozen SFT policy** via the low-variance k3 estimator,
  `exp(δ) − δ − 1`. With LoRA the reference needs no second copy of the weights
  — disabling the adapters recovers it.

### 4.4 What internalisation looks like in the logs

Watch `gen_tokens_mean` against `length_limit`. Success is generated length
**falling** as the clip tightens while reward holds or rises. In run 1 it did
not fall (654 → 746 over 9 steps) — but 9 steps at lr 1e-6 on 19 traces is far
too little to conclude anything. At ~97 s/step, a real attempt is **~13.5 h**.

---

## Part V — Latent collaboration at inference (LatentMAS §§3.1–3.2)

Optional layer on top of the internalised student. LatentMAS is training-free
and runs N separate models; here the same two mechanisms are applied to the
**one** student, giving explicit latent compute to the roles it absorbed.

**Latent thoughts (§3.1).** Instead of decoding a token, feed the last-layer
hidden state back as the next input embedding, for `m` steps. Zero tokens
emitted; one forward pass each.

**Input-output alignment (§3.1).** A raw hidden state is out-of-distribution as
an *input* embedding, so it is mapped back:

- `soft_embed` (default) — `e = softmax(h·W_out / T)·W_in` over the top-k
  logits. The exact soft-embedding of the token distribution the hidden state
  encodes; cheap, no `|V|×d_h` product.
- `linear` — precompute `W_a = W_out·W_in`, use `e = h·W_a`. The first-order
  expansion, closest to the paper's eq. (3).
- `identity` — feed hidden states straight back (COCONUT-style). Kept as the
  ablation LatentMAS predicts should fail; empirically it produces vectors ~40×
  the true embedding norm.

**Latent working memory (§3.2).** After a role's latent steps its layer-wise KV
cache is *kept*, and the next role generates conditioned on it. Lossless in the
sense of their Theorem 3.3: the successor sees the predecessor's actual
representations rather than a re-encoding through text.

Generation becomes:

```
prompt → <|agent din_sql|> → m latent steps
       → <|agent dail_sql|> → m latent steps
       → … → <|consensus|> → decode the SQL normally
```

---

## Part VI — Agent subspaces (IMAD §3)

The interpretability claim: after internalisation the student holds *separable*
directions for each teacher agent, rather than collapsing them into one
undifferentiated policy.

Method (Rimsky et al. 2024, difference-in-means / CAA):

1. For agent *i*, build contrastive pairs sharing context (question + debate
   history up to `<|agent i|>`). Positive = agent *i*'s real continuation;
   negative = another agent's response in the same slot.
2. Record residual-stream activations at a chosen layer; take
   `v_i = mean(pos) − mean(neg)`.
3. At inference, add `α·v_i` to that layer's output.

Two payoffs. **Interpretability**: if the effect is stronger after IMAD than on
the base model, the debate really was absorbed as structure. **Practical**:
steering toward `divide_conquer` vs `query_plan` is a cheap diversity knob,
recovering some of CHASE-SQL's multi-path generation from a *single* forward
pass per candidate.

---

## Part VII — Evaluation protocol

- **Metric**: execution accuracy (EX) — predicted rows vs gold rows, order-
  sensitive iff gold has `ORDER BY`. BIRD's VES optionally, averaged over BIRD
  rows only (it is a BIRD metric).
- **Size-matched baselines**: all of `single`, `din_sql`, `dail_sql`, `chess`,
  `chase_sql` run on the *same* backbone as the student.
- **Combined BIRD+Spider**: halves sampled independently and **interleaved by
  ratio**, so any prefix keeps the mixture — `--limit 200` on a 50/50 mix gives
  100+100, not 200 BIRD. Every example keeps its `dataset` field, and EX is
  reported per benchmark. A blended number alone is not comparable to any
  published result.
- **No leakage**: trace building reads the `train` splits, evaluation reads
  `dev`. DAIL-SQL few-shot examples are drawn only from the teacher-train pool.
- **Same everything**: one schema renderer, one executor, one scorer, so
  comparisons are apples-to-apples rather than numbers copied from four papers.

### The ablations a reviewer will demand

| Ablation | Switch | Objection it answers |
|---|---|---|
| Full trace vs. answer only | `train_on_full_trace: false` | "Is the transcript doing anything, or just the answers?" (DebateGPT; IMAD Table 1) |
| Panel vs. one format | `mas.agents: [query_plan]` | "Struct-SQL shows one execution-plan CoT is enough" |
| GRPO from base, no teacher | `03_grpo.py --from-examples` | "Arctic-Text2SQL-R1 hits 68.9% BIRD with only an execution reward — is the teacher earning its cost?" |

The third is the uncomfortable one and must be reported even if unflattering.

---

## Part VIII — Drawbacks and failure modes

Everything here was observed or measured on this codebase, not imagined. Each
item names the mitigation where one exists.

### 8.1 GRPO's degenerate-group problem — the most serious flaw

GRPO's advantage is group-relative: `A_i = (r_i − mean(r)) / std(r)` over the
`k` rollouts for one prompt. **If all `k` rollouts earn the same reward, every
advantage is exactly 0 and there is no gradient.** A group only teaches
something if it straddles success and failure.

The failure is worse than "no learning", because of *which* term goes flat
first. Reward is `w_fmt·R_fmt + 1.0·R_clip + …`. On a question the student
cannot yet solve, all `k` rollouts get `R_clip = 0`, so the only quantity left
with variance is `R_fmt` — **whose gradient points toward emitting more tags**.
The model is actively pushed to verbalise more, which is precisely the behaviour
the method exists to remove.

Measured in pilot run 1: the correctness term was flat at zero for the entire
group on **5 of 9 steps**, and generated length rose 654 → 746 tokens instead of
falling. Total weight movement over the whole GRPO stage was
`||ΔW||/||W|| = 6.3e-04` — the GRPO student was the SFT student to within noise,
and scored identically (54.17%).

*Mitigations*: larger `group_size` (8–16) so a group is more likely to straddle;
enough traces that easy questions are present; and see 8.2. The GRPO step log
now records `reward_std_mean` and `adv_nonzero_frac` so a dead group shows up
directly rather than having to be reconstructed after the fact.

### 8.1b Groups die from saturation too, and nothing rewards brevity

8.1 is the all-*wrong* group. Run 2 exposed its mirror image: when the student
gets **every** rollout right and formats them identically, the rewards are again
equal, `std = 0`, and the gradient is again zero. Measured at run-2 steps 8 and
9 (`R_clip = 1.0`, `adv_nonzero_frac = 0.0`).

Dense partial credit (8.1) does not help here - it only spreads *wrong*
rollouts. The gap is deeper: **no term in the reward distinguishes two correct
rollouts of different lengths.** Both earn `R_clip = 1`, so GRPO cannot learn
"be correct *and* shorter" from a saturated group - which is precisely the
internalisation objective.

IMAD's length pressure is supposed to come from the clip shrinking until long
rollouts fall outside it. That only works if the answer appears *late*. In
practice the student emits its query early and then keeps writing: at run-2 step
8, `gen_tokens_mean` was 1024 against a `length_limit` of 718, and all four
rollouts still scored `R_clip = 1`. The clip never bit.

*Mitigation*: a brevity term active among correct rollouts, so a terse correct
answer outscores a verbose one. Not yet implemented.

### 8.2 The annealing schedule is open-loop

`w_fmt` and `l` anneal on **step count**, not on whether the student is
succeeding. A student that cannot yet solve anything still has its curriculum
consumed on schedule: by the time `w_fmt` reaches 0.05 the format incentive is
gone, but if `R_clip` is still mostly zero there is now *no* usable signal at
all — the run ends having learned nothing from either term.

*Mitigation*: gate the anneal on a success criterion (e.g. hold `w_fmt` until
`R_clip > 0` on some fraction of groups) rather than on step index. Not
implemented; IMAD specifies a step-indexed schedule and this codebase follows it.

### 8.3 Corpus size is the product of two accuracies

`corpus = n_questions × teacher_accuracy × consensus_rate`, because
`require_correct` discards every question the teacher gets wrong. This is a
multiplicative penalty on the most expensive stage.

Measured: **19 of 81 kept (23.5%)**, of which 51 were dropped as incorrect and
11 for no consensus — **zero** for any other reason. Yield is bounded almost
entirely by teacher accuracy. That is **~17 usable traces per GPU-hour** with a
7B teacher and a four-system panel.

The bind: a better teacher means a bigger model, which is slower per question,
so raising accuracy costs throughput. There is no configuration that is cheap
and high-yield at once.

### 8.4 The filter systematically biases the corpus toward easy questions

`require_correct` keeps exactly the questions the teacher found easy. The
student therefore never sees a worked trace for a hard question — the cases
where multi-agent debate should help most are precisely the ones filtered out.
This is a distribution shift baked into the method, and it plausibly caps the
student's ceiling at "what the teacher already found easy".

*Mitigation*: none implemented. Options worth trying: keep incorrect traces with
a negative label for contrastive use; or retain no-consensus traces that contain
a correct candidate, since the debate content may be instructive even when the
selector failed.

### 8.5 Token savings and latency savings are not the same thing

The headline "fraction of the tokens" is about **total** tokens, which are
dominated by *prompt* tokens — the student reads one prompt where DIN-SQL reads
five. But **latency is dominated by output tokens**, which are generated
serially.

Measured in run 1, the student used fewer total tokens than DIN-SQL
(2 097 vs 3 186) but was **~4× slower** (24.1 s vs 6.4 s), because it emitted
more than twice the output tokens (675 vs 321) in a single autoregressive
stream. A multi-call baseline makes several short generations; the
pre-internalisation student makes one very long one.

This is not a permanent property — it is a direct consequence of 8.1. A student
that had actually internalised the debate would emit ~50 output tokens and be
the fastest method on the table. **Until internalisation is demonstrated, the
student is strictly worse on latency**, and reporting only the token ratio would
be misleading.

### 8.6 The student does not reliably close the debate

The student was the **only** method to emit unparseable output (2 of 24 in run
1). Run 2 quantified the deeper version: given a *3072*-token budget, with only
2 of 24 generations hitting the cap, the student still emitted `<|consensus|>`
on just **11 of 24** questions. The fallback extractor - last well-formed
```sql``` block - is therefore scraping a mid-debate candidate about half the
time, rather than the answer the debate converged on.

This is a **format-adherence** failure, not a budget failure. That distinction
was worth establishing: the tempting hypothesis is truncation, and re-running
run 2 at 1024 vs 3072 tokens gave *identical* accuracy (50.00% both times),
refuting it. Questions whose traces finish early are simply the easier ones.

*Mitigation*: `extract_final_sql` already falls back to the last well-formed
```sql``` block; a stricter constrained-decoding pass over the tag grammar would
remove the failure mode entirely.

### 8.7 The panel's cost is multiplicative

`rounds × panellists × (calls per panellist)`. The four-system panel runs ~27
LLM calls and ~50 s per question against the three-persona panel's ~14 calls and
~19 s — **2.7×** for what was, on a 4-question probe, the same accuracy. Its
advantage was in *consensus* (hence yield), not correctness, and that was
measured at n=4 and is not established.

Additionally the panellists are not independent: they share the front-end
(retrieval, linking, classification) and several share the same underlying
candidate agents, so their errors correlate more than four separate papers would
suggest.

### 8.8 Execution-based scoring fails silently

Because "correct" means "the rows match", **any defect upstream of execution
reads as low accuracy rather than as an error**. This project lost an entire
benchmark to it: canonical Spider `dev.json` stores the *parsed* query tree under
`sql` and the query string under `query`; the loader stringified the dict, so
every Spider gold was a Python repr that cannot execute, and all 39 Spider
questions scored 0.00% with no exception raised anywhere. It was caught only
because Spider scoring *below* BIRD is implausible — Spider is the easier
benchmark.

*Mitigation, now standard practice here*: assert that gold queries execute
before trusting any accuracy number, and treat an implausible per-benchmark
ordering as a bug signal rather than a result.

### 8.9 A weak teacher caps everything downstream

The teacher is itself a small local model. At 28.4% overall EX (54.8% BIRD /
0.0% Spider-with-the-bug), it is well below the GPT-4-class systems the four
papers were designed around. The student cannot exceed what the teacher
demonstrated, so every number in this project is bounded by the teacher's
ceiling, not by the method's.

### 8.10 Validation scale

n = 24 evaluation questions gives 95% Wilson intervals roughly ±19 points —
**every method's interval overlaps every other's**. Of those 24, six were solved
by all methods and seven by none, so only 11 discriminate at all. No ranking in
the current results table is statistically supported.

### 8.11 The latent layer is unvalidated for quality

LatentMAS mode is verified for *shapes, caches, hooks and schedules* — that
latent roles run, that working memory carries forward, that only the final query
is decoded. Nothing has been measured about whether it *helps*. The `identity`
aligner is known-bad by construction (it produces vectors ~40× the true
embedding norm, the out-of-distribution case LatentMAS §3.1 warns about), which
is why `soft_embed` is the default.

---

## Part IX — What is validated, and what is not

**Validated** (35 unit tests + a 28-check smoke test): the trace format
round-trips; every reward term behaves as IMAD eq. 1–2; length clipping clips;
group advantages normalise; prompt tokens are masked out of the SFT loss; GRPO
takes real optimizer steps under reward variance; latent roles run and only the
final query is decoded; latent working memory carries roles forward; steering
vectors extract and are separable; all four baselines run.

**Not validated**: the accuracy claim. As of run 1, the student has been
SFT'd on 19 traces and given 9 GRPO steps — enough to prove the pipeline
executes, nowhere near enough to internalise anything. See
[README §11c](README.md) for the recorded numbers and their caveats.

---

## Implementation map

| Concept | File |
|---|---|
| Panel, debate rounds, early exit | [`mas/orchestrator.py`](latentsql/mas/orchestrator.py) |
| The four systems as panellists (`SystemAgent`) | [`mas/agents.py`](latentsql/mas/agents.py) |
| Standalone re-implementations + `*_turn` twins | [`mas/baselines.py`](latentsql/mas/baselines.py) |
| Tag vocabulary, render/parse, `format_score` | [`mas/trace.py`](latentsql/mas/trace.py) |
| Filtering into the SFT corpus | [`distill/build_dataset.py`](latentsql/distill/build_dataset.py) |
| Stage 1 | [`distill/sft.py`](latentsql/distill/sft.py) |
| Stage 2, group advantages, KL | [`distill/grpo.py`](latentsql/distill/grpo.py) |
| Reward eq. (1)–(2), schedules, oracle | [`distill/rewards.py`](latentsql/distill/rewards.py) |
| Latent thoughts, aligners, working memory | [`distill/latent.py`](latentsql/distill/latent.py) |
| Agent subspaces / CAA | [`steering/caa.py`](latentsql/steering/caa.py) |
| LSH value retrieval | [`db/value_index.py`](latentsql/db/value_index.py) |
| Sandboxed execution, EX, VES | [`db/execute.py`](latentsql/db/execute.py) |
| BIRD/Spider/mixed loaders | [`data/loaders.py`](latentsql/data/loaders.py) |
| Per-sample / per-step run logs | [`utils/telemetry.py`](latentsql/utils/telemetry.py) |
| Report assembly | [`scripts/06_report.py`](scripts/06_report.py) |
