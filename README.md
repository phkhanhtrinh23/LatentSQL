# LatentSQL — internalising multi-agent Text-to-SQL into a small language model

Distil a DIN-SQL / DAIL-SQL / CHESS / CHASE-SQL–style multi-agent system into a
**single sub-7B model** that runs the debate in its **latent space** and emits
only the final query.

The target claim: *match or beat the explicit multi-agent system on BIRD and
Spider, with a small local model, at a fraction of the tokens.*

Method = **IMAD** (Yi et al., ACL 2026, "Latent Agents: A Post-Training
Procedure for Internalized Multi-Agent Debate") applied to Text-to-SQL, with
inference-time **LatentMAS** (Zou et al., ICML 2026, "Latent Collaboration in
Multi-Agent Systems") layered on top.

---

## 1. Contents

| Path | What it is |
|---|---|
| [COMPARISON.md](COMPARISON.md) | **LatentSQL vs. FINER-SQL** — what is actually different, what we borrowed, and what we had to change |
| [RESULTS.md](RESULTS.md) | **Recorded results** — accuracy, cost, GPU/wall-clock, and worked examples of what internalisation fixed |
| [METHOD.md](METHOD.md) | **The method in full detail** — the teacher panel, the trace format, both training stages, the latent layer, the evaluation protocol, and the measured drawbacks and failure modes |
| [latentsql/mas/](latentsql/mas/) | The teacher: agents, prompts, debate orchestration, trace format, and standalone re-implementations of the four published systems |
| [latentsql/distill/](latentsql/distill/) | IMAD stage 1 (SFT), stage 2 (GRPO + annealed reward), and the LatentMAS latent-thought module |
| [latentsql/steering/](latentsql/steering/) | Agent-subspace extraction and activation steering (IMAD §3) |
| [latentsql/db/](latentsql/db/) | Schema introspection, sandboxed execution, EX / VES metrics, LSH value retrieval |
| [latentsql/data/](latentsql/data/) | BIRD / Spider loaders + a synthetic fixture for offline testing |
| [scripts/](scripts/) | The five-step pipeline plus the smoke test |
| [configs/](configs/) | `bird_local.yaml` (main), `spider_local.yaml`, `smoke.yaml`, `teacher_openai.yaml` |
| [tests/](tests/) | 25 torch-free unit tests |

---

## 2. The idea in one picture

```
                    ┌─────────────── TEACHER (expensive, explicit) ───────────────┐
                    │                                                             │
 question + schema ─┼─▶ value retrieval ─▶ schema linking ─▶ classify/decompose ──┤
                    │        (CHESS/CHASE)     (DIN-SQL/CHESS)     (DIN-SQL)      │
                    │                                                             │
                    │   ┌────────────── debate, m = 2 rounds ─────────────┐       │
                    │   │  divide_conquer   query_plan     dail_icl       │       │
                    │   │   (CHASE §3.3)   (CHASE §3.3)   (DAIL §3.1-3.3) │       │
                    │   │        ↕ critique + execution feedback ↕        │       │
                    │   └─────────────────────────────────────────────────┘       │
                    │                      ↓ fixer (CHASE §3.4)                   │
                    │                      ↓ selector (CHESS Unit Tester)         │
                    └──────────────────────┬──────────────────────────────────────┘
                                           │  tagged debate transcript
                                           ▼
       ┌───────────────── STUDENT (< 7B, local) ─────────────────┐
       │  stage 1  SFT on the FULL transcript   (IMAD §2.2)      │
       │  stage 2  GRPO, w_fmt 1.0→0.05, l 2000→500  (IMAD §2.3) │
       │           ⇒ the debate stops being verbalised           │
       │  optional LatentMAS roles before decoding (§3.1-3.2)    │
       └────────────────────────┬────────────────────────────────┘
                                ▼
                          just the SQL
```

The mechanism that does the work is IMAD equation (1):

```
r(x, y) = w_fmt · R_fmt  +  w_clip · R(y; l)
```

`R_fmt` rewards emitting the debate structure tags; `R(y; l)` is 1 only if the
**correct** query appears within the first `l` tokens. Over training `w_fmt`
decays 1.0 → 0.05 and `l` shrinks 2000 → 500. Verbalising the debate stops
paying and stops fitting — so the only strategy left is to run the
multi-perspective analysis internally and answer directly.

For Text-to-SQL, "correct" means **execution match against the gold query**, so
`R(y; l)` runs SQL rather than string-matching (see
[rewards.py](latentsql/distill/rewards.py)).

---

## 3. Quick start

### Install

```bash
python -m venv .venv && . .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or /cpu
pip install -r requirements.txt
```

### Smoke test (no data, no GPU, no API key)

```bash
python scripts/smoke_test.py
```

Two tiers. **Tier 1** needs nothing but the standard library: it builds two
small SQLite databases, runs the teacher and all four baselines, and checks the
trace format and every reward term. **Tier 2** needs torch: it loads a tiny
model, runs an SFT step, a GRPO step with a real execution-based reward, latent
generation, and steering-vector extraction.

Tier 2 verifies **shapes, masks, caches, hooks and schedules** — not model
quality. A tiny model cannot write SQL; the point is that nothing is broken
before you spend a GPU-week.

See [§13 Smoke-test results](#13-smoke-test-results) for the full recorded run.

### See which local models you have

```bash
python scripts/00_list_models.py --hub /mnt/win_d/hub --max-params 7
```

Prints name / parameter count / path for every causal LM in the hub. Put a name
from that list into `llm.model` (teacher) and `sft.model_name_or_path` (student).

### Full pipeline on BIRD + Spider combined

[`configs/mix_local.yaml`](configs/mix_local.yaml) — a 7B teacher distilled
into a 3B student over one set drawn from **both** benchmarks:

```bash
python scripts/01_build_traces.py --config configs/mix_local.yaml --limit 600
python scripts/02_sft.py          --config configs/mix_local.yaml
python scripts/03_grpo.py         --config configs/mix_local.yaml
python scripts/04_eval.py         --config configs/mix_local.yaml \
    --methods single,din_sql,dail_sql,chess,chase_sql,fusion_mas,student,latent
```

### Full pipeline on BIRD

```bash
python scripts/01_build_traces.py --config configs/bird_local.yaml --limit 3000
python scripts/02_sft.py          --config configs/bird_local.yaml
python scripts/03_grpo.py         --config configs/bird_local.yaml
python scripts/04_eval.py         --config configs/bird_local.yaml \
    --methods single,din_sql,dail_sql,chess,chase_sql,fusion_mas,student,latent
python scripts/05_steering.py     --config configs/bird_local.yaml --compare-base
```

---

## 4. Local-first execution

Everything defaults to **open-weight models loaded from the local hub**
(`/mnt/win_d/hub`). No network call, no API key.

```yaml
llm:
  backend: hf          # default
  local_only: true     # refuse to fall back to the network
  model: Qwen/Qwen2.5-Coder-7B-Instruct
  model_fallbacks: [ ... ]   # tried in order if the first is not in the hub
```

`resolve_model_path()` searches the HF cache layout
(`models--org--model/snapshots/<rev>`) *and* plain mirrors (`<hub>/org/model`,
`<hub>/model`). If nothing matches it fails loudly and **lists the causal LMs it
did find**, rather than silently downloading.

The OpenAI-compatible backend is fully implemented
([openai_compat.py](latentsql/llm/openai_compat.py)) and drives the identical
agent code — it is simply never selected by default. Opt in with
`configs/teacher_openai.yaml`. The same file points at any OpenAI-compatible
server (vLLM, SGLang, Ollama) via `base_url`.

There is also a `mock` backend. **It is a test double, not a model**: it reads
the gold SQL and returns per-agent perturbations of it so the orchestration can
be exercised offline. Numbers produced under `backend: mock` measure plumbing
only and must never be reported.

---

## 5. The teacher: FusionSQL

Not a re-run of any one paper — a fusion that takes each component from the
system that does it best. Every module is in
[latentsql/mas/agents.py](latentsql/mas/agents.py) with the source section named
in its docstring.

| Stage | Taken from | Implementation |
|---|---|---|
| Keyword extraction + value retrieval | CHESS §3.1 "retrieve entity", CHASE §3.2 | MinHash-LSH over character trigrams, re-ranked by edit distance; optional embedding re-rank |
| Schema linking | DIN-SQL §4.1 | CoT prompt → `Tables:` / `Columns:` / `Conditions:`, parsed back onto real identifiers |
| Schema pruning | CHESS Schema Selector §3.1 | Skipped below 40 columns — CHESS §3.3 reports pruning *hurts* on small schemas. Join keys are always kept |
| Classification + decomposition | DIN-SQL §4.2 | EASY / NON-NESTED / NESTED; NESTED also gets sub-questions |
| Candidate agent 1 | CHASE-SQL §3.3 Alg. 1 | Divide-and-Conquer CoT |
| Candidate agent 2 | CHASE-SQL §3.3 | Query-Plan CoT |
| Candidate agent 3 | DAIL-SQL §3.1–3.3 | `CR^p` code representation, masked-question + query-skeleton example selection, DAIL organisation |
| Debate rounds | Du et al. 2023 | Each agent revises after seeing the others' answers **and their execution results** |
| Fixer | CHASE §3.4 / CHESS "revise" / DIN-SQL §4.4 | Execution-guided repair loop, ≤3 attempts |
| Selector | CHESS Unit Tester (default), CHASE §3.5 pairwise (option) | Discriminating assertions → vote; candidates grouped by execution signature first |

Two deliberate deviations, both because we have an executor the papers' prompt
chains do not always use:

1. **Execution feedback is inside the debate.** Round 2 agents see each other's
   *results*, not just their SQL. This is what makes the transcript worth
   distilling — the student learns which query shapes fail.
2. **Selection groups by execution signature before asking the LLM.** Candidates
   that return identical rows are one group, so the selector compares
   *semantically distinct* options, not near-duplicates.

The panel is intentionally **diverse rather than an ensemble of one prompt**:
CHASE-SQL §3.3 shows diversity is what makes multi-path generation pay, and
IMAD §3 shows distinct personas are what produce separable agent subspaces
after internalisation.

### Baselines

Each published system is also available standalone, on the same schema
rendering, executor and scorer, so comparisons are apples-to-apples rather than
numbers copied from four papers:

```bash
python scripts/04_eval.py --methods single,din_sql,dail_sql,chess,chase_sql,fusion_mas
```

### Option: make the four systems *be* the panel

The default panel is three prompt personas. The four published systems can
instead sit on the panel themselves, so the debate the student internalises is
an argument **between DIN-SQL, DAIL-SQL, CHESS and CHASE-SQL**:

```yaml
mas:
  agents: [din_sql, dail_sql, chess, chase_sql]
```

Round 1 runs each system's own published pipeline end to end and takes its
winning candidate; round 2+ uses the shared debate prompt, so the systems argue
over each other's answers with execution feedback attached. Tags come out as
`<|agent din_sql|>`, `<|agent chess|>`, … and the reasoning behind each
system's selected candidate is preserved
([`_pick`](latentsql/mas/baselines.py)), which is what makes the transcript
worth training on.

Why it is worth the cost with a 7B teacher: `trace.require_correct` throws away
every question the teacher gets wrong, so **trace yield is the binding
constraint**, and four independent pipelines give four shots at a correct
candidate instead of three prompt variants of one.

The price is that each panellist is itself multi-call. Measured on the
workstation (Qwen2.5-7B-Instruct, bf16, one RTX 4090, `rounds: 2`, one prompt
at a time through `model.generate`):

| Panel | s / question | LLM calls | Output tokens | 3 000-question pass |
|---|---:|---:|---:|---:|
| `[din_sql, dail_sql, chess, chase_sql]` | 49.6 | 26.2 | 2 134 | ~41 h |
| `[divide_conquer, query_plan, dail_icl]` | 18.6 | 13.8 | 968 | ~16 h |

Cost is strongly split-dependent — BIRD questions ran 62-84 s against Spider's
23-30 s, tracking schema size. Only the questions the teacher gets *right*
survive into the corpus, so size a pass by target trace count, not question
count.

On the same four questions both panels scored 1/4 correct, but only the
four-system panel also **reached consensus** on the one it got right, so it
yielded 1 usable trace against the persona panel's 0. That is n = 4 — an
illustration of where the extra 2.7x cost can pay (yield through
`require_consensus`, not raw accuracy), *not* evidence. Run a yield probe over
60-100 mixed questions before committing a multi-day pass.

The single biggest lever is not the panel: it is that `HFLocalBackend.chat`
generates **one prompt at a time**. Serving the same weights behind vLLM and
switching to `backend: openai` + `base_url` (already supported, see
[configs/teacher_openai.yaml](configs/teacher_openai.yaml)) buys continuous
batching on the same GPU. Do that before scaling the question count.

---

## 6. The trace format

The transcript the student learns to produce
([trace.py](latentsql/mas/trace.py)):

```
<|schema_link|> Tables: … Columns: … Values: … <|/schema_link|>
<|classify|> NON-NESTED <|/classify|>
<|round 1|>
  <|agent divide_conquer|> …reasoning… ```sql …``` <|exec|> …result/error… <|/exec|> <|/agent|>
  <|agent query_plan|>     …
  <|agent dail_icl|>       …
<|/round|>
<|round 2|>  …each agent revises after seeing the others…  <|/round|>
<|consensus|> ```sql <final query> ``` <|/consensus|>
<|endofdebate|>
```

IMAD §2.1 is explicit that the tags carry the method: without them "the
distinction between agents becomes difficult to make", and the §3 agent
subspaces come out less separable. `REQUIRED_TAGS` drives `R_fmt`, and
`extract_final_sql` degrades gracefully — after RL the student legitimately
drops the tags and emits a bare fenced query, which still parses.

**Filtering** ([build_dataset.py](latentsql/distill/build_dataset.py)) follows
IMAD §2.1 — no-consensus debates are discarded — and goes further, because gold
SQL gives a free execution label: traces whose consensus query is *wrong* are
dropped too. Reported as e.g.
`412/500 traces kept (dropped: no_consensus=31, incorrect=52, …)`.

---

## 7. Stage 1 — SFT (IMAD §2.2)

Next-token prediction over the **entire transcript**, prompt masked out of the
loss, LoRA by default.

Training on the full trace rather than only the final answer is the whole
difference between IMAD and DebateGPT (IMAD Table 1: DebateGPT is cheapest but
underperforms, because "without access to the full debate trace, it cannot
leverage the intermediate interactions"). That ablation is one flag:

```bash
python scripts/02_sft.py --consensus-only     # = DebateGPT
```

---

## 8. Stage 2 — GRPO internalisation (IMAD §2.3)

Self-contained GRPO ([grpo.py](latentsql/distill/grpo.py)) rather than a TRL
dependency, because the method *is* the two things TRL does not expose cleanly:
a reward whose weights change every step, and a length clip that shrinks every
step.

Per step: sample `group_size` completions → score at the current `(w_fmt, l)` →
advantage `(r − mean)/std` within the group → clipped policy gradient with a k3
KL penalty to the frozen SFT reference (recovered by disabling the LoRA
adapters, so no second copy of the weights).

Reward terms:

| Term | Weight | Meaning |
|---|---|---|
| `R_fmt` | `w_fmt`: 1.0 → 0.05 | fraction of required structure tags present |
| `R(y; l)` | `w_clip` = 1.0 | correct SQL **inside the first `l` tokens**, `l`: 2000 → 500 |
| `R_exec` | 0.1 | the clipped query at least parses and runs |
| `R_order` | 0.1 | tags appear in canonical order |

The two IMAD terms dominate by design; the SQL-specific shaping terms are small.
Clipping happens in **token** space via the tokenizer, not characters, so the
schedule is not model-dependent.

---

## 9. Latent collaboration (LatentMAS) — optional inference-time layer

LatentMAS is training-free and runs N separate models. We apply its two
mechanisms to the *one* internalised student, so the roles it absorbed get
explicit latent compute before it commits
([latent.py](latentsql/distill/latent.py)):

- **Latent thoughts (§3.1):** feed the last-layer hidden state back as the next
  input embedding for `m` steps. Zero tokens emitted, one forward pass each.
- **Input–output alignment (§3.1):** a raw hidden state is out of distribution
  as an input embedding. `soft_embed` (default) computes
  `e = softmax(h W_out / T) W_in` over the top-k logits — exact and cheap.
  `linear` precomputes the single `d_h × d_h` matrix.
  > **Note on eq. (3).** The paper writes `W_a ≈ W_out^† W_in` with a
  > pseudo-inverse. Under the convention `logits = h W_out`, the composition
  > mapping hidden → vocab → embedding is `W_out W_in`, which is what `linear`
  > implements (rescaled once so its output matches the mean embedding norm —
  > without that the latent steps are fed near-zero vectors). The two agree up
  > to logit temperature. `soft_embed` sidesteps the ambiguity, which is why it
  > is the default.
- **Latent working memory (§3.2):** each role's layer-wise KV cache is kept and
  the next role generates conditioned on it — lossless in the sense of their
  Theorem 3.3. `share_working_memory: false` ablates it (the smoke test
  confirms the decoder then attends to 7 positions instead of 630).

```bash
python scripts/04_eval.py --methods student,latent --latent-steps 8
```

There is also a COCONUT-style **latent-slot curriculum**
(`latent_slot_curriculum`) that progressively replaces agent blocks with
fixed-width latent slots during SFT — a second, discrete route to
internalisation, useful as an ablation against IMAD's purely reward-driven one.

---

## 10. Agent subspaces (IMAD §3)

Does internalisation preserve the agents, or collapse them?

```bash
python scripts/05_steering.py --compare-base
```

For each agent, build contrastive pairs sharing the same context (question +
debate history up to `<|agent i|>`) with the target agent's real response as
positive and another agent's as negative; take the difference in means at a
chosen layer; normalise. Reports pairwise cosine between the directions — lower
magnitude means more separable subspaces — and the change from base model to
internalised student.

Beyond interpretability this has a concrete use here: steering toward
`divide_conquer` vs `query_plan` at generation time is a cheap diversity knob,
recovering some of CHASE-SQL's multi-path candidate generation from a *single*
model.

---

## 11. Evaluation

Every method emits the same `Prediction` records, so accuracy and cost are
directly comparable. Output is an IMAD-shaped table — accuracy **next to** token
consumption:

| Method | Dataset | n | EX (%) | VES | Avg tokens | Avg output tokens | Tokens vs MAS |
|---|---|---:|---:|---:|---:|---:|---:|

- **EX**: Spider `eval_exec_match` semantics — row multisets compared
  order-insensitively unless the *gold* query has a **top-level** `ORDER BY`
  (an `ORDER BY` inside a sub-query does not count); column order ignored;
  numeric/string forms of the same value unified.
- **VES**: BIRD's `sqrt(t_gold / t_pred)` for correct queries, median of 3 runs.
- **Tokens**: input + output per question, matching IMAD Table 1.

Execution is sandboxed: read-only URI connections, a progress-handler interrupt
for runaway queries, and a guard that refuses any non-read-only statement
(including one smuggled in after a comment or a second statement).

---

## 11b. Run logs, quantization and the pilot

### One JSONL record per unit of work

Every stage writes an append-only log, flushed per record, so a killed run keeps
everything it had produced ([telemetry.py](latentsql/utils/telemetry.py)):

| File | One record per | Carries |
|---|---|---|
| `<run>/traces/teacher_samples.jsonl` | teacher question | latency, prompt/completion tokens, LLM calls, rounds, consensus, correct, trace size, GPU alloc/reserved/peak |
| `<run>/sft/sft_log_steps.jsonl` | optimizer step | loss, lr, tokens seen, elapsed, GPU |
| `<run>/grpo/grpo_log_steps.jsonl` | optimizer step | reward mean, `R_fmt`, `R_clip`, generated length, the `w_fmt`/`l` schedule, SQL executions, GPU |
| `<run>/eval*/eval_samples.jsonl` | question x method | correct, tokens, latency |

Each file opens with a header record naming the GPU, the torch/transformers
versions, the model and the quantization, so a log identifies the run that
produced it.

[`scripts/06_report.py`](scripts/06_report.py) folds all of it into one
`report.md`: accuracy per benchmark, token and latency cost against both the
teacher and each published baseline, and the training curves.

```bash
python scripts/06_report.py --run outputs/pilot
```

### Quantization

`llm.quantization` / `sft.quantization` / `grpo.quantization` take
`none | 4bit | 8bit` (bitsandbytes NF4 + double quant; `4bit` on the student is
QLoRA). On this workstation 4-bit is **faster**, not slower, for the teacher:

| 7B teacher, four-system panel | s / question | GPU |
|---|---:|---:|
| bf16 | 49.6 | 15.7 GB |
| 4-bit NF4 | 37.6 | 5.3 GB |

Single-stream decode is memory-bandwidth bound, so quartering the weight traffic
per token more than pays for bitsandbytes' dequantisation. Expect the ordering
to flip once generation is batched (vLLM), where compute dominates.

### The pilot

[`configs/pilot_mix.yaml`](configs/pilot_mix.yaml) +
[`scripts/run_pilot.sh`](scripts/run_pilot.sh) run the whole pipeline end to end
on real local models, sized for ~1.5 h on one 24GB card:

```bash
N_TRACE=90 N_EVAL=24 bash scripts/run_pilot.sh
```

Five stages, each a separate process so **exactly one model is resident on the
GPU at a time**: 7B teacher builds traces, 3B student takes SFT then GRPO, then
two separate evaluation passes - the four published systems driven by the *same
3B backbone as the student*, and the student itself. Trace building reads the
`train` splits and evaluation reads `dev`, so nothing scored was ever seen by
the teacher. Finished stages are skipped on rerun.

This is a pipeline proof and a log generator, **not a result** - the corpus it
produces is far too small to train on.

### GRPO memory

A GRPO group is `group_size` x (prompt + completion) tokens, and on BIRD the
prompt alone runs to thousands of tokens. Two things made that fit in 24GB:

- the group gradient accumulates in chunks of `grpo.micro_batch` rollouts
  (default 1) instead of one activation graph over the whole group;
- log-probabilities come from `cross_entropy` walked over the sequence in
  512-position blocks. The natural `log_softmax(logits.float())` materialises a
  `[batch, seq, vocab]` fp32 tensor - ~2.4GB per 4k-token sequence at Qwen's
  152k vocabulary, and the transpose forces a second copy. That single line was
  the difference between OOM and a 15GB peak.

Both are exact: forward outputs are identical and gradients agree to 1e-8.

---

## 11c. Pilot run 1 — recorded results (2026-08-23)

First end-to-end run on real local models: 7B teacher -> 3B student, one RTX
4090, 4-bit throughout, ~1.9 h wall-clock. Logs in `outputs/pilot/`, assembled
by [`06_report.py`](scripts/06_report.py).

**Read the caveats in "What this run does not show" before quoting any number.**

### Headline — the internalised 3B student vs. the four systems at the same 3B

24 held-out dev questions (12 BIRD + 12 Spider), same backbone
(`Qwen2.5-Coder-3B-Instruct`), same schema rendering, same executor, same
scorer. 95% Wilson intervals.

| Method | EX (%) | 95% CI | bird | spider | Tokens | Output | Latency | No SQL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single agent | 45.83 | 27.9-64.9 | 25.0 | 66.7 | 1 585 | 180 | 3.6 s | 0 |
| DIN-SQL | 54.17 | 35.1-72.1 | 33.3 | 75.0 | 3 186 | 321 | 6.4 s | 0 |
| **DAIL-SQL** | **62.50** | 42.7-78.8 | 41.7 | 83.3 | 4 697 | 349 | 7.1 s | 0 |
| CHESS | 41.67 | 24.5-61.2 | 16.7 | 66.7 | 7 253 | 1 079 | 26.6 s | 0 |
| CHASE-SQL | 50.00 | 31.4-68.6 | 33.3 | 66.7 | 6 744 | 653 | 13.0 s | 0 |
| **LatentSQL student (ours)** | 54.17 | 35.1-72.1 | 33.3 | 75.0 | 2 097 | 675 | 24.1 s | 2 |

The student ties DIN-SQL at 66% of its tokens and beats CHESS and CHASE-SQL at
~30% of theirs; it loses to DAIL-SQL. Against the 7B teacher panel that produced
its training data (26 710 tokens, 27.0 LLM calls per question) it answers in
**2 097 tokens and 1 call - 7.9% of the teacher's tokens**.

**Every interval overlaps every other.** At n = 24 this table cannot separate any
two methods; it shows the pipeline produces sane numbers, nothing more.

Per-question agreement is the sharper picture: of 24 questions, **6 were solved
by all six methods and 7 by none**, so only 11 discriminate at all. The student
solved **0 questions that no other method solved**, and lost 2 that DAIL-SQL got.
It also emitted 2 unparseable outputs, the only method to do so.

### Teacher

`Qwen2.5-7B-Instruct` 4-bit, panel `[din_sql, dail_sql, chess, chase_sql]`,
2 rounds, 81 train questions, 66.8 min, GPU peak 8.8 GB.

| | n | EX | Consensus | Latency (mean/med/p90) | Tokens (mean/med/p90) | Calls |
|---|---:|---:|---:|---:|---:|---:|
| all | 81 | 28.40% | 86.4% | 49.5 / 40.5 / 81.2 s | 26 710 / 23 148 / 48 759 | 27.0 |
| bird | 42 | **54.76%** | 76.2% | 63.5 / 47.1 / 116.3 s | 30 550 / 25 748 / 56 896 | 30.8 |
| spider | 39 | **0.00%** | 97.4% | 34.4 / 28.2 / 55.2 s | 22 574 / 19 388 / 34 661 | 22.8 |

Spider's 0.00% is the loader bug described below, not a property of Spider.
Filtering kept **19/81 (23.5%)**: 51 dropped incorrect, 11 no-consensus, 0 for
any other reason - i.e. yield is bounded almost entirely by teacher accuracy.
That is **~17 usable traces per GPU-hour** at this configuration.

### Student training

SFT: 19 traces, 14 steps, 35 s, 113k tokens, peak 8.9 GB, loss 0.63 -> 0.32.

GRPO: 9 steps, 14.5 min, 50 773 tokens generated, 152 real SQL executions,
peak 6.2 GB.

| step | w_fmt | clip | reward | R_fmt | R_clip | gen_len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.881 | 838 | 0.700 | 0.375 | 0.250 | 653 |
| 3 | 0.644 | 712 | 0.529 | 0.562 | 0.000 | 671 |
| 5 | 0.406 | 588 | 0.613 | 0.500 | 0.250 | 681 |
| 7 | 0.169 | 462 | 0.640 | 0.312 | 0.500 | 796 |
| 9 | 0.050 | 400 | 0.309 | 0.188 | 0.250 | 746 |

The schedule ran correctly - `w_fmt` annealed 0.88 -> 0.05 and the clip 838 ->
400 - but **generated length did not fall** (654 -> 746, no trend) and reward did
not rise. The GRPO student scores identically to the SFT-only student
(54.17%, 2 075 tokens). At 97 s/step, the several hundred steps this needs is
~13.5 h, not 15 minutes.

### What this run does *not* show

1. **No internalisation happened.** 9 optimizer steps on 19 traces at lr 1e-6.
   The mechanism is verified to run; it was not trained.
2. **The corpus is BIRD-only.** The teacher ran before the Spider fix, so all 19
   traces are BIRD. The student still scores 75% on Spider dev, which is
   cross-benchmark transfer, but it is not the mixed corpus the config asks for.
3. **n = 24 and 19 traces.** Nothing here is statistically meaningful.

### Two bugs this run surfaced

**Spider gold SQL was never loaded.** Canonical Spider `dev.json` stores the
*parsed* query tree under `sql` and the query *string* under `query`; the field
probe took `sql` first and stringified the dict, so every Spider gold was a
Python repr that cannot execute and every Spider question scored wrong -
silently. It showed up only as Spider reading 0.00% against BIRD's 54.76%, when
Spider is the *easier* benchmark. Fixed in `_get`, which now skips containers;
regression test in `tests/test_core.py`.

**GRPO OOMed on a 24GB card**, three compounding causes:

| Cause | Fix | Peak |
|---|---|---:|
| whole rollout group in one activation graph | accumulate per rollout (`grpo.micro_batch`) | 23.3 GB (OOM) -> 15.0 |
| `log_softmax(logits.float())` = fp32 `[1, 4k, 152k]` | time-chunked `cross_entropy` | -> 13.6 |
| **gradient checkpointing never wired into GRPO** | `gradient_checkpointing_enable()` + `enable_input_require_grads()` | -> **4.6** |

All three are exact: forward outputs identical, gradients agree to 1e-8.

---

## 11d. Pilot run 2 — the Spider fix, and the first sign GRPO works

Same pipeline, same seed, same 24 held-out questions; the only change upstream
is that Spider gold SQL now loads correctly. `outputs/pilot2/`.

| | run 1 | run 2 |
|---|---:|---:|
| teacher EX | 28.4% | **63.0%** |
| traces kept | 19/81 (23.5%) | **46/81 (56.8%)** |
| SFT final loss | 0.32 | **0.15** |
| GRPO steps with a live gradient | 4/9 (44%) | **13/15 (87%)** |
| SFT-only student | 54.17% | 41.67% |
| **GRPO student** | 54.17% (Δ 0.0) | 50.00% (**Δ +8.3**) |

**The result that matters: GRPO produced +8.3 EX in run 2, all of it on Spider
(58.3 → 75.0).** That is the first evidence here that stage 2 does anything at
all, and it appeared exactly when the gradient stopped being degenerate. In run
1, with 44% dead steps, GRPO moved the weights by 6e-04 and changed nothing.

The absolute number is still below run 1 because run 2's *SFT* is worse
(41.67%), despite 2.4x the traces and half the loss - it fits the transcripts
tightly and emits 993 output tokens against run 1's 653. Both runs sit inside
each other's confidence intervals at n = 24.

### Two failure modes the new diagnostics caught

`grpo_log_steps.jsonl` now records `reward_std_mean` and `adv_nonzero_frac`.
They exposed that a group can die from *either* end:

| step | R_clip | adv_nonzero_frac | what happened |
|---:|---:|---:|---|
| 8 | 1.00 | **0.00** | every rollout correct *and* identically formatted - no spread |
| 9 | 1.00 | **0.00** | fully saturated on every term |
| 12 | 0.00 | 1.00 | every rollout wrong; rescued only by a weak R_fmt spread (std 0.05) |

Run 1 died of all-wrong groups; run 2 also hits all-*correct* ones. Both give
zero gradient. This exposes a real gap: **nothing in the reward distinguishes
two correct rollouts of different lengths** - both score `R_clip = 1`, so GRPO
cannot learn "be correct *and* shorter" from a saturated group, which is the
entire internalisation objective. IMAD's length pressure only bites when the
answer appears *late*; here the student emits the query early and keeps writing.

### The student does not reliably close the debate

Re-evaluating with `--max-new-tokens 3072` instead of 1024:

| budget | EX | hit cap | reached `<\|consensus\|>` |
|---|---:|---:|---:|
| 1024 | 50.00% | 17/24 | 1/24 |
| 3072 | 50.00% | 2/24 | 11/24 |

Accuracy is *identical*. Truncation was not costing accuracy - a tempting
hypothesis, tested and refuted (questions with short traces are simply the
easier ones). What it does show is that even with ample room the student emits
`<|consensus|>` on only 11 of 24 questions, so the fallback extractor is
scraping a mid-debate query half the time. That is a format-adherence failure,
not a budget failure.

---

## 11e. A full worked example — real inference output, end to end

Everything below is copied verbatim from an actual completed run, not written
for illustration. Source: `outputs/imad_boost/eval/preds_student.jsonl`,
record `qid: spider_dev_430`, checkpoint `outputs/imad_boost/grpo` (**Run 5**,
`RESULTS.md` §5d - text-mode LatentSQL + speculative decoding, 62/252 GRPO
steps, 53.33% EX on n = 90). Note: speculative decoding accelerated the
*rollout generation during GRPO training* for this checkpoint; the eval run
that produced this transcript used plain, unaccelerated `model.generate()`
- see [§8](#8-stage-2--grpo-internalisation-imad-23) for where speculative
decoding actually sits in the pipeline.

### What actually happens at inference (recap)

Per the step-by-step explanation given during this project's development: the
question + schema are rendered into one prompt, tokenized once, and then
LatentSQL decodes **token by token, autoregressively** - a forward pass
produces logits, one token is picked (greedy at eval time, `temperature: 0.0`),
appended, and the cycle repeats until an end token or the length cap. Nothing
below is templated or hand-assembled; it is the literal sequence of tokens the
trained weights produced, one at a time, in that loop.

### The input

| | |
|---|---|
| Dataset | Spider dev, db `wta_1` |
| Question | *"How many players are there?"* |
| Gold SQL | `SELECT count(*) FROM players` |
| Prompt tokens | 1,232 |

### The full raw output (every token LatentSQL generated - the "thinking text")

```text
<|schema_link|>
Tables: players
Columns: players.player_id
Conditions: none
Values: none
<|/schema_link|>
<|classify|> EASY <|/classify|>
<|round 1|>
<|agent din_sql|>
1. The table `players` holds the requested information.
2. Scan the `players` table to count the number of rows.
3. Return the count as the result.
```sql
SELECT COUNT(*) FROM players
```
<|exec|> COUNT(*)
(100000,) <|/exec|>
<|/agent|>
<|agent dail_sql|>
This question is asking for the total number of records in the `players` table.
```sql
SELECT COUNT(*) FROM players
```
<|exec|> COUNT(*)
(100000,) <|/exec|>
<|/agent|>
<|agent chess|>
1. The table `players` holds the requested information.
2. Scan the `players` table to count the number of rows.
3. Return the count as the result.
```sql
SELECT COUNT(*) FROM players
```
<|exec|> COUNT(*)
(100000,) <|/exec|>
<|/agent|>
<|agent chase_sql|>
DIVIDE:
1. Count the number of players.

CONQUER:
1. SELECT COUNT(*) FROM players

ASSEMBLE:

OPTIMIZE:
```sql
SELECT COUNT(*) FROM players
```
<|exec|> COUNT(*)
(100000,) <|/exec|>
<|/agent|>
<|/round|>
<|round 2|>
<|agent din_sql|>
My initial query was correct and produced the right result. No changes are needed.
```sql
SELECT COUNT(*) FROM players
```
<|exec|> COUNT(*)
(100000,) <|/exec|>
<|/agent|>
<|agent dail_sql|>
My initial query was correct and produced the right result. No changes are needed.
```sql
SELECT COUNT(*) FROM players
```
<|exec|> COUNT(*)
(100000,) <|/exec|>
<|/agent|>
<|agent chess|>
My initial query was correct and produced the right result. No changes are needed.
```sql
SELECT COUNT(*) FROM players
```
<|exec|> COUNT(*)
(100000,) <|/exec|>
<|/agent|>
<|agent chase_sql|>
My initial query was correct and produced the right result. No changes are needed.
```sql
SELECT COUNT(*) FROM players
```
<|exec|> COUNT(*)
(100000,) <|/exec|>
<|/agent|>
<|/round|>
<|consensus|>
```sql
SELECT COUNT(*) FROM players
```
<|/consensus|>
<|endofdebate|>
```

### Extraction, execution, and the verdict

| | |
|---|---|
| Extracted SQL (`extract_final_sql`, pulls the last fenced block after `<\|consensus\|>`) | `SELECT COUNT(*) FROM players` |
| Executed against `wta_1.sqlite` | `(100000,)` |
| Gold executed | `(100000,)` |
| **EX verdict** | **correct** |
| Completion tokens | 645 |
| Latency | 20.89 s |

### Analysis

**What this demonstrates working.** The Run 5 checkpoint faithfully
reproduces all four teacher systems' *distinct reasoning styles* from one set
of weights: DIN-SQL's and CHESS's numbered decomposition, DAIL-SQL's short
direct justification, and CHASE-SQL's DIVIDE/CONQUER/ASSEMBLE/OPTIMIZE
framework are all present, in the correct order, with no cross-contamination
between agents' voices. All four independently reach the same correct query
in round 1; round 2 has every agent explicitly self-critique and confirm "no
changes are needed" rather than skip the step - LatentSQL learned the
debate's *shape*, not just its final answer. The `<|consensus|>` block and
`<|endofdebate|>` marker both fire cleanly, so the extractor did not need to
fall back to scraping a mid-debate query here.

**What this demonstrates NOT working - internalisation.** This is a one-line
`COUNT(*)` query that a single forward pass could answer directly, yet the
model still writes a full two-round, four-agent, 645-token transcript to get
there - the exact failure mode `RESULTS.md` §0.1 and §5.2 document: **EX
holds, but output tokens never collapse.** A genuinely internalised LatentSQL
would emit something close to the extracted SQL alone (a handful of tokens)
instead of re-deriving and re-confirming it eight times. This checkpoint's
compression anneal had only completed ~25% of its schedule when the 10-hour
training window ran out (`RESULTS.md` §5d), which is the direct, measured
reason the collapse has not happened yet - not evidence it cannot happen.

---

## 12. Data

Expected layout (auto-detected; vanilla BIRD/Spider layouts also work):

```
<bird_root>/dev/dev.json          + dev/dev_databases/<db_id>/<db_id>.sqlite
<bird_root>/train/train.json      + train/train_databases/<db_id>/<db_id>.sqlite
<spider_root>/dev.json            + database/<db_id>/<db_id>.sqlite
<spider_root>/train_spider.json   + database/<db_id>/<db_id>.sqlite
```

Override with `paths.bird_root` / `paths.spider_root` or
`$LATENTSQL_BIRD_ROOT` / `$LATENTSQL_SPIDER_ROOT`. The database root is chosen
per split, so `split: train` reads `train_databases` rather than silently
resolving train questions against the dev databases.

> **On this workstation:** `/mnt/win_d/data_FusionSQL/bird` contains only
> unfetched **git-lfs pointers** — 131-byte text files where the manifests and
> `.sqlite` files should be. The complete copy is
> `/mnt/win_d/data/sft_data_collections/{bird,spider}` (BIRD 1 534 dev / 9 428
> train, Spider 1 034 dev / 8 659 train), which is what `mix_local.yaml`
> points at. Manifest probing now rejects LFS pointers and zero-byte files
> instead of failing three frames deep in `json.load`.

### The combined BIRD + Spider set

`dataset: mix` (also `bird+spider`) builds one set from both benchmarks:

```yaml
mix:
  bird: 1500
  spider: 1500
  bird_split: train
  spider_split: train
```

Each half is sampled independently under `seed` and the two are **interleaved
by ratio**, so any prefix holds the configured mixture — `--limit 200` on a
50/50 mix gives 100 + 100, not 200 BIRD. Every example keeps its `dataset`
field through prediction and scoring, so `04_eval.py` reports EX per benchmark
next to the blended number:

```
| Method | Dataset | n | EX (%) | EX bird (%) | EX spider (%) | VES | ... |
```

That split matters: BIRD and Spider differ sharply in difficulty, only BIRD
carries the `evidence` hint, and VES is a BIRD metric — so it is averaged over
the BIRD rows only. A single combined EX number is not comparable to any
published result on either benchmark; report the per-benchmark columns.

DAIL-SQL few-shot examples are drawn **only from the teacher-train split**;
`split_train_dev` holds out a deterministic slice written to `heldout.jsonl`,
which `04_eval.py --heldout` consumes. Nothing evaluated is ever in the shot pool.

If neither benchmark is present, everything falls back to the synthetic fixture
(30 questions, 2 databases) and **says so in the output**, so a fallback can
never be mistaken for a real result.

---

## 13. Smoke-test results

Recorded run, this machine: **Windows 11, CPU-only, torch 2.13.0+cpu,
transformers 5.15.1, peft 0.20.0**, tier-2 model
`hf-internal-testing/tiny-random-LlamaForCausalLM` (auto-downloaded; a local hub
model is preferred when one is present). Raw logs are committed at
[outputs/smoke_test_output.txt](outputs/smoke_test_output.txt) and
[outputs/unit_test_output.txt](outputs/unit_test_output.txt).

```
$ python scripts/smoke_test.py --tier 2 --n 8
28/28 checks passed

$ python tests/test_core.py
Ran 25 tests ... OK
```

### Tier 1 — data, database layer (no torch)

| Check | Result |
|---|---|
| synthetic benchmark builds | 30 questions, 2 databases |
| gold queries all execute | 30/30 ok |
| execution accuracy metric is sound | match / mismatch / order-sensitivity all correct |
| write statements are refused | `DROP` rejected |
| schema introspection + `CR^p` rendering | 4 tables, 17 cols, 3 FKs |
| LSH value index retrieves cell values | exact + fuzzy hits (`singer.country = 'France'`) |

### Tier 1 — teacher multi-agent system

| Check | Result |
|---|---|
| FusionSQL teacher produces a labelled trace | EX=8/8, agents=`[dail_icl, divide_conquer, query_plan]`, max_rounds=2 |
| debate beats its own panellists | debate=8/8 vs best single agent=8/8 |
| trace renders, parses and round-trips through jsonl | 1720 chars, 2 rounds, 6 turns |
| SFT corpus filtering | 8/8 kept (no_consensus=0, incorrect=0, no_sql=0, exec_failed=0, too_long=0, duplicate=0) |
| all four baselines run | `single:5/8@462tok` · `din_sql:7/8@1250tok` · `dail_sql:7/8@2323tok` · `chess:8/8@2803tok` · `chase_sql:8/8@2719tok` |

### Tier 1 — IMAD reward function

| Check | Result |
|---|---|
| reward schedule anneals `w_fmt` and `l` | `w_fmt` 1.00→0.05, `l` 2000→500 |
| `R_fmt` + `R(y;l)` behave as IMAD eq. 1–2 | full=2.200, internalised=1.225, late_full=1.250, wrong=0.225 |
| length clipping actually clips | verbose trace earns 0 under a 12-token clip |
| GRPO group advantages normalise within a group | `[1.732, -0.577, -0.577, -0.577]` |

The reward row is the one worth reading closely, because it is the mechanism
of the whole method. At **step 0** the full transcript scores 2.200 and a bare
answer would score less — the model is paid to verbalise the debate. By the
**last step** the same full transcript is down to 1.250 while the internalised
bare answer holds 1.225, and a *wrong* bare answer collapses to 0.225. So the
gradient of the reward has flipped from "show the debate" to "be right, briefly"
— exactly IMAD §2.3 — and correctness, not brevity, is what carries the weight.

### Tier 2 — training, latent, steering (torch)

| Check | Result |
|---|---|
| student loads with LoRA attached | 1.1M params, 24K trainable |
| prompt tokens are masked out of the SFT loss | 653 prompt tokens masked; full-trace supervises 371 target tokens vs 36 for consensus-only |
| SFT runs and the loss is finite | 2 steps, loss=10.373 |
| GRPO takes a real optimizer step | 2 steps, 3 SQL executions, `l`=8 |
| GRPO updates parameters under reward variance | 28/28 LoRA tensors updated |
| input–output aligners map hidden→embedding space | `soft_embed` ‖e‖=0.02, `linear` ‖e‖=0.06, `identity` ‖e‖=3.18 (embedding reference 0.08) |
| latent roles run, only the final query is decoded | 6 latent steps emitted **0** tokens; 12 tokens decoded |
| latent working memory carries earlier roles forward | 684 attended positions shared vs 7 isolated, same 6 latent steps |
| latent-slot curriculum replaces agent blocks | 1720 → 1497 → 1185 chars |
| contrastive pairs are built from the traces | 8 pairs, context ends at the target agent tag |
| steering vectors extract and are separable | 3 unit vectors; pairwise cos `dail_icl\|divide_conquer=-0.51`, `dail_icl\|query_plan=+0.51`, `divide_conquer\|query_plan=-1.00` |
| steering hook changes generation | steering altered greedy decoding |
| student evaluation produces scored predictions | n=2, EX=0%, avg_tokens=672 |

Three of these rows deserve a caveat rather than a celebration:

- **`debate beats its own panellists` was 8/8 vs 8/8** — a tie, not a win. On 8
  synthetic questions with a mock teacher the panellists already saturate, so
  this check can only catch a *regression* (debate scoring below its best
  member), not demonstrate a gain. The real comparison is §11 on BIRD.
- **`GRPO takes a real optimizer step` reports `params_changed=False`.** That is
  correct: a random-init model earns reward 0 on every rollout, so all
  advantages are 0 and nothing should backpropagate. The separate
  `GRPO updates parameters under reward variance` check injects reward variance
  precisely so the gradient path is actually exercised — it moves 28/28 tensors.
- **`student evaluation` reports EX=0%.** Expected and meaningless: the tier-2
  model is randomly initialised. The check verifies the evaluation path
  (prompting, decoding, SQL extraction, scoring, token accounting), not accuracy.

`identity` alignment produces vectors ~40× the true embedding norm (3.18 vs
0.08), which is the out-of-distribution failure LatentMAS §3.1 warns about, and
why `soft_embed` is the default.

### What this does *not* establish

No accuracy claim. The teacher is a mock, the benchmark is a 30-question
fixture, and the tier-2 model is random-init. What passes is that every
component runs, composes and computes what it says it computes.

---

## 14. Required ablations

[NOVELTY_SEARCH.md](NOVELTY_SEARCH.md) surveys the neighbouring literature and
concludes the composition is unoccupied — but it also names three ablations that
a reviewer *will* demand, because a nearby paper already makes the competing
claim. All three are one flag or one config line.

| Ablation | Command | Answers the objection |
|---|---|---|
| **Full trace vs. final answer only** | `02_sft.py --consensus-only` | "is the debate transcript doing anything, or just the answers?" (DebateGPT, IMAD Table 1) |
| **Debate panel vs. one structured format** | `mas.agents: [query_plan]` | "Struct-SQL shows a single execution-plan CoT is enough" ([arXiv:2512.17053](https://arxiv.org/pdf/2512.17053)) |
| **GRPO from base, no teacher at all** | `03_grpo.py --from-examples bird --sft /nonexistent` | "Arctic-Text2SQL-R1 hits 68.9% BIRD with only an execution reward — is the teacher earning its cost?" ([arXiv:2505.20315](https://arxiv.org/pdf/2505.20315)) |

The third is the uncomfortable one and must be reported even if unflattering.
The honest bar for this project is not "beats the explicit MAS" — it is
**Arctic-Text2SQL-R1-7B at 68.9% BIRD dev** and **SLM-SQL-1.5B at 67.1%**.

---

## 15. Status and honest limits

**Verified on this machine:** all 28 smoke-test checks and all 25 unit tests
pass, CPU-only — see [§13](#13-smoke-test-results) for the recorded run. That
covers the teacher, the four baselines, trace serialisation, every reward term
and schedule, SFT (including loss masking and the DebateGPT ablation), a GRPO
optimizer step *and* the gradient path under non-zero advantage, all three
latent aligners, latent working memory, CAA extraction and steering hooks, and
the student evaluation path.

**Not yet run:** the real experiment. No BIRD/Spider results are reported here
because this machine has neither the datasets (`/mnt/win_d` does not exist on
it) nor the models nor a GPU. The table in §11 is a schema, not results. Nothing
in this repository should be cited as a number until `04_eval.py` has been run
on the workstation.

**Known limitations:**

- The teacher's quality bounds the student's. With a 7B local teacher, expect
  weaker traces than the GPT-4-class teachers in the original papers — the
  execution-based trace filter compensates by discarding wrong traces, but it
  cannot manufacture correct ones. If teacher EX is low, use
  `configs/teacher_openai.yaml` for trace generation only; the student stays local.
- GRPO rollouts are the bottleneck: `group_size` × prompts generations per step,
  each scored by executing SQL. Budget accordingly, and note that the
  `ExecutionOracle` cache is what makes it tractable.
- `latent` mode decodes token-by-token in Python (no KV-cache fast path through
  `generate`), so it is slower per token than `student` mode even though it
  emits far fewer tokens. Wall-clock comparisons should say which mode was used.
- The mock backend's numbers are plumbing checks, never results.

---

## 16. References

Run `scripts/fetch_papers.sh` to download all six into `pdfs/` (they are not
committed — copyright and repo size).

- Yi, Mueller, Lee. *Latent Agents: A Post-Training Procedure for Internalized
  Multi-Agent Debate.* ACL 2026. [aclanthology.org/2026.acl-long.709](https://aclanthology.org/2026.acl-long.709/)
- Zou et al. *Latent Collaboration in Multi-Agent Systems (LatentMAS).* ICML 2026. [arXiv:2511.20639](https://arxiv.org/abs/2511.20639)
- Pourreza & Rafiei. *DIN-SQL.* NeurIPS 2023. [arXiv:2304.11015](https://arxiv.org/abs/2304.11015)
- Gao et al. *Text-to-SQL Empowered by LLMs (DAIL-SQL).* VLDB 2024. [arXiv:2308.15363](https://arxiv.org/abs/2308.15363)
- Talaei et al. *CHESS.* 2024. [arXiv:2405.16755](https://arxiv.org/abs/2405.16755)
- Pourreza et al. *CHASE-SQL.* ICLR 2025. [arXiv:2410.01943](https://arxiv.org/abs/2410.01943)
- Du et al. *Improving Factuality and Reasoning via Multiagent Debate.* 2023.
- Shao et al. *DeepSeekMath (GRPO).* 2024.
- Rimsky et al. *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024.
