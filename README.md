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

## 12. Data

Expected layout (auto-detected; vanilla BIRD/Spider layouts also work):

```
/mnt/win_d/data_FusionSQL/bird/sft_bird_dev_text2sql.json
/mnt/win_d/data_FusionSQL/bird/dev/dev_databases/<db_id>/<db_id>.sqlite
/mnt/win_d/data_FusionSQL/spider/sft_spider_dev_text2sql.json
/mnt/win_d/data_FusionSQL/spider/database/<db_id>/<db_id>.sqlite
```

Override with `paths.bird_root` / `paths.spider_root` or
`$LATENTSQL_BIRD_ROOT` / `$LATENTSQL_SPIDER_ROOT`.

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
