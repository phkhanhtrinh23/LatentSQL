# Novelty search — does this idea already exist?

**Question asked:** has anyone already internalised a multi-agent Text-to-SQL
system (DIN-SQL / DAIL-SQL / CHESS / CHASE-SQL) into a single sub-7B model that
runs the debate in latent space?

**Short answer:** **No exact match found.** Every *ingredient* exists
separately, and two 2025–2026 Text-to-SQL papers come uncomfortably close on
adjacent axes. The specific composition appears unoccupied — but the novelty is
a composition, not a new primitive, and §5 is honest about what that means for
a thesis defence.

**Specifically on latent reasoning for Text-to-SQL** (re-asked and
re-searched 24 Aug 2026, ~30 queries total across both passes): no published
paper does continuous/activation-space reasoning — single-model or
multi-agent — for SQL *generation*. The closest thing found, LatentRefusal
(§2.6b), reads hidden states for a binary refusal *decision*, not generation.
See §2.8 for the full second-pass account, including five more multi-agent
latent-collaboration papers (the LatentMAS family has grown since Aug 2026)
and four single-model latent/implicit-reasoning papers, none of which
evaluates SQL.

*First pass 22 Aug 2026, adversarial second pass 24 Aug 2026. Method in §7.
This is a literature search, not a proof of absence — see §6 for what it
could not rule out.*

---

## 1. The claim being checked

| # | Component | Present in our design |
|---|---|---|
| C1 | Teacher = **fused** DIN-SQL + DAIL-SQL + CHESS + CHASE-SQL multi-agent system | yes |
| C2 | Distil the **whole debate transcript** (not just the final answer) into one model | yes |
| C3 | **Internalise** it: GRPO with decaying format reward + shrinking length clip | yes |
| C4 | Reward = **execution match** against gold SQL, not string match | yes |
| C5 | Student is a **sub-7B local** model | yes |
| C6 | **Latent** agent roles at inference (LatentMAS thoughts + KV working memory) | yes |
| C7 | **Agent-subspace steering** as a candidate-diversity mechanism for SQL | yes |

Nothing found combines more than **three** of these.

---

## 2. Nearest neighbours, and exactly where they diverge

Ordered by how close they get.

### 2.1 MATS — *A Multi-agent Text2SQL Framework using Small Language Models and Execution Feedback*
[arXiv:2512.18622](https://www.arxiv.org/pdf/2512.18622) (Hoang et al., Griffith / VinUniversity / HU Berlin / UQ, Dec 2025) · [code](https://github.com/thanhdath/mats-sql)

**The single most important related work.** Same motivation almost word for
word: privacy and cost rule out API LLMs, so use in-house SLMs (100M–5B); SLMs
are individually too weak for Text-to-SQL; therefore decompose into specialised
agents. Five agents — schema investigator, query planner, validator, fix agent,
selection agent — trained collaboratively with *Reinforcement Learning with
Execution Feedback*.

**Why it is not our idea:** MATS **keeps the multi-agent system at inference**.
Five SLM agents still run, still exchange text, still pay the token cost. It is
"make the MAS cheap by shrinking each agent", where we are "delete the MAS by
absorbing it into one agent". Concretely: has C1(partly), C4, C5. Lacks **C2,
C3, C6, C7** — no transcript distillation, no internalisation, no latent
communication, no token-reduction claim.

> **Action required:** this must be cited in the related-work section and
> explicitly contrasted, and ideally run as a baseline. Its agent decomposition
> is close enough to our teacher panel that a reviewer will ask.

### 2.2 Struct-SQL — *Knowledge Distillation with Structured Chain-of-Thought for Text-to-SQL*
[arXiv:2512.17053v3](https://arxiv.org/pdf/2512.17053) (Thaker & Bresler, Crater Labs, Mar 2026)

Distils reasoning from a large LLM into an SLM using the **query execution plan
as a formal blueprint** for structured CoT. Reports +8.1% absolute over
unstructured-CoT distillation, largely from fewer syntax errors.

**Overlap to be careful about:** their structured reasoning format is
essentially our `query_plan` agent (CHASE-SQL's Query Plan CoT). They show it is
a *better distillation target* than free-form CoT.

**Divergence:** single teacher, **no debate**, one reasoning format rather than
several competing ones, explicit text reasoning at inference, no RL, no
internalisation. Has C5 and a slice of C2. Lacks C1, C3, C4, C6, C7.

**Implication for us:** their result is evidence *for* our design (structure
helps distillation), and it also raises the bar — we should ablate
"query-plan-only distillation" to show the multi-perspective debate beats a
single structured format. That ablation is one config change
(`mas.agents: [query_plan]`).

### 2.3 SERL-SQL — *Selective Hindsight Distillation for Text-to-SQL Reinforcement Agentic Learning*
[arXiv:2608.00485](https://arxiv.org/html/2608.00485) (Aug 2026)

Closest on the **RL axis**. On-policy SQL trajectories; a training-only teacher
re-scores student actions with execution feedback as privileged hindsight; the
teacher–student likelihood gap becomes bounded masked weights that reweight
**GRPO advantages** on SQL and tool-action tokens. Evaluated on BIRD and Spider.

**Shares:** GRPO + execution grounding + BIRD/Spider + a teacher (C4, C5).
**Lacks:** multi-agent debate as the distillation source, format-reward decay,
length annealing, latent reasoning (C1, C2, C3, C6, C7). Their teacher supplies
*credit assignment*; ours supplies *a transcript to internalise*.

### 2.4 MAD-OPD — *Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate*
[arXiv:2605.01347](https://arxiv.org/html/2605.01347v1) (May 2026)

Closest on the **debate-distillation axis**. Multiple teachers debate over the
*student's own* sampled outputs; the debate transcript is privileged context
visible to teachers but not the student. A 4B student under 14B+8B teacher
debate beat the 14B teacher on code.

**Divergence:** the debate is a *teacher-side scoring device*, never something
the student learns to reproduce and then compress. Reasoning stays explicit
text; there is no token-reduction or latent objective. Tasks are agentic tool
use (BFCL-v4, τ²-Bench, VitaBench) and code (LiveCodeBench, MBPP+) —
**not Text-to-SQL**. Has C2(inverted), C5. Lacks C1, C3, C4, C6, C7.

### 2.5 IMAD / Latent Agents — the method we are porting
[ACL 2026](https://aclanthology.org/2026.acl-long.709/) · [arXiv:2604.24881](https://arxiv.org/abs/2604.24881) (Yi, Mueller, Lee; Boston University)

Two-stage SFT+GRPO internalisation with dynamic reward scheduling and length
clipping; matches or exceeds explicit debate with up to 93% fewer tokens;
activation steering reveals agent-specific subspaces.

**What is missing from it, and is therefore ours to add:** trained on
**arithmetic**, evaluated on GSM8K / MMLU-Pro / BBH. No Text-to-SQL, no
database, no schema linking, no executor. Its `R(y; l)` is a **string/answer
match**; there is no verifiable-execution reward and no way to filter traces by
correctness against a database. The agents are generic personas (CoT /
self-critique / PoT), not SQL-specialised roles from published systems.

### 2.6 LatentMAS — the latent mechanism we are borrowing
[ICML 2026 Spotlight](https://arxiv.org/html/2511.20639v1) · [code](https://github.com/Gen-Verse/LatentMAS)

Training-free latent collaboration: last-layer hidden states as latent thoughts,
KV-cache working memory transferred between agents. 70.8–83.7% fewer output
tokens, 4–4.3× faster.

**Two clean gaps.** (i) Its nine benchmarks are GSM8K, AIME24, AIME25,
GPQA-Diamond, MedQA, ARC-Easy, ARC-Challenge, MBPP-Plus, HumanEval-Plus —
verified from the PDF, **no SQL or database task**. (ii) It runs **N separate
models** at inference; it is not a distillation method at all. Applying its two
mechanisms *inside a single internalised student* is not something the paper
does or claims.

### 2.6b LatentRefusal — convergent evidence for "read hidden states, not tokens", from an adjacent Text-to-SQL problem
[arXiv:2601.10398v2](https://arxiv.org/pdf/2601.10398v2) (Ren, Hu, Lu, Huang; Fudan University, with Duan; Penn State Abington. Jan 2026)

Not a competitor to our claim - a **refusal gate**, not a generator. Given a
question + schema, it predicts *answerability* (should the system refuse
before producing any SQL at all) from a single frozen intermediate-layer
hidden state, via a small trained probe: the Tri-Residual Gated Encoder
(TRGE) - a 4-layer, 19M-parameter (<0.3% of an 8B backbone) Transformer-style
probe whose distinguishing feature is a third, SwiGLU-gated residual branch
that suppresses schema-token noise and amplifies the sparse tokens that carry
a question/schema mismatch. The base LLM (Qwen3-8B, Llama-3.1-8B) is entirely
frozen; only the probe is trained, on ~300 labelled examples, in ~10 minutes
on one A100. Result: 88.5% avg F1 across four answerability benchmarks
(TriageSQL, AMBROSIA, SQuAD 2.0, MD-Enterprise), beating both output-sampling
uncertainty methods (Semantic Entropy, Self-evaluation) and other
internal-state probes (CCS, Eigenscore, HaloScope, SAPLMA, TSV) - at **2ms**
probe overhead versus Semantic Entropy's 740ms (13.7x). Ablations: removing
the SwiGLU gate costs -4.1 F1 (MLP) to -16.7 F1 (linear probe); the best
signal sits at layer -16 of the backbone (middle-to-late), not the final
layer; probe depth peaks at 4 layers with diminishing returns beyond.

**Why it doesn't compete, and what it is worth to us anyway.** None of C1-C4
or C7 apply: single frozen LLM, no MAS teacher, no distilled transcript, no
RL/internalisation of anything, no execution reward, no agent personas. It
solves a different problem (safety gating for genuinely unanswerable/ambiguous
queries) on different benchmarks, and BIRD/Spider are not built to be
ambiguous, so it is not a target we need to beat. But it is independent
empirical support, in the *same problem domain*, for the premise our entire
latent-rollout mechanism rests on: that a transformer's intermediate hidden
states carry decision-relevant signal *before* any token is decoded, and that
a single forward pass without decoding is enough to extract it. We use that
premise for generation (`LatentReasoner`); they use it for a binary gate. Two
things are directly actionable for us:

1. **Layer choice.** Their signal peaks at layer -16 of ~32, i.e.
   middle-to-late but not final. Our `steering.layer` defaults to `-8`
   (IMAD's config, tuned for arithmetic, not SQL) - worth an ablation on our
   own agent-subspace extraction now that a second, independent Text-to-SQL
   result points away from the very last layers.
2. **A learned, gated aligner.** `LatentReasoner.align()` currently maps a
   role's hidden state back into embedding space with a *fixed, untrained*
   `soft_embed` transform. TRGE's central result is that a small
   input-conditioned SwiGLU gate substantially outperforms both a linear
   probe and a vanilla MLP at exactly this kind of task - extracting a sparse,
   task-relevant signal from a hidden state that is otherwise dominated by
   irrelevant (there: schema; here: prompt) content. A trained gated aligner
   in place of the fixed `soft_embed` map is a concrete, cheap experiment: it
   targets precisely the risk LatentMAS §3.1 flags for raw hidden-state
   reuse (out-of-distribution drift), by *learning* what to keep rather than
   relying on a fixed projection.

Also worth naming as a possible later extension, not a gap we need to close
now: a LatentRefusal-style gate sitting on top of our internalised student -
refuse before decoding when the latent debate itself shows internal
disagreement - would be a natural safety layer, since BIRD/Spider gold labels
already carry a `difficulty` field that could stand in for an answerability
signal in a pilot.

### 2.7 The SLM Text-to-SQL performance bar (not competing methods, but the numbers to beat)

| System | Size | BIRD dev EX | Approach |
|---|---|---|---|
| [Arctic-Text2SQL-R1](https://arxiv.org/pdf/2505.20315) | 7B | **68.9%** | RL, reward = execution correctness + syntax validity |
| [OmniSQL](https://github.com/HKUSTDial/NL2SQL_Handbook) | 7B | 68.9% | domain pre-training + large synthetic data |
| [SLM-SQL](https://arxiv.org/html/2507.22478v1) | 1.5B | 67.1% | SLM-focused fine-tuning + RL |
| SLM-SQL | 0.6B | 59.5% | " |
| CodeS | 15B | 58.5% | SQL-specific pre-training |

None distils a multi-agent system; none uses latent reasoning. **But this is the
honest bar.** A 3B internalised student that beats "explicit MAS" but lands
below Arctic-Text2SQL-R1-7B has not made the case. Arctic in particular is
awkward for us: it reaches 68.9% with *only* an execution-correctness reward and
no debate at all, which is close to our GRPO stage minus the whole teacher.

### 2.8 Latent-reasoning background (none Text-to-SQL) — re-verified 24 Aug 2026, deeper pass

[COCONUT](https://arxiv.org/abs/2412.06769) (hidden state fed back as next input
embedding), CODI (self-distillation into continuous space), SoftCoT, CoLaR,
iCoT. Our `latent_slot_curriculum` is COCONUT-style and is credited as such.
The original survey sweep of the [Awesome-Latent-CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT)
list found no Text-to-SQL entry (22 Aug 2026).

**A second, deliberately adversarial pass (24 Aug 2026, ~20 additional queries,
~14 papers fetched and read past the abstract) found no reason to revise that
conclusion — if anything it strengthens it,** because the pass specifically
targeted the two ways a paper could be missed by the first sweep: (a) it is
*about latent multi-agent collaboration* rather than latent CoT generally, or
(b) it is *very recent* and not yet well-indexed.

**(a) The LatentMAS family has grown since Aug 2026, and every new entrant was
checked for SQL by reading its actual benchmark list, not its abstract — none
has one.** [RecursiveMAS](https://arxiv.org/abs/2604.25917) (May 2026) casts a
whole heterogeneous multi-agent system as one latent recursive computation
(closing an inner/outer KV-cache loop across agents and rounds), +8.3% accuracy
and 34.6-75.6% fewer tokens — evaluated on MATH500, AIME2025, AIME2026,
GPQA-Diamond, MedQA, LiveCodeBench-v6, MBPP+, HotpotQA, Bamboogle. No SQL.
[Interlat / "Enabling Agents to Communicate Entirely in Latent Space"](https://arxiv.org/abs/2511.09149)
(ACL 2026) transmits last-layer hidden states directly between agents (up to
24× faster than NL communication) — evaluated on ALFWorld and MATH only.
[Cache Merging as a Convergent Replicated State for Multi-Agent Latent
Reasoning](https://arxiv.org/abs/2607.01308) (Jul 2026) fixes a correctness bug
in how multiple agents' KV-caches get concatenated (order-dependence), verified
on Qwen3-1.7B/4B — no task benchmark with SQL. Also checked and confirmed no
SQL: ["Beyond tokens": a unified framework for latent communication](https://arxiv.org/abs/2606.05711)
and ["Do Latent Channels Actually Communicate?"](https://arxiv.org/abs/2607.26773)
(the latter runs GSM8K/ARC-C/MATH-500). None of these five papers evaluates
Text-to-SQL, and none distils a *published, named* multi-agent Text-to-SQL
system (DIN-SQL/DAIL-SQL/CHESS/CHASE-SQL or otherwise) — they compose generic
agent roles, same as LatentMAS itself.

**(b) Single-model latent/implicit-reasoning papers, checked the same way:**
[SemCoT](https://arxiv.org/abs/2510.24940) (genuine continuous-vector implicit
tokens, contrastively aligned to explicit reasoning) — no SQL benchmark found.
[CoLT: Chain of Latent Tool Calls](https://arxiv.org/abs/2602.04246) — hidden
states trigger an external "unpacker" model per reasoning step, which is
architecturally the nearest thing to "SQL execution as a latent tool call" we
found anywhere — but its four benchmarks are all mathematical, SQL is not one
of its tools, and the base model still reasons in explicit tokens outside the
tool calls. [Dropout-GRPO](https://arxiv.org/abs/2606.10184) (Jun 2026) is the
closest **methodological** neighbor to our own stage-2 training design: it
combines GRPO with a COCONUT-style continuous-latent-reasoning policy,
addressing exactly the variance/exploration problem latent rollouts create for
group-relative RL. Evaluated on GSM8K only (27.29% → 29.01% pass@1 over a
Coconut baseline) — no SQL, and not a multi-agent or distillation method — but
worth citing for the GRPO-through-latent-space mechanics regardless of the
Text-to-SQL question. [A Survey on Latent Reasoning](https://arxiv.org/abs/2507.06203)
was also checked directly for its taxonomy of application domains: no
Text-to-SQL or database-query category appears anywhere in it.

**Confirmed false positives (colloquial "latent", not a real candidate) — read
past the abstract specifically to rule out an overloaded keyword hit:**
APEX-SQL uses "latent reasoning potential" as a figure of speech for untapped
test-time-compute headroom, not a mechanism; a Text-to-Python-vs-SQL benchmarking
paper's "latent domain knowledge" means unstated context, not hidden
activations; [YORO](https://arxiv.org/abs/2409.12172) "internalizes database
knowledge" by fine-tuning the schema into the model's *parametric* weights so
it need not be re-encoded per question — internalizing *content*, not a
reasoning *process*, and there is no debate, no transcript, no latent thought
step at inference.

**Net effect on the claim:** no work found, at any distance, does latent
reasoning *of any kind* — single-model or multi-agent — for Text-to-SQL
*generation*. LatentRefusal (§2.6b) remains the only paper touching both
"latent" and "Text-to-SQL" that survives a full read, and it is a
classification gate, not a generator. C1-C6 in combination (a fused,
*named* multi-agent Text-to-SQL system, distilled and internalised via
SFT+GRPO, into latent reasoning) has no known prior instance in either the
Text-to-SQL literature or the latent-reasoning literature, checked from both
directions.

### 2.9 Activation steering interacting with RL post-training — background for our own steering module (`caa.py`)

Prompted by a direct question ("are there any papers studying the steering of activation after applied RL like GRPO or more?"), a sweep of this adjacent literature turned up three distinct clusters, none of them Text-to-SQL, all of them relevant background for `latentsql/steering/caa.py` (the IMAD §3 difference-in-means agent-subspace module already in this codebase).

**(a) Steering applied on top of an already-RL-tuned model, without studying what the RL did.** [Angular Steering](https://proceedings.neurips.cc/paper_files/paper/2025/file/b0223cad0e73b793f31eb6cc41cefceb-Paper-Conference.pdf) (Vu & Nguyen, NeurIPS 2025) reformulates activation steering as a rotation within a fixed 2D subspace instead of vector addition or orthogonalization, evaluated on RLHF-instruct models (Qwen2.5-3B/7B/14B-Instruct, Llama-3.2-3B/3.1-8B-Instruct, Gemma-2-9b-it). It extracts its feature direction with the **exact same Difference-in-Means primitive** `caa.py` uses (their §4.4, citing [Belrose 2023, "Diff-in-means concept editing is worst-case optimal"] and the same [Arditi et al. 2024, "Refusal in LLMs is Mediated by a Single Direction"] lineage IMAD §3 draws on) — same tool, different target (refusal/emotion direction vs. our agent-persona subspaces). It also reports that smaller models (3B) are measurably more vulnerable to steering-induced incoherence than 7B+ ones (§6.2), a capacity-dependent finding worth keeping in mind given our own student is 3B.

**(b) Papers asking whether RL/safety alignment is "shallow" and can be exposed or undone by activation-level steering** — the closest match to "steering after applied RL" as a research question in its own right: [Why Do Aligned LLMs Remain Jailbreakable: Refusal-Escape Directions, Operator-Level Sources, and Safety-Utility Trade-off](https://arxiv.org/pdf/2605.08878), [Analysing the Safety Pitfalls of Steering Vectors](https://arxiv.org/pdf/2603.24543), [AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](https://arxiv.org/pdf/2506.07022), [SteeringControl: Holistic Evaluation of Alignment Steering](https://arxiv.org/pdf/2509.13450). The consistent finding across this cluster: RLHF/safety-RL often only reinforces a refusal *prefix* early in generation rather than durably reshaping deeper representations, so a steering vector applied post-hoc can re-expose pre-RL behavior — RL's effect on activations is real but shallower than the behavioral change suggests. Angular Steering's own §6.2 independently supports this: perplexity stays near baseline while steering aligns with the target feature, but drops below baseline as it moves toward the jailbroken region, suggesting harmful generations (learnt in pretraining) remain more probable under the surface than refusal responses (learnt in safety tuning).

**(c) Steering as a component *inside* an RL training loop**, closer to "steering after GRPO" read operationally: [ASRU: Activation Steering Meets Reinforcement Unlearning](https://arxiv.org/html/2605.15687v1) uses GRPO to learn a fine-grained unlearning boundary, with steering-derived direction identification feeding the reward; [Activation Reward Models for Few-Shot Model Alignment](https://arxiv.org/pdf/2507.01368) uses activation-space signals directly as the reward model rather than a separate scalar reward; goal-conditioned reward-model representations for RLHF steerability report +9.6% helpfulness over a plain SFT baseline by conditioning the reward model on representations of the desired end-state.

**Net effect on the claim:** none of (a)-(c) touch Text-to-SQL or multi-agent-debate internalization, so they do not change §4's novelty claim. They are recorded here because (a) shares our exact direction-extraction primitive and is the nearest published validation that Difference-in-Means subspace steering is a live, NeurIPS-2025-current technique, and (b)/(c) are the right literature to cite if a reviewer asks whether our DPO-on-panellist-preference-pairs stage (which also acts on the model's *behavior* post-hoc, from teacher disagreement rather than a safety objective) risks the same "shallow" fragility this cluster documents for RLHF.

---

## 3. Coverage matrix

| Work | C1 fused MAS teacher | C2 distil transcript | C3 internalise (RL) | C4 exec reward | C5 SLM | C6 latent | C7 agent steering |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **This project** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MATS | ◐ | ❌ | ◐ | ✅ | ✅ | ❌ | ❌ |
| Struct-SQL | ❌ | ◐ | ❌ | ❌ | ✅ | ❌ | ❌ |
| SERL-SQL | ❌ | ❌ | ◐ | ✅ | ✅ | ❌ | ❌ |
| MAD-OPD | ❌ | ◐ | ◐ | ❌ | ✅ | ❌ | ❌ |
| IMAD | ❌ | ✅ | ✅ | ❌ | ✅ | ◐ | ✅ |
| LatentMAS | ❌ | ❌ | ❌ | ❌ | ◐ | ✅ | ❌ |
| Arctic-Text2SQL-R1 | ❌ | ❌ | ◐ | ✅ | ✅ | ❌ | ❌ |
| LatentRefusal | ❌ | ❌ | ❌ | ❌ | ◐ | ✅ | ❌ |

◐ = partial / different mechanism. Max overlap by any single work: **3 full + 2 partial**.
LatentRefusal's C6 tick is a different *use* of the latent channel (a refusal
classifier, not generation) - it does not compete with C1-C4 or C7 at all, and
is included for completeness rather than as a threat to novelty (see §2.6b).

---

## 4. What is defensibly new

Ranked by how well each would survive a hostile reviewer.

1. **Execution-grounded internalisation reward (strong).** IMAD's `R(y; l)` is
   answer matching. Ours executes the clipped prefix against the database and
   compares result sets. This is not cosmetic: it makes the reward *verifiable*,
   it makes trace filtering possible (drop traces whose consensus query is
   wrong), and it is what lets the length clip shrink without the model learning
   to emit plausible-looking wrong SQL early. No prior work anneals a
   format-vs-length schedule against an execution oracle.

2. **Agent subspaces as a diversity mechanism for candidate generation (strong,
   and the most novel single idea here).** CHASE-SQL's gain comes from
   generating *diverse* candidates via multiple prompts and higher temperature.
   If internalisation really creates separable per-agent directions (IMAD §3),
   then steering along `divide_conquer` vs `query_plan` recovers that diversity
   from **one** model with no extra prompting. I found no work applying
   activation steering to Text-to-SQL at all, let alone as a substitute for
   multi-path generation. This is a testable, falsifiable claim.

3. **Latent roles inside a distilled student (moderate).** LatentMAS does latent
   collaboration across N models; IMAD does implicit internalisation in one.
   Doing LatentMAS's explicit latent-thought budget + KV working memory *inside*
   the IMAD student is a real combination neither paper performs. Risk: a
   reviewer may call it "obvious composition", and it may not help empirically
   once the debate is already internalised — the ablation must be run honestly.

4. **First Text-to-SQL port of debate internalisation (moderate).** Legitimate
   but weakest as a standalone claim: "method X applied to domain Y" needs the
   domain to bite back. It does here — schema linking, value retrieval and
   execution feedback have no analogue in arithmetic — but this should be framed
   as *enabling* 1–3, not as the contribution.

5. **Fused DIN/DAIL/CHESS/CHASE teacher (weak as novelty, useful as engineering).**
   Presented as a contribution it invites "so it is an ensemble". Presented as
   *the instrument that produces a maximally diverse transcript to distil* it is
   defensible. Note the four systems are individually re-implemented on shared
   infra, which is itself a fair contribution for comparability.

---

## 5. Risks a reviewer will raise

- **"This is IMAD with a new dataset."** The answer must be the execution
  reward + the steering-as-diversity result, not the port itself.
- **"MATS already did multi-agent Text2SQL with SLMs."** True, and it must be
  cited prominently. The distinction (they keep the MAS at inference, we delete
  it) is real but needs to be stated in the abstract, not buried.
- **"Struct-SQL shows one structured format is enough."** Must be met with the
  single-agent ablation (`mas.agents: [query_plan]`), not with argument.
- **"Arctic-Text2SQL-R1 gets 68.9% on BIRD with just an execution reward."**
  The sharpest objection. If our GRPO stage alone (no teacher, no debate) gets
  close to our full pipeline, the teacher is not earning its cost. **This
  ablation must be run and reported even if unflattering:** SFT-free GRPO from
  the base model, same reward, same budget.
- **Teacher quality ceiling.** With a 7B local teacher, traces will be weaker
  than the GPT-4-class teachers in DIN/CHESS/CHASE. If teacher EX is low, the
  headline "beats the MAS" is beating a *weak* MAS. Report teacher EX
  prominently and, if needed, generate traces with a stronger teacher while
  keeping the student local.

---

## 6. What this search could **not** rule out

- **Non-English and non-arXiv venues.** Search was English-language and
  arXiv/ACL-weighted.
- **Very recent preprints** (last ~4 weeks before 22 Aug 2026) that may not be
  indexed yet.
- **Industrial systems.** Snowflake, Databricks, Google and AWS all ship
  Text-to-SQL; internal distillation of an agent pipeline into one served model
  is a natural cost optimisation and would not necessarily be published.
- **Theses and workshop papers**, which are poorly indexed by general search.
- I could not fully read SQuaD-SQL's body text (PDF encoding); its abstract and
  intro are clear enough to place it (single-teacher synthetic-data KD on
  WikiSQL, explicit reasoning), but its details are taken on the abstract's word.

**Recommended follow-ups before submission:** a re-run of this search one week
before the deadline, and periodic re-checks of the LatentMAS-family papers in
§2.8 in case a future revision adds a Text-to-SQL benchmark. The
citation-graph sweep and the `"text-to-SQL" AND latent-reasoning` full-text
search recommended after the first pass were both carried out on 24 Aug 2026
(§2.8, §7) and did not surface a hit; Semantic Scholar's own "cited by" UI
would not render through automated fetch, so that specific citation list is
still unverified and worth a manual check nearer submission.

---

## 7. Method

Searches issued 22 Aug 2026 (all English, US region):

1. `internalize multi-agent debate distillation text-to-SQL small language model latent`
2. `latent reasoning text-to-SQL continuous thoughts hidden states SQL generation 2026`
3. `distill multi-agent text-to-SQL pipeline into single small model DIN-SQL CHESS distillation`
4. `text-to-SQL small language model distillation 7B DTS-SQL SENSE CodeS OmniSQL Arctic-Text2SQL`
5. `"text-to-SQL" latent space reasoning COCONUT continuous chain-of-thought SQL`
6. `multi-agent debate distillation single model agent subspace activation steering 2026 internalized collaboration`
7. `text-to-SQL multi-agent debate distillation internalized latent agent SQL 2026 thesis`
8. `"NL2SQL" OR "text-to-SQL" implicit reasoning compression fewer tokens internalize chain-of-thought efficiency 2026`
9. `LatentMAS latent working memory KV cache multi-agent benchmarks does it include SQL database`
10. `arxiv text-to-SQL activation steering representation engineering schema linking latent direction`
11. `latent multi-agent collaboration database agent SQL hidden state communication 2026 application`
12. `SERL-SQL selective hindsight distillation text-to-SQL agentic reinforcement learning teacher`

Full PDFs retrieved and read for: MATS (2512.18622), Struct-SQL (2512.17053),
SQuaD-SQL (2607.08161), MAD-OPD (2605.01347), plus the six papers already in
[pdfs/](pdfs/). LatentMAS's benchmark list was verified against the paper text
rather than a summary. LatentRefusal (2601.10398) was supplied directly by the
user on 24 Aug 2026, downloaded and read in full (9 pages) rather than found
by search - added to §2.6b.

**Second pass, 24 Aug 2026** — user asked directly "are there no papers
studying latent reasoning for Text-to-SQL?" after LatentRefusal surfaced;
treated as a request to stress-test §2.8 rather than accept the first pass's
conclusion. ~20 further queries (web search + direct arXiv/paper fetches, not
abstracts alone):

13. `latent reasoning text-to-SQL continuous chain-of-thought SQL generation`
14. `"text-to-SQL" implicit reasoning hidden state SQL generation arxiv 2026`
15. `continuous chain of thought SQL generation database query arxiv`
16. `"text2sql" latent multi-agent debate internalize KV cache`
17. `"Cache Merging" "Multi-Agent Latent Reasoning" arxiv 2607.01308`
18. `Interlat inter-agent latent message adapter hidden states benchmark`
19. `RecursiveMAS recursive latent computation heterogeneous multi-agent system`
20. `looped transformer recurrent depth reasoning SQL generation text-to-SQL`
21. `arxiv 2026 semantic parsing latent reasoning continuous thought logical form generation`
22. `knowledge graph question answering KGQA latent reasoning continuous space query generation`
23. `API call generation latent reasoning continuous space tool use internalize`
24. `site:arxiv.org "text-to-SQL" "latent" 2026`
25. `"internalize" OR "internalized" multi-agent Text-to-SQL debate distillation SQL 2026`
26. `soft thought SQL query generation continuous embedding agent collaboration database`
27. `activation steering text-to-SQL agent subspace SQL generation interpretability`
28. `Arctic-Text2SQL-R1 OR OmniSQL OR "SLM-SQL" related work "latent reasoning" OR "continuous chain-of-thought" OR COCONUT`
29. `"implicit reasoning" text-to-SQL fewer tokens compress chain of thought 2026`
30. `arxiv.org full text search text-to-SQL "latent reasoning" OR "continuous reasoning"`
31. `YORO "You Only Read Once" internalize database knowledge text-to-SQL mechanism fine-tuning`

Papers fetched and read past the abstract (benchmark list and mechanism
checked directly against the paper text, not inferred from a search snippet):
RecursiveMAS (2604.25917), Interlat/ACL2026 (2511.09149), Cache Merging
(2607.01308), "Beyond tokens" (2606.05711), "Do Latent Channels Actually
Communicate?" (2607.26773), Loop-Think-Generalize (2604.07822), SemCoT
(2510.24940), CoLT (2602.04246), Dropout-GRPO (2606.10184), A Survey on Latent
Reasoning (2507.06203), APEX-SQL (2602.16720), the Text-to-Python-vs-SQL paper
(2601.15728), YORO (2409.12172), and N-rep/"Cheaper Better Faster Stronger"
(2505.14174) — the latter four specifically to rule out colloquial "latent"
keyword hits before excluding them. Semantic Scholar's citation list for
LatentMAS and for IMAD would not render through automated fetch (JS-rendered
UI); that specific citation-graph check remains open, see §6.

---

## Sources

- [Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate — ACL 2026](https://aclanthology.org/2026.acl-long.709/) · [arXiv:2604.24881](https://arxiv.org/abs/2604.24881)
- [Latent Collaboration in Multi-Agent Systems (LatentMAS) — arXiv:2511.20639](https://arxiv.org/html/2511.20639v1) · [code](https://github.com/Gen-Verse/LatentMAS)
- [A Multi-agent Text2SQL Framework using Small Language Models and Execution Feedback (MATS) — arXiv:2512.18622](https://www.arxiv.org/pdf/2512.18622) · [code](https://github.com/thanhdath/mats-sql)
- [Knowledge Distillation with Structured Chain-of-Thought for Text-to-SQL (Struct-SQL) — arXiv:2512.17053](https://arxiv.org/pdf/2512.17053)
- [SERL-SQL: Selective Hindsight Distillation for Text-to-SQL Reinforcement Agentic Learning — arXiv:2608.00485](https://arxiv.org/abs/2608.00485)
- [MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate — arXiv:2605.01347](https://arxiv.org/html/2605.01347v1)
- [SQuaD-SQL: Efficient Text-to-SQL with Small Language Models via LLM-Guided Knowledge Distillation — arXiv:2607.08161](https://arxiv.org/pdf/2607.08161)
- [Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL — arXiv:2505.20315](https://arxiv.org/pdf/2505.20315) · [Snowflake blog](https://www.snowflake.com/en/blog/engineering/arctic-text2sql-r1-sql-generation-benchmark/)
- [SLM-SQL: An Exploration of Small Language Models for Text-to-SQL — arXiv:2507.22478](https://arxiv.org/html/2507.22478v1)
- [LatentRefusal: Latent-Signal Refusal for Unanswerable Text-to-SQL Queries — arXiv:2601.10398](https://arxiv.org/pdf/2601.10398v2)
- [Training Large Language Models to Reason in a Continuous Latent Space (COCONUT) — arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- [A Survey on Latent Reasoning — arXiv:2507.06203](https://arxiv.org/abs/2507.06203)
- [Recursive Multi-Agent Systems (RecursiveMAS) — arXiv:2604.25917](https://arxiv.org/abs/2604.25917)
- [Enabling Agents to Communicate Entirely in Latent Space (Interlat) — ACL 2026 · arXiv:2511.09149](https://arxiv.org/abs/2511.09149)
- [Cache Merging as a Convergent Replicated State for Multi-Agent Latent Reasoning — arXiv:2607.01308](https://arxiv.org/abs/2607.01308)
- [Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems — arXiv:2606.05711](https://arxiv.org/abs/2606.05711)
- [Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication — arXiv:2607.26773](https://arxiv.org/abs/2607.26773)
- [SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens — arXiv:2510.24940](https://arxiv.org/abs/2510.24940)
- [CoLT: Reasoning with Chain of Latent Tool Calls — arXiv:2602.04246](https://arxiv.org/abs/2602.04246)
- [Dropout-GRPO: Variational Stochasticity for Continuous Latent Reasoning — arXiv:2606.10184](https://arxiv.org/abs/2606.10184)
- [You Only Read Once (YORO): Learning to Internalize Database Knowledge for Text-to-SQL — NAACL 2025 · arXiv:2409.12172](https://arxiv.org/abs/2409.12172)
- [Awesome-Latent-CoT reading list](https://github.com/EIT-NLP/Awesome-Latent-CoT)
- [NL2SQL Handbook](https://github.com/HKUSTDial/NL2SQL_Handbook)
- [BAPPA: Benchmarking Agents, Plans, and Pipelines for Automated Text-to-SQL Generation — arXiv:2511.04153](https://arxiv.org/html/2511.04153)
- [DCMM-SQL: Automated Data-Centric Pipeline and Multi-Model Collaboration Training for Text-to-SQL — arXiv:2510.23284](https://arxiv.org/pdf/2510.23284)
- [Angular Steering: Behavior Control via Rotation in Activation Space — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/b0223cad0e73b793f31eb6cc41cefceb-Paper-Conference.pdf) · [code](https://github.com/lone17/angular-steering/)
- [Refusal in Language Models Is Mediated by a Single Direction — arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- [Why Do Aligned LLMs Remain Jailbreakable: Refusal-Escape Directions, Operator-Level Sources, and Safety-Utility Trade-off — arXiv:2605.08878](https://arxiv.org/pdf/2605.08878)
- [Analysing the Safety Pitfalls of Steering Vectors — arXiv:2603.24543](https://arxiv.org/pdf/2603.24543)
- [AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint — arXiv:2506.07022](https://arxiv.org/pdf/2506.07022)
- [SteeringControl: Holistic Evaluation of Alignment Steering in LLMs — arXiv:2509.13450](https://arxiv.org/pdf/2509.13450)
- [ASRU: Activation Steering Meets Reinforcement Unlearning for Multimodal LLMs — arXiv:2605.15687](https://arxiv.org/html/2605.15687v1)
- [Activation Reward Models for Few-Shot Model Alignment — arXiv:2507.01368](https://arxiv.org/pdf/2507.01368)
