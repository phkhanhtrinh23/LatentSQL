# Novelty search — does this idea already exist?

**Question asked:** has anyone already internalised a multi-agent Text-to-SQL
system (DIN-SQL / DAIL-SQL / CHESS / CHASE-SQL) into a single sub-7B model that
runs the debate in latent space?

**Short answer:** **No exact match found.** Every *ingredient* exists
separately, and two 2025–2026 Text-to-SQL papers come uncomfortably close on
adjacent axes. The specific composition appears unoccupied — but the novelty is
a composition, not a new primitive, and §5 is honest about what that means for
a thesis defence.

*Search performed 22 Aug 2026. Method in §7. This is a literature search, not a
proof of absence — see §6 for what it could not rule out.*

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

### 2.8 Latent-reasoning background (none Text-to-SQL)

[COCONUT](https://arxiv.org/abs/2412.06769) (hidden state fed back as next input
embedding), CODI (self-distillation into continuous space), SoftCoT, CoLaR,
iCoT. Our `latent_slot_curriculum` is COCONUT-style and is credited as such.
A survey sweep of the [Awesome-Latent-CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT)
list surfaced no Text-to-SQL entry.

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

◐ = partial / different mechanism. Max overlap by any single work: **3 full + 2 partial**.

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

**Recommended follow-ups before submission:** a Semantic Scholar / Google
Scholar *citation-graph* sweep forward from IMAD and LatentMAS (anyone citing
both is the highest-risk collision); an arXiv full-text search for
`"text-to-SQL" AND ("latent reasoning" OR "internalize")`; and a re-run of this
search one week before the deadline.

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
rather than a summary.

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
- [Training Large Language Models to Reason in a Continuous Latent Space (COCONUT) — arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- [Awesome-Latent-CoT reading list](https://github.com/EIT-NLP/Awesome-Latent-CoT)
- [NL2SQL Handbook](https://github.com/HKUSTDial/NL2SQL_Handbook)
- [BAPPA: Benchmarking Agents, Plans, and Pipelines for Automated Text-to-SQL Generation — arXiv:2511.04153](https://arxiv.org/html/2511.04153)
- [DCMM-SQL: Automated Data-Centric Pipeline and Multi-Model Collaboration Training for Text-to-SQL — arXiv:2510.23284](https://arxiv.org/pdf/2510.23284)
