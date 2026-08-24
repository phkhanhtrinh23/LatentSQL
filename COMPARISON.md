# LatentSQL vs. FINER-SQL

Both train a **≤3B model for Text-to-SQL with GRPO**, on **BIRD and Spider**, on
**one consumer GPU**. From a distance they look like the same paper. They are
not — they optimise for opposite things, and that difference explains every
other difference between them.

- **FINER-SQL** — *Boosting Small Language Models for Text-to-SQL with
  Fine-Grained Execution Feedback and Cost-Efficient Rewards*, Hoang, Huynh,
  Weidlich, Nguyen, Chen, Yin, Nguyen. ICDE 2026.
  [arXiv:2605.03465](https://arxiv.org/abs/2605.03465) ·
  [code](https://github.com/thanhdath/finer-sql)
- **LatentSQL** — this repo. IMAD (ACL 2026) + LatentMAS (ICML 2026) applied to
  Text-to-SQL. See [METHOD.md](METHOD.md).

---

## 1. The one difference that matters

|  | FINER-SQL | LatentSQL |
|---|---|---|
| **Optimises** | accuracy | accuracy **per output token** |
| **Reasoning is** | the *tool* — more and better reasoning is good | the *cost* — reasoning must disappear |
| **Wants the model to** | keep writing its chain of thought | stop writing, and think internally instead |
| **Succeeds when** | EX goes up | EX holds **while generated length collapses** |

FINER-SQL never asks the model to be shorter. LatentSQL spends its entire
second training stage trying to make the model *stop* producing the reasoning
it was taught in the first stage. That is the whole thesis: the multi-agent
debate should end up in the activation space, not on the page.

This is why the two papers' rewards pull in different directions, and why
borrowing from FINER-SQL required care (§4).

---

## 2. Side by side

|  | FINER-SQL | LatentSQL |
|---|---|---|
| Multi-agent anything | **no** — one model, one pass | **yes** — teacher is DIN-SQL + DAIL-SQL + CHESS + CHASE-SQL debating for 2 rounds |
| SFT data | reasoning distilled from a model pool (deepseek-reasoner), filtered on `is_execution_correct` | the **full tagged debate transcript**: `<\|schema_link\|>`, `<\|round 1\|>`, `<\|agent chess\|>` …, `<\|consensus\|>` |
| What SFT teaches | how to reason toward one query | how four systems argue, revise against each other's **execution results**, and converge |
| GRPO reward | **dense, static**: format + execution + atomic-op overlap + memory similarity | **sparse, annealed**: `w_fmt·R_fmt + w_clip·R(y;l)`, with `w_fmt` 1.0→0.05 and clip `l` 2000→500 |
| Role of the schedule | none — weights are fixed | **the mechanism.** Verbalising stops paying (`w_fmt`↓) *and* stops fitting (`l`↓), so internal computation is the only strategy left |
| Inference | plain decoding | optional **LatentMAS** layer: latent thoughts + shared KV working memory, decoding only the final query |
| Interpretability claim | none | agent subspaces recoverable by difference-in-means steering (IMAD §3) |
| Headline metric | EX (plus Recall@30, majority-vote) | EX **and** tokens/latency vs. the explicit system |
| Reported result | **67.8% BIRD, 85.0% Spider @3B** (50.6 / 75.0 @0.5B) | not yet demonstrated — see §5 |

---

## 2b. Does FINER-SQL have pipeline stages? No — it is flat

**FINER-SQL has no schema linking, no value retrieval, no schema pruning, no
query classification, no fixer and no selector agent.** Grepping their repo for
any of those concepts returns nothing; the only hit is an unrelated
`majority_voting.py`. There is one prompt and one completion:

```
SYSTEM: "You are a meticulous SQL expert. Generate a single, correct SQL query
         for the user question and the provided database schema.
         Rules: output exactly one SQL statement; executable on SQLite;
         no explanatory text ..."
USER:    question + the database schema

MODEL:   <think>  ... reasoning ...  </think>
         SELECT ...
```

The schema goes in **whole**, unpruned. The model reasons inside `<think>` and
then emits raw SQL. That is the entire inference path — one forward pass, one
call. Their format reward simply enforces this shape: nothing before `<think>`,
exactly one `<think>`/`</think>` pair, a thought longer than 100 characters, and
an extractable query.

### How this compares to us

The honest comparison has to separate **teacher** from **student**, because we
have stages and they do not — but only on the teacher side:

|  | FINER-SQL | LatentSQL teacher | LatentSQL student |
|---|---|---|---|
| value retrieval (LSH over cell values) | — | yes | — |
| schema linking | — | yes | — |
| schema pruning | — | yes (skipped <40 cols) | — |
| query classification / decomposition | — | yes | — |
| multiple candidate generators | — | 4 published systems | — |
| debate rounds w/ execution feedback | — | 2 | — |
| fixer / selector agent | — | yes / yes | — |
| **LLM calls per question** | **1** | **~27** | **1** |

So **at inference both methods are a single call.** The difference is not
pipeline depth at test time — it is what the single call was *trained to
produce*. FINER-SQL trains the model to emit good reasoning then SQL. We train
the model to emit a whole four-agent debate then SQL, and then try to compress
that debate away. All of our stages live in the teacher and are discarded once
traces are built.

### The caveat that matters for reading their numbers

**FINER-SQL's headline 67.8% BIRD is not single-greedy-decode.** From
`scripts/reproduce.sh`: *"Each run = vLLM n=30 @ temperature=1.0 → vav majority
voting → official EX."* They sample **30 candidates per question** and pick by
majority vote over **execution results** (candidates whose queries return
identical rows are grouped; the largest group wins). Their own table separates
the two:

| | 3B BIRD | 0.5B BIRD |
|---|---:|---:|
| Recall@30 (any of 30 correct) | 81.23% | 68.32% |
| **majority vote → official EX** | **67.80%** | **50.59%** |

So 67.8% is a **30-sample** number. Every LatentSQL figure in
[RESULTS.md](RESULTS.md) is a **single greedy decode**. The two are not directly
comparable, and the 30× sampling cost is exactly the kind of inference-time
expense our method is trying to remove. Any honest head-to-head has to either
give our student 30 samples and majority voting too, or quote their
single-sample number — we have not yet run either.

## 3. Why FINER-SQL's dense rewards exist

Their diagnosis is sparse reward. Binary "right or wrong" gives GRPO nothing to
learn from when every rollout in a group is wrong, so they add graded credit:

```
exec matches gold exactly        → exec_score = 2.0
else if the query at least runs  → exec_score = 1.0 + atomic_jaccard(pred, gold)   ∈ [1,2)
memory reward, ONLY when exec_score < 2.0  →  similarity of the reasoning to
                                              verified-correct reasoning paths
```

- **Atomic reward** — Jaccard over atomic SQL operations (sqlglot), shaped by a
  5th-power penalty so near-misses are not over-rewarded.
- **Memory reward** — cosine similarity against a ChromaDB bank of reasoning
  traces from *correct* answers only (`--only-correct` defaults to true).

The key structural point: **both terms score only wrong rollouts.** A correct
rollout already earns full credit and cannot earn more by resembling something.

---

## 4. What we took, and the one thing we had to change

We hit exactly the failure FINER-SQL describes. In pilot run 1, on 5 of 9 GRPO
steps every rollout in the group was wrong, so all rewards were equal, every
advantage was 0, and those steps taught the model nothing — total weight
movement across the whole stage was `6.3e-04`. Worse, the only term left with
variance was `R_fmt`, whose gradient rewards emitting **more** debate — the
opposite of the objective.

So we implemented both terms in
[`dense_rewards.py`](latentsql/distill/dense_rewards.py), locally: sqlglot for
atomic ops, a local MiniLM plus a numpy cosine instead of ChromaDB (at a few
thousand traces a matrix multiply beats a vector DB and adds no dependency).
Our memory bank is built from **our own kept traces** — verified multi-agent
debates — so the term measures "does this reasoning resemble a debate that
actually worked", which is closer to our objective than a generic reasoning
prior.

**The change we had to make: the memory weight is annealed on `w_fmt`'s
schedule.** FINER-SQL never wants to compress reasoning, so paying for
trace-like reasoning is free for them. For us a constant memory reward would
fight the length anneal directly — one term paying the model to write reasoning
while another makes writing it impossible. Tying `w_memory` to `w_fmt` lets it
guide early and fade as verbalisation stops paying. The atomic term scores the
*query*, not the prose, so it has no such tension and stays constant.

We also added a term FINER-SQL has no reason to want: **brevity among correct
rollouts** (`w_brevity`). Their reward has nothing to say about two correct
answers of different lengths — and neither did ours, which is why a saturated
group (all rollouts correct) produced zero gradient and the debate never
compressed.

### Measured effect of the borrowed terms

Controlled ablation, same SFT checkpoint and same data, only the reward
differing ([RESULTS.md §5b](RESULTS.md)):

| group type | binary reward | + atomic & memory |
|---|---:|---:|
| **all-wrong**: dead steps | 1 | **0** |
| **all-wrong**: mean `reward_std` | 0.029 | **0.091** (3.2×) |
| all-correct: mean `reward_std` | 0.052 | 0.054 (1.0×) |

FINER-SQL's fix works, and works precisely where they say it does — on failing
groups, not on saturated ones.

---

## 5. Honest standing

**FINER-SQL is a validated result.** 67.8% BIRD / 85.0% Spider at 3B, published,
reproducible from their repo.

**LatentSQL is not, yet.** Across three runs the student has never compressed
its output — generated length went 1024 → 922 across a fully annealed schedule
— so the "fraction of the tokens" half of the claim is untested. Our student
currently uses *fewer total tokens* than the multi-call baselines but is
**slower**, because total tokens are dominated by prompt tokens while latency
tracks output tokens.

The two are not competitors so much as different bets:

- FINER-SQL bets that **dense reward shaping** is what small models need. It has
  the evidence.
- LatentSQL bets that a **multi-agent debate can be moved into activation
  space**, buying the accuracy of an explicit multi-agent system at
  single-forward-pass cost. That bet is still open.

A fair reading of our results so far is that FINER-SQL's contribution is
*complementary and necessary* — without dense partial credit, our GRPO stage
has no gradient on the hardest questions — but not *sufficient*, because it
cannot address the saturated-group case that internalisation specifically
requires.

---

## 6. If you only remember one thing

> FINER-SQL makes a small model **reason better**.
> LatentSQL tries to make a small model **stop reasoning out loud** while
> keeping the benefit.
>
> That is why FINER-SQL rewards reasoning similarity and we have to anneal it
> away; why they have no length schedule and ours is the entire mechanism; and
> why their headline number is EX while ours has to be EX *per token*.
