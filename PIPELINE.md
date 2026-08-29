# LatentSQL Training Pipeline

Step-by-step documentation of the full pipeline currently running: internalizing a
multi-agent Text-to-SQL debate into latent reasoning inside a single model, via
plain-text SFT followed by a COCONUT-style curriculum latent-SFT stage. Written to
reflect the pipeline as it stands after this session's debugging and COCONUT-fidelity
fixes (see RESULTS.md for the experiment log this pipeline produced).

## Stage 0 — Teacher trace generation

**Script**: `scripts/01_build_traces.py`
**Orchestrator**: `FusionSQL` in `latentsql/mas/orchestrator.py`
**Config**: `configs/latent_sft_1000.yaml` (or `configs/latent_sft_1000_7b.yaml` for the
7B student — only `sft.model_name_or_path` differs between the two)

The teacher is not a re-run of any single paper's system — `FusionSQL` fuses the
component each source system is strongest on. It has **five stages**, only one of which
(the debate itself) becomes part of the latent-reasoning curriculum — see the next
section for exactly which.

1. **Context building** (`FusionSQL._build_context` + `solve`, stage 1 in the code):
   - `InformationRetriever` (CHESS-style value retrieval: LSH + edit distance over
     column values) — runs only if `mas.use_value_retrieval` (on in our configs).
   - `SchemaLinker` (DIN-SQL-style schema linking, narrowed by the CHESS Schema
     Selector) — always runs.
   - `QueryClassifier` (DIN-SQL's easy / non-nested / nested difficulty split) — runs
     only if `mas.use_classifier` (on in our configs).
   - These populate `trace.schema_prompt`, `trace.schema_links`, `trace.value_hints`,
     `trace.difficulty_class` — all of which end up baked into the **student's plain
     prompt text** (schema + value hints), not into any latent role.
2. **Few-shot example selection**: DAIL-SQL's masked-similarity in-context example
   selection (`select_dail_examples`), picking `mas.dail_shots` examples for the
   DAIL-SQL panellist specifically.
3. **The debate** (`mas.rounds` rounds, `mas.agents` panel — see next section): each
   panellist proposes a candidate SQL in round 0, then revises given its own and the
   other panellists' previous turns in later rounds. `QueryFixer` (CHASE-SQL / CHESS
   revise-style self-correction) is applied to every turn if `mas.use_fixer`. The debate
   can early-exit once every agent already agrees and has run at least 2 rounds.
4. **Selection**: `Selector` — either the CHESS Unit Tester (`mas.selector:
   unit_test`, the default, executes `mas.n_unit_tests` generated tests against each
   candidate) or CHASE-SQL's pairwise comparison — picks the final `consensus_sql` from
   the last round's candidates.
5. **Labeling**: the consensus SQL (and every individual turn) is checked against the
   real gold SQL by execution accuracy, *after* the fact — the agents themselves never
   see the gold answer. A trace is kept only if it reached consensus, that consensus
   executes, and it's correct (`trace.require_consensus`, `require_execution`,
   `require_correct`); `trace.rescue_correct_agent` salvages a trace where one agent was
   individually correct even without full-panel consensus.

Run over a 50/50 BIRD/Spider question mix (`mix.bird: 500`, `mix.spider: 500` — 900 for
the teacher, 100 held out for eval), with a 7B teacher LLM (`Qwen/Qwen2.5-7B-Instruct`,
4-bit — a general instruct model, used only to *run* the agents above, not to be
internalized itself).

Output: `outputs/latent_sft_1000/traces/traces.jsonl` (full transcripts, 677 kept of 900
attempted — 78.11% teacher execution accuracy) and `.../traces/sft.jsonl` (flattened
prompt/completion pairs for Stage 1).

```
LATENTSQL_GPU_FRACTION=0.72 LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 scripts/01_build_traces.py --config configs/latent_sft_1000.yaml
```

## Stage 1 — Plain-text SFT

**Script**: `scripts/02_sft.py`

Ordinary supervised fine-tuning (LoRA, r=32/alpha=64, 4-bit base, gradient
checkpointing): the student reads a plain prompt (schema + question) and is trained to
produce the *whole written debate transcript* verbatim, ending in the gold SQL. This is
the "internalize IMAD" step — the student learns to imitate the debate structure in
ordinary text, with no latent reasoning involved yet.

```
LATENTSQL_GPU_FRACTION=0.72 LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 scripts/02_sft.py --config configs/latent_sft_1000_7b.yaml \
  --data outputs/latent_sft_1000/traces/sft.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --out outputs/latent_sft_1000/sft_7b
```

Output: a LoRA adapter checkpoint (e.g. `outputs/latent_sft_1000/sft_7b`) that becomes
the starting point for Stage 1.5.

## Stage 1.5 — Curriculum latent-SFT

**Core mechanism**: `LatentReasoner.build_prefix()` in `latentsql/distill/latent.py`
**Training loop**: `train_latent_sft()` in `latentsql/distill/latent_sft.py`
**Script**: `scripts/02c_latent_sft.py`

This is the actual "debate in continuous space" step.

### Which pipeline steps actually become latent "roles" — and which don't

Of the five teacher-pipeline stages above, **only stage 3 (the debate) is represented
as latent roles.** Concretely, `cfg.latent.roles` is:

```
[din_sql, dail_sql, chess, chase_sql, din_sql, dail_sql, chess, chase_sql, consensus]
```

— the same 4-agent panel appearing twice (once per debate round, `mas.rounds: 2`),
followed by one final `consensus` marker. That is **8 latent-trainable roles + 1 fixed
consensus tag**, nothing more.

Everything else from Stage 0 is **not** a latent role:

- `InformationRetriever`, `SchemaLinker`, `QueryClassifier` (context building) and the
  DAIL-SQL few-shot selection never run at latent-training time at all — their output
  (`schema_prompt`, `value_hints`) was already baked into the **plain prompt text** the
  student reads at the very start (`student_prompt()` in `latentsql/mas/prompts.py`),
  before any latent step begins. The student sees them as ordinary tokens every time,
  in every curriculum stage, latent or not.
- `QueryFixer` and `Selector` never appear to the student as a role either — they only
  shaped which candidate SQL each debate turn's *text* ended up containing (a fixer
  correction, or which candidate the selector eventually chose as `consensus_sql`) when
  the teacher trace was generated. There is no separate "fixer role" or "selector role"
  in `cfg.latent.roles`; the fixer/selector's effect is already absorbed into the debate
  turns' own rendered text.

So "how many steps, and which ones, for latent reasoning" — **8 debate-panel turns (4
agents × 2 rounds) plus 1 consensus marker, and nothing from context building,
retrieval, classification, fixing, or selection.**

For each of those 8 roles, the student either:

- **runs latent** — appends the role's tag, then auto-regresses `latent_steps` times on
  hidden states, emitting no tokens (LatentMAS-style: the hidden state is fed back as
  the next input embedding); or
- **runs explicit text** — is fed the teacher's real rendered turn for that role,
  ordinary token embeddings, teacher-forced.

A curriculum ramps `n_latent_roles` from 0 (every role explicit text — what plain SFT
already trained) up to 8 (every role latent — the regime GRPO/eval actually need),
instead of demanding the full jump in one step.

### COCONUT-fidelity fixes (this session)

The curriculum was checked against the actual reference implementation
(`facebookresearch/coconut`) and three real divergences were found and fixed, each
behind its own flag so defaults (GRPO/eval) are unaffected:

1. **Curriculum direction** — COCONUT replaces reasoning steps with latent thoughts
   front-to-back (earliest steps go latent first). An earlier version of this code went
   back-to-front. Fixed: `is_latent = idx < n_latent_roles`.
2. **`--supervise-text`** — COCONUT's loss is not restricted to the final answer; every
   token from wherever latent leaves off through the answer gets real cross-entropy
   supervision, so the model keeps being trained to actually *produce* whatever
   reasoning text hasn't yet been replaced by latent thoughts. Previously, explicit-text
   roles here were only ever read as conditioning, never trained to be generated. Now
   `_text_steps(..., supervise=True)` adds a real next-token generation loss over each
   text role's own tokens, bridged from the preceding latent role's own prediction.
3. **`latent.align`** — COCONUT feeds the raw hidden state back with no learned
   projection. This codebase has an existing LatentMAS-derived `align()` layer
   (`soft_embed` by default) plus an `identity` mode for exactly this ablation. Tried
   `identity`: produced degenerate token-soup output (garbage, no structure) — this
   codebase's own docstring had already flagged `identity` as the out-of-distribution
   risk case. **Reverted to `soft_embed`**; the switch stays available via config.

Also added, available but not necessarily on in every run:

- **`--decode-compare`** — an auxiliary loss forcing a latent role's hidden state to be
  decodable (via the model's own already-computed logits, no extra forward pass) toward
  the first `latent_steps` tokens of what that agent's real turn would have said.
- **Eval-time OOM protection** (`latentsql/eval/run_eval.py`) — a single sample that
  OOMs during generation is now caught, logged, replaced with a `pred_sql=None`
  placeholder (scores as incorrect), and the run continues, instead of crashing the
  whole eval. Forward-only, so unlike training-time OOM this recovery is safe (no
  `torch.utils.checkpoint` recompute state to corrupt).
- **Live-streamed predictions** — every eval run now writes
  `preds_<method>_live.jsonl` incrementally as each sample is generated, so real
  generated text is inspectable while a long eval is still running, not just at the end.

### Why training runs as a subprocess chain, not one long process

A backward pass that OOMs inside a checkpointed latent region leaves PyTorch/CUDA state
corrupted for the rest of that process — every later step then OOMs identically
regardless of content, which looks like (but isn't) a Python-level memory leak. The fix
is architectural: train one batch (a `grad_accum`-sized group of records) per **fresh
subprocess**, chaining the LoRA checkpoint *and* AdamW momentum (`--optim-state`)
forward across processes via disk, so only the corruption-prone CUDA/checkpoint state
resets each time, not the model or the optimizer's memory of past gradients.

**Scripts**:
- `scripts/run_curriculum_chain_batched.sh` — one pass over the corpus, batched,
  momentum-persisting. Parameters (positional): `N BATCH_SIZE START_BATCH BASE N_ROLES
  DATA CONFIG EVAL_EVERY EVAL_N LR WIPE LATENT_STEPS RUNDIR GPU_FRACTION BASE_MODEL`.
- `scripts/run_curriculum_chain_batched_multipass.sh` — wraps the above to run
  `N_PASSES` full monotonic curriculum ramps back to back, checkpoint and momentum
  carried across passes (only the ramp position itself resets to batch 0 each pass).
- `scripts/run_curriculum_chain.sh` — the original one-record-per-process variant
  (no batching/momentum), kept for reference.
- `scripts/run_curriculum_chain_decode_compare.sh` — batched variant with
  `--decode-compare` turned on.

Every batch also applies `--max-prompt-chars 3000 --max-role-chars 600`: a record whose
prompt or any individual role turn is too long falls back to the always-cheap
fully-latent path for that record (a record with no usable role-turn text runs latent
regardless of curriculum depth), keeping peak memory bounded and batch-poisoning risk
(one bad record failing the whole grad-accum group) low.

### Current run

7B student (`Qwen/Qwen2.5-Coder-7B-Instruct`, 4-bit, LoRA), continuing from the Stage 1
checkpoint `outputs/latent_sft_1000/sft_7b`:

```
bash scripts/run_curriculum_chain_batched_multipass.sh 2 813 16 \
  outputs/latent_sft_1000/sft_7b 8 \
  outputs/latent_sft_1000/traces/traces.jsonl configs/latent_sft_1000_7b.yaml \
  10 30 1e-4 5 outputs/latent_sft_1000/curriculum_chain_7b 0.893 \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

- 813 records, batch size 16 (51 batches/pass), 2 passes (102 batches total)
- `latent_steps=5` per role (40 total at full depth — COCONUT's own demonstrated max is
  6 total ever; the first attempt at this pipeline used `latent_steps=12` (96 total),
  16x beyond anything COCONUT has shown works, and produced consistent collapse)
- `lr=1e-4` (COCONUT's own value), `--supervise-text` on, `align=soft_embed`
- GPU capped at `0.893` fraction ≈ 21GB of the 23.5GB card
- Intermediate eval every 10 batches (n=30, held-out dev split), full eval (n=90) at
  the end

Output: `outputs/latent_sft_1000/curriculum_chain_7b/latent_sft` (final adapter),
`outputs/latent_sft_1000/curriculum_chain_7b.log` (per-step training log),
`outputs/latent_sft_1000/curriculum_chain_7b_eval_progress/batch_N/` (each
intermediate eval's real predictions + scoring).

## Stage 2 (not yet run for this corpus) — GRPO

**Script**: `scripts/03_grpo.py` (referenced in configs, not launched this session for
the 1000-question/7B line of experiments — gated on Stage 1.5 actually producing a
usable, non-collapsed checkpoint first).

## Evaluation

**Script**: `scripts/04_eval.py`, driven through `latentsql/eval/run_eval.py`

```
LATENTSQL_GPU_FRACTION=0.893 LATENTSQL_CPU_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 scripts/04_eval.py --config configs/latent_sft_1000_7b.yaml \
  --methods latent \
  --student outputs/latent_sft_1000/curriculum_chain_7b/latent_sft \
  --base Qwen/Qwen2.5-Coder-7B-Instruct \
  --out outputs/latent_sft_1000/eval_7b
```

`--methods latent` runs `LatentReasoner.generate()` — always fully latent (all 8 roles,
`latent_steps` steps each — the regime the curriculum is meant to prepare the model
for), n=90 held-out questions (45 BIRD, 45 Spider) by default. Reports EX (execution
accuracy) split by dataset, average tokens, and now writes real generated text
incrementally as it runs (see "live-streamed predictions" above), not just the final
aggregate score.

## Known results so far

See `RESULTS.md` for the full experiment log. In short: every variant tried before the
COCONUT-fidelity fixes and the 7B/shorter-latent-horizon combination collapsed to 0% EX,
either as incoherent token-soup output, off-task hallucination, or memorization to one
fixed (wrong) answer regardless of the actual question. The 7B run documented here is
the first to combine all three COCONUT fixes with a validated-coherent latent horizon
(short, not the original 96-step one) and a bigger model; its result is not yet known as
of this document.
