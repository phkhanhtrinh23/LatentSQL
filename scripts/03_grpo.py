#!/usr/bin/env python
"""Stage 2: internalisation via GRPO with dynamic reward scheduling (IMAD 2.3).

    python scripts/03_grpo.py --config configs/bird_local.yaml \
        --data outputs/traces/sft.jsonl --sft outputs/sft
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.distill.build_dataset import load_records
from latentsql.distill.grpo import GRPOTrainer, load_reference, tasks_from_records
from latentsql.distill.rewards import ExecutionOracle
from latentsql.distill.sft import prepare_student
from latentsql.llm.hf_local import load_model
from latentsql.utils.common import banner, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default="outputs/traces/sft.jsonl")
    ap.add_argument("--sft", default=None, help="SFT adapter dir (default: sft.output_dir)")
    ap.add_argument("--base", default=None, help="base model (default: sft.model_name_or_path)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--batch-prompts", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume-consumed-prompts", type=int, default=0,
                    help="after --limit, keep only prompts not consumed by the seeded GRPO order")
    ap.add_argument("--max-prompt-chars", type=int, default=None,
                    help="drop GRPO records whose prompt text is longer than this")
    ap.add_argument("--prefer-hard", action="store_true",
                    help="prioritize likely-hard trace records before applying --limit")
    ap.add_argument("--hard-preds", default=None,
                    help="optional preds jsonl; wrong/unparseable qids are prioritized if they overlap")
    ap.add_argument("--len-start", type=int, default=None)
    ap.add_argument("--len-end", type=int, default=None)
    ap.add_argument("--w-fmt-start", type=float, default=None)
    ap.add_argument("--w-fmt-end", type=float, default=None)
    ap.add_argument("--w-brevity", type=float, default=None,
                    help="reward weight for shorter correct latent rollouts")
    ap.add_argument("--w-atomic", type=float, default=None)
    ap.add_argument("--w-memory", type=float, default=None)
    ap.add_argument("--w-spurious", type=float, default=None)
    ap.add_argument("--n-stress-instances", type=int, default=None)
    ap.add_argument("--cleanup-stress-cache", action="store_true")
    ap.add_argument("--cegis", action="store_true",
                    help="grow each question's instance pool with counterexamples found "
                         "against the student's own spurious rollouts (paper Eq. 22; "
                         "--reward-mode paper only)")
    ap.add_argument("--cegis-k-max", type=int, default=0,
                    help="pool size cap, counting the fixed stress instances; 0 = no cap")
    ap.add_argument("--cegis-n-delta", type=int, default=3, help="rows an edit may touch")
    ap.add_argument("--cegis-budget", type=int, default=40,
                    help="candidate edits per mutation search")
    ap.add_argument("--verieql-python", default=None,
                    help="python of VeriEQL's environment; with --verieql-dir enables VeriEQL "
                         "as the fallback verifier when mutation search finds nothing")
    ap.add_argument("--verieql-dir", default=None, help="VeriEQL checkout")
    ap.add_argument("--verieql-bound", type=int, default=2, help="max rows per table")
    ap.add_argument("--verieql-timeout", type=float, default=60.0, help="seconds per pair")
    ap.add_argument("--brevity-ref-tokens", type=int, default=None,
                    help="token count that receives zero brevity reward")
    ap.add_argument("--latent-steps", type=int, default=None)
    ap.add_argument("--latent-role-list", default=None)
    ap.add_argument("--spec-draft-model", default=None)
    ap.add_argument("--spec-k", type=int, default=None)
    ap.add_argument("--no-spec-decode", action="store_true",
                    help="disable speculative decoding for this run")
    ap.add_argument("--reward-mode", choices=("weighted", "paper"), default=None,
                    help="paper = main_latentsql.pdf Eq. (13): R_atomic + 1[EX=1](R_spur + R_brev)")
    ap.add_argument("--role-credit", action="store_true",
                    help="also reward SQL decoded right after each earlier latent role")
    ap.add_argument("--std-adv", action="store_true",
                    help="original GRPO (r-mean)/std advantages instead of Dr. GRPO r-mean")
    ap.add_argument(
        "--from-examples", default=None,
        help="build GRPO prompts straight from a benchmark (bird|spider|synthetic) "
             "instead of from teacher traces. With --sft pointing nowhere this is "
             "the no-teacher ablation: RL from the base model on execution reward only.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    sft_dir = args.sft or cfg.sft.output_dir
    base = args.base or cfg.sft.model_name_or_path
    if args.out:
        cfg.grpo.output_dir = args.out
    if args.epochs is not None:
        cfg.grpo.epochs = args.epochs
    if args.group_size:
        cfg.grpo.group_size = args.group_size
    if args.batch_prompts:
        cfg.grpo.batch_prompts = args.batch_prompts
    if args.grad_accum:
        cfg.grpo.grad_accum = args.grad_accum
    if args.max_new_tokens:
        cfg.grpo.max_new_tokens = args.max_new_tokens
    if args.lr is not None:
        cfg.grpo.lr = args.lr
    if args.temperature is not None:
        cfg.grpo.temperature = args.temperature
    if args.top_p is not None:
        cfg.grpo.top_p = args.top_p
    if args.len_start:
        cfg.grpo.reward.len_start = args.len_start
    if args.len_end:
        cfg.grpo.reward.len_end = args.len_end
    if args.w_fmt_start is not None:
        cfg.grpo.reward.w_fmt_start = args.w_fmt_start
    if args.w_fmt_end is not None:
        cfg.grpo.reward.w_fmt_end = args.w_fmt_end
    if args.w_brevity is not None:
        cfg.grpo.reward.w_brevity = args.w_brevity
    if args.w_atomic is not None:
        cfg.grpo.reward.w_atomic = args.w_atomic
    if args.w_memory is not None:
        cfg.grpo.reward.w_memory = args.w_memory
    if args.w_spurious is not None:
        cfg.grpo.reward.w_spurious = args.w_spurious
    if args.n_stress_instances is not None:
        cfg.grpo.reward.n_stress_instances = args.n_stress_instances
    if args.cleanup_stress_cache:
        cfg.grpo.reward.cleanup_stress_cache = True
    if args.brevity_ref_tokens is not None:
        cfg.grpo.reward.brevity_ref_tokens = args.brevity_ref_tokens
    if args.latent_steps:
        cfg.latent.latent_steps = args.latent_steps
    if args.latent_role_list:
        cfg.latent.roles = [r.strip() for r in args.latent_role_list.split(",") if r.strip()]
    if args.spec_draft_model:
        cfg.grpo.spec_draft_model = args.spec_draft_model
        cfg.grpo.spec_decode = True
    if args.spec_k:
        cfg.grpo.spec_k = args.spec_k
    if args.no_spec_decode:
        cfg.grpo.spec_decode = False
    if args.std_adv:
        cfg.grpo.std_normalize_advantages = True
    if args.reward_mode:
        cfg.grpo.reward.reward_mode = args.reward_mode
    if args.role_credit:
        if cfg.grpo.spec_decode:
            ap.error("--role-credit is not implemented for speculative decoding")
        cfg.grpo.role_credit = True
    set_seed(cfg.grpo.seed)

    banner("loading tasks")
    if args.from_examples:
        from latentsql.db import schema as schema_mod
        from latentsql.distill.grpo import tasks_from_examples
        from latentsql.utils.common import resolve_examples

        # Prompts here carry the full, untruncated schema, so one BIRD database
        # can produce a ~19k-token prompt - 12x longer than anything the latent
        # SFT corpus contains (capped at 4000 chars / 1800 tokens). The GRPO
        # loss backpropagates through the prompt prefill, so a single such
        # prompt OOMs the backward regardless of the memory cap. Apply the
        # same --max-prompt-chars filter the --data branch has, over-sampling
        # so --limit is still met after the long prompts are dropped.
        want = args.limit
        fetch = want * 2 if (want and args.max_prompt_chars) else want
        examples, used = resolve_examples(
            cfg, dataset=args.from_examples, limit=fetch, split="train"
        )

        def _render(ex):
            return schema_mod.render_schema(
                schema_mod.load_schema_cached(ex.db_path, ex.db_id), style="code"
            )

        tasks = tasks_from_examples(
            examples, _render, retrieval_cfg=cfg.mas,
            cache_dir=os.path.join(cfg.paths.cache_dir, "value_indexes"),
        )
        if args.max_prompt_chars:
            before = len(tasks)
            tasks = [t for t in tasks if len(t.prompt) <= args.max_prompt_chars]
            print(f"prompt length filter kept {len(tasks)}/{before} "
                  f"(max_prompt_chars={args.max_prompt_chars})")
        if want:
            tasks = tasks[:want]
        print(f"no-teacher ablation: {len(tasks)} prompts built directly from {used}")
    else:
        records = load_records(args.data)
        if args.max_prompt_chars:
            before = len(records)
            records = [
                r for r in records
                if len(getattr(r, "prompt", "") or "") <= args.max_prompt_chars
            ]
            print(
                f"prompt length filter kept {len(records)}/{before} "
                f"(max_prompt_chars={args.max_prompt_chars})"
            )
        if args.prefer_hard:
            hard_qids = set()
            if args.hard_preds and os.path.exists(args.hard_preds):
                with open(args.hard_preds, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("correct") is False or not row.get("pred_sql"):
                            hard_qids.add(row.get("qid", ""))

            def _hard_key(r):
                qid = getattr(r, "qid", "")
                is_bird = 1 if qid.startswith("bird_") else 0
                # High teacher-token traces usually reflect larger schemas or
                # harder debates; prompt length catches cases where the trace
                # token count is missing.
                return (
                    1 if qid in hard_qids else 0,
                    is_bird,
                    int(getattr(r, "teacher_tokens", 0) or 0),
                    len(getattr(r, "prompt", "") or ""),
                )

            records = sorted(records, key=_hard_key, reverse=True)
        if args.limit:
            records = records[: args.limit]
        if args.resume_consumed_prompts:
            if args.resume_consumed_prompts >= len(records):
                print("resume-consumed-prompts leaves no records")
                return 1
            # GRPOTrainer starts with this exact seeded shuffle and consumes
            # indices from the end. Retain only the prompts not represented by
            # the saved adapter, then let the continuation reshuffle that set.
            import random as _random

            order = list(range(len(records)))
            _random.Random(cfg.grpo.seed).shuffle(order)
            consumed = set(order[-args.resume_consumed_prompts:])
            records = [record for i, record in enumerate(records) if i not in consumed]
            print(
                f"resume selection kept {len(records)} prompts after excluding "
                f"{args.resume_consumed_prompts} already consumed"
            )
        tasks = tasks_from_records(
            records, retrieval_cfg=cfg.mas,
            cache_dir=os.path.join(cfg.paths.cache_dir, "value_indexes"),
        )
    if not tasks:
        print("no GRPO tasks: prompts need db_path and gold_sql")
        return 1
    print(f"{len(tasks)} prompts | group_size={cfg.grpo.group_size}")

    banner(f"loading SFT student ({base} + {sft_dir})")
    from latentsql.llm.hf_local import resolve_model_path

    base_path = resolve_model_path(
        base, hub_dir=cfg.paths.hf_hub, local_only=False
    )
    bundle = load_model(
        base_path,
        device=cfg.grpo.device,
        dtype=("bfloat16" if cfg.grpo.bf16 else "auto"),
        hub_dir=cfg.paths.hf_hub,
        adapter_path=(sft_dir if os.path.exists(os.path.join(sft_dir, "adapter_config.json")) else None),
        trainable=True,
        quantization=getattr(cfg.grpo, "quantization", "none"),
    )
    if not hasattr(bundle.model, "disable_adapter"):
        # no LoRA found: attach fresh adapters so GRPO stays parameter-efficient
        from peft import LoraConfig, get_peft_model

        bundle.model = get_peft_model(
            bundle.model,
            LoraConfig(
                r=cfg.sft.lora_r, lora_alpha=cfg.sft.lora_alpha,
                lora_dropout=cfg.sft.lora_dropout,
                target_modules=cfg.sft.lora_target_modules,
                task_type="CAUSAL_LM", bias="none",
            ),
        )
    for n, p in bundle.model.named_parameters():
        p.requires_grad_("lora_" in n)

    if getattr(cfg.grpo, "gradient_checkpointing", True):
        # Without this the backward retains every layer's MLP activations for
        # the whole prompt+completion; on BIRD (schema prompts of thousands of
        # tokens) that alone is ~8GB per sequence and OOMs a 24GB card partway
        # through training. Checkpointing only engages while the model is in
        # training mode, so the eval-mode rollout generation is unaffected.
        bundle.model.gradient_checkpointing_enable()
        # LoRA freezes the embeddings, so without this no input to the first
        # checkpointed block requires grad and the backward silently breaks
        if hasattr(bundle.model, "enable_input_require_grads"):
            bundle.model.enable_input_require_grads()
        bundle.model.config.use_cache = False

    banner("GRPO")
    # the brevity reference is the generation budget: a rollout that uses the
    # whole budget scores 0, one that answers immediately scores ~1
    paper_reward = getattr(cfg.grpo.reward, "reward_mode", "weighted") == "paper"
    if (paper_reward or getattr(cfg.grpo.reward, "w_brevity", 0.0) > 0) and \
            not getattr(cfg.grpo.reward, "brevity_ref_tokens", 0):
        cfg.grpo.reward.brevity_ref_tokens = cfg.grpo.max_new_tokens

    oracle = ExecutionOracle(
        timeout=cfg.mas.execution_timeout,
        stress_cache_dir=cfg.grpo.reward.stress_cache_dir
        or os.path.join(cfg.paths.cache_dir, "stress"),
        n_stress_instances=cfg.grpo.reward.n_stress_instances,
        cleanup_stress_cache=cfg.grpo.reward.cleanup_stress_cache,
    )
    if args.cegis:
        if args.reward_mode != "paper":
            raise SystemExit("--cegis needs --reward-mode paper")
        from latentsql.db.counterexample import CounterexamplePool

        oracle.cegis = CounterexamplePool(
            os.path.join(oracle.stress_cache_dir, "cegis"), k_max=args.cegis_k_max,
            n_delta=args.cegis_n_delta, budget=args.cegis_budget,
            verieql_python=args.verieql_python, verieql_dir=args.verieql_dir,
            verieql_bound=args.verieql_bound, verieql_timeout=args.verieql_timeout,
        )
        print(f"[cegis] counterexample pools on (k_max={args.cegis_k_max}, "
              f"n_delta={args.cegis_n_delta}, budget={args.cegis_budget}, "
              f"verieql={'on' if args.verieql_python else 'off'})")
    if float(getattr(cfg.grpo.reward, "w_spurious", 0.0)) > 0:
        print(f"[spurious] w_spurious={cfg.grpo.reward.w_spurious} enabled - "
              f"up to {cfg.grpo.reward.n_stress_instances} stress instances per gold "
              f"query, cached under {oracle.stress_cache_dir}")

    # the memory bank is built from the *kept* teacher traces - transcripts
    # whose final query was confirmed correct by execution - so the term scores
    # "does this reasoning resemble a debate that actually worked"
    memory = None
    if float(getattr(cfg.grpo.reward, "w_memory", 0.0)) > 0:
        from latentsql.distill.dense_rewards import MemoryReward
        from latentsql.mas.trace import load_traces

        trace_path = os.path.join(os.path.dirname(args.data), "traces.jsonl")
        if os.path.exists(trace_path):
            banked = load_traces(trace_path)
            kept = {t.qid for t in banked if t.correct and t.consensus_reached}
            memory = MemoryReward.from_traces(
                [t for t in banked if t.qid in kept],
                model_name=cfg.grpo.reward.memory_model,
                hub_dir=cfg.paths.hf_hub,
                device="cpu",              # tiny encoder; keep the GPU for the policy
                top_k=cfg.grpo.reward.memory_top_k,
                same_db_only=cfg.grpo.reward.memory_same_db_only,
            )
            print(f"[memory] bank of {len(memory._emb) if memory.ready else 0} "
                  f"verified reasoning paths from {trace_path}")
        else:
            print(f"[memory] w_memory > 0 but {trace_path} not found; term disabled")

    latent = None
    if getattr(cfg.grpo, "latent_rollouts", False):
        from latentsql.distill.latent import LatentReasoner

        latent = LatentReasoner(bundle, cfg.latent)
        if latent.extras is not None:
            for prm in latent.extras.parameters():
                prm.requires_grad_(False)
            print(f"latent extras loaded: align_r1={latent.align_r1} (frozen during GRPO)")
        print(f"[latent] rollouts run {cfg.latent.latent_steps} latent steps per role "
              f"over {cfg.latent.roles}; only the query is decoded")

    spec = None
    if getattr(cfg.grpo, "spec_decode", False):
        from latentsql.distill.speculative import SpeculativeSampler, load_draft

        draft = load_draft(cfg.grpo.spec_draft_model, hub_dir=cfg.paths.hf_hub,
                           quantization=getattr(cfg.grpo, "quantization", "none"))
        spec = SpeculativeSampler(bundle, draft, k=cfg.grpo.spec_k,
                                  temperature=cfg.grpo.temperature)
        print(f"[spec] draft={cfg.grpo.spec_draft_model} k={cfg.grpo.spec_k} "
              f"- lossless: rollouts still follow the policy distribution")

    trainer = GRPOTrainer(bundle, cfg.grpo, oracle, memory=memory, latent=latent, spec=spec)
    ref = load_reference(bundle)
    result = trainer.train(
        tasks, ref, log_path=os.path.join(cfg.grpo.output_dir, "grpo_log.json")
    )
    if getattr(oracle, "cegis", None) is not None:
        pools = oracle.cegis.pools
        st = oracle.cegis.stats
        result["cegis"] = {**st, "questions_with_counterexamples": len(pools),
                           "counterexamples": sum(len(v) for v in pools.values()),
                           "largest_pool": max((len(v) for v in pools.values()), default=0),
                           "search_seconds_per_step": round(
                               (st["search_seconds"] + st["pool_check_seconds"])
                               / max(1, len(result.get("history", []))), 3)}
        print(f"[cegis] {result['cegis']}")
        oracle.cegis.close(remove=True)
    write_json(result, os.path.join(cfg.grpo.output_dir, "grpo_result.json"))
    print(f"\nsaved internalised student to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
