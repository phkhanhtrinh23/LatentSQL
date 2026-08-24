#!/usr/bin/env python
"""Stage 2: internalisation via GRPO with dynamic reward scheduling (IMAD 2.3).

    python scripts/03_grpo.py --config configs/bird_local.yaml \
        --data outputs/traces/sft.jsonl --sft outputs/sft
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--len-start", type=int, default=None)
    ap.add_argument("--len-end", type=int, default=None)
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
    if args.len_start:
        cfg.grpo.reward.len_start = args.len_start
    if args.len_end:
        cfg.grpo.reward.len_end = args.len_end
    set_seed(cfg.grpo.seed)

    banner("loading tasks")
    if args.from_examples:
        from latentsql.db import schema as schema_mod
        from latentsql.distill.grpo import tasks_from_examples
        from latentsql.utils.common import resolve_examples

        examples, used = resolve_examples(
            cfg, dataset=args.from_examples, limit=args.limit, split="train"
        )

        def _render(ex):
            return schema_mod.render_schema(
                schema_mod.load_schema_cached(ex.db_path, ex.db_id), style="code"
            )

        tasks = tasks_from_examples(examples, _render)
        print(f"no-teacher ablation: prompts built directly from {used}")
    else:
        records = load_records(args.data)
        if args.limit:
            records = records[: args.limit]
        tasks = tasks_from_records(records)
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
    if getattr(cfg.grpo.reward, "w_brevity", 0.0) > 0 and \
            not getattr(cfg.grpo.reward, "brevity_ref_tokens", 0):
        cfg.grpo.reward.brevity_ref_tokens = cfg.grpo.max_new_tokens

    oracle = ExecutionOracle(timeout=cfg.mas.execution_timeout)

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
        print(f"[latent] rollouts run {cfg.latent.latent_steps} latent steps per role "
              f"over {cfg.latent.roles}; only the query is decoded")

    trainer = GRPOTrainer(bundle, cfg.grpo, oracle, memory=memory, latent=latent)
    ref = load_reference(bundle)
    result = trainer.train(
        tasks, ref, log_path=os.path.join(cfg.grpo.output_dir, "grpo_log.json")
    )
    write_json(result, os.path.join(cfg.grpo.output_dir, "grpo_result.json"))
    print(f"\nsaved internalised student to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
