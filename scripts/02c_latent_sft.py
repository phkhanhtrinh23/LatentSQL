#!/usr/bin/env python
"""Stage 1.5b (optional): latent SFT - a worked example of success inside the
exact latent-primed conditioning GRPO/eval use, so GRPO does not have to
bootstrap the skill from a 0%-correct starting point. See
latentsql/distill/latent_sft.py for the full rationale, including the
COCONUT-style curriculum mode (--curriculum).

    python scripts/02c_latent_sft.py --config configs/latent_sft_big.yaml \
        --data outputs/latent_sft_big/traces/traces.jsonl \
        --sft outputs/latent_sft_big/sft \
        --out outputs/latent_sft_big/latent_sft \
        --curriculum --latent-steps 12 \
        --roles din_sql,dail_sql,chess,chase_sql,din_sql,dail_sql,chess,chase_sql,consensus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.mas.trace import load_traces
from latentsql.utils.common import banner, limit_resources, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="traces.jsonl")
    ap.add_argument("--sft", required=True, help="starting adapter (the plain-text SFT checkpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--curriculum", action="store_true",
                    help="COCONUT-style: ramp n_latent_roles 0 -> all across training, "
                         "instead of always running fully latent")
    ap.add_argument("--decode-compare", action="store_true",
                    help="add a decode-and-compare auxiliary loss on latent roles that still "
                         "have real teacher text available - see LatentReasoner.build_prefix's "
                         "docstring. Requires --curriculum (needs role_turns).")
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="weight of the decode-and-compare auxiliary loss relative to the main "
                         "gold-SQL loss (only used with --decode-compare)")
    ap.add_argument("--supervise-text", action="store_true",
                    help="COCONUT-fidelity fix: train the model to actually GENERATE each "
                         "explicit-text role's own tokens (real next-token loss), not just read "
                         "them as conditioning - see LatentReasoner.build_prefix's docstring. "
                         "Requires --curriculum (needs role_turns).")
    ap.add_argument("--coconut-objective", action="store_true",
                    help="COCONUT curriculum objective: replace the requested leading roles "
                         "with latent steps, supervise remaining explicit role turns, then "
                         "supervise SQL after <|consensus|>.")
    ap.add_argument("--sql-ready-role", type=int, default=None,
                    help="strict COCONUT only: after this 1-indexed latent role, add an "
                         "auxiliary probe that feeds <|consensus|> and supervises the "
                         "first SQL-answer tokens. Useful when a newly latent role causes "
                         "schema/table-list loops instead of SQL generation.")
    ap.add_argument("--sql-ready-weight", type=float, default=0.0,
                    help="weight for --sql-ready-role auxiliary loss")
    ap.add_argument("--sql-ready-tokens", type=int, default=16,
                    help="number of SQL-answer target tokens used by the sql-readiness probe")
    ap.add_argument("--chain-format", choices=("debate_roles", "chess_steps"),
                    default="debate_roles",
                    help="which ordered step chain to derive from traces for curriculum metadata")
    ap.add_argument("--require-role-turns", action="store_true",
                    help="drop records without a complete ordered role-turn transcript before "
                         "offset/limit slicing; needed for strict staged COCONUT training.")
    ap.add_argument("--text-weight", type=float, default=1.0,
                    help="weight of the text-generation loss relative to the main gold-SQL loss "
                         "(only used with --supervise-text)")
    ap.add_argument("--latent-steps", type=int, default=None)
    ap.add_argument("--roles", default=None, help="comma-separated role list")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0, help="skip this many records before taking --limit")
    ap.add_argument("--limit", type=int, default=None, help="cap n records")
    ap.add_argument("--max-prompt-chars", type=int, default=None,
                    help="skip records with a larger schema+question prompt (keeps peak memory bounded)")
    ap.add_argument("--max-role-chars", type=int, default=None,
                    help="curriculum: fall back to bare tags for records with an oversized "
                         "single role turn (keeps peak memory bounded at low curriculum depth)")
    ap.add_argument("--max-total-tokens", type=int, default=None,
                    help="curriculum: fall back to bare tags for records whose prompt + all "
                         "role turns exceed this TOKEN count (more accurate than the char caps "
                         "above - SQL/schema text can tokenize far less efficiently than its "
                         "char length suggests)")
    ap.add_argument("--force-n-latent-roles", type=int, default=None,
                    help="curriculum chain mode: use this exact depth instead of computing it "
                         "from this process' own step count (always 1 in one-record-per-process "
                         "chain mode) - see scripts/run_curriculum_chain.sh")
    ap.add_argument("--optim-state", default=None,
                    help="chain mode: path to persist/restore AdamW momentum (exp_avg, "
                         "exp_avg_sq) across subprocess boundaries - without this every chain "
                         "step starts from zero momentum, which is far noisier than a real "
                         "training run's optimizer")
    args = ap.parse_args()

    limits = limit_resources()
    print(f"[limits] {limits}", flush=True)

    cfg = load_config(args.config)
    set_seed(getattr(cfg, "seed", 0))
    if args.latent_steps:
        cfg.latent.latent_steps = args.latent_steps
    if args.roles:
        cfg.latent.roles = [r.strip() for r in args.roles.split(",")]

    banner("loading the SFT student")
    from latentsql.llm.hf_local import load_model, resolve_model_path

    base_path = resolve_model_path(cfg.sft.model_name_or_path, hub_dir=cfg.paths.hf_hub, local_only=False)
    adapter = args.sft if os.path.exists(os.path.join(args.sft, "adapter_config.json")) else None
    if adapter is None:
        raise SystemExit(f"no SFT adapter at {args.sft}")
    bundle = load_model(base_path, device="auto", dtype=("bfloat16" if cfg.sft.bf16 else "auto"),
                        hub_dir=cfg.paths.hf_hub, adapter_path=adapter, trainable=True,
                        quantization=getattr(cfg.sft, "quantization", "4bit"))
    if getattr(cfg.sft, "gradient_checkpointing", False) and hasattr(bundle.model, "gradient_checkpointing_enable"):
        bundle.model.gradient_checkpointing_enable()
        if hasattr(bundle.model, "enable_input_require_grads"):
            bundle.model.enable_input_require_grads()
    for n, prm in bundle.model.named_parameters():
        prm.requires_grad_("lora_" in n)
    print(f"student: base={base_path} adapter={adapter}")

    banner("loading traces")
    from latentsql.distill.latent_sft import build_records_from_traces

    with open(args.data, "r", encoding="utf-8") as probe_fh:
        first_record = json.loads(next(line for line in probe_fh if line.strip()))
    if {"system", "prompt", "gold_sql", "role_turns"}.issubset(first_record):
        with open(args.data, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        print(f"[latent_sft] loaded {len(records)} prepared retrieval-consistent records")
    else:
        traces = load_traces(args.data)
        records = build_records_from_traces(
            traces, cfg, max_prompt_chars=args.max_prompt_chars,
            max_role_chars=args.max_role_chars, tokenizer=bundle.tokenizer,
            max_total_tokens=args.max_total_tokens, chain_format=args.chain_format,
        )
    if args.require_role_turns:
        before = len(records)
        records = [r for r in records if r.get("role_turns") is not None]
        print(f"[latent_sft] strict role-turn filter kept {len(records)}/{before} records",
              flush=True)
    if args.offset:
        records = records[args.offset:]
    if args.limit:
        records = records[: args.limit]
    n_with_turns = sum(1 for r in records if r.get("role_turns") is not None)
    print(f"[latent_sft] {len(records)} usable records, {n_with_turns} with full role-turn text "
          f"(needed for --curriculum)", flush=True)
    if not records:
        print("no usable records - nothing to train on")
        return 1
    if args.require_role_turns and n_with_turns == 0:
        print("[latent_sft] strict role-turn training found no full role-turn records in this slice",
              flush=True)
        return 1
    if args.curriculum and n_with_turns == 0:
        print("[latent_sft] warning: --curriculum requested but this slice has no full "
              "role-turn text; training will fall back to bare role tags for explicit-text "
              "roles", flush=True)

    banner("latent SFT" + (" (curriculum)" if args.curriculum else ""))
    from latentsql.distill.latent_sft import train_latent_sft

    run_cfg = SimpleNamespace(
        latent=cfg.latent, lr=args.lr, epochs=args.epochs, grad_accum=args.grad_accum,
        logging_steps=args.logging_steps, save_steps=args.save_steps, output_dir=args.out,
        seed=getattr(cfg, "seed", 0), curriculum=args.curriculum,
        force_n_latent_roles=args.force_n_latent_roles,
        decode_compare=args.decode_compare, aux_weight=args.aux_weight,
        supervise_text=args.supervise_text,
        text_weight=args.text_weight,
        coconut_objective=args.coconut_objective,
        sql_ready_role=args.sql_ready_role,
        sql_ready_weight=args.sql_ready_weight,
        sql_ready_tokens=args.sql_ready_tokens,
        require_role_turns=args.require_role_turns,
        optim_state_path=args.optim_state,
    )
    result = train_latent_sft(bundle, records, run_cfg, log_path=f"{args.out}_result.json")
    write_json({k: v for k, v in result.items() if k != "history"},
               os.path.join(os.path.dirname(args.out) or ".", "latent_sft_summary.json"))
    print(f"\nfinal loss {result['final_loss']} over {result['steps']} steps")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
