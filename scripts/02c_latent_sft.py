#!/usr/bin/env python
"""Stage 1.5b (optional): latent SFT - a worked example of success inside the
exact latent-primed conditioning GRPO/eval use, so GRPO does not have to
bootstrap the skill from a 0%-correct starting point. See
latentsql/distill/latent_sft.py for the full rationale.

    python scripts/02c_latent_sft.py --config configs/imad_boost.yaml \
        --data outputs/imad_boost/traces/sft.jsonl \
        --sft outputs/imad_boost/sft \
        --out outputs/latent_sft_probe/latent_sft \
        --latent-steps 12 --roles din_sql,dail_sql,chess,chase_sql,din_sql,dail_sql,chess,chase_sql,consensus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.utils.common import banner, limit_resources, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="sft.jsonl with system/prompt/gold_sql")
    ap.add_argument("--sft", required=True, help="starting adapter (the plain-text SFT checkpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent-steps", type=int, default=None)
    ap.add_argument("--roles", default=None, help="comma-separated role list")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0, help="skip this many records before taking --limit")
    ap.add_argument("--limit", type=int, default=None, help="cap n records - keep this a light warm-up")
    ap.add_argument("--max-prompt-chars", type=int, default=None,
                    help="skip records with a larger schema+question prompt (keeps peak memory bounded)")
    args = ap.parse_args()

    limits = limit_resources()
    print(f"[limits] {limits}", flush=True)

    cfg = load_config(args.config)
    set_seed(getattr(cfg, "seed", 0))
    if args.latent_steps:
        cfg.latent.latent_steps = args.latent_steps
    if args.roles:
        cfg.latent.roles = [r.strip() for r in args.roles.split(",")]

    banner("loading records")
    records = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    records = [r for r in records if r.get("gold_sql") and r.get("system") and r.get("prompt")]
    if args.max_prompt_chars:
        before = len(records)
        records = [r for r in records if len(r["prompt"]) <= args.max_prompt_chars]
        print(f"[latent_sft] dropped {before - len(records)} records over "
              f"{args.max_prompt_chars} prompt chars", flush=True)
    if args.offset:
        records = records[args.offset:]
    if args.limit:
        records = records[: args.limit]
    print(f"[latent_sft] {len(records)} usable records (system+prompt+gold_sql present)", flush=True)
    if not records:
        print("no usable records - nothing to train on")
        return 1

    banner("loading the SFT student")
    from latentsql.llm.hf_local import load_model, resolve_model_path

    base_path = resolve_model_path(cfg.sft.model_name_or_path, hub_dir=cfg.paths.hf_hub, local_only=False)
    adapter = args.sft if os.path.exists(os.path.join(args.sft, "adapter_config.json")) else None
    if adapter is None:
        raise SystemExit(f"no SFT adapter at {args.sft}")
    bundle = load_model(base_path, device="auto", dtype=("bfloat16" if cfg.sft.bf16 else "auto"),
                        hub_dir=cfg.paths.hf_hub, adapter_path=adapter, trainable=True,
                        quantization=getattr(cfg.sft, "quantization", "4bit"))
    for n, prm in bundle.model.named_parameters():
        prm.requires_grad_("lora_" in n)
    print(f"student: base={base_path} adapter={adapter}")

    banner("latent SFT")
    from latentsql.distill.latent_sft import train_latent_sft

    run_cfg = SimpleNamespace(
        latent=cfg.latent, lr=args.lr, epochs=args.epochs, grad_accum=args.grad_accum,
        logging_steps=args.logging_steps, save_steps=args.save_steps, output_dir=args.out,
        seed=getattr(cfg, "seed", 0),
    )
    result = train_latent_sft(bundle, records, run_cfg, log_path=f"{args.out}_result.json")
    write_json({k: v for k, v in result.items() if k != "history"},
               os.path.join(os.path.dirname(args.out) or ".", "latent_sft_summary.json"))
    print(f"\nfinal loss {result['final_loss']} over {result['steps']} steps")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
