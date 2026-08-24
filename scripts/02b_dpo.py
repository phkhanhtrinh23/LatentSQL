#!/usr/bin/env python
"""Stage 1.5: preference optimisation on the teacher's wrong pipelines.

    python scripts/02b_dpo.py --config configs/x.yaml --traces outputs/x/traces/traces.jsonl

Pairs a correct panellist's turn against a wrong one on the same question, so
the student learns what *not* to do - which SFT cannot express, because raising
the likelihood of a wrong trace is exactly the wrong update.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.distill.dpo import build_pairs, save_pairs, train_dpo
from latentsql.mas.trace import load_traces
from latentsql.utils.common import banner, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--sft", default=None, help="SFT adapter to start from")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.out:
        cfg.dpo.output_dir = args.out
    set_seed(cfg.dpo.seed)

    banner("building preference pairs")
    traces = load_traces(args.traces)
    pairs = build_pairs(traces, round_one_only=cfg.dpo.round_one_only,
                        max_per_question=cfg.dpo.max_pairs_per_question,
                        seed=cfg.dpo.seed)
    print(f"{len(pairs)} pairs from {len(traces)} traces "
          f"(round_one_only={cfg.dpo.round_one_only})")
    if not pairs:
        print("no (correct, wrong) pairs available - nothing to optimise")
        return 1
    save_pairs(pairs, os.path.join(cfg.dpo.output_dir, "pairs.jsonl"))

    banner("loading the SFT student")
    from latentsql.llm.hf_local import load_model, resolve_model_path

    sft_dir = args.sft or cfg.sft.output_dir
    base_path = resolve_model_path(cfg.sft.model_name_or_path,
                                   hub_dir=cfg.paths.hf_hub, local_only=False)
    adapter = sft_dir if os.path.exists(
        os.path.join(sft_dir, "adapter_config.json")) else None
    bundle = load_model(base_path, device=cfg.dpo.device,
                        dtype=("bfloat16" if cfg.sft.bf16 else "auto"),
                        hub_dir=cfg.paths.hf_hub, adapter_path=adapter,
                        trainable=True,
                        quantization=getattr(cfg.sft, "quantization", "none"))
    if adapter is None:
        raise SystemExit(f"no SFT adapter at {sft_dir}; DPO needs one as reference")
    for n, prm in bundle.model.named_parameters():
        prm.requires_grad_("lora_" in n)
    print(f"student: base={base_path} adapter={adapter}")

    banner("DPO")
    res = train_dpo(bundle, pairs, cfg.dpo,
                    log_path=os.path.join(cfg.dpo.output_dir, "dpo_log.json"))
    write_json({k: v for k, v in res.items() if k != "history"},
               os.path.join(cfg.dpo.output_dir, "dpo_result.json"))
    print(f"\nfinal margin {res['final_margin']} | mean preference acc {res['mean_acc']}")
    print(f"wrote {cfg.dpo.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
