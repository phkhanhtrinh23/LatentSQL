#!/usr/bin/env python
"""Stage 1: debate structure learning (IMAD 2.2).

    python scripts/02_sft.py --config configs/bird_local.yaml \
        --data outputs/traces/sft.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.distill.build_dataset import load_records
from latentsql.distill.sft import train_sft
from latentsql.utils.common import banner, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default="outputs/traces/sft.jsonl")
    ap.add_argument("--model", default=None, help="override sft.model_name_or_path")
    ap.add_argument("--out", default=None, help="override sft.output_dir")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--consensus-only", action="store_true",
                    help="DebateGPT ablation: supervise only the final answer")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.sft.model_name_or_path = args.model
    if args.out:
        cfg.sft.output_dir = args.out
    if args.epochs is not None:
        cfg.sft.epochs = args.epochs
    if args.max_seq_len:
        cfg.sft.max_seq_len = args.max_seq_len
    if args.consensus_only:
        cfg.sft.train_on_full_trace = False
    set_seed(cfg.sft.seed)

    banner(f"loading {args.data}")
    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} training traces | model={cfg.sft.model_name_or_path}")
    print(f"train_on_full_trace={cfg.sft.train_on_full_trace} (False = DebateGPT ablation)")

    banner("supervised fine-tuning")
    result = train_sft(
        records, cfg.sft, hub_dir=cfg.paths.hf_hub,
        log_path=os.path.join(cfg.sft.output_dir, "sft_log.json"),
    )
    write_json(result, os.path.join(cfg.sft.output_dir, "sft_result.json"))
    print(f"\nsaved student to {result['output_dir']}  (final loss {result['final_loss']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
