#!/usr/bin/env python
"""Appendix C.4, Figure 9: what the latent steps hold.

    python scripts/analysis/probe_latent.py --config configs/qwen3_8b.yaml \
        --model outputs/.../rl/final --benchmarks spider --limit 5

The latent chain is stopped after each role (r1, r2, ...) and the model then
continues in language mode. The continuations are diagnostic only and are not
part of inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from latentsql.config import load_config
from latentsql.data.inputs import student_input
from latentsql.data.loaders import load_benchmarks
from latentsql.latent.reasoner import LatentReasoner
from latentsql.latent.student import load_student
from latentsql.utils.common import add_common_args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", default="spider")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default=None, help="jsonl output")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    examples = load_benchmarks(cfg, "test", args.benchmarks.split(","), limit=args.limit)
    bundle = load_student(cfg, adapter_dir=args.model, trainable=False, hub_dir=cfg.paths.hf_hub)
    reasoner = LatentReasoner(bundle, cfg.latent, adapter_dir=args.model)
    rows = []
    for ex in examples:
        inp = student_input(ex, cfg)
        reasoner.set_schema(ex.db_path, ex.db_id)
        row = {"qid": ex.qid, "question": ex.question, "gold_sql": ex.gold_sql}
        for j, role in enumerate(cfg.latent.roles[:-1], 1):
            row[f"after_{role}"] = reasoner.language_mode(
                inp["system"], inp["prompt"], n_latent_roles=j, max_new_tokens=args.max_new_tokens)
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
