#!/usr/bin/env python
"""Appendix C.3, Table 6: Recall@k of the latent schema distribution pihat.

    python scripts/analysis/schema_recall.py --config configs/qwen3_8b.yaml \
        --model outputs/.../rl/final --benchmarks bird

Runs role r1 only and reads pihat (Eq. 14) at its first and last latent step.
Recall@k is the share of gold tables and columns C* among the top-k items,
averaged over questions. ``--set latent.schema_beta=0`` gives the "w/o FK
propagation" row.
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
from latentsql.latent.schema_align import gold_schema_items
from latentsql.latent.student import load_student
from latentsql.utils.common import add_common_args, write_json


def recall_at(pi, keys, gold, k):
    top = {keys[i].lower() for i in pi.reshape(-1).topk(min(k, len(keys))).indices.tolist()}
    return len(top & gold) / len(gold)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", default="bird")
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if not cfg.latent.schema_alignment:
        raise SystemExit("schema recall needs latent.schema_alignment = true")
    ks = [int(k) for k in args.ks.split(",")]

    examples = load_benchmarks(cfg, "test", args.benchmarks.split(","),
                               limit=cfg.data.max_eval_per_benchmark)
    bundle = load_student(cfg, adapter_dir=args.model, trainable=False, hub_dir=cfg.paths.hf_hub)
    reasoner = LatentReasoner(bundle, cfg.latent, adapter_dir=args.model)
    sums = {f"{w}_recall@{k}": 0.0 for w in ("first_step", "last_step") for k in ks}
    n = 0
    for ex in examples:
        gold = {g.lower() for g in gold_schema_items(ex.gold_sql, ex.db_path, ex.db_id)}
        if not gold:
            continue
        inp = student_input(ex, cfg)
        reasoner.set_schema(ex.db_path, ex.db_id)
        reasoner.prefix(inp["system"], inp["prompt"], n_latent_roles=1, add_sql_tag=False)
        pis, keys = reasoner.schema_pis, reasoner.schema_keys()
        if not pis:
            continue
        for k in ks:
            sums[f"first_step_recall@{k}"] += recall_at(pis[0], keys, gold, k)
            sums[f"last_step_recall@{k}"] += recall_at(pis[-1], keys, gold, k)
        n += 1
    result = {"n": n, "beta": cfg.latent.schema_beta,
              **{k: round(100 * v / max(1, n), 1) for k, v in sums.items()}}
    print(json.dumps(result, indent=2))
    if args.out:
        write_json(result, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
