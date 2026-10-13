#!/usr/bin/env python
"""Step 4 (Sec. 4.4, 5): evaluate a latent student on the test splits.

    python scripts/evaluate.py --config configs/qwen3_8b.yaml --model outputs/.../rl/final

Writes ``predictions.jsonl`` and ``report.json`` (EX per benchmark and per
difficulty, generated tokens, latent steps, reasoning KV positions, latency).
``--latent-roles k`` decodes after the first k roles (early decoding).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.data.loaders import dataset_counts, load_benchmarks
from latentsql.eval.evaluate import evaluate
from latentsql.eval.metrics import report, save_predictions
from latentsql.latent.reasoner import LatentReasoner
from latentsql.latent.student import load_student
from latentsql.utils.common import add_common_args, banner, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--model", default=None, help="adapter dir (default <output_dir>/rl/final)")
    ap.add_argument("--out", default=None, help="default <output_dir>/eval")
    ap.add_argument("--benchmarks", default=None, help="comma-separated (default data.eval_benchmarks)")
    ap.add_argument("--latent-roles", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    model = args.model or os.path.join(cfg.paths.output_dir, "rl", "final")
    out = args.out or os.path.join(cfg.paths.output_dir, "eval")
    names = args.benchmarks.split(",") if args.benchmarks else None

    examples = load_benchmarks(cfg, "test", names, limit=cfg.data.max_eval_per_benchmark)
    print(f"{len(examples)} test questions: {dataset_counts(examples)}")

    banner(f"latent student {model}")
    bundle = load_student(cfg, adapter_dir=model, trainable=False, hub_dir=cfg.paths.hf_hub)
    reasoner = LatentReasoner(bundle, cfg.latent, adapter_dir=model)
    preds = evaluate(examples, reasoner, cfg, n_latent_roles=args.latent_roles,
                     stream_path=os.path.join(out, "predictions.partial.jsonl"))
    save_predictions(preds, os.path.join(out, "predictions.jsonl"))
    os.remove(os.path.join(out, "predictions.partial.jsonl"))
    rep = report(preds)
    write_json({"model": model, "latent_roles": args.latent_roles, "benchmarks": rep},
               os.path.join(out, "report.json"))
    print(json.dumps({k: {"n": v["n"], "ex": v["ex"]} for k, v in rep.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
