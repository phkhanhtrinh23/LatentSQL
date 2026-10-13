#!/usr/bin/env python
"""Step 3 (Sec. 4.3): Dr. GRPO with dense and spurious rewards.

    python scripts/train_rl.py --config configs/qwen3_8b.yaml

Starts from ``<output_dir>/latent/final`` and writes ``<output_dir>/rl/final``.
RL uses the training questions of ``data.train_benchmarks`` with their gold
queries; no teacher is needed.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.data.inputs import student_input
from latentsql.data.loaders import dataset_counts, load_benchmarks
from latentsql.latent.reasoner import LatentReasoner
from latentsql.latent.student import load_student
from latentsql.rl.grpo import GRPOTrainer
from latentsql.rl.rewards import ExecutionOracle, LatentSQLReward
from latentsql.utils.common import add_common_args, banner, set_seed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--init", default=None, help="default <output_dir>/latent/final")
    ap.add_argument("--out", default=None, help="default <output_dir>/rl")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.rl.seed)
    init = args.init or os.path.join(cfg.paths.output_dir, "latent", "final")
    out = args.out or os.path.join(cfg.paths.output_dir, "rl")
    os.makedirs(out, exist_ok=True)
    cfg.dump(os.path.join(out, "config.json"))

    banner("RL questions")
    examples = load_benchmarks(cfg, "train", limit=cfg.data.max_train_per_benchmark)
    if cfg.rl.max_train_examples:
        import random

        random.Random(cfg.rl.seed).shuffle(examples)
        examples = examples[: cfg.rl.max_train_examples]
    print(f"{len(examples)} questions: {dataset_counts(examples)}")
    tasks = []
    for ex in examples:
        inp = student_input(ex, cfg)
        tasks.append({"qid": ex.qid, "system": inp["system"], "prompt": inp["prompt"],
                      "db_path": ex.db_path, "db_id": ex.db_id, "gold_sql": ex.gold_sql})

    banner(f"student from {init}")
    bundle = load_student(cfg, adapter_dir=init, trainable=True, hub_dir=cfg.paths.hf_hub)
    reasoner = LatentReasoner(bundle, cfg.latent, adapter_dir=init)
    oracle = ExecutionOracle(cfg.rl, cache_dir=os.path.join(cfg.paths.cache_dir, "rl"),
                             timeout=cfg.eval.execution_timeout)
    trainer = GRPOTrainer(bundle, reasoner, LatentSQLReward(cfg.rl, oracle), cfg.rl)
    result = trainer.train(tasks, out)
    print(f"{result['optimizer_steps']} steps; final student -> {os.path.join(out, 'final')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
