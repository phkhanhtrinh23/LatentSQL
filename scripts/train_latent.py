#!/usr/bin/env python
"""Step 2 (Sec. 4.2): warm-up on the verified traces, then the text-to-latent curriculum.

    python scripts/train_latent.py --config configs/qwen3_8b.yaml

Reads ``<output_dir>/teacher/traces.jsonl``; writes one checkpoint per stage to
``<output_dir>/latent/stage_k`` and the final student to
``<output_dir>/latent/final``. Finished stages are skipped on a re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.latent.curriculum import build_records, stage_schedule, train_curriculum
from latentsql.latent.student import load_student
from latentsql.teacher.trace import load_traces
from latentsql.utils.common import add_common_args, banner, set_seed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--out", default=None, help="default <output_dir>/latent")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.curriculum.seed)
    traces_path = args.traces or os.path.join(cfg.paths.output_dir, "teacher", "traces.jsonl")
    out = args.out or os.path.join(cfg.paths.output_dir, "latent")
    os.makedirs(out, exist_ok=True)
    cfg.dump(os.path.join(out, "config.json"))

    banner(f"student {cfg.student.model}")
    bundle = load_student(cfg, trainable=True, hub_dir=cfg.paths.hf_hub)

    banner("training records from verified teacher traces")
    traces = load_traces(traces_path)
    records = build_records(traces, cfg, tokenizer=bundle.tokenizer)
    print(f"{len(records)} records from {len(traces)} traces; roles={cfg.latent.roles}, "
          f"m={cfg.latent.latent_steps}, stages={stage_schedule(cfg)}")
    if not records:
        print("no usable records (no trace with EX = 1?)")
        return 1

    summary = train_curriculum(bundle, records, cfg, out)
    print(json.dumps(summary, indent=2))
    print(f"final latent student -> {os.path.join(out, 'final')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
