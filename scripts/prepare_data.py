#!/usr/bin/env python
"""Convert one benchmark's official release into data/<benchmark>/{train,test}.jsonl.

    python scripts/prepare_data.py --benchmark spider --raw /path/to/spider
    python scripts/prepare_data.py --benchmark livesqlbench --raw /path/to/lsb --gt lsb_gt.jsonl

See the README ("Data") for the expected layout of each ``--raw`` directory.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import BENCHMARKS
from latentsql.data.convert import CONVERTERS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--raw", required=True, help="directory of the official download")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--gt", default=None, help="LiveSQLBench: file with the released solutions")
    ap.add_argument("--heldout-fraction", type=float, default=0.3,
                    help="KaggleDBQA / LiveSQLBench test fraction (no training split)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.abspath(os.path.join(args.data_dir, args.benchmark))
    os.makedirs(out, exist_ok=True)
    counts = CONVERTERS[args.benchmark](
        os.path.abspath(args.raw), out, gt=args.gt,
        heldout_fraction=args.heldout_fraction, seed=args.seed,
    )
    print(f"{args.benchmark}: train={counts['train']} test={counts['test']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
