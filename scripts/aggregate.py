#!/usr/bin/env python
"""Mean EX with 95% CI across runs (Table 1, Table 2).

    python scripts/aggregate.py --runs outputs/qwen3_8b_seed*/eval --out results/qwen3_8b.md

Each ``--runs`` entry is an evaluation directory containing ``report.json``.
The paper reports 10 runs per method.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import BENCHMARKS
from latentsql.eval.metrics import mean_ci95


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="LatentSQL")
    ap.add_argument("--out", default=None, help="write a markdown table here")
    args = ap.parse_args()

    dirs = sorted({d for pat in args.runs for d in glob.glob(pat)})
    reports = []
    for d in dirs:
        path = os.path.join(d, "report.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                reports.append(json.load(fh)["benchmarks"])
    if not reports:
        print("no report.json found")
        return 1
    seen = {b for r in reports for b in r}
    names = [b for b in BENCHMARKS if b in seen] + sorted(seen - set(BENCHMARKS))

    rows = {}
    for b in names:
        vals = [r[b]["ex"] for r in reports if b in r]
        stats = mean_ci95(vals)
        stats["avg_generated_tokens"] = round(
            sum(r[b]["avg_generated_tokens"] for r in reports if b in r) / len(vals), 1)
        stats["avg_latency_s"] = round(
            sum(r[b]["avg_latency_s"] for r in reports if b in r) / len(vals), 3)
        rows[b] = stats

    head = "| Method | " + " | ".join(names) + " |\n|---|" + "---:|" * len(names) + "\n"
    cells = " | ".join(f"{rows[b]['mean']:.2f}±{rows[b]['ci95']:.1f}" for b in names)
    table = head + f"| {args.name} | {cells} |\n"
    print(f"{len(reports)} runs\n")
    print(table)
    print(json.dumps(rows, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(f"EX (%) over {len(reports)} runs, mean ± 95% CI\n\n{table}\n")
            fh.write("```json\n" + json.dumps(rows, indent=2) + "\n```\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
