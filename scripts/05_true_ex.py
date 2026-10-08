#!/usr/bin/env python3
"""Audit ordinary execution accuracy on deterministic stress databases.

True EX credits a prediction only when it matches the gold query on the
original database and on every usable stress instance.  Stress databases are
shared across arms for each question and can be deleted immediately after the
question is scored, keeping the audit practical for large source databases.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.db.execute import execute_sql, gold_is_ordered, result_match
from latentsql.db.spurious import generate_stress_instances


def _load(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _safe_remove_instance_folders(paths: list[str], cache_dir: str) -> None:
    root = os.path.abspath(cache_dir)
    for folder in {os.path.abspath(os.path.dirname(path)) for path in paths}:
        if os.path.commonpath((root, folder)) == root and folder != root:
            shutil.rmtree(folder, ignore_errors=True)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="final predictions JSONL; repeat once per checkpoint",
    )
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.pred:
        if "=" not in spec:
            ap.error(f"--pred must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        rows = _load(path)
        by_qid = {str(row["qid"]): row for row in rows}
        if len(by_qid) != len(rows):
            raise ValueError(f"duplicate qid in {path}")
        arms[label] = by_qid

    qid_sets = [set(rows) for rows in arms.values()]
    if not qid_sets or any(qids != qid_sets[0] for qids in qid_sets[1:]):
        raise ValueError("all prediction files must contain the same qids")

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    details: dict[str, list[dict[str, Any]]] = {label: [] for label in arms}

    ordered_qids = list(next(iter(arms.values())))
    for index, qid in enumerate(ordered_qids, 1):
        reference = next(iter(arms.values()))[qid]
        db_path = reference["db_path"]
        gold_sql = reference["gold_sql"]
        paths: list[str] = []
        if any(bool(rows[qid].get("correct")) for rows in arms.values()):
            paths = generate_stress_instances(
                db_path,
                gold_sql,
                args.cache_dir,
                n=args.instances,
                seed=0,
                timeout=args.timeout,
            )

        gold_results = []
        for path in paths:
            result = execute_sql(path, gold_sql, timeout=args.timeout)
            if result.ok:
                gold_results.append((path, result))

        for label, rows in arms.items():
            row = rows[qid]
            ordinary = bool(row.get("correct"))
            agree = 0
            if ordinary:
                for path, gold_result in gold_results:
                    pred_result = execute_sql(path, row.get("pred_sql"), timeout=args.timeout)
                    if result_match(
                        pred_result,
                        gold_result,
                        ordered=gold_is_ordered(gold_sql),
                    ):
                        agree += 1
            checked = len(gold_results) if ordinary else 0
            robust = ordinary and (checked == 0 or agree == checked)
            details[label].append(
                {
                    "qid": qid,
                    "dataset": row.get("dataset", ""),
                    "db_id": row.get("db_id", ""),
                    "ordinary_correct": ordinary,
                    "stress_instances": checked,
                    "stress_agree": agree,
                    "stress_agreement": (agree / checked) if checked else None,
                    "robust_correct": robust,
                    "spurious_correct": ordinary and checked > 0 and agree < checked,
                }
            )

        if paths and not args.keep_cache:
            _safe_remove_instance_folders(paths, args.cache_dir)
        print(f"[true-ex] {index}/{len(ordered_qids)} {qid} stress={len(gold_results)}", flush=True)

    summary: dict[str, Any] = {}
    for label, rows in details.items():
        n = len(rows)
        ordinary = sum(bool(row["ordinary_correct"]) for row in rows)
        robust = sum(bool(row["robust_correct"]) for row in rows)
        spurious = sum(bool(row["spurious_correct"]) for row in rows)
        stressable = sum(
            bool(row["ordinary_correct"]) and int(row["stress_instances"]) > 0 for row in rows
        )
        by_dataset: dict[str, dict[str, Any]] = {}
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row["dataset"] or "unknown")].append(row)
        for dataset, bucket in sorted(buckets.items()):
            ds_n = len(bucket)
            ds_ordinary = sum(bool(row["ordinary_correct"]) for row in bucket)
            ds_robust = sum(bool(row["robust_correct"]) for row in bucket)
            by_dataset[dataset] = {
                "n": ds_n,
                "ordinary_correct": ds_ordinary,
                "ordinary_ex": ds_ordinary / ds_n,
                "robust_correct": ds_robust,
                "true_ex": ds_robust / ds_n,
            }
        summary[label] = {
            "n": n,
            "ordinary_correct": ordinary,
            "ordinary_ex": ordinary / n,
            "stressable_ordinary_correct": stressable,
            "spurious_correct": spurious,
            "spurious_rate_among_ordinary_correct": spurious / ordinary if ordinary else 0.0,
            "robust_correct": robust,
            "true_ex": robust / n,
            "by_dataset": by_dataset,
        }
        with open(os.path.join(args.out, f"details_{label}.jsonl"), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4)

    lines = [
        "# True execution accuracy",
        "",
        f"Each ordinarily correct prediction was checked on up to {args.instances} deterministic stress instances.",
        "",
        "| Checkpoint | Ordinary EX | Spurious among ordinary-correct | Robust correct | True EX |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in summary.items():
        rate = item["spurious_rate_among_ordinary_correct"]
        lines.append(
            f"| {label} | {_pct(item['ordinary_ex'])} ({item['ordinary_correct']}/{item['n']}) "
            f"| {item['spurious_correct']}/{item['ordinary_correct']} ({_pct(rate)}) "
            f"| {item['robust_correct']}/{item['n']} | {_pct(item['true_ex'])} |"
        )
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
