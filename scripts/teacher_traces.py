#!/usr/bin/env python
"""Step 1 (Sec. 4.1): run the three-role teacher on the training questions.

    python scripts/teacher_traces.py --config configs/qwen3_8b.yaml

Writes ``<output_dir>/teacher/traces.jsonl`` (every trace, with ``correct`` =
EX of the teacher's final query) and ``summary.json``. Re-running resumes:
questions already in ``traces.jsonl`` are skipped, so an interrupted API run
loses nothing. The value indexes are built on first use and cached.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.data.loaders import dataset_counts, load_benchmarks
from latentsql.teacher.roles import run_teacher
from latentsql.utils.common import add_common_args, banner, make_teacher_backend, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out", default=None, help="traces.jsonl (default <output_dir>/teacher/)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    out = args.out or os.path.join(cfg.paths.output_dir, "teacher", "traces.jsonl")

    banner("loading training questions")
    examples = load_benchmarks(cfg, "train", limit=cfg.data.max_train_per_benchmark)
    print(f"{len(examples)} questions: {dataset_counts(examples)}")

    banner(f"teacher: {cfg.teacher.backend} / {cfg.teacher.model}")
    llm = make_teacher_backend(cfg, examples)
    traces = run_teacher(examples, cfg, llm, out)

    kept = [t for t in traces if t.correct]
    per_ds = {}
    for t in traces:
        d = per_ds.setdefault(t.dataset, {"n": 0, "kept": 0})
        d["n"] += 1
        d["kept"] += int(t.correct)
    summary = {
        "n_traces": len(traces), "n_kept": len(kept),
        "keep_rate": round(len(kept) / max(1, len(traces)), 4),
        "repaired_by_reviser": sum(1 for t in kept if (t.draft_sql or "") != (t.final_sql or "")),
        "per_benchmark": per_ds,
        "avg_teacher_tokens": round(sum(t.teacher_tokens for t in traces) / max(1, len(traces)), 1),
        "teacher": {"backend": cfg.teacher.backend, "model": cfg.teacher.model},
        "traces": out,
    }
    write_json(summary, os.path.join(os.path.dirname(out), "summary.json"))
    print(f"kept {len(kept)}/{len(traces)} traces with EX = 1 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
