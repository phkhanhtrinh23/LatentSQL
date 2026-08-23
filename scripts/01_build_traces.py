#!/usr/bin/env python
"""Stage 0: run the FusionSQL teacher and build the distillation corpus.

    python scripts/01_build_traces.py --config configs/bird_local.yaml --limit 2000

Writes ``traces.jsonl`` (full transcripts, used later for steering analysis)
and ``sft.jsonl`` (filtered prompt/completion pairs for stage 1).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.data.loaders import split_train_dev
from latentsql.distill.build_dataset import build_sft_records, run_teacher, save_records
from latentsql.mas.trace import save_traces
from latentsql.utils.common import banner, make_backend, resolve_examples, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="bird | spider | synthetic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--backend", default=None, help="override llm.backend")
    ap.add_argument("--model", default=None, help="override llm.model")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.llm.backend = args.backend
    if args.model:
        cfg.llm.model = args.model
    set_seed(cfg.seed)

    out_dir = args.out or os.path.join(cfg.paths.output_dir, "traces")
    os.makedirs(out_dir, exist_ok=True)

    banner("loading data")
    examples, used = resolve_examples(
        cfg, dataset=args.dataset, limit=args.limit, split=args.split
    )
    train, dev = split_train_dev(examples, dev_frac=args.dev_frac, seed=cfg.seed)
    print(f"dataset={used}  total={len(examples)}  teacher-train={len(train)}  held-out={len(dev)}")

    banner(f"running the teacher (backend={cfg.llm.backend}, model={cfg.llm.model})")
    llm = make_backend(cfg, examples)
    traces = run_teacher(
        train, cfg, llm,
        example_pool=train,                      # DAIL-SQL shots come from train only
        cache_dir=cfg.paths.cache_dir,
        log_path=os.path.join(out_dir, "teacher_samples.jsonl"),
    )

    trace_path = os.path.join(out_dir, "traces.jsonl")
    save_traces(traces, trace_path)

    banner("filtering into the SFT corpus")
    records, stats = build_sft_records(traces, cfg.trace)
    for r in records:
        if not r.db_path:
            match = next((e for e in train if e.qid == r.qid), None)
            if match:
                r.db_path = match.db_path
    sft_path = os.path.join(out_dir, "sft.jsonl")
    save_records(records, sft_path)

    n_correct = sum(1 for t in traces if t.correct)
    summary = {
        "dataset": used,
        "n_examples": len(train),
        "teacher_ex": round(n_correct / max(1, len(traces)), 4),
        "avg_teacher_tokens": round(
            sum(t.teacher_tokens for t in traces) / max(1, len(traces)), 1
        ),
        "avg_teacher_calls": round(
            sum(t.teacher_calls for t in traces) / max(1, len(traces)), 2
        ),
        "filter": stats.as_dict(),
        "traces_path": trace_path,
        "sft_path": sft_path,
        "backend": cfg.llm.backend,
        "model": cfg.llm.model,
        "quantization": getattr(cfg.llm, "quantization", "none"),
        "panel": list(cfg.mas.agents),
        "rounds": cfg.mas.rounds,
        "sample_log": os.path.join(out_dir, "teacher_samples.jsonl"),
    }
    write_json(summary, os.path.join(out_dir, "summary.json"))
    from latentsql.data.loaders import save_examples

    save_examples(dev, os.path.join(out_dir, "heldout.jsonl"))

    print(f"\nteacher execution accuracy: {summary['teacher_ex'] * 100:.2f}%")
    print(f"avg teacher tokens/question: {summary['avg_teacher_tokens']}")
    print(stats.summary())
    print(f"\nwrote {trace_path}\n      {sft_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
