#!/usr/bin/env python
"""Evaluate any subset of methods on the same held-out questions.

    python scripts/04_eval.py --config configs/bird_local.yaml \
        --methods single,din_sql,dail_sql,chess,chase_sql,fusion_mas,student,latent \
        --student outputs/grpo --limit 100

Produces the IMAD-style table: execution accuracy next to token consumption.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.data.loaders import load_examples
from latentsql.eval.metrics import render_table, save_predictions, save_reports
from latentsql.eval.run_eval import run
from latentsql.utils.common import banner, make_backend, resolve_examples, set_seed

STUDENT_METHODS = {"student", "latent", "sft_trace"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--methods", default="single,fusion_mas,student")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--heldout", default=None, help="jsonl written by 01_build_traces.py")
    ap.add_argument("--student", default=None, help="adapter dir for student/latent modes")
    ap.add_argument("--base", default=None, help="base model for the student")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ves", action="store_true", help="also compute BIRD VES (slow)")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--latent-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.llm.backend = args.backend
    if args.limit:
        cfg.eval.n_samples = args.limit
    if args.latent_steps:
        cfg.latent.latent_steps = args.latent_steps
    set_seed(cfg.seed)
    out_dir = args.out or os.path.join(cfg.paths.output_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    banner("loading held-out questions")
    if args.heldout and os.path.exists(args.heldout):
        examples = load_examples(args.heldout)[: args.limit or None]
        used = examples[0].dataset if examples else "unknown"
    else:
        examples, used = resolve_examples(
            cfg, dataset=args.dataset, limit=args.limit or cfg.eval.n_samples, split="dev"
        )
    print(f"dataset={used}  n={len(examples)}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    needs_llm = any(m not in STUDENT_METHODS for m in methods)
    needs_student = any(m in STUDENT_METHODS for m in methods)

    llm = make_backend(cfg, examples) if needs_llm else None
    bundle = None
    if needs_student:
        from latentsql.llm.hf_local import load_model, resolve_model_path

        student_dir = args.student or cfg.grpo.output_dir
        base = args.base or cfg.sft.model_name_or_path
        adapter = student_dir if os.path.exists(
            os.path.join(student_dir, "adapter_config.json")
        ) else None
        base_path = resolve_model_path(base, hub_dir=cfg.paths.hf_hub, local_only=False)
        print(f"student: base={base_path} adapter={adapter}")
        bundle = load_model(
            base_path, device=cfg.eval.seed and "auto" or "auto",
            hub_dir=cfg.paths.hf_hub, adapter_path=adapter,
        )

    reports = []
    for method in methods:
        banner(f"evaluating: {method}")
        if llm is not None:
            llm.reset_usage()
        preds, report = run(
            method, examples, cfg, llm=llm, bundle=bundle,
            pool=[], cache_dir=cfg.paths.cache_dir, compute_ves=args.ves,
        )
        save_predictions(preds, os.path.join(out_dir, f"preds_{method}.jsonl"))
        reports.append(report)
        print(report.as_row())

    save_reports(reports, os.path.join(out_dir, "reports.json"))
    table = render_table(reports)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Results ({used}, n={len(examples)})\n\n{table}\n")
    banner("results")
    print(table)
    print(f"\nwrote {out_dir}/results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
