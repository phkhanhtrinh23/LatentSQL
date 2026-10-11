#!/usr/bin/env python3
"""Prepare retrieval-consistent latent curriculum records once.

The staged latent trainer runs many fresh subprocesses for OOM isolation.
Precomputing prevents every subprocess from rebuilding value retrieval and
tokenizing the entire trace corpus before selecting its small training slice.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.distill.latent_sft import (
    build_records_from_traces, chess_reviser_turns_for, chess_step_turns_for,
    reviser_repair_turns_for,
)
from latentsql.llm.hf_local import resolve_model_path
from latentsql.mas.trace import load_traces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--max-prompt-chars", type=int, default=4000)
    ap.add_argument("--max-role-chars", type=int, default=2000)
    ap.add_argument("--max-total-tokens", type=int, default=1600)
    ap.add_argument("--require-role-turns", action="store_true")
    ap.add_argument("--chain-format", choices=("chess_steps", "chess_reviser", "reviser_repair"),
                    default="chess_steps",
                    help="chess_reviser = three teacher roles with an executing reviser r3; "
                         "reviser_repair = the same, with repairs drawn from every teacher")
    ap.add_argument("--repeat-fixes", type=int, default=1,
                    help="write each record whose reviser turn repairs the draft this many times")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.latent.roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    from transformers import AutoTokenizer

    base = args.base or cfg.sft.model_name_or_path
    base_path = resolve_model_path(base, hub_dir=cfg.paths.hf_hub, local_only=False)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    traces = load_traces(args.traces)
    turns_for = {"chess_reviser": chess_reviser_turns_for,
                 "reviser_repair": reviser_repair_turns_for}.get(args.chain_format,
                                                                 chess_step_turns_for)
    if args.require_role_turns:
        traces = [
            trace for trace in traces
            if turns_for(trace, cfg.latent.roles) is not None
        ]
        print(f"pre-filtered to {len(traces)} traces with complete required roles")
    records = build_records_from_traces(
        traces, cfg,
        max_prompt_chars=args.max_prompt_chars,
        max_role_chars=args.max_role_chars,
        tokenizer=tokenizer,
        max_total_tokens=args.max_total_tokens,
        chain_format=args.chain_format,
    )
    if args.require_role_turns:
        records = [record for record in records if record.get("role_turns") is not None]
    n_fix = sum(any("Diagnosis and fix" in t for t in (r.get("role_turns") or []))
                for r in records)
    n_err = sum(any("Execute the draft once.\nError:" in t for t in (r.get("role_turns") or []))
                for r in records)
    print(f"{len(records)} questions: {n_fix} repaired drafts ({n_err} from an execution "
          f"error), {len(records) - n_fix} kept drafts")
    if args.repeat_fixes > 1:
        records = records + [
            r for r in records
            if any("Diagnosis and fix" in t for t in (r.get("role_turns") or []))
        ] * (args.repeat_fixes - 1)
        # batches are cut in file order, so mix the copies in
        random.Random(0).shuffle(records)
        print(f"repeated repaired records x{args.repeat_fixes}: {len(records)} records (shuffled)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with_hints = sum(bool(record.get("retrieved_values")) for record in records)
    print(
        f"wrote {len(records)} records to {args.out}; "
        f"external retrieval returned values for {with_hints}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
