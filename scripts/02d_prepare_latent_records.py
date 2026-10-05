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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.distill.latent_sft import build_records_from_traces, chess_step_turns_for
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.latent.roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    from transformers import AutoTokenizer

    base = args.base or cfg.sft.model_name_or_path
    base_path = resolve_model_path(base, hub_dir=cfg.paths.hf_hub, local_only=False)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    traces = load_traces(args.traces)
    if args.require_role_turns:
        traces = [
            trace for trace in traces
            if chess_step_turns_for(trace, cfg.latent.roles) is not None
        ]
        print(f"pre-filtered to {len(traces)} traces with complete required roles")
    records = build_records_from_traces(
        traces, cfg,
        max_prompt_chars=args.max_prompt_chars,
        max_role_chars=args.max_role_chars,
        tokenizer=tokenizer,
        max_total_tokens=args.max_total_tokens,
        chain_format="chess_steps",
    )
    if args.require_role_turns:
        records = [record for record in records if record.get("role_turns") is not None]
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
