#!/usr/bin/env python
"""Agent-subspace analysis (IMAD section 3).

    python scripts/05_steering.py --config configs/bird_local.yaml \
        --traces outputs/traces/traces.jsonl --student outputs/grpo

Extracts one difference-in-means direction per teacher agent, reports how
separable they are, and compares the base model against the internalised
student on the same probes.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.llm.hf_local import load_model, resolve_model_path
from latentsql.mas.trace import load_traces
from latentsql.steering.caa import (
    cosine_separation,
    extract_steering_vectors,
    generate_steered,
)
from latentsql.utils.common import banner, set_seed, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--traces", default="outputs/traces/traces.jsonl")
    ap.add_argument("--student", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--compare-base", action="store_true",
                    help="also extract vectors from the un-tuned base model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.layer is not None:
        cfg.steering.layer = args.layer
    set_seed(cfg.seed)
    out_dir = args.out or os.path.join(cfg.paths.output_dir, "steering")
    os.makedirs(out_dir, exist_ok=True)

    banner("loading traces")
    traces = load_traces(args.traces)
    agents = sorted({t.agent for tr in traces for t in tr.turns})
    print(f"{len(traces)} traces | agents: {agents}")

    base = args.base or cfg.sft.model_name_or_path
    base_path = resolve_model_path(base, hub_dir=cfg.paths.hf_hub, local_only=False)
    student_dir = args.student or cfg.grpo.output_dir
    adapter = student_dir if os.path.exists(
        os.path.join(student_dir, "adapter_config.json")
    ) else None

    results = {}
    banner(f"extracting agent subspaces from the student (layer {cfg.steering.layer})")
    bundle = load_model(base_path, hub_dir=cfg.paths.hf_hub, adapter_path=adapter)
    vecs = extract_steering_vectors(bundle, traces, agents, cfg.steering)
    vecs.save(os.path.join(out_dir, "steering_student.pt"))
    results["student"] = {
        "layer": vecs.layer,
        "n_pairs": {k: v["n_pairs"] for k, v in vecs.meta.items()},
        "pairwise_cosine": cosine_separation(vecs),
    }
    print("pairwise cosine between agent directions (lower = more separable):")
    for k, v in results["student"]["pairwise_cosine"].items():
        print(f"  {k:<36} {v:+.4f}")

    if args.compare_base:
        banner("same probes on the un-tuned base model")
        base_bundle = load_model(base_path, hub_dir=cfg.paths.hf_hub)
        bvecs = extract_steering_vectors(base_bundle, traces, agents, cfg.steering)
        bvecs.save(os.path.join(out_dir, "steering_base.pt"))
        results["base"] = {
            "layer": bvecs.layer,
            "pairwise_cosine": cosine_separation(bvecs),
        }
        print("base-model pairwise cosine:")
        for k, v in results["base"]["pairwise_cosine"].items():
            print(f"  {k:<36} {v:+.4f}")

        s = results["student"]["pairwise_cosine"]
        b = results["base"]["pairwise_cosine"]
        shared = set(s) & set(b)
        if shared:
            delta = sum(abs(b[k]) - abs(s[k]) for k in shared) / len(shared)
            results["separability_gain"] = round(delta, 4)
            print(
                f"\nmean |cos| drop after internalisation: {delta:+.4f} "
                f"(positive = more separable agent subspaces)"
            )

    write_json(results, os.path.join(out_dir, "steering_report.json"))
    print(f"\nwrote {out_dir}/steering_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
