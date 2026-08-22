#!/usr/bin/env python
"""Show which causal LMs are available in the local HuggingFace hub.

    python scripts/00_list_models.py --hub /mnt/win_d/hub --max-params 7

Prints name, parameter count and path, so you can pick a student for
``sft.model_name_or_path`` and a teacher for ``llm.model`` without guessing.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.llm.hf_local import scan_hub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--hub", default=None, help="override paths.hf_hub")
    ap.add_argument("--max-params", type=float, default=7.0,
                    help="hide models above this many billion params (0 = no filter)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hub = args.hub or cfg.paths.hf_hub
    if not os.path.isdir(hub):
        print(f"hub directory not found: {hub}")
        print("Point --hub (or paths.hf_hub) at your HuggingFace cache.")
        return 1

    models = scan_hub(hub, max_params_b=args.max_params)
    if not models:
        print(f"no causal LMs found under {hub}")
        return 1

    print(f"{'params(B)':>10}  {'name':<48} {'arch':<28} path")
    print("-" * 130)
    for m in models:
        p = f"{m['params_b']:.2f}" if m["params_b"] else "?"
        print(f"{p:>10}  {m['name']:<48} {m['architecture']:<28} {m['path']}")
    print(f"\n{len(models)} model(s). Use a name above for llm.model / sft.model_name_or_path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
