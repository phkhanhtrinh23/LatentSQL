"""Small shared helpers: seeding, JSON io, script plumbing."""
from __future__ import annotations

import json
import os
import random
from typing import Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # optional cap: LATENTSQL_GPU_FRACTION=0.5 lets two jobs share one GPU
        frac = os.environ.get("LATENTSQL_GPU_FRACTION")
        if frac and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(frac))
    except Exception:
        pass


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=4, default=str)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


def add_common_args(ap: Any) -> None:
    """``--config`` and repeatable ``--set section.key=value`` overrides."""
    ap.add_argument("--config", default=None, help="YAML config (see configs/)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override one config value, e.g. --set rl.lr=2e-6 (repeatable)")


def make_teacher_backend(cfg: Any, examples: Any = ()) -> Any:
    from ..llm.base import build_backend

    llm = build_backend(cfg.teacher)
    if cfg.teacher.backend == "mock":
        llm.set_oracle(examples)
    return llm
