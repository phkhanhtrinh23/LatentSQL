"""Small shared helpers: seeding, JSON io, and dataset resolution."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    except Exception:
        pass


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_examples(
    cfg: Any,
    dataset: Optional[str] = None,
    limit: Optional[int] = None,
    split: str = "dev",
    fallback_to_synthetic: bool = True,
    synthetic_root: Optional[str] = None,
    shuffle: bool = True,
) -> Tuple[List[Any], str]:
    """Load BIRD/Spider if present, otherwise the synthetic fixture.

    Returns ``(examples, dataset_name_actually_used)`` so callers can label
    their results honestly when they fell back.
    """
    from ..data.loaders import available, load_dataset
    from ..data.synthetic import load_synthetic

    name = (dataset or cfg.eval.dataset or "bird").lower()
    roots = {"bird": cfg.paths.bird_root, "spider": cfg.paths.spider_root}

    if name in roots and available(roots[name], name):
        return (
            load_dataset(
                roots[name], dataset=name, split=split, limit=limit,
                shuffle=shuffle, seed=cfg.seed,
            ),
            name,
        )
    if name in roots and not fallback_to_synthetic:
        raise FileNotFoundError(
            f"{name} not found under {roots[name]!r}. Set paths.{name}_root "
            f"(or $LATENTSQL_{name.upper()}_ROOT) to the FusionSQL directory."
        )
    root = synthetic_root or os.path.join(cfg.paths.output_dir, "synthetic")
    return load_synthetic(root, limit=limit), "synthetic"


def make_backend(cfg: Any, examples: Sequence[Any] = ()) -> Any:
    """Build the teacher backend; seed the mock oracle when in mock mode."""
    from ..llm.base import build_backend

    llm = build_backend(cfg.llm)
    if cfg.llm.backend == "mock" and examples:
        llm.set_oracle(examples)
    return llm


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)
