"""Small shared helpers: seeding, JSON io, and dataset resolution."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def limit_resources() -> Dict[str, Any]:
    """Cap GPU memory and CPU threads so a long run cannot destabilise the box.

    An RTX 4090 draws transient spikes well above its steady wattage, which is a
    common cause of sudden PSU shutdowns during multi-hour runs, and torch will
    otherwise spin up one CPU thread per core for tokenisation and SQL scoring.
    Both caps are advisory defaults, overridable per run:

    * ``LATENTSQL_GPU_FRACTION`` - fraction of VRAM this process may allocate
      (default 0.5, i.e. half the card).
    * ``LATENTSQL_CPU_THREADS``  - torch/OMP thread cap (default: a quarter of
      the cores, at least 2, at most 8).

    Note this bounds *allocation*, not power draw; lowering the board power
    limit needs root (``sudo nvidia-smi -pl <watts>``).
    """
    info: Dict[str, Any] = {}

    n_cpu = os.cpu_count() or 4
    threads = int(os.environ.get("LATENTSQL_CPU_THREADS", max(2, min(8, n_cpu // 4))))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    info["cpu_threads"] = threads

    try:
        import torch

        torch.set_num_threads(threads)
        if torch.cuda.is_available():
            frac = float(os.environ.get("LATENTSQL_GPU_FRACTION", "0.5"))
            if 0.0 < frac < 1.0:
                torch.cuda.set_per_process_memory_fraction(frac)
                total = torch.cuda.get_device_properties(0).total_memory / 2**30
                info["gpu_fraction"] = frac
                info["gpu_cap_gb"] = round(frac * total, 2)
    except Exception:
        pass
    return info


def set_seed(seed: int) -> None:
    caps = limit_resources()
    if caps:
        print(f"[limits] {caps}", flush=True)
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
        json.dump(obj, fh, indent=4, default=str)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


#: names that select the combined BIRD + Spider set
MIX_ALIASES = frozenset({"mix", "mixed", "bird+spider", "spider+bird", "combined"})


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
    from ..data.loaders import available, load_dataset, load_mixed
    from ..data.synthetic import load_synthetic

    name = (dataset or cfg.eval.dataset or "bird").lower()
    roots = {"bird": cfg.paths.bird_root, "spider": cfg.paths.spider_root}

    if name in MIX_ALIASES:
        mix = getattr(cfg, "mix", None)
        n_bird = getattr(mix, "bird", None) if mix else None
        n_spider = getattr(mix, "spider", None) if mix else None
        if available(roots["bird"], "bird") or available(roots["spider"], "spider"):
            return (
                load_mixed(
                    roots["bird"], roots["spider"],
                    n_bird=n_bird, n_spider=n_spider,
                    bird_split=(getattr(mix, "bird_split", None) if mix else None) or split,
                    spider_split=(getattr(mix, "spider_split", None) if mix else None) or split,
                    limit=limit, seed=cfg.seed,
                    shuffle=getattr(mix, "shuffle", shuffle) if mix else shuffle,
                ),
                "mix",
            )
        if not fallback_to_synthetic:
            raise FileNotFoundError(
                f"neither bird ({roots['bird']!r}) nor spider ({roots['spider']!r}) "
                f"is readable; set paths.bird_root / paths.spider_root"
            )
    elif name in roots and available(roots[name], name):
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
