"""One example format for all eight benchmarks.

``scripts/prepare_data.py`` converts every benchmark once into::

    <data_dir>/<benchmark>/train.jsonl
    <data_dir>/<benchmark>/test.jsonl
    <data_dir>/<benchmark>/<database files or links>

Each line is an :class:`Example`. ``db_path`` is stored relative to the
benchmark folder, so the converted data can be moved as a whole.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Example:
    qid: str
    db_id: str
    question: str
    gold_sql: str
    #: external knowledge K (BIRD evidence, LiveSQLBench knowledge, ...)
    evidence: str = ""
    #: interaction history H: earlier questions of the same dialogue (SParC, CoSQL)
    history: List[str] = field(default_factory=list)
    dataset: str = ""
    db_path: str = ""
    difficulty: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Example":
        keys = set(Example.__dataclass_fields__)
        return Example(**{k: v for k, v in d.items() if k in keys})


def save_examples(examples: Sequence[Example], path: str, relative_to: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in examples:
            d = e.to_dict()
            if relative_to and d["db_path"] and os.path.isabs(d["db_path"]):
                d["db_path"] = os.path.relpath(d["db_path"], relative_to)
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def load_examples(path: str, root: Optional[str] = None) -> List[Example]:
    """Read a jsonl split; relative ``db_path`` values are resolved against ``root``."""
    root = root if root is not None else os.path.dirname(os.path.abspath(path))
    out: List[Example] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            ex = Example.from_dict(json.loads(line))
            if ex.db_path and not os.path.isabs(ex.db_path):
                ex.db_path = os.path.normpath(os.path.join(root, ex.db_path))
            out.append(ex)
    return out


def split_path(data_dir: str, benchmark: str, split: str) -> str:
    return os.path.join(data_dir, benchmark, f"{split}.jsonl")


def load_split(data_dir: str, benchmark: str, split: str, limit: Optional[int] = None,
               keep_unanswerable: bool = False, require_db: bool = True) -> List[Example]:
    """Load ``<data_dir>/<benchmark>/<split>.jsonl`` (split = train | test)."""
    path = split_path(data_dir, benchmark, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Convert the benchmark first: "
            f"python scripts/prepare_data.py --benchmark {benchmark} --raw <raw dir>")
    examples = load_examples(path)
    if not keep_unanswerable:
        examples = [e for e in examples if e.gold_sql and e.gold_sql.strip().lower() not in ("nan", "null")]
    if require_db:
        examples = [e for e in examples if e.db_path and os.path.exists(e.db_path)]
    return examples[:limit] if limit else examples


def load_benchmarks(cfg: Any, split: str, benchmarks: Optional[Sequence[str]] = None,
                    limit: Optional[int] = None) -> List[Example]:
    """Concatenate one split of several benchmarks, ``limit`` questions each.

    ``benchmarks = ["synthetic"]`` builds the offline fixture instead.
    """
    names = list(benchmarks or (cfg.data.train_benchmarks if split == "train"
                                else cfg.data.eval_benchmarks))
    out: List[Example] = []
    for name in names:
        if name == "synthetic":
            from .synthetic import load_synthetic

            out += load_synthetic(os.path.join(cfg.paths.data_dir, "synthetic"), split=split,
                                  limit=limit)
            continue
        out += load_split(cfg.paths.data_dir, name, split, limit=limit,
                          keep_unanswerable=cfg.data.keep_unanswerable)
    return out


def random_split(examples: Sequence[Example], heldout_fraction: float = 0.3,
                 seed: int = 0) -> Tuple[List[Example], List[Example]]:
    """Fixed-seed random split for benchmarks without a training split (Appendix A.1)."""
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_test = int(round(len(idx) * heldout_fraction))
    test_idx = set(idx[:n_test])
    train = [e for i, e in enumerate(examples) if i not in test_idx]
    test = [e for i, e in enumerate(examples) if i in test_idx]
    return train, test


def dataset_counts(examples: Sequence[Example]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in examples:
        counts[e.dataset] = counts.get(e.dataset, 0) + 1
    return counts
