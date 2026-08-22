"""BIRD / Spider loaders for the FusionSQL directory layout.

Expected on the workstation::

    /mnt/win_d/data_FusionSQL/bird/sft_bird_dev_text2sql.json
    /mnt/win_d/data_FusionSQL/bird/dev/dev_databases/<db_id>/<db_id>.sqlite
    /mnt/win_d/data_FusionSQL/spider/sft_spider_dev_text2sql.json
    /mnt/win_d/data_FusionSQL/spider/database/<db_id>/<db_id>.sqlite

Both the manifest filename and the database root are auto-detected, so the
vanilla BIRD (``dev.json`` + ``dev_databases/``) and Spider
(``dev.json`` + ``database/``) layouts also load without configuration.
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Example:
    qid: str
    db_id: str
    question: str
    gold_sql: str
    evidence: str = ""            # BIRD "external knowledge" hint
    dataset: str = "bird"
    db_path: str = ""
    difficulty: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# path discovery
# --------------------------------------------------------------------------
_MANIFESTS = {
    "bird": [
        "sft_bird_dev_text2sql.json",
        "dev/dev.json",
        "dev.json",
        "mini_dev_sqlite.json",
        "sft_bird_train_text2sql.json",
        "train/train.json",
        "train.json",
    ],
    "spider": [
        "sft_spider_dev_text2sql.json",
        "dev.json",
        "sft_spider_train_text2sql.json",
        "train_spider.json",
    ],
}

_DB_ROOTS = {
    "bird": ["dev/dev_databases", "dev_databases", "database", "train/train_databases"],
    "spider": ["database", "db", "test_database"],
}

#: a git-lfs pointer is a small text file, not the data it stands for. Treating
#: one as a manifest fails with a JSONDecodeError three frames deep, so reject
#: it (and zero-byte placeholders) while probing instead.
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def _is_placeholder(path: str) -> bool:
    """True for an empty file or an unfetched git-lfs pointer."""
    try:
        if os.path.isdir(path):
            return False
        if os.path.getsize(path) == 0:
            return True
        with open(path, "rb") as fh:
            return fh.read(len(_LFS_MAGIC)) == _LFS_MAGIC
    except OSError:
        return True


def _first_existing(root: str, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.exists(p) and not _is_placeholder(p):
            return p
    return None


def find_manifest(root: str, dataset: str, split: str = "dev") -> Optional[str]:
    cands = list(_MANIFESTS.get(dataset, []))
    if split != "dev":
        cands = [c for c in cands if split in c] + cands
    hit = _first_existing(root, cands)
    if hit:
        return hit
    # last resort: any json at the top level that looks like a manifest
    for p in sorted(glob.glob(os.path.join(root, "*.json"))):
        if "table" in os.path.basename(p).lower() or _is_placeholder(p):
            continue
        return p
    return None


def find_db_root(root: str, dataset: str, split: str = "dev") -> Optional[str]:
    """Locate the SQLite root, preferring the one belonging to ``split``.

    BIRD ships ``dev/dev_databases`` and ``train/train_databases`` side by
    side. Probing in a fixed order would resolve train questions against the
    dev databases, where ``require_db`` then silently drops nearly all of
    them - so the split's own root is tried first.
    """
    cands = list(_DB_ROOTS.get(dataset, []))
    cands = [c for c in cands if split in c] + cands
    hit = _first_existing(root, cands)
    if hit:
        return hit
    for p in glob.glob(os.path.join(root, "**", "*.sqlite"), recursive=True):
        if _is_placeholder(p):
            continue
        # <db_root>/<db_id>/<db_id>.sqlite
        return os.path.dirname(os.path.dirname(p))
    return None


def resolve_db_path(db_root: str, db_id: str) -> str:
    """Locate ``<db_id>.sqlite`` under a benchmark's database root."""
    for cand in (
        os.path.join(db_root, db_id, f"{db_id}.sqlite"),
        os.path.join(db_root, db_id, f"{db_id}.db"),
        os.path.join(db_root, f"{db_id}.sqlite"),
        os.path.join(db_root, f"{db_id}.db"),
    ):
        if os.path.exists(cand):
            return cand
    matches = glob.glob(os.path.join(db_root, "**", f"{db_id}.sqlite"), recursive=True)
    return matches[0] if matches else os.path.join(db_root, db_id, f"{db_id}.sqlite")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _get(rec: Dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = rec.get(k)
        if v not in (None, ""):
            return str(v)
    return default


def load_dataset(
    root: str,
    dataset: str = "bird",
    split: str = "dev",
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
    require_db: bool = True,
) -> List[Example]:
    """Load a benchmark into ``Example`` records with resolved SQLite paths."""
    manifest = find_manifest(root, dataset, split)
    if manifest is None:
        raise FileNotFoundError(
            f"no {dataset} manifest under {root!r}. Expected one of "
            f"{_MANIFESTS.get(dataset)}"
        )
    db_root = find_db_root(root, dataset, split)
    if db_root is None and require_db:
        raise FileNotFoundError(f"no SQLite database root under {root!r}")

    with open(manifest, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):  # some dumps wrap the list
        raw = raw.get("data") or raw.get("questions") or []

    out: List[Example] = []
    for i, rec in enumerate(raw):
        db_id = _get(rec, "db_id", "database_id")
        gold = _get(rec, "SQL", "sql", "query", "gold_sql")
        question = _get(rec, "question", "text", "nl")
        if not (db_id and question):
            continue
        ex = Example(
            qid=_get(rec, "question_id", "qid", default=str(i)),
            db_id=db_id,
            question=question,
            gold_sql=gold,
            evidence=_get(rec, "evidence", "hint", "knowledge"),
            dataset=dataset,
            difficulty=_get(rec, "difficulty"),
            db_path=resolve_db_path(db_root, db_id) if db_root else "",
        )
        ex.qid = f"{dataset}_{split}_{ex.qid}"
        out.append(ex)

    if require_db:
        out = [e for e in out if os.path.exists(e.db_path) and not _is_placeholder(e.db_path)]
    if shuffle:
        random.Random(seed).shuffle(out)
    if limit is not None:
        out = out[:limit]
    return out


def load_bird(root: str, **kw) -> List[Example]:
    return load_dataset(root, dataset="bird", **kw)


def load_spider(root: str, **kw) -> List[Example]:
    return load_dataset(root, dataset="spider", **kw)


def interleave_by_ratio(*groups: Sequence[Example]) -> List[Example]:
    """Merge lists so that *every prefix* holds the same mixture as the whole.

    Each element gets the key ``(i + 0.5) / len(group)`` and everything is
    sorted on it. Taking the first ``k`` of the result therefore yields ~the
    configured BIRD:Spider ratio for any ``k`` - which is what makes
    ``--limit`` safe on a mixed set. Ties break on group order, so the result
    is deterministic.
    """
    keyed: List[tuple] = []
    for gi, g in enumerate(groups):
        n = len(g)
        for i, ex in enumerate(g):
            keyed.append((((i + 0.5) / n) if n else 0.0, gi, i, ex))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in keyed]


def load_mixed(
    bird_root: str,
    spider_root: str,
    n_bird: Optional[int] = None,
    n_spider: Optional[int] = None,
    bird_split: str = "dev",
    spider_split: str = "dev",
    limit: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = True,
    require_db: bool = True,
) -> List[Example]:
    """One combined BIRD + Spider set, drawn from both benchmarks.

    Each half is sampled independently (shuffled under ``seed``, then cut to
    ``n_bird`` / ``n_spider``) and the two are interleaved by ratio. Every
    ``Example`` keeps its ``dataset`` field, so the scorer can still report
    BIRD and Spider separately - a combined headline number alone would hide
    that the two benchmarks have very different difficulty and that only BIRD
    carries the ``evidence`` hint.

    Missing benchmarks are skipped rather than fatal; the caller can see what
    it got from the ``dataset`` fields.
    """
    halves: List[List[Example]] = []
    for root, name, split, n in (
        (bird_root, "bird", bird_split, n_bird),
        (spider_root, "spider", spider_split, n_spider),
    ):
        if not (root and available(root, name)):
            continue
        halves.append(
            load_dataset(
                root, dataset=name, split=split, limit=n,
                shuffle=shuffle, seed=seed, require_db=require_db,
            )
        )
    if not halves:
        raise FileNotFoundError(
            f"neither bird ({bird_root!r}) nor spider ({spider_root!r}) is readable"
        )
    out = interleave_by_ratio(*halves)
    return out[:limit] if limit is not None else out


def dataset_counts(examples: Sequence[Example]) -> Dict[str, int]:
    """How many questions came from each benchmark."""
    counts: Dict[str, int] = {}
    for e in examples:
        counts[e.dataset] = counts.get(e.dataset, 0) + 1
    return counts


def split_train_dev(
    examples: Sequence[Example], dev_frac: float = 0.1, seed: int = 0
) -> tuple[List[Example], List[Example]]:
    """Deterministic split used to build the distillation corpus vs. held-out eval."""
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_dev = max(1, int(len(idx) * dev_frac)) if examples else 0
    dev_idx = set(idx[:n_dev])
    train = [e for i, e in enumerate(examples) if i not in dev_idx]
    dev = [e for i, e in enumerate(examples) if i in dev_idx]
    return train, dev


def save_examples(examples: Sequence[Example], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def load_examples(path: str) -> List[Example]:
    out: List[Example] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Example(**json.loads(line)))
    return out


def available(root: str, dataset: str, split: str = "dev") -> bool:
    """Cheap check used by the smoke test to decide whether to fall back."""
    try:
        return (
            find_manifest(root, dataset, split) is not None
            and find_db_root(root, dataset, split) is not None
        )
    except Exception:
        return False
