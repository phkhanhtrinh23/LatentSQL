"""Database-value retrieval with locality-sensitive hashing (Sec. 4.1).

As in CHESS: every distinct cell value is indexed once with MinHash LSH over
character trigrams, the index is probed with keywords extracted from the
question, and the shortlist is re-ranked by edit-distance similarity. The
retrieved values are appended to the input x in training, RL and inference.

Only the standard library is used, so it runs anywhere. Embedding-based
re-ranking is an optional third stage - pass an ``embed_fn`` to enable it.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1

_QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "between",
    "both", "by", "can", "did", "do", "does", "each", "for", "from", "had",
    "has", "have", "how", "in", "into", "is", "it", "its", "least", "less",
    "list", "many", "more", "most", "name", "names", "not", "number", "of",
    "on", "or", "other", "show", "than", "that", "the", "their", "there",
    "these", "those", "to", "was", "were", "what", "when", "where", "which",
    "who", "whose", "with", "would",
}


@dataclass
class ValueHit:
    table: str
    column: str
    value: str
    similarity: float

    def as_text(self) -> str:
        return f"{self.table}.{self.column} = {self.value!r}"


def _shingles(text: str, k: int = 3) -> Set[str]:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    if len(t) < k:
        return {t} if t else set()
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def _hash32(s: str, seed: int) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8, salt=seed.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little") & _MAX_HASH


class MinHashLSH:
    """Minimal MinHash + banded LSH index over string values."""

    def __init__(self, num_perm: int = 64, bands: int = 16):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.buckets: List[Dict[str, List[int]]] = [defaultdict(list) for _ in range(bands)]
        self.entries: List[Tuple[str, str, str]] = []  # (table, column, value)

    def signature(self, text: str) -> List[int]:
        sh = _shingles(text)
        if not sh:
            return [0] * self.num_perm
        return [min(_hash32(s, seed) for s in sh) for seed in range(self.num_perm)]

    def add(self, table: str, column: str, value: str) -> None:
        idx = len(self.entries)
        self.entries.append((table, column, value))
        sig = self.signature(value)
        for b in range(self.bands):
            key = ",".join(map(str, sig[b * self.rows : (b + 1) * self.rows]))
            self.buckets[b][key].append(idx)

    def probe(self, text: str, limit: int = 200) -> List[int]:
        sig = self.signature(text)
        seen: List[int] = []
        got: Set[int] = set()
        for b in range(self.bands):
            key = ",".join(map(str, sig[b * self.rows : (b + 1) * self.rows]))
            for idx in self.buckets[b].get(key, ()):
                if idx not in got:
                    got.add(idx)
                    seen.append(idx)
                    if len(seen) >= limit:
                        return seen
        return seen


def _edit_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class ValueIndex:
    """Per-database index of distinct textual cell values."""

    #: characters of the leading prefix used as a typo-tolerant fallback key
    PREFIX_LEN = 2

    def __init__(self, num_perm: int = 64, bands: int = 16):
        self.lsh = MinHashLSH(num_perm=num_perm, bands=bands)
        # exact/substring fallback for the (common) case where LSH misses
        self.by_token: Dict[str, List[int]] = defaultdict(list)
        # transposition fallback: trigram MinHash shares *no* shingles between
        # e.g. "france" and "frnace", so a purely shingle-based index misses
        # exactly the typos CHESS/CHASE claim robustness to. Bucketing by
        # leading characters recovers them at negligible cost.
        self.by_prefix: Dict[str, List[int]] = defaultdict(list)
        self.db_id = ""

    # -- build ------------------------------------------------------------
    @classmethod
    def build(
        cls,
        db_path: str,
        db_id: str = "",
        max_values_per_column: int = 5000,
        max_value_len: int = 80,
        num_perm: int = 64,
        bands: int = 16,
    ) -> "ValueIndex":
        idx = cls(num_perm=num_perm, bands=bands)
        idx.db_id = db_id or os.path.splitext(os.path.basename(db_path))[0]
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [r[0] for r in cur.fetchall()]
            for t in tables:
                try:
                    cur.execute(f'PRAGMA table_info("{t}")')
                    cols = [(r[1], (r[2] or "").upper()) for r in cur.fetchall()]
                except sqlite3.Error:
                    continue
                for cname, ctype in cols:
                    # numeric columns are matched by the SQL itself, not by string search
                    if any(k in ctype for k in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")):
                        continue
                    try:
                        cur.execute(
                            f'SELECT DISTINCT "{cname}" FROM "{t}" '
                            f'WHERE "{cname}" IS NOT NULL LIMIT {int(max_values_per_column)}'
                        )
                    except sqlite3.Error:
                        continue
                    for (v,) in cur.fetchall():
                        s = str(v).strip()
                        if not s or len(s) > max_value_len:
                            continue
                        idx._add(t, cname, s)
        finally:
            conn.close()
        return idx

    def _add(self, table: str, column: str, value: str) -> None:
        i = len(self.lsh.entries)
        self.lsh.add(table, column, value)
        for tok in set(re.findall(r"[a-z0-9]+", value.lower())):
            if len(tok) >= 2:
                self.by_token[tok].append(i)
                self.by_prefix[tok[: self.PREFIX_LEN]].append(i)

    # -- query ------------------------------------------------------------
    def search(
        self,
        keyword: str,
        top_k: int = 5,
        min_similarity: float = 0.72,
        embed_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
        probe_limit: int = 200,
    ) -> List[ValueHit]:
        kw = (keyword or "").strip()
        if not kw:
            return []
        tokens = set(re.findall(r"[a-z0-9]+", kw.lower()))
        cand = set(self.lsh.probe(kw, limit=probe_limit))
        for tok in tokens:
            cand.update(self.by_token.get(tok, ())[:probe_limit])
        if not cand:
            # shingle- and token-level lookups both missed: fall back to the
            # prefix buckets so transposed/misspelled keywords still resolve
            for tok in tokens:
                cand.update(self.by_prefix.get(tok[: self.PREFIX_LEN], ())[:probe_limit])
        if not cand:
            return []

        scored: List[ValueHit] = []
        for i in cand:
            table, column, value = self.lsh.entries[i]
            sim = _edit_sim(kw, value)
            # a keyword contained verbatim in a cell is a strong signal even
            # when the cell is much longer than the keyword
            if kw.lower() in value.lower():
                sim = max(sim, 0.9)
            if sim >= min_similarity:
                scored.append(ValueHit(table, column, value, sim))

        scored.sort(key=lambda h: -h.similarity)
        shortlist = scored[: max(top_k * 4, top_k)]

        if embed_fn is not None and shortlist:
            try:
                vecs = list(embed_fn([kw] + [h.value for h in shortlist]))
                q, rest = vecs[0], vecs[1:]
                for h, v in zip(shortlist, rest):
                    h.similarity = 0.5 * h.similarity + 0.5 * _cosine(q, v)
                shortlist.sort(key=lambda h: -h.similarity)
            except Exception:
                pass  # embedding re-rank is best-effort
        return shortlist[:top_k]

    def search_many(
        self, keywords: Iterable[str], top_k: int = 5, min_similarity: float = 0.72, **kw
    ) -> List[ValueHit]:
        seen: Set[Tuple[str, str, str]] = set()
        out: List[ValueHit] = []
        for k in keywords:
            for hit in self.search(k, top_k=top_k, min_similarity=min_similarity, **kw):
                key = (hit.table, hit.column, hit.value)
                if key not in seen:
                    seen.add(key)
                    out.append(hit)
        out.sort(key=lambda h: -h.similarity)
        return out

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "ValueIndex":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


_INDEX_CACHE: Dict[str, ValueIndex] = {}


def get_value_index(
    db_path: str,
    db_id: str = "",
    cache_dir: Optional[str] = None,
    max_values_per_column: int = 5000,
    num_perm: int = 64,
    bands: int = 16,
) -> ValueIndex:
    """Build (or load) the value index for a database, memoised per process."""
    key = os.path.abspath(db_path)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    disk = None
    if cache_dir:
        stem = hashlib.md5(key.encode()).hexdigest()[:12]
        disk = os.path.join(cache_dir, f"values_{db_id or 'db'}_{stem}.pkl")
        if os.path.exists(disk):
            try:
                idx = ValueIndex.load(disk)
                _INDEX_CACHE[key] = idx
                return idx
            except Exception:
                pass

    idx = ValueIndex.build(
        db_path, db_id=db_id, max_values_per_column=max_values_per_column,
        num_perm=num_perm, bands=bands,
    )
    if disk:
        try:
            idx.save(disk)
        except Exception:
            pass
    _INDEX_CACHE[key] = idx
    return idx


def extract_value_keywords(question: str, evidence: str = "", max_keywords: int = 40) -> List[str]:
    """Extract deterministic database-value probes without an LLM.

    Retrieval is preprocessing, not a latent reasoning role.  Keeping keyword
    extraction deterministic lets latent-SFT, GRPO, and evaluation construct
    exactly the same prompt without calling a teacher model at test time.
    Quoted strings and proper-name spans are prioritised, followed by useful
    1--4-grams from the question/evidence.  The value index's similarity
    threshold rejects schema words and generic phrases that are not cell
    values.
    """
    text = " ".join(part.strip() for part in (question, evidence) if part and part.strip())
    candidates: List[str] = []

    candidates.extend(
        match.strip() for match in re.findall(r"['\"]([^'\"]{2,80})['\"]", text)
    )
    candidates.extend(
        match.strip()
        for match in re.findall(
            r"\b(?:[A-Z][\w'-]*(?:\s+|$)){1,5}", text
        )
    )

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", text)
    content = [
        word for word in words
        if len(word) >= 2 and word.lower() not in _QUESTION_STOPWORDS
    ]
    for width in (4, 3, 2, 1):
        for start in range(0, max(0, len(content) - width + 1)):
            candidates.append(" ".join(content[start:start + width]))

    out: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        clean = re.sub(r"\s+", " ", candidate).strip(" .,:;!?\t\r\n")
        key = clean.lower()
        if len(clean) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= max_keywords:
            break
    return out


def retrieve_value_hints(
    db_path: str,
    db_id: str,
    question: str,
    evidence: str = "",
    *,
    cfg: Optional[object] = None,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """Return externally retrieved value hints for a student prompt.

    This helper is the single train/GRPO/eval entry point.  It never invokes
    an LLM and returns an empty list when retrieval is disabled or unavailable.
    """
    if cfg is not None and not bool(getattr(cfg, "enabled", True)):
        return []
    if not db_path or not os.path.exists(db_path):
        return []
    try:
        index = get_value_index(
            db_path,
            db_id=db_id,
            cache_dir=cache_dir,
            max_values_per_column=int(
                getattr(cfg, "max_values_per_column", 5000)
            ),
            num_perm=int(getattr(cfg, "lsh_signature_size", 64)),
            bands=int(getattr(cfg, "lsh_bands", 16)),
        )
        hits = index.search_many(
            extract_value_keywords(question, evidence),
            top_k=int(getattr(cfg, "top_k", 5)),
            min_similarity=float(getattr(cfg, "min_similarity", 0.72)),
        )
        return [hit.as_text() for hit in hits[: int(getattr(cfg, "top_k", 5)) * 2]]
    except Exception:
        return []
