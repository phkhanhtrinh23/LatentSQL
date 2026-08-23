"""Dense reward terms, after FINER-SQL (Hoang et al., ICDE 2026).

IMAD's ``R(y; l)`` is binary: the query is right or it is not. GRPO normalises
advantages *within* a rollout group, so when every rollout in a group is wrong
all rewards are equal, the advantage is identically zero, and **the step
contributes no gradient at all**. Worse, the only term left with variance is
then ``R_fmt``, whose gradient points toward emitting *more* debate — the exact
behaviour internalisation is meant to remove. Pilot run 1 hit this on 5 of 9
steps and the weights moved 6e-04 in total.

FINER-SQL's fix is graded partial credit, so a nearly-right rollout outscores a
badly-wrong one and the group always has variance:

* **Atomic reward** - Jaccard overlap of *atomic SQL operations* against gold.
  Pure parsing, no model, no network.
* **Memory reward** - semantic similarity of the rollout's reasoning to
  verified-correct reasoning traces.

Both are scored **only when the rollout is wrong**, exactly as FINER-SQL applies
them: a correct rollout already earns full credit and must not be able to earn
more by resembling something.

One deviation from FINER-SQL, forced by our different objective. FINER-SQL never
wants to compress its reasoning, so rewarding trace-like reasoning is free for
it. IMAD spends stage 2 trying to make the model *stop* writing reasoning, so a
constant memory reward would fight the length anneal directly. We therefore
anneal the memory weight on the same schedule as ``w_fmt``: it guides early,
when the model is still learning what good reasoning looks like, and fades as
verbalisation stops paying. The atomic reward has no such tension — it scores
the *query*, not the prose — so it stays constant.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

# --------------------------------------------------------------------------
# atomic operations
# --------------------------------------------------------------------------
_AGG = {"COUNT", "SUM", "AVG", "MIN", "MAX", "TOTAL"}


def sql_to_ops(sql: str, dialect: str = "sqlite") -> List[str]:
    """Decompose a query into canonical atomic operations.

    Table aliases are erased and column references reduced to bare names, so
    two queries that differ only in aliasing produce identical op sets. What
    survives is the *structure*: which tables are touched, which joins, which
    predicates, which aggregates, grouping, ordering and limiting.

    Returns ``[]`` for anything unparseable, which scores 0 overlap - the
    correct outcome for a malformed query.
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:  # pragma: no cover - depends on env
        return []

    try:
        root = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return []
    if root is None:
        return []

    ops: List[str] = []

    def name_of(node: Any) -> str:
        try:
            return (node.alias_or_name or node.name or "").lower()
        except Exception:
            return ""

    def col(node: Any) -> str:
        # bare column name: alias-insensitive by construction
        try:
            return (node.name or "").lower()
        except Exception:
            return ""

    try:
        for t in root.find_all(exp.Table):
            n = (t.name or "").lower()
            if n:
                ops.append(f"TABLE({n})")

        for j in root.find_all(exp.Join):
            kind = (j.args.get("kind") or j.args.get("side") or "INNER").upper()
            ops.append(f"JOIN({kind})")
            on = j.args.get("on")
            if on is not None:
                for c in on.find_all(exp.Column):
                    ops.append(f"JOIN_KEY({col(c)})")

        for sel in root.find_all(exp.Select):
            for e in sel.expressions:
                if isinstance(e, exp.Star):
                    ops.append("SELECT(*)")
                    continue
                fn = None
                distinct = False
                for f in e.find_all(exp.Func):
                    nm = f.sql_name().upper() if hasattr(f, "sql_name") else ""
                    if nm in _AGG:
                        fn = nm
                        # COUNT(DISTINCT x) carries the DISTINCT *inside* the
                        # aggregate, not on the SELECT, and dropping it is a
                        # classic near-miss that must not score as identical
                        distinct = any(True for _ in f.find_all(exp.Distinct))
                        break
                cols = [col(c) for c in e.find_all(exp.Column) if col(c)]
                target = cols[0] if cols else "expr"
                if fn:
                    ops.append(f"AGG({fn},{'DISTINCT ' if distinct else ''}{target})")
                else:
                    ops.append(f"SELECT({target})")
            if sel.args.get("distinct"):
                ops.append("DISTINCT")

        for w in root.find_all(exp.Where):
            for pred in w.find_all(exp.Binary):
                op = type(pred).__name__.upper()
                cols = [col(c) for c in pred.find_all(exp.Column) if col(c)]
                if cols:
                    ops.append(f"WHERE({op},{cols[0]})")

        for g in root.find_all(exp.Group):
            for e in g.expressions:
                ops.append(f"GROUP_BY({col(e) or 'expr'})")
        for h in root.find_all(exp.Having):
            ops.append("HAVING")
        for o in root.find_all(exp.Order):
            for e in o.expressions:
                desc = "DESC" if e.args.get("desc") else "ASC"
                cols = [col(c) for c in e.find_all(exp.Column) if col(c)]
                ops.append(f"ORDER_BY({cols[0] if cols else 'expr'},{desc})")
        for l in root.find_all(exp.Limit):
            ops.append("LIMIT")
        for s in root.find_all(exp.Subquery):
            ops.append("SUBQUERY")
        for u in root.find_all(exp.Union):
            ops.append("UNION")
    except Exception:
        return ops

    return ops


class AtomicOpsReward:
    """Jaccard overlap of atomic ops between a predicted and a gold query.

    ``power`` shapes the raw Jaccard so that near-misses are not over-rewarded:
    a query sharing half its structure with gold is not half-correct. FINER-SQL
    uses ``e*s + (1-e)*beta*s**gamma``; the same shape is used here with the
    same defaults.
    """

    def __init__(self, dialect: str = "sqlite", e: float = 0.30,
                 beta: float = 0.71, gamma: float = 5.0):
        self.dialect = dialect
        self.e, self.beta, self.gamma = e, beta, gamma
        self._cache: Dict[str, Set[str]] = {}

    def ops(self, sql: Optional[str]) -> Set[str]:
        key = (sql or "").strip()
        if not key:
            return set()
        if key not in self._cache:
            if len(self._cache) > 50_000:
                self._cache.clear()
            self._cache[key] = set(sql_to_ops(key, self.dialect))
        return self._cache[key]

    def jaccard(self, pred_sql: Optional[str], gold_sql: str) -> float:
        a, b = self.ops(pred_sql), self.ops(gold_sql)
        if not a and not b:
            return 0.0
        union = len(a | b)
        return (len(a & b) / union) if union else 0.0

    def score(self, pred_sql: Optional[str], gold_sql: str) -> float:
        """Shaped overlap in [0, 1]."""
        s = self.jaccard(pred_sql, gold_sql)
        return float(self.e * s + (1.0 - self.e) * self.beta * (s ** self.gamma))


# --------------------------------------------------------------------------
# memory over verified reasoning
# --------------------------------------------------------------------------
class MemoryReward:
    """Similarity of a rollout's reasoning to verified-correct reasoning.

    The bank is built from the *kept* teacher traces — transcripts whose final
    query was confirmed correct by execution. That makes this reward
    specifically "does this reasoning resemble a debate that actually worked",
    which is closer to the internalisation objective than a generic
    reasoning-quality prior would be.

    FINER-SQL stores its bank in ChromaDB. With a corpus in the thousands a
    dense matrix and one matmul is faster, exact, and adds no dependency, so
    that is what is used here.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hub_dir: Optional[str] = None,
        device: str = "cpu",
        top_k: int = 5,
        same_db_only: bool = False,
    ):
        self.model_name = model_name
        self.hub_dir = hub_dir
        self.device = device
        self.top_k = top_k
        self.same_db_only = same_db_only
        self._model = None
        self._emb = None                      # (n, d) L2-normalised
        self._db_ids: List[str] = []

    # -- construction -----------------------------------------------------
    def _encoder(self):
        if self._model is None:
            import os

            if self.hub_dir and os.path.isdir(self.hub_dir):
                os.environ.setdefault("HF_HOME", self.hub_dir)
                os.environ.setdefault("HF_HUB_CACHE", self.hub_dir)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def build(self, texts: Sequence[str], db_ids: Sequence[str] = ()) -> "MemoryReward":
        """Embed the verified reasoning bank. Call once before training."""
        import numpy as np

        texts = [t for t in texts if t and t.strip()]
        if not texts:
            self._emb = None
            return self
        vecs = self._encoder().encode(
            list(texts), batch_size=32, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        self._emb = np.asarray(vecs, dtype="float32")
        self._db_ids = list(db_ids) if len(db_ids) == len(texts) else [""] * len(texts)
        return self

    @classmethod
    def from_traces(cls, traces: Sequence[Any], **kw) -> "MemoryReward":
        """Build from ``DebateTrace``-like records, using the agents' reasoning."""
        texts, dbs = [], []
        for t in traces:
            turns = getattr(t, "turns", None)
            if turns is None and isinstance(t, dict):
                turns = t.get("turns", [])
            parts = []
            for turn in turns or []:
                r = getattr(turn, "reasoning", None)
                if r is None and isinstance(turn, dict):
                    r = turn.get("reasoning", "")
                if r:
                    parts.append(r.strip())
            if parts:
                texts.append("\n".join(parts))
                dbs.append(getattr(t, "db_id", "") or (t.get("db_id", "") if isinstance(t, dict) else ""))
        return cls(**kw).build(texts, dbs)

    @property
    def ready(self) -> bool:
        return self._emb is not None and len(self._emb) > 0

    # -- scoring ----------------------------------------------------------
    def score(self, reasoning: str, db_id: str = "") -> float:
        """Mean cosine similarity to the top-k nearest verified traces, in [0, 1]."""
        if not self.ready or not reasoning or len(reasoning.strip()) < 30:
            return 0.0
        import numpy as np

        q = self._encoder().encode(
            [reasoning], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype("float32")

        emb, idx = self._emb, None
        if self.same_db_only and db_id:
            keep = [i for i, d in enumerate(self._db_ids) if d == db_id]
            if keep:
                idx = np.asarray(keep)
                emb = self._emb[idx]
        sims = emb @ q                                  # cosine, both normalised
        k = min(self.top_k, len(sims))
        top = np.partition(sims, -k)[-k:] if k < len(sims) else sims
        # cosine is in [-1, 1]; clamp to [0, 1] so the term never subtracts
        return float(max(0.0, min(1.0, float(top.mean()))))
