"""The dense atomic reward R_atomic (Sec. 4.3), following FINER-SQL.

A query is parsed with sqlglot into a set O(Y) of atomic operations (tables,
selected columns, joins, predicates, aggregates, grouping, ordering, limits,
set operations). The reward is the shaped Jaccard overlap of O(Y) and O(Y*)
as defined in FINER-SQL, so a query that is wrong in one clause still scores
above an unrelated one and a group of wrong rollouts keeps a learning signal.
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
