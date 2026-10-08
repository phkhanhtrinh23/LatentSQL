"""Spurious-correctness stress testing for RL rewards (EX-Ext, adapted).

Plain execution accuracy (EX) checks a prediction against exactly one
database: the one shipped with the benchmark. Two SQL queries that mean
different things can still return the same rows there by pure coincidence, so
EX silently rewards a wrong query as if it were right.
``pool_text2sql_crowd_source_mixture/main_pool_eval.pdf`` Appendix B ("EX and
EX-Ext") names this *spurious correctness* and fixes it, for offline
evaluation, by building several modified copies of the database and requiring
the prediction to agree with gold on *all* of them, not just the original -
an LLM (their GPT-5.4) hand-picks which row to insert into each copy, per
query pair, to expose the disagreement.

That hand-picking does not scale to RL. GRPO calls the reward function on a
fresh, unforeseen rollout at every training step, so there is no (gold,
prediction) pair to design a row for in advance - only the *gold* query is
known ahead of time, at trace-build time. This module replaces the
LLM-in-the-loop step with two automatic, gold-query-only methods, plus an
always-available fallback:

  Method A (:func:`build_boundary_instance`) - parse gold's WHERE/HAVING/ON
  predicates and insert synthetic rows straddling literal boundaries. Equi-
  join keys get dangling and duplicate-key rows. Needs ``sqlglot`` (optional
  - see requirements.txt). This is the main mechanism: it can manufacture
  the specific counterexample row a predicate or join bug needs, even when
  that row combination does not exist in the real data.

  Method B (:func:`_is_discriminating`) - a filter, not a generator. Keep a
  Method-A instance only if some small mutation of gold's own query (flip a
  comparison operator, nudge a literal, swap AND/OR) disagrees with gold on
  it. This is a no-LLM stand-in for the paper's "GPT-5.4 guided the selection
  of inserted rows to expose differences" - the same judgement call, made by
  trying plausible mutants instead of asking a model.

  Method C (:func:`bootstrap_instance`) - no parsing at all: duplicate, drop,
  or numerically jitter a few existing rows at random. It cannot manufacture a missing
  counterexample row the way Method A can, so it is blind to predicate/filter
  bugs. What it *is* good at is anything sensitive to row *multiplicity* - a
  JOIN that silently fans out, a forgotten DISTINCT, an off-by-one COUNT -
  because duplicating or dropping rows that already exist directly perturbs
  multiplicity without needing to know in advance what a wrong query would
  look like. Always available (standard library only); used to fill in when
  ``sqlglot`` is missing, or when Method A finds nothing to work with (e.g. a
  query with no WHERE/HAVING clause at all, such as a bare aggregate).

Generated instances are cached on disk under ``cache_dir``, keyed by a hash of
(db path, gold SQL): building them involves several extra SQL executions and
file copies, far too slow to repeat on every GRPO step for every rollout.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .execute import execute_sql, gold_is_ordered, result_match
from .schema import DBSchema, load_schema

try:
    import sqlglot
    from sqlglot import exp

    _HAVE_SQLGLOT = True
except Exception:  # pragma: no cover - exercised by the no-sqlglot fallback tests
    _HAVE_SQLGLOT = False

_COMPARISON_TYPES: Tuple[type, ...] = ()
if _HAVE_SQLGLOT:
    _COMPARISON_TYPES = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)

_MANIFEST = "_manifest.json"
_CACHE_VERSION = 2
#: databases above this size get no stress instances (each one is a full copy)
MAX_STRESS_DB_BYTES = 512 * 2**20


# --------------------------------------------------------------------------
# Method A: extract the boundary a predicate literal sits on
# --------------------------------------------------------------------------
@dataclass
class BoundaryLiteral:
    table: str
    column: str
    op: str                      # '=', '!=', '>', '>=', '<', '<=', 'BETWEEN'
    value: Any
    value2: Any = None           # second bound, BETWEEN only


@dataclass
class JoinBoundary:
    """One equality key from an ON clause, oriented by JOIN semantics."""

    left_table: str              # preserved side for INNER/LEFT JOIN
    left_column: str
    right_table: str             # table introduced by this JOIN
    right_column: str
    join_type: str = "INNER"


def _table_alias_map(tree: "exp.Expression") -> Dict[str, str]:
    """``{alias_or_name: real_table_name}`` for every table in the query."""
    out: Dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        name = t.name
        out[name] = name
        if t.alias:
            out[t.alias] = name
    return out


def _resolve_table(column: "exp.Column", alias_map: Dict[str, str],
                   schema: DBSchema) -> Optional[str]:
    """Which real table a (possibly qualified) column belongs to.

    An explicit qualifier (``T1.salary``) is resolved through the alias map.
    An unqualified column is resolved by asking the schema which table (if
    exactly one) actually has a column with that name - ambiguous or unknown
    columns are skipped rather than guessed at.
    """
    if column.table:
        return alias_map.get(column.table)
    hits = [t.name for t in schema.tables if t.column(column.name) is not None]
    return hits[0] if len(hits) == 1 else None


def extract_literals(sql: str, schema: DBSchema) -> List[BoundaryLiteral]:
    """Pull ``(table, column, op, literal)`` triples out of WHERE/HAVING.

    Only comparisons against a *literal* are kept - ``T1.id = T2.id`` is a
    join condition, not a predicate boundary, and is skipped by checking that
    the other side of the comparison is an ``exp.Literal``, not a column.
    Returns ``[]`` if ``sqlglot`` is unavailable or the query fails to parse;
    callers fall back to Method C in that case.
    """
    if not _HAVE_SQLGLOT or not sql or not sql.strip():
        return []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []

    alias_map = _table_alias_map(tree)
    out: List[BoundaryLiteral] = []

    for node in tree.find_all(_COMPARISON_TYPES):
        col = node.this
        other = node.expression
        reverse = False
        if isinstance(col, exp.Literal) and isinstance(other, exp.Column):
            col, other = other, col
            reverse = True
        if not isinstance(col, exp.Column) or not isinstance(other, exp.Literal):
            continue  # a column-to-column comparison (a join key), not a predicate
        table = _resolve_table(col, alias_map, schema)
        if table is None:
            continue
        op = {"eq": "=", "neq": "!=", "gt": ">", "gte": ">=",
              "lt": "<", "lte": "<="}[node.key]
        if reverse:
            op = {">": "<", ">=": "<=", "<": ">", "<=": ">="}.get(op, op)
        value = _literal_value(other)
        out.append(BoundaryLiteral(table=table, column=col.name, op=op, value=value))

    for node in tree.find_all(exp.Between):
        col = node.this
        if not isinstance(col, exp.Column):
            continue
        low, high = node.args.get("low"), node.args.get("high")
        if not isinstance(low, exp.Literal) or not isinstance(high, exp.Literal):
            continue
        table = _resolve_table(col, alias_map, schema)
        if table is None:
            continue
        out.append(BoundaryLiteral(
            table=table, column=col.name, op="BETWEEN",
            value=_literal_value(low), value2=_literal_value(high),
        ))
    return out


def extract_join_boundaries(sql: str, schema: DBSchema) -> List[JoinBoundary]:
    """Extract column-equality keys from ``JOIN ... ON`` clauses.

    Literal predicates and join keys need different perturbations.  This
    function keeps the join target on the right so a dangling row can be
    inserted on the side preserved by a LEFT JOIN, while a duplicate-key row
    on the introduced table exercises fan-out and missing-DISTINCT errors.
    """
    if not _HAVE_SQLGLOT or not sql or not sql.strip():
        return []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []

    alias_map = _table_alias_map(tree)
    out: List[JoinBoundary] = []
    for join in tree.find_all(exp.Join):
        target = join.this
        if not isinstance(target, exp.Table):
            continue
        right_table = alias_map.get(target.alias_or_name, target.name)
        on = join.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            lhs, rhs = eq.this, eq.expression
            if not isinstance(lhs, exp.Column) or not isinstance(rhs, exp.Column):
                continue
            lhs_table = _resolve_table(lhs, alias_map, schema)
            rhs_table = _resolve_table(rhs, alias_map, schema)
            if lhs_table is None or rhs_table is None or lhs_table == rhs_table:
                continue
            if lhs_table.lower() == right_table.lower():
                left_table, left_col = rhs_table, rhs.name
                right_col = lhs.name
            elif rhs_table.lower() == right_table.lower():
                left_table, left_col = lhs_table, lhs.name
                right_col = rhs.name
            else:
                continue
            out.append(JoinBoundary(
                left_table=left_table,
                left_column=left_col,
                right_table=right_table,
                right_column=right_col,
                join_type=(join.side or join.kind or "INNER").upper(),
            ))
    return out


def _literal_value(lit: "exp.Literal") -> Any:
    if lit.is_string:
        return str(lit.this)
    text = str(lit.this)
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


# --------------------------------------------------------------------------
# Method B: is a generated instance actually discriminating?
# --------------------------------------------------------------------------
def mutate_query(sql: str, rng: random.Random, max_mutants: int = 12) -> List[str]:
    """A handful of 1-edit mutants of ``sql`` itself (never of the prediction).

    Flips one comparison operator, nudges one numeric literal by +-1, or swaps
    one AND/OR. Used only to test whether a candidate stress instance is worth
    keeping (:func:`_is_discriminating`) - these mutants are never executed
    against anything but the instance under test, and never scored on their
    own. Returns ``[]`` if ``sqlglot`` is unavailable.
    """
    if not _HAVE_SQLGLOT or not sql or not sql.strip():
        return []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []

    _flip = {
        "eq": exp.NEQ, "neq": exp.EQ, "gt": exp.GTE,
        "gte": exp.GT, "lt": exp.LTE, "lte": exp.LT,
    }
    groups: List[List[str]] = [[], [], [], []]

    # Index-based correspondence: tree.copy() preserves traversal order, so
    # the i-th comparison/literal found in a *copy* is the same node as the
    # i-th one found in the original. Matching by rendered SQL text instead
    # would be fragile whenever two unrelated nodes happen to render
    # identically (e.g. two different "salary > 1000" conditions).
    comparisons = list(tree.find_all(_COMPARISON_TYPES))
    for i, node in enumerate(comparisons):
        if node.key not in _flip:
            continue
        mutant = tree.copy()
        target = list(mutant.find_all(_COMPARISON_TYPES))[i]
        target.replace(_flip[node.key](this=target.this.copy(),
                                       expression=target.expression.copy()))
        groups[0].append(mutant.sql(dialect="sqlite"))

    literals = list(tree.find_all(exp.Literal))
    for i, node in enumerate(literals):
        if node.is_string:
            continue
        try:
            n = int(str(node.this))
        except ValueError:
            continue
        mutant = tree.copy()
        target = list(mutant.find_all(exp.Literal))[i]
        target.set("this", str(n + 1))
        groups[1].append(mutant.sql(dialect="sqlite"))

    booleans = [node for node in tree.walk() if isinstance(node, (exp.And, exp.Or))]
    for i, node in enumerate(booleans):
        mutant = tree.copy()
        targets = [x for x in mutant.walk() if isinstance(x, (exp.And, exp.Or))]
        target = targets[i]
        replacement = exp.Or if isinstance(node, exp.And) else exp.And
        target.replace(replacement(this=target.this.copy(),
                                   expression=target.expression.copy()))
        groups[2].append(mutant.sql(dialect="sqlite"))

    joins = list(tree.find_all(exp.Join))
    for i, node in enumerate(joins):
        mutant = tree.copy()
        target = list(mutant.find_all(exp.Join))[i]
        if (node.side or "").upper() == "LEFT":
            target.set("side", None)
            target.set("kind", "INNER")
        elif not node.side or (node.kind or "").upper() == "INNER":
            target.set("side", "LEFT")
            target.set("kind", None)
        else:
            continue
        groups[3].append(mutant.sql(dialect="sqlite"))

    # Preserve mutation diversity when max_mutants is small: take one from
    # each available category before filling the remaining slots.
    for group in groups:
        rng.shuffle(group)
    candidates: List[str] = []
    for group in groups:
        if group and len(candidates) < max_mutants:
            candidates.append(group.pop())
    remainder = [candidate for group in groups for candidate in group]
    rng.shuffle(remainder)
    candidates.extend(remainder[:max(0, max_mutants - len(candidates))])
    return candidates


def _is_discriminating(gold_sql: str, mutants: Sequence[str], inst_path: str,
                        timeout: float) -> bool:
    """Does at least one mutant of gold disagree with gold on this instance?

    If it does, the instance is capable of separating gold from *some*
    plausible nearby query - the automatic analogue of the paper's "GPT-5.4
    guided the selection of inserted rows to expose differences". An instance
    that passes this check is kept; one where every mutant still agrees with
    gold would never actually test anything, so it is discarded.
    """
    if not mutants:
        return True  # nothing to check against - keep it; Method A already targeted a boundary
    gold_res = execute_sql(inst_path, gold_sql, timeout=timeout)
    if not gold_res.ok:
        return False
    ordered = gold_is_ordered(gold_sql)
    for mutant_sql in mutants:
        mutant_res = execute_sql(inst_path, mutant_sql, timeout=timeout)
        if not mutant_res.ok:
            continue
        if not result_match(mutant_res, gold_res, ordered=ordered):
            return True
    return False


# --------------------------------------------------------------------------
# row synthesis shared by Method A and Method C
# --------------------------------------------------------------------------
def _sample_typed_values(conn: sqlite3.Connection, table: str, column: str,
                         n: int = 20) -> List[Any]:
    try:
        cur = conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT ?',
            (n,),
        )
        return [row[0] for row in cur.fetchall()]
    except sqlite3.Error:
        return []


def _placeholder_for(col_type: str) -> Any:
    t = (col_type or "").upper()
    if "INT" in t:
        return 0
    if any(k in t for k in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return 0.0
    return "stress_placeholder"


def _numeric_step(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        return 1
    return max(abs(value) * 1e-6, 1e-6)


def _neighbor(value: Any) -> Any:
    step = _numeric_step(value)
    if step is not None:
        return value + step
    return f"{value}_stress_neighbor"


def _satisfying_value(lit: BoundaryLiteral) -> Optional[Any]:
    """Return a value satisfying a simple literal predicate, if possible."""
    step = _numeric_step(lit.value)
    if lit.op == "=":
        return lit.value
    if lit.op == "!=":
        return _neighbor(lit.value)
    if lit.op == ">" and step is not None:
        return lit.value + step
    if lit.op == ">=":
        return lit.value
    if lit.op == "<" and step is not None:
        return lit.value - step
    if lit.op == "<=":
        return lit.value
    if lit.op == "BETWEEN":
        if isinstance(lit.value, (int, float)) and isinstance(lit.value2, (int, float)):
            return (lit.value + lit.value2) / 2
        return lit.value
    return None


def _other_filter_overrides(literals: Sequence[BoundaryLiteral],
                            target: Optional[BoundaryLiteral],
                            table_name: str,
                            protected_columns: Sequence[str] = ()) -> Dict[str, Any]:
    """Values that make other same-table gold predicates true.

    A target/join-key column is protected because its deliberately chosen
    boundary value must win. Multiple constraints on another column are
    solved together, so a synthesized row cannot satisfy one while silently
    violating another conjunct on that same column.
    """
    protected = {name.lower() for name in protected_columns}
    grouped: Dict[str, List[BoundaryLiteral]] = {}
    for lit in literals:
        if lit is target or lit.table.lower() != table_name.lower():
            continue
        if lit.column.lower() in protected:
            continue
        grouped.setdefault(lit.column, []).append(lit)

    overrides: Dict[str, Any] = {}
    for column, constraints in grouped.items():
        candidates: List[Any] = []
        for constraint in constraints:
            value = _satisfying_value(constraint)
            if value is not None:
                candidates.append(value)
            candidates.extend([constraint.value, _neighbor(constraint.value)])
            if constraint.value2 is not None:
                candidates.extend([constraint.value2, _neighbor(constraint.value2)])
        numeric = [value for value in candidates
                   if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if numeric:
            candidates.append((min(numeric) + max(numeric)) / 2)
        chosen = next(
            (value for value in candidates
             if all(_value_satisfies(value, constraint) for constraint in constraints)),
            None,
        )
        if chosen is not None:
            overrides[column] = chosen
    return overrides


def _value_satisfies(value: Any, lit: BoundaryLiteral) -> bool:
    try:
        if lit.op == "=":
            return value == lit.value
        if lit.op == "!=":
            return value != lit.value
        if lit.op == ">":
            return value > lit.value
        if lit.op == ">=":
            return value >= lit.value
        if lit.op == "<":
            return value < lit.value
        if lit.op == "<=":
            return value <= lit.value
        if lit.op == "BETWEEN":
            return lit.value <= value <= lit.value2
    except TypeError:
        return False
    return False


def _synthesize_row(conn: sqlite3.Connection, schema: DBSchema, table_name: str,
                    overrides: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Build one full row for ``table_name``, honouring ``overrides`` on the
    columns a boundary literal cares about and filling every other column from
    a real sampled value where possible (falling back to a type-consistent
    placeholder). Returns ``None`` if the table cannot be found in the schema.
    """
    table = schema.table(table_name)
    if table is None:
        return None
    row: Dict[str, Any] = {}
    for col in table.columns:
        if col.name in overrides:
            row[col.name] = overrides[col.name]
            continue
        if col.pk and "INT" in col.type.upper():
            continue  # rowid alias - omit so SQLite autoincrements it, avoiding collisions
        samples = _sample_typed_values(conn, table_name, col.name)
        if samples:
            row[col.name] = rng.choice(samples)
        elif col.notnull:
            row[col.name] = _placeholder_for(col.type)
        # else: leave unset -> NULL, which is fine for a nullable column
    return row


def _insert_row(conn: sqlite3.Connection, table_name: str, row: Dict[str, Any]) -> bool:
    if not row:
        try:
            conn.execute(f'INSERT INTO "{table_name}" DEFAULT VALUES')
            return True
        except sqlite3.Error:
            return False
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(f'"{c}"' for c in cols)
    try:
        conn.execute(
            f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})',
            [row[c] for c in cols],
        )
        return True
    except sqlite3.Error:
        # a NOT NULL/foreign-key/unique constraint we did not model - skip this
        # row rather than crash the whole stress-instance build
        return False


# --------------------------------------------------------------------------
# Method A: build one boundary-straddling instance
# --------------------------------------------------------------------------
def _straddling_values(lit: BoundaryLiteral) -> List[Tuple[Any, Any]]:
    """``[(matching_value, near_miss_value), ...]`` for one literal's boundary."""
    if lit.op == "=":
        return [(lit.value, _neighbor(lit.value))]
    if lit.op == "!=":
        return [(_neighbor(lit.value), lit.value)]
    step = _numeric_step(lit.value)
    if lit.op == ">" and step is not None:
        return [(lit.value + step, lit.value)]
    if lit.op == ">=" and step is not None:
        return [(lit.value, lit.value - step)]
    if lit.op == "<" and step is not None:
        return [(lit.value - step, lit.value)]
    if lit.op == "<=" and step is not None:
        return [(lit.value, lit.value + step)]
    if lit.op == "BETWEEN":
        lo, hi = lit.value, lit.value2
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            lo_step, hi_step = _numeric_step(lo), _numeric_step(hi)
            return [(lo, lo - lo_step), (hi, hi + hi_step)]
        return []
    return []


def build_boundary_instance(src_db: str, dst_path: str, schema: DBSchema,
                            lit: BoundaryLiteral, rng: random.Random,
                            all_literals: Sequence[BoundaryLiteral] = ()) -> bool:
    """Copy ``src_db`` to ``dst_path`` and insert rows straddling ``lit``.

    Mirrors the paper's Table 2: one row that satisfies the predicate
    (the "matching" value) and one that lies just the wrong side of the
    boundary (the "near-miss" value), both with every other column filled
    from real sampled data so the row is otherwise unremarkable. Returns
    ``True`` if at least one row was actually inserted.
    """
    pairs = _straddling_values(lit)
    if not pairs:
        return False
    shutil.copy2(src_db, dst_path)
    conn = sqlite3.connect(dst_path)
    inserted = False
    try:
        for matching, near_miss in pairs:
            for value in (matching, near_miss):
                overrides = _other_filter_overrides(
                    all_literals, lit, lit.table, protected_columns=(lit.column,)
                )
                overrides[lit.column] = value
                row = _synthesize_row(conn, schema, lit.table, overrides, rng)
                if row is not None and _insert_row(conn, lit.table, row):
                    inserted = True
        conn.commit()
    finally:
        conn.close()
    return inserted


def _fresh_join_value(conn: sqlite3.Connection, table: str, column: str) -> Any:
    values = _sample_typed_values(conn, table, column, n=100)
    numeric = [value for value in values
               if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if numeric:
        return max(numeric) + 1
    base = str(values[0]) if values else "stress_key"
    candidate = base + "_unmatched"
    existing = set(values)
    while candidate in existing:
        candidate += "_x"
    return candidate


def build_join_instance(src_db: str, dst_path: str, schema: DBSchema,
                        join: JoinBoundary, rng: random.Random,
                        all_literals: Sequence[BoundaryLiteral] = ()) -> bool:
    """Create dangling-left and duplicate-right rows for one equi-join."""
    shutil.copy2(src_db, dst_path)
    conn = sqlite3.connect(dst_path)
    inserted = False
    try:
        # INNER-vs-LEFT: a row on the preserved side with no counterpart.
        dangling_key = _fresh_join_value(conn, join.left_table, join.left_column)
        left_overrides = _other_filter_overrides(
            all_literals, None, join.left_table,
            protected_columns=(join.left_column,),
        )
        left_overrides[join.left_column] = dangling_key
        left_row = _synthesize_row(conn, schema, join.left_table, left_overrides, rng)
        if left_row is not None and _insert_row(conn, join.left_table, left_row):
            inserted = True

        # Fan-out/missing DISTINCT: add a second introduced-side row carrying
        # a key that already matches the preserved side.  If the right key is
        # unique (usually a PK), try the reverse side instead.
        candidates = [
            (join.right_table, join.right_column, join.left_table, join.left_column),
            (join.left_table, join.left_column, join.right_table, join.right_column),
        ]
        for dup_table, dup_col, source_table, source_col in candidates:
            table_schema = schema.table(dup_table)
            column_schema = table_schema.column(dup_col) if table_schema else None
            if column_schema is not None and column_schema.pk:
                continue
            key_row = conn.execute(
                f'SELECT s."{source_col}" FROM "{source_table}" AS s '
                f'JOIN "{dup_table}" AS d ON s."{source_col}" = d."{dup_col}" '
                f'WHERE s."{source_col}" IS NOT NULL LIMIT 1'
            ).fetchone()
            if key_row is None:
                continue
            overrides = _other_filter_overrides(
                all_literals, None, dup_table, protected_columns=(dup_col,)
            )
            overrides[dup_col] = key_row[0]
            row = _synthesize_row(conn, schema, dup_table, overrides, rng)
            if row is not None and _insert_row(conn, dup_table, row):
                inserted = True
                break
        conn.commit()
    finally:
        conn.close()
    return inserted


# --------------------------------------------------------------------------
# Method C: untargeted bootstrap resampling
# --------------------------------------------------------------------------
def bootstrap_instance(src_db: str, dst_path: str, rng: random.Random,
                       schema: Optional[DBSchema] = None,
                       drop_frac: float = 0.15, dup_frac: float = 0.1,
                       jitter_frac: float = 0.05) -> bool:
    """Copy ``src_db`` to ``dst_path``, then drop/duplicate/jitter rows of each
    table at random. No SQL parsing at all - the always-available fallback.

    Duplicating a row is what gives this method its one real strength: it
    perturbs row *multiplicity* (how many times a row appears), which is
    exactly what a forgotten DISTINCT or a fan-out JOIN is sensitive to, and
    which dropping/inserting a *new* row combination (Method A) does not
    exercise at all.

    ``schema`` (optional) lets a duplicate skip an ``INTEGER PRIMARY KEY``
    (rowid-alias) column so SQLite assigns the copy a fresh rowid instead of
    colliding with the original's - without it, every duplicate on a table
    with such a column would hit a uniqueness violation and silently become a
    no-op, which is not a hypothetical: it is the common case for BIRD/Spider
    schemas (``id INTEGER PRIMARY KEY`` on nearly every table).
    """
    shutil.copy2(src_db, dst_path)
    conn = sqlite3.connect(dst_path)
    changed = False
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [r[0] for r in cur.fetchall()]
        for table in tables:
            try:
                rowids = [r[0] for r in conn.execute(f'SELECT rowid FROM "{table}"').fetchall()]
            except sqlite3.Error:
                continue  # a WITHOUT ROWID table - skip rather than special-case it
            if not rowids:
                continue
            n_drop = int(len(rowids) * drop_frac)
            n_dup = int(len(rowids) * dup_frac)

            # Sample the drop set ONCE and reuse it: calling rng.sample() twice
            # with the same arguments advances the RNG and returns a *different*
            # subset each time, which would let the "remaining" computation
            # below silently include rows that were already deleted.
            dropped = set(rng.sample(rowids, min(n_drop, len(rowids))))
            for rid in dropped:
                try:
                    conn.execute(f'DELETE FROM "{table}" WHERE rowid = ?', (rid,))
                    changed = True
                except sqlite3.Error:
                    pass

            remaining = [r for r in rowids if r not in dropped]
            if not remaining:
                continue
            all_cols = [d[0] for d in conn.execute(f'SELECT * FROM "{table}" LIMIT 1').description]
            rowid_alias = None
            if schema is not None:
                tbl = schema.table(table)
                if tbl is not None:
                    rowid_alias = next(
                        (c.name for c in tbl.columns if c.pk and "INT" in c.type.upper()), None
                    )
            cols = [c for c in all_cols if c != rowid_alias]
            col_list = ",".join(f'"{c}"' for c in cols)
            col_index = {name: i for i, name in enumerate(all_cols)}
            for rid in rng.sample(remaining, min(n_dup, len(remaining))):
                try:
                    src_row = conn.execute(f'SELECT * FROM "{table}" WHERE rowid = ?', (rid,)).fetchone()
                    if src_row is None:
                        continue
                    values = [src_row[col_index[c]] for c in cols]
                    placeholders = ",".join(["?"] * len(cols))
                    conn.execute(
                        f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})', values
                    )
                    changed = True
                except sqlite3.Error:
                    pass

            # Jitter ordinary numeric attributes, but never identities or
            # foreign keys: changing keys would turn this cheap fallback into
            # an accidental referential-integrity perturbation.
            numeric_cols: List[str] = []
            if schema is not None and tbl is not None:
                fk_cols = {
                    fk.from_column.lower() for fk in schema.foreign_keys
                    if fk.from_table.lower() == table.lower()
                }
                numeric_cols = [
                    col.name for col in tbl.columns
                    if not col.pk
                    and col.name.lower() not in fk_cols
                    and any(token in col.type.upper()
                            for token in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC"))
                ]
            n_jitter = min(len(remaining), int(len(remaining) * jitter_frac))
            if jitter_frac > 0 and n_jitter == 0 and remaining and numeric_cols:
                n_jitter = 1
            for rid in rng.sample(remaining, n_jitter):
                if not numeric_cols:
                    break
                column = rng.choice(numeric_cols)
                try:
                    old = conn.execute(
                        f'SELECT "{column}" FROM "{table}" WHERE rowid = ?', (rid,)
                    ).fetchone()
                    if old is None or old[0] is None or isinstance(old[0], bool):
                        continue
                    step = _numeric_step(old[0])
                    if step is None:
                        continue
                    delta = step if rng.random() < 0.5 else -step
                    conn.execute(
                        f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?',
                        (old[0] + delta, rid),
                    )
                    changed = True
                except sqlite3.Error:
                    pass
        conn.commit()
    finally:
        conn.close()
    return changed


# --------------------------------------------------------------------------
# entry point: build (or load from cache) the full set of stress instances
# --------------------------------------------------------------------------
def _cache_key(db_path: str, gold_sql: str) -> str:
    h = hashlib.sha1()
    h.update(f"v{_CACHE_VERSION}".encode("ascii"))
    h.update(b"\0")
    h.update(os.path.abspath(db_path).encode("utf-8"))
    h.update(b"\0")
    h.update((gold_sql or "").encode("utf-8"))
    return h.hexdigest()[:20]


def generate_stress_instances(db_path: str, gold_sql: str, cache_dir: str,
                              n: int = 5, seed: int = 0,
                              timeout: float = 10.0,
                              max_db_bytes: int = MAX_STRESS_DB_BYTES) -> List[str]:
    """Build up to ``n`` perturbed copies of ``db_path`` for ``gold_sql``.

    Cached on disk under ``cache_dir/<hash>/`` so repeated calls across GRPO
    steps and epochs are free after the first. Method A literal and join
    instances (filtered by Method B) are tried first; Method C fills any
    remaining slots, and is the *only* source when ``sqlglot`` is unavailable
    or the gold query has no targetable predicate or join.

    Every instance is a full file copy, so a database larger than
    ``max_db_bytes`` is skipped (returns ``[]``, i.e. no spurious check):
    three copies of a 5 GB BIRD database filled the disk and stalled GRPO.
    """
    try:
        if max_db_bytes and os.path.getsize(db_path) > max_db_bytes:
            return []
    except OSError:
        return []
    key = _cache_key(db_path, gold_sql)
    folder = os.path.join(cache_dir, key)
    manifest_path = os.path.join(folder, _MANIFEST)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            paths = [os.path.join(folder, name) for name in manifest.get("instances", [])]
            if all(os.path.exists(p) for p in paths):
                return paths
        except (OSError, json.JSONDecodeError):
            pass  # corrupt cache entry - fall through and rebuild

    os.makedirs(folder, exist_ok=True)
    rng = random.Random(seed)
    instances: List[str] = []

    try:
        schema = load_schema(db_path, n_examples=0)
    except (FileNotFoundError, sqlite3.Error):
        schema = DBSchema(db_id="", path=db_path)

    literals = extract_literals(gold_sql, schema)
    joins = extract_join_boundaries(gold_sql, schema)
    rng.shuffle(literals)
    rng.shuffle(joins)
    mutants = mutate_query(gold_sql, rng)

    for lit in literals:
        if len(instances) >= n:
            break
        idx = len(instances)
        dst = os.path.join(folder, f"instance_{idx}.sqlite")
        try:
            built = build_boundary_instance(
                db_path, dst, schema, lit, rng, all_literals=literals
            )
        except (sqlite3.Error, OSError):
            built = False
        if not built:
            if os.path.exists(dst):
                os.remove(dst)
            continue
        if _is_discriminating(gold_sql, mutants, dst, timeout):
            instances.append(dst)
        else:
            os.remove(dst)

    n_boundary = len(instances)
    for join in joins:
        if len(instances) >= n:
            break
        idx = len(instances)
        dst = os.path.join(folder, f"instance_{idx}.sqlite")
        try:
            built = build_join_instance(
                db_path, dst, schema, join, rng, all_literals=literals
            )
        except (sqlite3.Error, OSError):
            built = False
        if not built:
            if os.path.exists(dst):
                os.remove(dst)
            continue
        if _is_discriminating(gold_sql, mutants, dst, timeout):
            instances.append(dst)
        else:
            os.remove(dst)

    n_join = len(instances) - n_boundary

    attempts = 0
    while len(instances) < n and attempts < 2 * n:
        attempts += 1
        idx = len(instances)
        dst = os.path.join(folder, f"instance_{idx}.sqlite")
        try:
            built = bootstrap_instance(db_path, dst, rng, schema=schema)
        except (sqlite3.Error, OSError):
            built = False
        if built:
            instances.append(dst)
        elif os.path.exists(dst):
            os.remove(dst)

    manifest = {
        "instances": [os.path.basename(p) for p in instances],
        "n_boundary": n_boundary,
        "n_join": n_join,
        "n_bootstrap": len(instances) - n_boundary - n_join,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=4)
    return instances
