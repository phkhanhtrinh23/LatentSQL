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

  Method A (:func:`boundary_instances`) - parse gold's own WHERE/HAVING
  literals and insert synthetic rows straddling each one's boundary. Needs
  ``sqlglot`` (optional - see requirements.txt). This is the main mechanism:
  it is the only one of the three that can *manufacture* the specific
  counterexample row a predicate-level bug needs, because that exact row
  combination may not exist anywhere in the real data.

  Method B (:func:`_is_discriminating`) - a filter, not a generator. Keep a
  Method-A instance only if some small mutation of gold's own query (flip a
  comparison operator, nudge a literal, swap AND/OR) disagrees with gold on
  it. This is a no-LLM stand-in for the paper's "GPT-5.4 guided the selection
  of inserted rows to expose differences" - the same judgement call, made by
  trying plausible mutants instead of asking a model.

  Method C (:func:`bootstrap_instance`) - no parsing at all: duplicate or
  drop a few existing rows at random. It cannot manufacture a missing
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
        if not isinstance(col, exp.Column):
            continue
        other = node.expression
        if not isinstance(other, exp.Literal):
            continue  # a column-to-column comparison (a join key), not a predicate
        table = _resolve_table(col, alias_map, schema)
        if table is None:
            continue
        op = {"eq": "=", "neq": "!=", "gt": ">", "gte": ">=",
              "lt": "<", "lte": "<="}[node.key]
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
def mutate_query(sql: str, rng: random.Random, max_mutants: int = 4) -> List[str]:
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
    candidates: List[str] = []

    # Index-based correspondence: tree.copy() preserves traversal order, so
    # the i-th comparison/literal found in a *copy* is the same node as the
    # i-th one found in the original. Matching by rendered SQL text instead
    # would be fragile whenever two unrelated nodes happen to render
    # identically (e.g. two different "salary > 1000" conditions).
    comparisons = list(tree.find_all(_COMPARISON_TYPES))
    for i, node in enumerate(comparisons):
        if node.key not in _flip or len(candidates) >= max_mutants:
            continue
        mutant = tree.copy()
        target = list(mutant.find_all(_COMPARISON_TYPES))[i]
        target.replace(_flip[node.key](this=target.this.copy(),
                                       expression=target.expression.copy()))
        candidates.append(mutant.sql(dialect="sqlite"))

    literals = list(tree.find_all(exp.Literal))
    for i, node in enumerate(literals):
        if len(candidates) >= max_mutants:
            break
        if node.is_string:
            continue
        try:
            n = int(str(node.this))
        except ValueError:
            continue
        mutant = tree.copy()
        target = list(mutant.find_all(exp.Literal))[i]
        target.set("this", str(n + 1))
        candidates.append(mutant.sql(dialect="sqlite"))

    rng.shuffle(candidates)
    return candidates[:max_mutants]


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
    if lit.op in ("=", "!="):
        if isinstance(lit.value, str):
            return [(lit.value, lit.value + "_stress_neighbor")]
        return [(lit.value, lit.value + 1)]
    if lit.op in (">", ">="):
        return [(lit.value, lit.value - 1 if isinstance(lit.value, (int, float)) else lit.value)]
    if lit.op in ("<", "<="):
        return [(lit.value, lit.value + 1 if isinstance(lit.value, (int, float)) else lit.value)]
    if lit.op == "BETWEEN":
        lo, hi = lit.value, lit.value2
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return [(lo, lo - 1), (hi, hi + 1)]
        return []
    return []


def build_boundary_instance(src_db: str, dst_path: str, schema: DBSchema,
                            lit: BoundaryLiteral, rng: random.Random) -> bool:
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
                row = _synthesize_row(conn, schema, lit.table, {lit.column: value}, rng)
                if row is not None and _insert_row(conn, lit.table, row):
                    inserted = True
        conn.commit()
    finally:
        conn.close()
    return inserted


# --------------------------------------------------------------------------
# Method C: untargeted bootstrap resampling
# --------------------------------------------------------------------------
def bootstrap_instance(src_db: str, dst_path: str, rng: random.Random,
                       schema: Optional[DBSchema] = None,
                       drop_frac: float = 0.15, dup_frac: float = 0.1) -> bool:
    """Copy ``src_db`` to ``dst_path``, then drop/duplicate a few rows of each
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
        conn.commit()
    finally:
        conn.close()
    return changed


# --------------------------------------------------------------------------
# entry point: build (or load from cache) the full set of stress instances
# --------------------------------------------------------------------------
def _cache_key(db_path: str, gold_sql: str) -> str:
    h = hashlib.sha1()
    h.update(os.path.abspath(db_path).encode("utf-8"))
    h.update(b"\0")
    h.update((gold_sql or "").encode("utf-8"))
    return h.hexdigest()[:20]


def generate_stress_instances(db_path: str, gold_sql: str, cache_dir: str,
                              n: int = 5, seed: int = 0,
                              timeout: float = 10.0) -> List[str]:
    """Build up to ``n`` perturbed copies of ``db_path`` for ``gold_sql``.

    Cached on disk under ``cache_dir/<hash>/`` so repeated calls across GRPO
    steps and epochs are free after the first. Method A instances (filtered by
    Method B) are tried first; Method C fills any remaining slots, and is the
    *only* source when ``sqlglot`` is unavailable or the gold query has no
    literal comparison to anchor on (e.g. a bare ``SELECT COUNT(*) FROM t``).
    """
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
    rng.shuffle(literals)
    mutants = mutate_query(gold_sql, rng)

    for lit in literals:
        if len(instances) >= n:
            break
        idx = len(instances)
        dst = os.path.join(folder, f"instance_{idx}.sqlite")
        try:
            built = build_boundary_instance(db_path, dst, schema, lit, rng)
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

    manifest = {"instances": [os.path.basename(p) for p in instances],
                "n_from_method_a": sum(1 for _ in literals[: len(instances)])}
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return instances
