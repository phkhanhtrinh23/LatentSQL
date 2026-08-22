"""SQLite execution plus the Spider/BIRD execution-accuracy and VES metrics.

Execution runs in a worker thread with a SQLite progress-handler interrupt so a
runaway query (a cartesian product on BIRD's larger DBs) cannot wedge the
pipeline. Comparison follows the official Spider ``eval_exec_match`` semantics:
row multisets are compared order-insensitively unless the *gold* query has a
top-level ORDER BY, in which case order matters.
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


@dataclass
class ExecResult:
    ok: bool
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    error: str = ""
    elapsed: float = 0.0
    truncated: bool = False

    @property
    def empty(self) -> bool:
        return self.ok and len(self.rows) == 0

    def preview(self, n: int = 3, max_chars: int = 400) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        if not self.rows:
            return "(0 rows)"
        head = ", ".join(self.columns) if self.columns else ""
        body = "\n".join(str(r) for r in self.rows[:n])
        text = (head + "\n" + body) if head else body
        return text[:max_chars] + ("..." if len(text) > max_chars else "")


_WRITE = re.compile(
    r"(?is)\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b"
)


def is_read_only(sql: str) -> bool:
    """Reject anything that could mutate the benchmark databases."""
    stripped = re.sub(r"(?s)/\*.*?\*/", " ", sql or "")
    stripped = re.sub(r"(?m)--.*$", " ", stripped)
    # a leading WITH ... CREATE is still a write; look at the whole statement
    return not _WRITE.search(stripped)


def execute_sql(
    db_path: str,
    sql: str,
    timeout: float = 30.0,
    max_rows: int = 10000,
    read_only: bool = True,
) -> ExecResult:
    """Run ``sql`` against ``db_path`` and return rows or the error message."""
    if not sql or not sql.strip():
        return ExecResult(ok=False, error="empty query")
    if read_only and not is_read_only(sql):
        return ExecResult(ok=False, error="refused: non read-only statement")
    if not os.path.exists(db_path):
        return ExecResult(ok=False, error=f"missing database: {db_path}")

    box: dict = {}
    start = time.time()
    deadline = start + timeout

    def _run() -> None:
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            # interrupt long-running queries from inside the VM loop
            conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 2000)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            cols = [d[0] for d in cur.description] if cur.description else []
            box["res"] = ExecResult(
                ok=True, rows=[tuple(r) for r in rows[:max_rows]],
                columns=cols, truncated=truncated,
            )
        except Exception as exc:  # sqlite3.Error, OperationalError on interrupt, ...
            box["res"] = ExecResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout + 1.0)
    res = box.get("res") or ExecResult(ok=False, error=f"timeout after {timeout}s")
    res.elapsed = time.time() - start
    return res


# --------------------------------------------------------------------------
# execution accuracy
# --------------------------------------------------------------------------
_ORDER_BY = re.compile(r"(?i)\border\s+by\b")


def gold_is_ordered(gold_sql: str) -> bool:
    """True when the gold query's result order is semantically meaningful.

    ORDER BY inside a sub-query does not make the *outer* result ordered, so we
    only look at the tail of the statement, following the Spider evaluator.
    """
    if not gold_sql:
        return False
    depth, last_top_level = 0, -1
    for m in re.finditer(r"[()]|\border\s+by\b", gold_sql, flags=re.IGNORECASE):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            last_top_level = m.start()
    return last_top_level >= 0


def _norm_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return "nan"
        return round(f, 6)
    s = str(v).strip()
    # numeric strings compare as numbers ("3" == 3 == 3.0)
    try:
        return round(float(s), 6)
    except ValueError:
        return s.lower()


def _norm_rows(rows: Sequence[Sequence[Any]], ordered: bool):
    """Column order is not part of Spider's EX metric, so sort within each row."""
    norm = [tuple(sorted((str(_norm_cell(c)) for c in row))) for row in rows]
    return norm if ordered else sorted(norm)


def result_match(
    pred: ExecResult, gold: ExecResult, ordered: bool = False
) -> bool:
    if not (pred.ok and gold.ok):
        return False
    if len(pred.rows) != len(gold.rows):
        return False
    return _norm_rows(pred.rows, ordered) == _norm_rows(gold.rows, ordered)


def execution_accuracy(
    db_path: str,
    pred_sql: Optional[str],
    gold_sql: str,
    timeout: float = 30.0,
) -> Tuple[bool, ExecResult, ExecResult]:
    """Return ``(is_correct, pred_result, gold_result)``."""
    gold_res = execute_sql(db_path, gold_sql, timeout=timeout)
    if not pred_sql:
        return False, ExecResult(ok=False, error="no prediction"), gold_res
    pred_res = execute_sql(db_path, pred_sql, timeout=timeout)
    ok = result_match(pred_res, gold_res, ordered=gold_is_ordered(gold_sql))
    return ok, pred_res, gold_res


def valid_efficiency_score(
    db_path: str,
    pred_sql: Optional[str],
    gold_sql: str,
    timeout: float = 30.0,
    repeats: int = 3,
) -> float:
    """BIRD's VES: sqrt(t_gold / t_pred) for correct queries, else 0.

    Timing is the median of ``repeats`` runs to damp OS noise.
    """
    ok, _, _ = execution_accuracy(db_path, pred_sql, gold_sql, timeout=timeout)
    if not ok:
        return 0.0

    def _median_time(sql: str) -> float:
        ts = []
        for _ in range(max(1, repeats)):
            r = execute_sql(db_path, sql, timeout=timeout)
            if not r.ok:
                return float("nan")
            ts.append(max(r.elapsed, 1e-6))
        ts.sort()
        return ts[len(ts) // 2]

    t_gold, t_pred = _median_time(gold_sql), _median_time(pred_sql or "")
    if not (t_gold > 0 and t_pred > 0) or math.isnan(t_gold) or math.isnan(t_pred):
        return 0.0
    return math.sqrt(t_gold / t_pred)
