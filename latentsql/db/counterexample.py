"""Counterexample-guided database instances (main_latentsql.pdf Sec. 4.3, Eq. 22).

The fixed stress instances of ``spurious.py`` are built from the gold query
alone, so they can miss the particular mistake a student keeps making. Here,
during RL, a rollout Y that is correct on D and on every instance in the pool
but whose atomic operations differ from the gold's is attacked directly: we
search for a small edit Delta of D (at most ``n_delta`` inserted or modified
rows) such that

    exec(Y, D + Delta) != exec(Y*, D + Delta).

A found edit joins the question's pool (up to ``k_max`` instances counting the
fixed ones) and every later rollout of that question is checked against it.
If the budget runs out, Y is treated as equivalent to Y* (CEGIS,
Solar-Lezama et al., 2006).

An instance is stored as its edit script, not a database copy: one writable
copy of D per database is kept, and each check applies the edit inside a
SAVEPOINT and rolls it back. Candidate values come from the constants and
boundary values of the predicates in Y and Y*; edits never write primary-key
or foreign-key columns except to give an inserted row a fresh integer key, so
key and FK constraints hold. Training-only: evaluation uses the original
databases and the gold-only instances.

When mutation search finds nothing, an optional second verifier runs:
VeriEQL (He et al., OOPSLA 2024), an SMT-based bounded equivalence checker,
in its own Python environment (``scripts/verieql_worker.py``). Given the
involved tables, their primary and foreign keys and both queries, it either
proves the pair equivalent for every database with at most ``bound`` rows per
table or returns such a database on which they differ. That small database is
written to SQLite with the original CREATE TABLE statements, re-checked by
executing both queries (VeriEQL works in its own SQL semantics, e.g. it
upper-cases string literals), and joins the pool as a separate instance.
Equivalent, unsupported or timed-out pairs keep their reward.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import subprocess
import random
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .execute import ExecResult, gold_is_ordered, is_read_only, result_match
from .spurious import (MAX_STRESS_DB_BYTES, BoundaryLiteral, _neighbor, extract_literals)

Edit = Tuple[str, Tuple[Any, ...]]          # (sql, params)


def _q(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def run_on(conn: sqlite3.Connection, sql: str, timeout: float = 5.0) -> ExecResult:
    """Execute a read-only query on an open connection with a time limit."""
    if not is_read_only(sql):
        return ExecResult(ok=False, error="refused: non read-only statement")
    deadline = time.time() + timeout
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(100_000)
        cols = [d[0] for d in (cur.description or [])]
        return ExecResult(ok=True, rows=[tuple(r) for r in rows], columns=cols)
    except Exception as exc:          # sqlite3.Error, interrupt, bad SQL
        return ExecResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        conn.set_progress_handler(None, 0)


@dataclass
class _TableInfo:
    name: str
    columns: List[str]
    int_pk: Optional[str]                      # single INTEGER primary key, if any
    frozen: set = field(default_factory=set)   # PK / FK columns: never written
    has_pk: bool = False


class CounterexamplePool:
    """Per-question pools of separating edits, grown during GRPO."""

    def __init__(self, cache_dir: str, k_max: int = 10, n_delta: int = 3,
                 budget: int = 40, timeout: float = 5.0, seed: int = 0,
                 max_db_bytes: int = MAX_STRESS_DB_BYTES,
                 verieql_python: Optional[str] = None, verieql_dir: Optional[str] = None,
                 verieql_bound: int = 3, verieql_timeout: float = 60.0):
        self.cache_dir = cache_dir
        self.verieql_python, self.verieql_dir = verieql_python, verieql_dir
        self.verieql_bound, self.verieql_timeout = int(verieql_bound), float(verieql_timeout)
        self._inst_conns: Dict[str, sqlite3.Connection] = {}
        self.k_max, self.n_delta, self.budget = int(k_max), int(n_delta), int(budget)
        self.timeout = float(timeout)
        self.max_db_bytes = max_db_bytes
        self.rng = random.Random(seed)
        # an entry is an edit script of D (List[Edit]) or {"db": path} for a
        # separate counterexample database from VeriEQL
        self.pools: Dict[Tuple[str, str], List[Any]] = {}
        self._conns: Dict[str, sqlite3.Connection] = {}
        self._tables: Dict[str, Dict[str, _TableInfo]] = {}
        self.stats = {"searches": 0, "found": 0, "pool_rejections": 0, "candidates_tried": 0,
                      "search_seconds": 0.0, "pool_check_seconds": 0.0,
                      "found_mutation": 0, "found_verieql": 0,
                      "verieql_calls": 0, "verieql_equivalent": 0, "verieql_counterexample": 0,
                      "verieql_unconfirmed": 0, "verieql_unsupported": 0,
                      "verieql_seconds": 0.0}

    # -- working copies ------------------------------------------------------
    def _conn(self, db_path: str) -> Optional[sqlite3.Connection]:
        key = os.path.abspath(db_path)
        if key in self._conns:
            return self._conns[key]
        try:
            if self.max_db_bytes and os.path.getsize(db_path) > self.max_db_bytes:
                return None
        except OSError:
            return None
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        work = os.path.join(self.cache_dir, f"work_{h}.sqlite")
        if not os.path.exists(work):
            shutil.copyfile(db_path, work + ".tmp")
            os.replace(work + ".tmp", work)
        conn = sqlite3.connect(work, isolation_level=None)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        self._conns[key] = conn
        self._tables[key] = self._read_tables(conn)
        return conn

    @staticmethod
    def _read_tables(conn: sqlite3.Connection) -> Dict[str, _TableInfo]:
        out: Dict[str, _TableInfo] = {}
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        fk_targets: Dict[str, set] = {}
        for t in names:
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                fk_targets.setdefault(fk[2].lower(), set()).add(str(fk[4] or "").lower())
        for t in names:
            info = list(conn.execute(f"PRAGMA table_info({_q(t)})"))
            cols = [r[1] for r in info]
            pks = [r for r in info if r[5]]
            int_pk = pks[0][1] if len(pks) == 1 and "INT" in str(pks[0][2]).upper() else None
            frozen = {r[1].lower() for r in pks}
            frozen |= {str(fk[3]).lower() for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})")}
            frozen |= fk_targets.get(t.lower(), set())
            out[t.lower()] = _TableInfo(t, cols, int_pk, frozen, has_pk=bool(pks))
        return out

    def close(self, remove: bool = False) -> None:
        for conn in self._inst_conns.values():
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._inst_conns.clear()
        for conn in self._conns.values():
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._conns.clear()
        if remove and os.path.isdir(self.cache_dir):
            for name in os.listdir(self.cache_dir):
                if name.startswith(("work_", "veq_")):
                    try:
                        os.remove(os.path.join(self.cache_dir, name))
                    except OSError:
                        pass

    # -- checking ----------------------------------------------------------------
    def _separates(self, conn: sqlite3.Connection, edit: Sequence[Edit],
                   pred: str, gold: str, ordered: bool) -> Optional[bool]:
        """True if the edit makes pred and gold disagree; None if it is unusable."""
        conn.execute("SAVEPOINT cx")
        try:
            for sql, params in edit:
                conn.execute(sql, params)
            g = run_on(conn, gold, self.timeout)
            if not g.ok:
                return None                      # the edit broke gold itself
            p = run_on(conn, pred, self.timeout)
            return not result_match(p, g, ordered=ordered)
        except sqlite3.Error:
            return None
        finally:
            conn.execute("ROLLBACK TO cx")
            conn.execute("RELEASE cx")

    def pool_size(self, db_path: str, gold: str) -> int:
        return len(self.pools.get((os.path.abspath(db_path), gold), []))

    def passes_pool(self, db_path: str, pred: str, gold: str) -> bool:
        """False if pred disagrees with gold on any counterexample instance."""
        t0 = time.time()
        try:
            return self._passes_pool(db_path, pred, gold)
        except Exception as exc:
            self.stats["errors"] = self.stats.get("errors", 0) + 1
            self.stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return True
        finally:
            self.stats["pool_check_seconds"] += time.time() - t0

    def _passes_pool(self, db_path: str, pred: str, gold: str) -> bool:
        pool = self.pools.get((os.path.abspath(db_path), gold), [])
        if not pool:
            return True
        ordered = gold_is_ordered(gold)
        for entry in pool:
            if isinstance(entry, dict):
                differs = self._differs_on(entry["db"], pred, gold, ordered)
            else:
                conn = self._conn(db_path)
                differs = conn is not None and self._separates(conn, entry, pred, gold, ordered)
            if differs:
                self.stats["pool_rejections"] += 1
                return False
        return True

    def _differs_on(self, inst_path: str, pred: str, gold: str, ordered: bool) -> bool:
        conn = self._inst_conns.get(inst_path)
        if conn is None:
            conn = sqlite3.connect(f"file:{inst_path}?mode=ro", uri=True)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            self._inst_conns[inst_path] = conn
        g = run_on(conn, gold, self.timeout)
        if not g.ok:
            return False
        return not result_match(run_on(conn, pred, self.timeout), g, ordered=ordered)

    # -- search ---------------------------------------------------------------------
    def _tables_of(self, sql: str, tables: Dict[str, _TableInfo]) -> List[_TableInfo]:
        try:
            import sqlglot
            from sqlglot import exp

            names = {t.name.lower() for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}
        except Exception:
            names = {k for k in tables if k in (sql or "").lower()}
        return [tables[n] for n in names if n in tables]

    def _literals(self, db_path: str, sqls: Sequence[str]) -> List[BoundaryLiteral]:
        from .schema import load_schema_cached

        try:
            schema = load_schema_cached(db_path)
        except Exception:
            return []
        out: List[BoundaryLiteral] = []
        for s in sqls:
            try:
                out += extract_literals(s, schema)
            except Exception:
                continue
        return out

    def _candidate_edits(self, conn, tables: List[_TableInfo],
                         literals: List[BoundaryLiteral]) -> List[Edit]:
        """Single-row edits: duplicates, inserts and updates at predicate values."""
        edits: List[Edit] = []
        for t in tables:
            try:
                rows = list(conn.execute(
                    f"SELECT rowid, * FROM {_q(t.name)} WHERE rowid IN "
                    f"(SELECT rowid FROM {_q(t.name)} LIMIT 200) ORDER BY RANDOM() LIMIT 3"))
            except sqlite3.Error:
                continue
            if not rows:
                continue
            values: Dict[str, List[Any]] = {}
            real = {c.lower(): c for c in t.columns}
            for lit in literals:
                if lit.table.lower() != t.name.lower() or lit.column.lower() in t.frozen:
                    continue
                col = real.get(lit.column.lower())
                if col is None:
                    continue
                vs = [lit.value, _neighbor(lit.value)]
                if lit.value2 is not None:
                    vs += [lit.value2, _neighbor(lit.value2)]
                values.setdefault(col, []).extend(v for v in vs if v is not None)
            for c in t.columns:
                if c.lower() not in t.frozen:
                    values.setdefault(c, []).append(None)      # NULLs expose COUNT/IS NULL bugs
            col_list = ", ".join(_q(c) for c in t.columns)
            placeholders = ", ".join("?" for _ in t.columns)

            def insert(row: Sequence[Any], override: Dict[str, Any]) -> Optional[Edit]:
                vals = list(row)
                for c, v in override.items():
                    vals[t.columns.index(c)] = v
                if t.int_pk:
                    # fresh key computed inside the savepoint, so it stays unique
                    i = t.columns.index(t.int_pk)
                    cols = [c for j, c in enumerate(t.columns) if j != i]
                    vals = [v for j, v in enumerate(vals) if j != i]
                    sql = (f"INSERT INTO {_q(t.name)} ({_q(t.int_pk)}, "
                           f"{', '.join(_q(c) for c in cols)}) SELECT "
                           f"COALESCE(MAX({_q(t.int_pk)}), 0) + 1, "
                           f"{', '.join('?' for _ in cols)} FROM {_q(t.name)}")
                    return (sql, tuple(vals))
                if t.has_pk:
                    # composite or non-integer key: a copy would duplicate it
                    return None
                return (f"INSERT INTO {_q(t.name)} ({col_list}) VALUES ({placeholders})",
                        tuple(vals))

            for row in rows:
                rowid, data = row[0], row[1:]
                e = insert(data, {})                           # duplicated row
                if e:
                    edits.append(e)
                for c, vs in values.items():
                    for v in vs:
                        e = insert(data, {c: v})
                        if e:
                            edits.append(e)
                        edits.append((f"UPDATE {_q(t.name)} SET {_q(c)} = ? WHERE rowid = ?",
                                      (v, rowid)))
        return edits

    def search(self, db_path: str, pred: str, gold: str) -> bool:
        """Look for a separating edit; on success add it to the pool."""
        t0 = time.time()
        try:
            return self._search(db_path, pred, gold)
        except Exception as exc:          # never let the verifier crash training
            self.stats["errors"] = self.stats.get("errors", 0) + 1
            self.stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return False
        finally:
            self.stats["search_seconds"] += time.time() - t0

    def _search(self, db_path: str, pred: str, gold: str) -> bool:
        self.stats["searches"] += 1
        if self._mutation_search(db_path, pred, gold):
            self.stats["found_mutation"] += 1
            return True
        if self.verieql_python and self.verieql_dir and self._verieql_search(db_path, pred, gold):
            self.stats["found_verieql"] += 1
            self.stats["found"] += 1
            return True
        return False

    def _mutation_search(self, db_path: str, pred: str, gold: str) -> bool:
        key = (os.path.abspath(db_path), gold)
        conn = self._conn(db_path)
        if conn is None:
            return False
        tables = self._tables[os.path.abspath(db_path)]
        involved = self._tables_of(pred, tables) + [
            t for t in self._tables_of(gold, tables) if t not in self._tables_of(pred, tables)]
        singles = self._candidate_edits(conn, involved, self._literals(db_path, [pred, gold]))
        if not singles:
            return False
        self.rng.shuffle(singles)
        ordered = gold_is_ordered(gold)
        candidates: List[List[Edit]] = [[e] for e in singles]
        for n in range(2, self.n_delta + 1):
            combos = list(itertools.islice(itertools.combinations(singles, n), 4 * self.budget))
            self.rng.shuffle(combos)
            candidates += [list(c) for c in combos]
        for edit in candidates[: self.budget]:
            self.stats["candidates_tried"] += 1
            if self._separates(conn, edit, pred, gold, ordered):
                pool = self.pools.setdefault(key, [])
                pool.append(edit)
                self.stats["found"] += 1
                return True
        return False

    # -- VeriEQL fallback -------------------------------------------------------------
    def _schema_tables(self, db_path: str) -> Dict[str, _TableInfo]:
        """Table info without needing a working copy (works for large databases)."""
        key = os.path.abspath(db_path)
        if key not in self._tables:
            ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                self._tables[key] = self._read_tables(ro)
            finally:
                ro.close()
        return self._tables[key]

    @staticmethod
    def _veq_type(decl: str) -> str:
        d = (decl or "").upper()
        if "DATE" in d or "TIME" in d:
            return "DATE"
        if "BOOL" in d:
            return "BOOLEAN"
        if any(t in d for t in ("CHAR", "TEXT", "CLOB", "STRING")) or not d:
            return "VARCHAR"
        return "INT"    # VeriEQL encodes the remaining numeric types as integers

    def _verieql_search(self, db_path: str, pred: str, gold: str) -> bool:
        tables_all = self._schema_tables(db_path)
        involved = {t.name.lower(): t for t in self._tables_of(pred, tables_all)}
        involved.update({t.name.lower(): t for t in self._tables_of(gold, tables_all)})
        if not involved:
            return False
        ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            decl: Dict[str, Dict[str, str]] = {}
            create_sql: Dict[str, str] = {}
            constraints: List[Dict[str, Any]] = []
            for low, t in involved.items():
                info = list(ro.execute(f"PRAGMA table_info({_q(t.name)})"))
                decl[t.name] = {r[1]: self._veq_type(r[2]) for r in info}
                create_sql[t.name] = ro.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (t.name,)).fetchone()[0]
                pks = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
                if pks:
                    constraints.append({"primary": [
                        {"value": f"{t.name.upper()}__{c.upper()}"} for c in pks]})
                for fk in ro.execute(f"PRAGMA foreign_key_list({_q(t.name)})"):
                    if fk[2].lower() in involved and fk[4]:
                        constraints.append({"foreign": [
                            {"value": f"{t.name.upper()}__{str(fk[3]).upper()}"},
                            {"value": f"{involved[fk[2].lower()].name.upper()}__{str(fk[4]).upper()}"},
                        ]})
        finally:
            ro.close()
        req = {
            "sql1": gold, "sql2": pred,
            "schema": {t.upper(): {c.upper(): ty for c, ty in cols.items()}
                       for t, cols in decl.items()},
            "constraints": constraints, "bound": self.verieql_bound,
            "verieql_dir": self.verieql_dir,
        }
        worker = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "scripts", "verieql_worker.py")
        self.stats["verieql_calls"] += 1
        t0 = time.time()
        try:
            proc = subprocess.run([self.verieql_python, worker], input=json.dumps(req),
                                  capture_output=True, text=True, timeout=self.verieql_timeout)
            out = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            out = {"state": "ERR"}
        finally:
            self.stats["verieql_seconds"] += time.time() - t0
        state = out.get("state")
        if state == "EQU":
            self.stats["verieql_equivalent"] += 1
            return False
        if state != "NEQ" or not out.get("tables"):
            self.stats["verieql_unsupported"] += 1
            return False
        self.stats["verieql_counterexample"] += 1
        path = self._write_instance(db_path, gold, pred, out["tables"], decl, create_sql)
        if path is None or not self._differs_on(path, pred, gold, gold_is_ordered(gold)):
            self.stats["verieql_unconfirmed"] += 1
            if path:
                self._drop_instance(path)
            return False
        self.pools.setdefault((os.path.abspath(db_path), gold), []).append({"db": path})
        return True

    def _drop_instance(self, path: str) -> None:
        conn = self._inst_conns.pop(path, None)
        if conn is not None:
            conn.close()
        try:
            os.remove(path)
        except OSError:
            pass

    def _write_instance(self, db_path: str, gold: str, pred: str,
                        tables: Dict[str, List[Dict[str, str]]],
                        decl: Dict[str, Dict[str, str]], create_sql: Dict[str, str]) -> Optional[str]:
        """VeriEQL's counterexample rows -> a SQLite file with the original DDL."""
        literals = {m.upper(): m for m in re.findall(r"'((?:[^']|'')*)'", f"{gold} {pred}")}
        by_upper = {t.upper(): t for t in decl}

        def value(lit: str, col_type: str) -> Any:
            if lit is None or lit.upper() == "NULL":
                return None
            if len(lit) >= 2 and lit[0] == lit[-1] == "'":
                text = lit[1:-1]
                return literals.get(text.upper(), text)
            try:
                return int(lit)
            except ValueError:
                try:
                    return float(lit)
                except ValueError:
                    return lit

        h = hashlib.sha1(f"{db_path}\0{gold}\0{pred}".encode("utf-8")).hexdigest()[:20]
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"veq_{h}.sqlite")
        if os.path.exists(path):
            os.remove(path)
        conn = sqlite3.connect(path)
        try:
            for name, ddl in create_sql.items():
                conn.execute(ddl)
            for up, rows in tables.items():
                name = by_upper.get(up.upper())
                if name is None:
                    continue
                cols = {c.upper(): c for c in decl[name]}
                for row in rows:
                    names = [cols[c.upper()] for c in row if c.upper() in cols]
                    vals = [value(v, decl[name][cols[c.upper()]]) for c, v in row.items()
                            if c.upper() in cols]
                    conn.execute(
                        f"INSERT INTO {_q(name)} ({', '.join(_q(c) for c in names)}) "
                        f"VALUES ({', '.join('?' for _ in names)})", vals)
            conn.commit()
        except sqlite3.Error:
            conn.close()
            os.remove(path)
            return None
        conn.close()
        return path
