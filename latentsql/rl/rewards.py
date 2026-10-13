"""The RL reward (Sec. 4.3, Eq. 20-21)::

    s(Y) = R_atomic(Y) + 1[EX = 1] * R_spur(Y)
    R_spur(Y) = prod_{D' in I(Y*)} EX(Y, Y*, D')

``R_atomic`` is the dense atomic-operation overlap with the gold query
(``atomic.py``). ``I(Y*)`` starts with ``n_initial_instances`` database
instances built from the gold query (``db/instances.py``) and grows by
counterexample-guided synthesis (``db/counterexample.py``): when a rollout
passes every instance but ``O(Y) != O(Y*)``, the verifier searches for a
database on which Y and Y* disagree, and a found one joins the pool (up to
``max_pool_size`` instances).

Ablations of Table 2: ``use_atomic = False`` drops R_atomic, ``use_spurious =
False`` replaces R_spur by 1 (plain EX), ``use_counterexamples = False`` keeps
the initial instances only.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..db.execute import ExecResult, execute_sql, gold_is_ordered, result_match
from ..prompts import normalize_sql
from .atomic import AtomicOpsReward


class ExecutionOracle:
    """Cached execution against D and against the instance pool I(Y*)."""

    def __init__(self, cfg: Any, cache_dir: str, timeout: float = 30.0):
        self.cfg = cfg
        self.timeout = timeout
        self.cache_dir = cache_dir
        self._gold: Dict[Tuple[str, str], ExecResult] = {}
        self._ex: Dict[Tuple[str, str, str], bool] = {}
        self._instances: Dict[Tuple[str, str], List[str]] = {}
        self.n_exec = 0
        self.pool = None
        if cfg.use_spurious and cfg.use_counterexamples:
            from ..db.counterexample import CounterexamplePool

            self.pool = CounterexamplePool(
                os.path.join(cache_dir, "counterexamples"), k_max=cfg.max_pool_size,
                verifier=cfg.verifier, verieql_python=cfg.verieql_python,
                verieql_dir=cfg.verieql_dir, verieql_bound=cfg.verieql_bound,
                verieql_timeout=cfg.verieql_timeout, seed=cfg.seed,
            )

    def gold_result(self, db_path: str, gold_sql: str) -> ExecResult:
        key = (db_path, normalize_sql(gold_sql))
        if key not in self._gold:
            self._gold[key] = execute_sql(db_path, gold_sql, timeout=self.timeout)
            self.n_exec += 1
        return self._gold[key]

    def ex(self, db_path: str, pred_sql: Optional[str], gold_sql: str) -> bool:
        """EX(Y, Y*, D) of Eq. 10."""
        pred = normalize_sql(pred_sql)
        if not pred:
            return False
        key = (db_path, pred, normalize_sql(gold_sql))
        if key not in self._ex:
            gold = self.gold_result(db_path, gold_sql)
            res = execute_sql(db_path, pred, timeout=self.timeout)
            self.n_exec += 1
            self._ex[key] = gold.ok and result_match(res, gold, ordered=gold_is_ordered(gold_sql))
        return self._ex[key]

    def instances(self, db_path: str, gold_sql: str) -> List[str]:
        """The initial instances of I(Y*), built once per gold query."""
        key = (db_path, normalize_sql(gold_sql))
        if key not in self._instances:
            from ..db.instances import generate_stress_instances

            self._instances[key] = generate_stress_instances(
                db_path, gold_sql, os.path.join(self.cache_dir, "instances"),
                n=self.cfg.n_initial_instances, seed=self.cfg.seed, timeout=self.timeout,
            )
        return self._instances[key]

    def robust(self, db_path: str, pred_sql: str, gold_sql: str) -> bool:
        """``prod EX`` over the current pool (initial instances + counterexamples)."""
        for inst in self.instances(db_path, gold_sql):
            gold = self.gold_result(inst, gold_sql)
            if not gold.ok:
                continue          # the edit broke the gold query itself; not a valid test
            self.n_exec += 1
            if not result_match(execute_sql(inst, pred_sql, timeout=self.timeout), gold,
                                ordered=gold_is_ordered(gold_sql)):
                return False
        if self.pool is not None and not self.pool.passes_pool(db_path, pred_sql, gold_sql):
            return False
        return True

    def pool_size(self, db_path: str, gold_sql: str) -> int:
        n = len(self.instances(db_path, gold_sql))
        return n + (self.pool.pool_size(db_path, gold_sql) if self.pool is not None else 0)

    def release(self, db_path: str, gold_sql: str) -> None:
        """Drop one question's initial instances from disk after its rollout group."""
        if not self.cfg.cleanup_instances:
            return
        paths = self._instances.pop((db_path, normalize_sql(gold_sql)), [])
        for p in paths:
            self._gold.pop((p, normalize_sql(gold_sql)), None)
        for folder in {os.path.dirname(p) for p in paths}:
            shutil.rmtree(folder, ignore_errors=True)


@dataclass
class Reward:
    total: float
    ex: float
    r_atomic: float = 0.0
    r_spur: float = 1.0
    counterexample_found: bool = False
    sql: Optional[str] = None


class LatentSQLReward:
    """``s(Y) = R_atomic(Y) + 1[EX = 1] R_spur(Y)`` (Eq. 21)."""

    def __init__(self, cfg: Any, oracle: ExecutionOracle):
        self.cfg = cfg
        self.oracle = oracle
        self.atomic = AtomicOpsReward()

    def __call__(self, sql: Optional[str], db_path: str, gold_sql: str) -> Reward:
        sql = normalize_sql(sql) or None
        ex = 1.0 if (sql and self.oracle.ex(db_path, sql, gold_sql)) else 0.0
        r_atomic = self.atomic.score(sql, gold_sql) if (sql and self.cfg.use_atomic) else 0.0
        r_spur, found = 1.0, False
        if ex and self.cfg.use_spurious:
            r_spur = 1.0 if self.oracle.robust(db_path, sql, gold_sql) else 0.0
            pool = self.oracle.pool
            if (r_spur and pool is not None
                    and self.atomic.ops(sql) != self.atomic.ops(gold_sql)
                    and self.oracle.pool_size(db_path, gold_sql) < self.cfg.max_pool_size):
                found = pool.search(db_path, sql, gold_sql)
                r_spur = 0.0 if found else 1.0
        return Reward(total=r_atomic + ex * r_spur, ex=ex, r_atomic=r_atomic,
                      r_spur=r_spur, counterexample_found=found, sql=sql)


def group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> List[float]:
    """Dr. GRPO advantage: ``r_i - mean(r)``, with no division by the group std."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    if max(rewards) - min(rewards) < eps:
        return [0.0] * n
    return [r - mean for r in rewards]
