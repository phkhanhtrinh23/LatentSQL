"""IMAD's reward function and its dynamic schedules (paper section 2.3).

    r(x, y) = w_fmt * R_fmt  +  w_clip * R(y; l)                        (eq. 1)

    R(y; l) = 1 if the correct final answer is present in the first l tokens
              of y, else 0                                              (eq. 2)

Two things are annealed over training:

* ``w_fmt``: 1.0 -> 0.05, removing the incentive to *verbalise* the debate.
* ``l``:     l0 -> l*,   making verbalisation impossible while still earning
  the correctness reward.

The paper's insight is that the interaction of the two is what forces the model
to move the debate into its latent space: "the only viable strategy for the
model is to perform the multi-perspective analysis learned during SFT
internally, then directly generate the final answer."

For Text-to-SQL, "correct answer" means *execution* match against the gold
query, not string match - so ``R(y; l)`` runs SQL. Results are cached per
(db, sql) pair because GRPO re-scores the same queries constantly.
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..db.execute import ExecResult, execute_sql, gold_is_ordered, result_match
from ..mas.trace import (
    REQUIRED_TAGS,
    extract_final_sql,
    format_score,
    normalize_sql,
    tag_order_score,
)


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------
def _progress(step: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return min(1.0, max(0.0, step / float(total - 1)))


def anneal(start: float, end: float, step: int, total: int, kind: str = "linear") -> float:
    """Interpolate ``start`` -> ``end`` across training."""
    p = _progress(step, total)
    if kind == "cosine":
        p = 0.5 * (1.0 - math.cos(math.pi * p))
    elif kind == "step":
        p = 0.0 if p < 0.5 else 1.0
    return start + (end - start) * p


@dataclass
class RewardSchedule:
    """Current ``(w_fmt, l)`` for a training step."""

    cfg: Any
    total_steps: int = 1000

    def w_fmt(self, step: int) -> float:
        return anneal(
            self.cfg.w_fmt_start, self.cfg.w_fmt_end, step, self.total_steps, self.cfg.schedule
        )

    def length_limit(self, step: int) -> int:
        return int(round(anneal(
            float(self.cfg.len_start), float(self.cfg.len_end),
            step, self.total_steps, self.cfg.schedule,
        )))

    def snapshot(self, step: int) -> Dict[str, float]:
        return {"step": step, "w_fmt": self.w_fmt(step), "length_limit": self.length_limit(step)}


# --------------------------------------------------------------------------
# execution cache
# --------------------------------------------------------------------------
class ExecutionOracle:
    """Caches gold results and predicted-vs-gold verdicts across GRPO steps."""

    def __init__(self, timeout: float = 30.0, max_cache: int = 200_000,
                stress_cache_dir: Optional[str] = None,
                n_stress_instances: int = 5, stress_seed: int = 0,
                cleanup_stress_cache: bool = False):
        self.timeout = timeout
        self.max_cache = max_cache
        self._gold: Dict[Tuple[str, str], ExecResult] = {}
        self._verdict: Dict[Tuple[str, str, str], bool] = {}
        self.n_exec = 0
        # spurious-correctness stress testing (EX-Ext, adapted - see
        # latentsql/db/spurious.py). Only used when a caller asks for
        # spurious_agreement(); building nothing here keeps every existing
        # ExecutionOracle() call (smoke test, scripts, is_correct-only paths)
        # byte-for-byte unaffected.
        self.stress_cache_dir = stress_cache_dir
        self.n_stress_instances = n_stress_instances
        self.stress_seed = stress_seed
        self.cleanup_stress_cache = cleanup_stress_cache
        self._stress_paths: Dict[Tuple[str, str], List[str]] = {}
        self._spurious: Dict[Tuple[str, str, str], float] = {}

    def gold_result(self, db_path: str, gold_sql: str) -> ExecResult:
        key = (db_path, normalize_sql(gold_sql))
        if key not in self._gold:
            self._gold[key] = execute_sql(db_path, gold_sql, timeout=self.timeout)
            self.n_exec += 1
        return self._gold[key]

    def is_correct(self, db_path: str, pred_sql: Optional[str], gold_sql: str) -> bool:
        pred = normalize_sql(pred_sql)
        if not pred:
            return False
        key = (db_path, pred, normalize_sql(gold_sql))
        if key in self._verdict:
            return self._verdict[key]
        gold_res = self.gold_result(db_path, gold_sql)
        if not gold_res.ok:
            self._verdict[key] = False
            return False
        pred_res = execute_sql(db_path, pred, timeout=self.timeout)
        self.n_exec += 1
        ok = result_match(pred_res, gold_res, ordered=gold_is_ordered(gold_sql))
        if len(self._verdict) < self.max_cache:
            self._verdict[key] = ok
        return ok

    def executes(self, db_path: str, pred_sql: Optional[str]) -> bool:
        pred = normalize_sql(pred_sql)
        if not pred:
            return False
        res = execute_sql(db_path, pred, timeout=self.timeout)
        self.n_exec += 1
        return res.ok

    def _stress_instances(self, db_path: str, gold_sql: str) -> List[str]:
        key = (db_path, normalize_sql(gold_sql))
        if key not in self._stress_paths:
            from ..db.spurious import generate_stress_instances

            cache_dir = self.stress_cache_dir or os.path.join(
                os.path.dirname(os.path.abspath(db_path)) or ".", ".stress_cache"
            )
            self._stress_paths[key] = generate_stress_instances(
                db_path, gold_sql, cache_dir,
                n=self.n_stress_instances, seed=self.stress_seed, timeout=self.timeout,
            )
        return self._stress_paths[key]

    def spurious_agreement(self, db_path: str, pred_sql: Optional[str], gold_sql: str) -> float:
        """Fraction of EX-Ext-style stress instances (``latentsql/db/spurious.py``,
        adapting ``pool_text2sql_crowd_source_mixture/main_pool_eval.pdf``
        Appendix B) on which ``pred_sql`` still agrees with ``gold_sql``.

        Only meaningful to call on a rollout that already passed
        :meth:`is_correct` on the original database - this is a check on
        *why* it passed, not a second chance for one that failed. Returns
        ``1.0`` (fully robust - no penalty) if there are no usable stress
        instances, e.g. an un-parseable gold query with no literal for
        Method A and a bootstrap that happened to change nothing.
        """
        pred = normalize_sql(pred_sql)
        if not pred:
            return 0.0
        key = (db_path, pred, normalize_sql(gold_sql))
        if key in self._spurious:
            return self._spurious[key]
        instances = self._stress_instances(db_path, gold_sql)
        if not instances:
            self._spurious[key] = 1.0
            return 1.0
        agree, checked = 0, 0
        ordered = gold_is_ordered(gold_sql)
        for inst_path in instances:
            # gold_result's own cache is keyed by (path, sql), and a stress
            # instance's path is distinct from db_path, so this transparently
            # caches gold's result per instance across rollouts for free.
            gold_res = self.gold_result(inst_path, gold_sql)
            if not gold_res.ok:
                continue  # the perturbation broke gold itself - don't hold pred to that instance
            checked += 1
            pred_res = execute_sql(inst_path, pred, timeout=self.timeout)
            self.n_exec += 1
            if result_match(pred_res, gold_res, ordered=ordered):
                agree += 1
        score = (agree / checked) if checked else 1.0
        if len(self._spurious) < self.max_cache:
            self._spurious[key] = score
        return score

    def release_stress_instances(self, db_path: str, gold_sql: str) -> None:
        """Release one query's stress DBs after its rollout group is finished."""
        if not self.cleanup_stress_cache:
            return
        normalized_gold = normalize_sql(gold_sql)
        key = (db_path, normalized_gold)
        paths = self._stress_paths.pop(key, [])
        path_set = set(paths)
        for cache_key in list(self._gold):
            if cache_key[0] in path_set:
                self._gold.pop(cache_key, None)
        for cache_key in list(self._verdict):
            if cache_key[0] in path_set:
                self._verdict.pop(cache_key, None)
        for cache_key in list(self._spurious):
            if cache_key[0] == db_path and cache_key[2] == normalized_gold:
                self._spurious.pop(cache_key, None)
        folders = {os.path.dirname(path) for path in paths}
        cache_root = os.path.abspath(self.stress_cache_dir) if self.stress_cache_dir else None
        for folder in folders:
            absolute = os.path.abspath(folder)
            if cache_root and os.path.commonpath((cache_root, absolute)) == cache_root:
                shutil.rmtree(absolute, ignore_errors=True)


# --------------------------------------------------------------------------
# reward terms
# --------------------------------------------------------------------------
@dataclass
class RewardBreakdown:
    total: float = 0.0
    r_fmt: float = 0.0
    r_clip: float = 0.0
    r_exec_ok: float = 0.0
    r_tag_order: float = 0.0
    r_atomic: float = 0.0
    r_memory: float = 0.0
    r_brevity: float = 0.0
    r_spurious: float = 1.0
    w_fmt: float = 0.0
    length_limit: int = 0
    n_tokens: int = 0
    clipped_has_answer: bool = False
    final_sql: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _reasoning_of(text: str) -> str:
    """The prose of a trace with its SQL blocks removed.

    The memory bank holds *reasoning*, so comparing against a string dominated
    by SQL would score syntax rather than argument.
    """
    import re as _re

    return _re.sub(r"```sql.*?```", " ", text or "", flags=_re.S | _re.I).strip()


def clip_tokens(token_ids: Sequence[int], limit: int) -> Sequence[int]:
    return token_ids[: max(0, limit)]


class DebateReward:
    """Callable that scores one GRPO rollout.

    ``decode`` maps token ids back to text; passing the tokenizer's decoder
    keeps the length clipping *in token space*, exactly as IMAD eq. (2)
    specifies (clipping in characters would make the schedule model-dependent).
    """

    def __init__(
        self,
        cfg: Any,
        oracle: ExecutionOracle,
        schedule: RewardSchedule,
        decode: Optional[Callable[[Sequence[int]], str]] = None,
        required_tags: Sequence[str] = (),
        memory: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.oracle = oracle
        self.schedule = schedule
        self.decode = decode
        self.required_tags = tuple(required_tags or cfg.required_tags or REQUIRED_TAGS)
        self.memory = memory
        self.atomic = None
        if float(getattr(cfg, "w_atomic", 0.0)) > 0:
            from .dense_rewards import AtomicOpsReward

            self.atomic = AtomicOpsReward()

    def __call__(
        self,
        text: str,
        token_ids: Optional[Sequence[int]],
        db_path: str,
        gold_sql: str,
        step: int,
    ) -> RewardBreakdown:
        w_fmt = self.schedule.w_fmt(step)
        limit = self.schedule.length_limit(step)

        # --- R_fmt: are the structure tags there? (token matching, per 2.3) ---
        r_fmt = format_score(text, self.required_tags)
        r_order = tag_order_score(text, self.required_tags)

        # --- R(y; l): correct answer inside the first l tokens ----------------
        if token_ids is not None and self.decode is not None:
            clipped_text = self.decode(clip_tokens(token_ids, limit))
            n_tokens = len(token_ids)
        else:  # character fallback (~4 chars/token) when no tokenizer is around
            clipped_text = text[: limit * 4]
            n_tokens = max(1, len(text) // 4)

        clipped_sql = extract_final_sql(clipped_text)
        r_clip = 1.0 if self.oracle.is_correct(db_path, clipped_sql, gold_sql) else 0.0

        # --- small shaping terms, specific to SQL -----------------------------
        full_sql = extract_final_sql(text)
        r_exec = 1.0 if (clipped_sql and self.oracle.executes(db_path, clipped_sql)) else 0.0

        # --- brevity, awarded ONLY to CORRECT rollouts ------------------------
        # Mirror of the dense terms below. A saturated group (every rollout
        # correct) otherwise has identical rewards and therefore no gradient,
        # so GRPO can never learn "correct AND shorter" - the objective of
        # internalisation. Scoring 1 - n_tokens/ref among correct rollouts
        # makes a terse right answer beat a verbose right answer.
        r_brevity = 0.0
        if r_clip >= 1.0 and float(getattr(self.cfg, "w_brevity", 0.0)) > 0:
            ref = int(getattr(self.cfg, "brevity_ref_tokens", 0)) or max(1, limit * 2)
            r_brevity = max(0.0, 1.0 - float(n_tokens) / float(ref))

        # --- spurious correctness, checked ONLY on CORRECT rollouts -----------
        # A pass under r_clip only means the prediction agreed with gold on the
        # *one* database the benchmark ships - two queries that mean different
        # things can agree there by coincidence (see Example in the proof doc,
        # Part X, "Spurious correctness"). This re-executes gold vs. the
        # prediction on several synthetic stress instances
        # (latentsql/db/spurious.py, EX-Ext adapted for RL) and discounts the
        # credit by how many of them still agree. r_spurious stays at its
        # default of 1.0 (no penalty) whenever the feature is off or the
        # rollout was already wrong, so behaviour is unchanged unless
        # w_spurious is explicitly set.
        r_spurious = 1.0
        if r_clip >= 1.0 and float(getattr(self.cfg, "w_spurious", 0.0)) > 0 and clipped_sql:
            r_spurious = self.oracle.spurious_agreement(db_path, clipped_sql, gold_sql)

        # --- dense partial credit, awarded ONLY to wrong rollouts -------------
        # A correct rollout already earns full credit; letting it also earn
        # similarity credit would let a verbose-but-right answer outscore a
        # terse-but-right one, which is backwards for internalisation.
        r_atomic = r_memory = 0.0
        if r_clip < 1.0:
            if self.atomic is not None and clipped_sql:
                r_atomic = self.atomic.score(clipped_sql, gold_sql)
            if self.memory is not None and float(getattr(self.cfg, "w_memory", 0.0)) > 0:
                r_memory = self.memory.score(_reasoning_of(clipped_text))

        w_mem = float(getattr(self.cfg, "w_memory", 0.0))
        if w_mem and getattr(self.cfg, "anneal_memory_with_fmt", True):
            # scale on w_fmt's trajectory so the incentive to look like a trace
            # decays together with the incentive to write one
            span = max(1e-9, self.cfg.w_fmt_start - self.cfg.w_fmt_end)
            w_mem *= (w_fmt - self.cfg.w_fmt_end) / span

        total = (
            w_fmt * r_fmt
            + self.cfg.w_clip * r_clip
            + self.cfg.w_exec_ok * r_exec
            + self.cfg.w_tag_order * r_order
            + float(getattr(self.cfg, "w_atomic", 0.0)) * r_atomic
            + w_mem * r_memory
            + float(getattr(self.cfg, "w_brevity", 0.0)) * r_brevity
            + float(getattr(self.cfg, "w_spurious", 0.0)) * (r_spurious - 1.0)
        )
        return RewardBreakdown(
            total=total,
            r_atomic=r_atomic,
            r_memory=r_memory,
            r_brevity=r_brevity,
            r_spurious=r_spurious,
            r_fmt=r_fmt,
            r_clip=r_clip,
            r_exec_ok=r_exec,
            r_tag_order=r_order,
            w_fmt=w_fmt,
            length_limit=limit,
            n_tokens=n_tokens,
            clipped_has_answer=bool(clipped_sql),
            final_sql=full_sql,
        )


# --------------------------------------------------------------------------
# GRPO advantage
# --------------------------------------------------------------------------
def group_advantages(rewards: Sequence[float], eps: float = 1e-4,
                     std_normalize: bool = False) -> List[float]:
    """Group-relative advantage, Dr. GRPO style: r - mean, NOT divided by std.

    Dividing by the group std (original GRPO) blows tiny reward gaps up to
    full size: four correct rollouts that differ by one token of brevity
    (std ~4e-4) got advantages of about +1.7/-0.6 - as large as a right-vs-wrong
    group - so most steps trained "be one token shorter". Without the std a
    one-token gap stays ~1e-3 and a right/wrong group gets ~+/-0.5
    (Liu et al., 2025, "Understanding R1-Zero-Like Training").

    With a degenerate group (all rewards equal within eps) every advantage is
    0 - there is no preference signal to learn from.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    if std_normalize:
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
        if std < eps:
            return [0.0] * n
        return [(r - mean) / std for r in rewards]
    if max(rewards) - min(rewards) < eps:
        return [0.0] * n
    return [r - mean for r in rewards]
