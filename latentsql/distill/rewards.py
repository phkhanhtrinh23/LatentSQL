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

    def __init__(self, timeout: float = 30.0, max_cache: int = 200_000):
        self.timeout = timeout
        self.max_cache = max_cache
        self._gold: Dict[Tuple[str, str], ExecResult] = {}
        self._verdict: Dict[Tuple[str, str, str], bool] = {}
        self.n_exec = 0

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
        )
        return RewardBreakdown(
            total=total,
            r_atomic=r_atomic,
            r_memory=r_memory,
            r_brevity=r_brevity,
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
def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> List[float]:
    """GRPO's group-relative advantage: (r - mean) / std within the group.

    With a degenerate group (all rewards equal) every advantage is 0, which is
    correct - there is no preference signal to learn from.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < eps:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]
