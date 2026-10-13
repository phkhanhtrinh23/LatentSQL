"""Execution accuracy, efficiency counters, and mean +- 95% CI over runs."""
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..db.execute import execution_accuracy


@dataclass
class Prediction:
    qid: str
    dataset: str
    db_id: str
    db_path: str
    question: str
    gold_sql: str
    pred_sql: Optional[str]
    difficulty: str = ""
    correct: Optional[bool] = None
    prompt_tokens: int = 0
    #: generated tokens (the SQL only, for LatentSQL)
    completion_tokens: int = 0
    latent_steps: int = 0
    #: KV-cache positions spent on reasoning (role tags, latent steps, <sql>)
    reasoning_positions: int = 0
    latency_s: float = 0.0
    raw_output: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score(preds: Sequence[Prediction], timeout: float = 30.0) -> None:
    """Fill ``correct`` = EX(Y, Y*, D) in place."""
    for p in preds:
        if p.correct is None:
            p.correct, _, _ = execution_accuracy(p.db_path, p.pred_sql, p.gold_sql, timeout=timeout)


def _mean(xs: Sequence[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else 0.0


def report(preds: Sequence[Prediction]) -> Dict[str, Any]:
    """EX and efficiency per benchmark, with a per-difficulty breakdown."""
    out: Dict[str, Any] = {}
    by_ds: Dict[str, List[Prediction]] = {}
    for p in preds:
        by_ds.setdefault(p.dataset, []).append(p)
    for ds, ps in sorted(by_ds.items()):
        diff: Dict[str, Dict[str, float]] = {}
        for p in ps:
            d = diff.setdefault(p.difficulty or "all", {"n": 0, "correct": 0})
            d["n"] += 1
            d["correct"] += int(bool(p.correct))
        for d in diff.values():
            d["ex"] = round(100.0 * d["correct"] / d["n"], 2)
        out[ds] = {
            "n": len(ps),
            "ex": round(100.0 * sum(bool(p.correct) for p in ps) / len(ps), 2),
            "by_difficulty": diff,
            "avg_prompt_tokens": _mean([p.prompt_tokens for p in ps]),
            "avg_generated_tokens": _mean([p.completion_tokens for p in ps]),
            "avg_latent_steps": _mean([p.latent_steps for p in ps]),
            "avg_reasoning_positions": _mean([p.reasoning_positions for p in ps]),
            "avg_latency_s": _mean([p.latency_s for p in ps]),
            "total_latency_s": round(sum(p.latency_s for p in ps), 2),
            "n_unparseable": sum(1 for p in ps if not p.pred_sql),
        }
    return out


def save_predictions(preds: Sequence[Prediction], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_predictions(path: str) -> List[Prediction]:
    with open(path, "r", encoding="utf-8") as fh:
        return [Prediction(**json.loads(line)) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# mean and 95% confidence interval across runs (Table 1, Table 2)
# --------------------------------------------------------------------------
#: two-sided 97.5% quantiles of Student's t, by degrees of freedom
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t_quantile(df: int) -> float:
    if df in _T975:
        return _T975[df]
    if df > 30:
        return 1.96 + 2.4 / df          # close to the normal quantile
    return _T975[max(k for k in _T975 if k <= df)]


def mean_ci95(values: Sequence[float]) -> Dict[str, float]:
    """Mean and half-width of the 95% t-interval (0 for a single run)."""
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n_runs": 0}
    m = statistics.fmean(values)
    if n == 1:
        return {"mean": round(m, 2), "ci95": 0.0, "n_runs": 1}
    sd = statistics.stdev(values)
    return {"mean": round(m, 2), "ci95": round(t_quantile(n - 1) * sd / math.sqrt(n), 2),
            "n_runs": n}
