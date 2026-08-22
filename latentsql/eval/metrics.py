"""Scoring: execution accuracy, VES, and the token-efficiency numbers.

IMAD's headline table reports accuracy *and* average token consumption
(input + output) per question, because the claim is "match the multi-agent
system at a fraction of the cost". We report the same pair, plus BIRD's VES and
a per-difficulty breakdown.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from ..db.execute import execution_accuracy, valid_efficiency_score


@dataclass
class Prediction:
    qid: str
    db_id: str
    db_path: str
    question: str
    gold_sql: str
    pred_sql: Optional[str]
    correct: Optional[bool] = None
    ves: Optional[float] = None
    difficulty: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw_output: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    method: str
    dataset: str
    n: int
    ex: float                      # execution accuracy
    ves: Optional[float] = None
    avg_total_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_latency_s: float = 0.0
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_unparseable: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_row(self) -> str:
        ves = f"{self.ves:.3f}" if self.ves is not None else "  -  "
        return (
            f"{self.method:<18} {self.dataset:<10} n={self.n:<5} "
            f"EX={self.ex * 100:6.2f}%  VES={ves}  "
            f"tok={self.avg_total_tokens:8.1f}  out={self.avg_completion_tokens:7.1f}  "
            f"{self.avg_latency_s:6.2f}s"
        )


def score_predictions(
    preds: Sequence[Prediction],
    timeout: float = 30.0,
    compute_ves: bool = False,
) -> None:
    """Fill in ``correct`` (and optionally ``ves``) in place."""
    for p in preds:
        if p.correct is None:
            ok, _pr, _gr = execution_accuracy(p.db_path, p.pred_sql, p.gold_sql, timeout=timeout)
            p.correct = ok
        if compute_ves and p.ves is None:
            p.ves = valid_efficiency_score(p.db_path, p.pred_sql, p.gold_sql, timeout=timeout)


def build_report(
    preds: Sequence[Prediction],
    method: str,
    dataset: str,
    compute_ves: bool = False,
    notes: str = "",
) -> EvalReport:
    n = len(preds)
    if n == 0:
        return EvalReport(method=method, dataset=dataset, n=0, ex=0.0, notes=notes)

    ex = sum(1 for p in preds if p.correct) / n
    ves = (sum(p.ves or 0.0 for p in preds) / n) if compute_ves else None

    by_diff: Dict[str, Dict[str, float]] = {}
    for p in preds:
        key = p.difficulty or "all"
        bucket = by_diff.setdefault(key, {"n": 0.0, "correct": 0.0})
        bucket["n"] += 1
        bucket["correct"] += 1.0 if p.correct else 0.0
    for key, bucket in by_diff.items():
        bucket["ex"] = round(bucket["correct"] / bucket["n"], 4) if bucket["n"] else 0.0

    return EvalReport(
        method=method,
        dataset=dataset,
        n=n,
        ex=ex,
        ves=ves,
        avg_total_tokens=round(statistics.fmean(p.total_tokens for p in preds), 2),
        avg_completion_tokens=round(statistics.fmean(p.completion_tokens for p in preds), 2),
        avg_latency_s=round(statistics.fmean(p.latency_s for p in preds), 3),
        by_difficulty=by_diff,
        n_unparseable=sum(1 for p in preds if not p.pred_sql),
        notes=notes,
    )


def save_predictions(preds: Sequence[Prediction], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def save_reports(reports: Sequence[EvalReport], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in reports], fh, indent=2)


def render_table(reports: Sequence[EvalReport]) -> str:
    """A markdown table in IMAD's shape: accuracy next to token consumption."""
    head = (
        "| Method | Dataset | n | EX (%) | VES | Avg tokens | Avg output tokens | "
        "Tokens vs MAS |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    base = next(
        (r.avg_total_tokens for r in reports if r.method in ("fusion_mas", "mas")), None
    )
    rows = []
    for r in reports:
        rel = (
            f"{100.0 * r.avg_total_tokens / base:.1f}%"
            if base and base > 0 else "-"
        )
        ves = f"{r.ves:.3f}" if r.ves is not None else "-"
        rows.append(
            f"| {r.method} | {r.dataset} | {r.n} | {r.ex * 100:.2f} | {ves} | "
            f"{r.avg_total_tokens:.1f} | {r.avg_completion_tokens:.1f} | {rel} |"
        )
    return head + "\n".join(rows)
