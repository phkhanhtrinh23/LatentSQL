"""Turn teacher transcripts into the SFT corpus (IMAD section 2.1).

The filtering follows the paper: debates where the agents never reached a
consensus are discarded. For Text-to-SQL we can filter harder than IMAD could
on arithmetic, because the gold query gives us a free execution-based label -
so by default we also drop traces whose consensus query is wrong.

Each surviving record is ``{prompt, completion}`` where the completion is the
*entire* tagged transcript, not just the final SQL. That distinction is the
whole difference between IMAD and DebateGPT (IMAD 2.2, Table 1): training on
the full trace is what transfers the intermediate agent interactions.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..mas.prompts import student_prompt
from ..mas.trace import DebateTrace


@dataclass
class SFTRecord:
    qid: str
    db_id: str
    db_path: str
    question: str
    evidence: str
    gold_sql: str
    system: str
    prompt: str
    completion: str
    n_rounds: int = 0
    correct: Optional[bool] = None
    teacher_tokens: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    no_consensus: int = 0
    incorrect: int = 0
    no_sql: int = 0
    exec_failed: int = 0
    too_long: int = 0
    duplicate: int = 0

    def as_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()

    def summary(self) -> str:
        return (
            f"{self.kept}/{self.total} traces kept "
            f"(dropped: no_consensus={self.no_consensus}, incorrect={self.incorrect}, "
            f"no_sql={self.no_sql}, exec_failed={self.exec_failed}, "
            f"too_long={self.too_long}, duplicate={self.duplicate})"
        )


def build_sft_records(
    traces: Sequence[DebateTrace],
    cfg: Any,
    db_paths: Optional[Dict[str, str]] = None,
) -> tuple[List[SFTRecord], FilterStats]:
    """Filter traces and render them as prompt/completion pairs."""
    stats = FilterStats(total=len(traces))
    seen: set[str] = set()
    out: List[SFTRecord] = []

    for tr in traces:
        if not tr.consensus_sql:
            stats.no_sql += 1
            continue
        if cfg.require_consensus and not tr.consensus_reached:
            stats.no_consensus += 1
            continue
        if cfg.require_correct and tr.correct is False:
            stats.incorrect += 1
            continue
        if cfg.require_execution and not any(
            t.exec_ok for t in tr.turns if t.sql == tr.consensus_sql
        ):
            # the consensus query never executed cleanly in any turn
            stats.exec_failed += 1
            continue

        completion = tr.render()
        if len(completion) > cfg.max_trace_chars:
            stats.too_long += 1
            continue

        if cfg.dedup:
            key = hashlib.md5(
                (tr.db_id + "|" + tr.question.strip().lower()).encode("utf-8")
            ).hexdigest()
            if key in seen:
                stats.duplicate += 1
                continue
            seen.add(key)

        sp = student_prompt(
            schema=tr.schema_prompt,
            question=tr.question,
            evidence=tr.evidence,
            value_hints="\n".join(f"- {v}" for v in tr.value_hints),
        )
        out.append(
            SFTRecord(
                qid=tr.qid,
                db_id=tr.db_id,
                db_path=(db_paths or {}).get(tr.qid, tr.meta.get("db_path", "")),
                question=tr.question,
                evidence=tr.evidence,
                gold_sql=tr.gold_sql,
                system=sp["system"],
                prompt=sp["user"],
                completion=completion,
                n_rounds=tr.n_rounds,
                correct=tr.correct,
                teacher_tokens=tr.teacher_tokens,
                meta={"selector": tr.meta.get("selector", "")},
            )
        )
        stats.kept += 1
    return out, stats


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def save_records(records: Sequence[SFTRecord], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_records(path: str) -> List[SFTRecord]:
    out: List[SFTRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(SFTRecord(**json.loads(line)))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_teacher(
    examples: Sequence[Any],
    cfg: Any,
    llm: Any,
    example_pool: Sequence[Any] = (),
    cache_dir: Optional[str] = None,
    on_trace: Optional[Any] = None,
    progress: bool = True,
    log_path: Optional[str] = None,
) -> List[DebateTrace]:
    """Run the fusion teacher over ``examples`` and collect the transcripts."""
    import time

    from ..mas.orchestrator import FusionSQL
    from ..utils.telemetry import RunLog, gpu_stats, reset_gpu_peak, run_environment

    samplelog = None
    if log_path:
        samplelog = RunLog(
            log_path,
            header=run_environment(
                stage="teacher", model=cfg.llm.model, backend=cfg.llm.backend,
                quantization=getattr(cfg.llm, "quantization", "none"),
                panel=list(cfg.mas.agents), rounds=cfg.mas.rounds,
                n_examples=len(examples),
            ),
        )
    reset_gpu_peak()

    mas = FusionSQL(llm, cfg.mas, example_pool=example_pool, cache_dir=cache_dir)
    traces: List[DebateTrace] = []
    for i, ex in enumerate(examples):
        before = (llm.usage.prompt_tokens, llm.usage.completion_tokens, llm.usage.calls)
        t0 = time.time()
        try:
            tr = mas.solve(ex)
        except Exception as exc:  # one bad DB must not kill a long teacher run
            tr = DebateTrace(
                qid=getattr(ex, "qid", str(i)),
                db_id=getattr(ex, "db_id", ""),
                question=getattr(ex, "question", ""),
                gold_sql=getattr(ex, "gold_sql", ""),
                meta={"error": f"{type(exc).__name__}: {exc}"},
            )
        tr.meta["db_path"] = getattr(ex, "db_path", "")
        traces.append(tr)
        if samplelog is not None:
            samplelog.write({
                "record": "sample",
                "i": i,
                "qid": getattr(ex, "qid", str(i)),
                "dataset": getattr(ex, "dataset", ""),
                "db_id": getattr(ex, "db_id", ""),
                "difficulty": getattr(ex, "difficulty", ""),
                "latency_s": round(time.time() - t0, 3),
                "prompt_tokens": llm.usage.prompt_tokens - before[0],
                "completion_tokens": llm.usage.completion_tokens - before[1],
                "llm_calls": llm.usage.calls - before[2],
                "n_rounds": tr.n_rounds,
                "consensus": tr.consensus_reached,
                "correct": tr.correct,
                "has_sql": bool(tr.consensus_sql),
                "trace_chars": len(tr.render()) if tr.consensus_sql else 0,
                "error": tr.meta.get("error", ""),
                **gpu_stats(),
            })
        if on_trace is not None:
            on_trace(tr)
        if progress and (i + 1) % 10 == 0:
            n_ok = sum(1 for t in traces if t.correct)
            print(
                f"  [teacher] {i + 1}/{len(examples)}  correct={n_ok}  "
                f"{round(time.time() - t0, 1)}s/q", flush=True,
            )
    if samplelog is not None:
        samplelog.close()
    return traces
