"""The three-role teacher (Sec. 4.1, Figure 2-1).

For every training question:

1. retrieve database values with the precomputed LSH index and add them to x;
2. r1, schema selector: tables, columns, joins and conditions  -> T1;
3. r2, candidate generator: a draft query, step by step         -> T2;
4. r3, reviser: execute the draft on D, read the result or the
   error, keep the draft or fix it                              -> T3, Y_T;
5. keep the trace iff EX(Y_T, Y*, D) = 1 (Eq. 10).

The teacher runs only to build training data. The student never executes a
query at inference; internalising r3 teaches it to anticipate the errors that
the teacher's execution feedback recorded.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable, List, Optional, Sequence

from ..data.inputs import retrieved_values, teacher_input
from ..db.execute import ExecResult, execute_sql, execution_accuracy
from ..prompts import (CANDIDATE_GENERATOR_SYSTEM, DRAFT_BLOCK, REVISER_SYSTEM,
                       SCHEMA_SELECTOR_SYSTEM, SELECTED_SCHEMA, extract_sql, normalize_sql)
from .trace import TeacherTrace


def execution_report(res: ExecResult, max_chars: int = 400) -> str:
    """What the reviser reads after executing the draft."""
    if not res.ok:
        return f"error - {res.error.strip()[:max_chars]}"
    if not res.rows:
        return "the query ran and returned no rows"
    return (f"the query ran and returned {len(res.rows)}{'+' if res.truncated else ''} row(s); "
            f"first rows:\n{res.preview(n=3, max_chars=max_chars)}")


class ThreeRoleTeacher:
    def __init__(self, llm: Any, cfg: Any):
        self.llm = llm
        self.cfg = cfg

    def solve(self, ex: Any) -> TeacherTrace:
        tcfg = self.cfg.teacher
        before = (self.llm.usage.total_tokens, self.llm.usage.calls)
        values = retrieved_values(ex, self.cfg)
        ctx = teacher_input(ex, self.cfg, values)

        # r1: schema selection
        t1 = self.llm.ask(SCHEMA_SELECTOR_SYSTEM, ctx + "\n\nSchema selection:",
                          tag="schema_selector").strip()
        selected = SELECTED_SCHEMA.format(t1=t1)

        # r2: candidate generation
        t2 = self.llm.ask(CANDIDATE_GENERATOR_SYSTEM, ctx + selected + "\n\nCandidate query:",
                          tag="candidate_generator").strip()
        draft = extract_sql(t2)

        # r3: revision with execution feedback
        current = draft or ""
        parts: List[str] = []
        for _ in range(max(1, tcfg.max_revisions)):
            report = execution_report(execute_sql(ex.db_path, current, timeout=tcfg.execution_timeout))
            out = self.llm.ask(
                REVISER_SYSTEM,
                ctx + selected + DRAFT_BLOCK.format(draft=current, report=report) + "\n\nReview:",
                tag="reviser",
            ).strip()
            parts.append(f"Execution: {report}\n{out}")
            revised = extract_sql(out) or current
            if normalize_sql(revised) == normalize_sql(current):
                break
            current = revised
        final = normalize_sql(current) or None

        correct = False
        if final:
            correct, _, _ = execution_accuracy(ex.db_path, final, ex.gold_sql,
                                               timeout=tcfg.execution_timeout)
        return TeacherTrace(
            qid=ex.qid, dataset=ex.dataset, db_id=ex.db_id, db_path=ex.db_path,
            question=ex.question, evidence=ex.evidence, history=list(ex.history),
            gold_sql=ex.gold_sql, values=values,
            turns={"schema_selector": t1, "candidate_generator": t2, "reviser": "\n".join(parts)},
            draft_sql=draft, final_sql=final, correct=bool(correct),
            teacher_tokens=self.llm.usage.total_tokens - before[0],
            teacher_calls=self.llm.usage.calls - before[1],
        )


def run_teacher(examples: Sequence[Any], cfg: Any, llm: Any, out_path: str,
                progress_every: int = 20) -> List[TeacherTrace]:
    """Run the teacher on ``examples``, appending to ``out_path`` (resumable)."""
    from .trace import load_traces

    done = {t.qid: t for t in load_traces(out_path)} if os.path.exists(out_path) else {}
    teacher = ThreeRoleTeacher(llm, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    t0 = time.time()
    traces: List[TeacherTrace] = []
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, ex in enumerate(examples, 1):
            if ex.qid in done:
                traces.append(done[ex.qid])
                continue
            try:
                tr = teacher.solve(ex)
            except Exception as exc:  # an API failure must not lose earlier traces
                print(f"[teacher] {ex.qid} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            traces.append(tr)
            fh.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            if i % progress_every == 0 or i == len(examples):
                kept = sum(t.correct for t in traces)
                print(f"[teacher] {i}/{len(examples)} kept={kept} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return traces
