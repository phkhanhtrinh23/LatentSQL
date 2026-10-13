"""The teacher trace T(x) = <r1> T1 <r2> T2 <r3> T3 <sql> Y_T (Eq. 9)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..prompts import SQL_TAG, role_tag, sql_target


@dataclass
class TeacherTrace:
    qid: str = ""
    dataset: str = ""
    db_id: str = ""
    db_path: str = ""
    question: str = ""
    evidence: str = ""
    history: List[str] = field(default_factory=list)
    gold_sql: str = ""
    #: retrieved database values added to x
    values: List[str] = field(default_factory=list)
    #: role -> textual reasoning T_j
    turns: Dict[str, str] = field(default_factory=dict)
    draft_sql: Optional[str] = None
    #: Y_T, the teacher's final query
    final_sql: Optional[str] = None
    #: EX(Y_T, Y*, D); a trace is kept only when this is true
    correct: bool = False
    teacher_tokens: int = 0
    teacher_calls: int = 0

    def role_turns(self, roles: Sequence[str]) -> Optional[List[str]]:
        """``[T_1, ..., T_J]`` for ``roles``, or None if one is missing."""
        out = []
        for r in roles:
            t = (self.turns.get(r) or "").strip()
            if not t:
                return None
            out.append(t)
        return out

    def render(self, roles: Sequence[str]) -> str:
        """The full written trace, as the warm-up stage sees it."""
        parts = [f"{role_tag(r)}\n{(self.turns.get(r) or '').strip()}" for r in roles]
        return "\n".join(parts) + f"\n{SQL_TAG}{sql_target(self.final_sql or '')}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TeacherTrace":
        keys = set(TeacherTrace.__dataclass_fields__)
        return TeacherTrace(**{k: v for k, v in d.items() if k in keys})


def save_traces(traces: Sequence[TeacherTrace], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def load_traces(path: str) -> List[TeacherTrace]:
    with open(path, "r", encoding="utf-8") as fh:
        return [TeacherTrace.from_dict(json.loads(line)) for line in fh if line.strip()]
