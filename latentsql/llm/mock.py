"""Deterministic offline stand-in for the teacher LLM (smoke test only).

It looks up the gold query of the question it is asked about and answers each
teacher role from it: the schema selector lists the gold tables and columns,
the candidate generator writes the gold query (or, for one question in three,
a draft with an unknown column, so the reviser has an error to fix), and the
reviser keeps or repairs the draft. It is a test double, not a model; numbers
produced with it measure plumbing only.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional, Sequence

from .base import LLMBackend, Usage


def _broken(sql: str) -> str:
    """The draft's mistake: reference a column that does not exist."""
    return re.sub(r"(?i)^SELECT\s+", "SELECT missing_column, ", sql.strip(), count=1)


class MockBackend(LLMBackend):
    def __init__(self, cfg: Any = None):
        super().__init__()
        self.cfg = cfg
        self.oracle: Dict[str, Any] = {}

    def set_oracle(self, examples: Sequence[Any]) -> None:
        self.oracle = {e.question.strip(): e for e in examples}

    def _example(self, text: str) -> Optional[Any]:
        m = re.search(r"Question: (.*?)(?:\n\n|$)", text, re.S)
        return self.oracle.get(m.group(1).strip()) if m else None

    def chat(self, messages: Sequence[Dict[str, str]], **kw: Any) -> str:
        user = messages[-1]["content"]
        ex = self._example(user)
        tag = kw.get("tag", "")
        gold = ex.gold_sql if ex else "SELECT 1"
        if tag == "schema_selector":
            out = self._schema(ex)
        elif tag == "candidate_generator":
            h = int(hashlib.md5(gold.encode()).hexdigest(), 16)
            draft = _broken(gold) if h % 3 == 0 else gold
            out = f"Find the requested rows, then apply the conditions.\n```sql\n{draft}\n```"
        elif tag == "reviser":
            if "error" in user.split("Execution:")[-1][:20]:
                out = ("The draft references a column that does not exist. "
                       f"Drop it.\n```sql\n{gold}\n```")
            else:
                draft = re.search(r"Draft query:\n```sql\n(.*?)\n```", user, re.S)
                out = f"The draft answers the question; keep it.\n```sql\n{draft.group(1) if draft else gold}\n```"
        else:
            out = f"```sql\n{gold}\n```"
        self.usage.add(Usage(prompt_tokens=self.approx_tokens(user),
                             completion_tokens=self.approx_tokens(out), calls=1))
        return out

    @staticmethod
    def _schema(ex: Optional[Any]) -> str:
        if ex is None:
            return "Tables: none\nColumns: none\nJoins: none\nConditions: none"
        from ..latent.schema_align import gold_schema_items

        items = gold_schema_items(ex.gold_sql, ex.db_path, ex.db_id)
        tables = [i for i in items if "." not in i]
        cols = [i for i in items if "." in i]
        return (f"Tables: {', '.join(tables) or 'none'}\nColumns: {', '.join(cols) or 'none'}\n"
                f"Joins: none\nConditions: none")
