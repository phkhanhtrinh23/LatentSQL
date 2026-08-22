"""The structured debate transcript that the student is trained to reproduce.

IMAD section 2.1 is explicit that the *structure tags* are what make
internalisation work: without them "the distinction between agents becomes
difficult to make", and the agent subspaces in section 3 come out less
separable. We keep that idea and specialise the tag vocabulary to Text-to-SQL,
so the trace also records the schema-linking and execution-feedback steps that
DIN-SQL / CHESS / CHASE-SQL contribute.

Canonical shape::

    <|schema_link|> tables + columns + retrieved values <|/schema_link|>
    <|classify|> EASY | NON-NESTED | NESTED <|/classify|>
    <|round 1|>
    <|agent divide_conquer|> reasoning ... ```sql ...``` <|/agent|>
    <|agent query_plan|>     reasoning ... ```sql ...``` <|/agent|>
    <|agent dail_icl|>       reasoning ... ```sql ...``` <|/agent|>
    <|/round|>
    <|round 2|>
    ... each agent revises after seeing the others + execution feedback ...
    <|/round|>
    <|consensus|> ```sql <final> ``` <|/consensus|>
    <|endofdebate|>

Everything here is pure string handling so rewards can be computed without a
tokenizer or a model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# tag vocabulary
# --------------------------------------------------------------------------
T_SCHEMA_OPEN, T_SCHEMA_CLOSE = "<|schema_link|>", "<|/schema_link|>"
T_CLASS_OPEN, T_CLASS_CLOSE = "<|classify|>", "<|/classify|>"
T_ROUND_CLOSE = "<|/round|>"
T_AGENT_CLOSE = "<|/agent|>"
T_EXEC_OPEN, T_EXEC_CLOSE = "<|exec|>", "<|/exec|>"
T_CONSENSUS_OPEN, T_CONSENSUS_CLOSE = "<|consensus|>", "<|/consensus|>"
T_END = "<|endofdebate|>"

AGENT_IDS = ("divide_conquer", "query_plan", "dail_icl")


def round_tag(i: int) -> str:
    return f"<|round {i}|>"


def agent_tag(name: str) -> str:
    return f"<|agent {name}|>"


def special_tokens(agents: Sequence[str] = AGENT_IDS, max_rounds: int = 4) -> List[str]:
    """Every literal tag, for optional ``add_special_tokens`` on the student."""
    toks = [
        T_SCHEMA_OPEN, T_SCHEMA_CLOSE, T_CLASS_OPEN, T_CLASS_CLOSE,
        T_ROUND_CLOSE, T_AGENT_CLOSE, T_EXEC_OPEN, T_EXEC_CLOSE,
        T_CONSENSUS_OPEN, T_CONSENSUS_CLOSE, T_END,
    ]
    toks += [round_tag(i) for i in range(1, max_rounds + 1)]
    toks += [agent_tag(a) for a in agents]
    return toks


#: tags every well-formed trace must contain, in this order.
#: Used by ``format_reward`` (IMAD's R_fmt) and ``tag_order_reward``.
REQUIRED_TAGS: Tuple[str, ...] = (
    T_SCHEMA_OPEN,
    round_tag(1),
    T_CONSENSUS_OPEN,
    T_END,
)


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
@dataclass
class AgentTurn:
    agent: str
    round: int
    reasoning: str
    sql: Optional[str]
    exec_ok: bool = False
    exec_error: str = ""
    exec_preview: str = ""
    correct: Optional[bool] = None   # vs gold; only known at trace-build time

    def render(self, with_exec: bool = True) -> str:
        body = self.reasoning.strip()
        if self.sql:
            body += f"\n```sql\n{self.sql}\n```"
        if with_exec and (self.exec_error or self.exec_preview):
            note = self.exec_error or self.exec_preview
            body += f"\n{T_EXEC_OPEN} {note.strip()[:200]} {T_EXEC_CLOSE}"
        return f"{agent_tag(self.agent)}\n{body}\n{T_AGENT_CLOSE}"


@dataclass
class DebateTrace:
    """One teacher transcript for one question."""

    qid: str = ""
    db_id: str = ""
    question: str = ""
    evidence: str = ""
    gold_sql: str = ""
    schema_prompt: str = ""          # the rendered schema shown to the agents
    schema_links: str = ""
    value_hints: List[str] = field(default_factory=list)
    difficulty_class: str = ""
    turns: List[AgentTurn] = field(default_factory=list)
    consensus_sql: Optional[str] = None
    consensus_reached: bool = False
    correct: Optional[bool] = None
    n_rounds: int = 0
    teacher_tokens: int = 0
    teacher_calls: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- rendering --------------------------------------------------------
    def render(self, with_exec: bool = True) -> str:
        """Serialise to the tagged transcript the student learns to emit."""
        parts: List[str] = []
        links = self.schema_links.strip()
        if self.value_hints:
            links += "\nValues: " + "; ".join(self.value_hints)
        parts.append(f"{T_SCHEMA_OPEN}\n{links}\n{T_SCHEMA_CLOSE}")
        if self.difficulty_class:
            parts.append(f"{T_CLASS_OPEN} {self.difficulty_class} {T_CLASS_CLOSE}")

        by_round: Dict[int, List[AgentTurn]] = {}
        for t in self.turns:
            by_round.setdefault(t.round, []).append(t)
        for r in sorted(by_round):
            inner = "\n".join(t.render(with_exec=with_exec) for t in by_round[r])
            parts.append(f"{round_tag(r + 1)}\n{inner}\n{T_ROUND_CLOSE}")

        final = self.consensus_sql or ""
        parts.append(f"{T_CONSENSUS_OPEN}\n```sql\n{final}\n```\n{T_CONSENSUS_CLOSE}")
        parts.append(T_END)
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DebateTrace":
        turns = [AgentTurn(**t) for t in d.get("turns", [])]
        d = {k: v for k, v in d.items() if k != "turns"}
        return DebateTrace(turns=turns, **d)


# --------------------------------------------------------------------------
# parsing (used by rewards and by evaluation of student output)
# --------------------------------------------------------------------------
_SQL_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_CONSENSUS = re.compile(
    re.escape(T_CONSENSUS_OPEN) + r"(.*?)(?:" + re.escape(T_CONSENSUS_CLOSE) + r"|$)",
    re.DOTALL,
)
_AGENT_BLOCK = re.compile(
    r"<\|agent\s+([a-z_0-9]+)\|>(.*?)(?:" + re.escape(T_AGENT_CLOSE) + r"|(?=<\|)|$)",
    re.DOTALL | re.IGNORECASE,
)
_ROUND_BLOCK = re.compile(r"<\|round\s+(\d+)\|>(.*?)(?:" + re.escape(T_ROUND_CLOSE) + r"|(?=<\|round)|$)",
                          re.DOTALL | re.IGNORECASE)
_SQL_START = re.compile(r"(?is)\b(WITH|SELECT)\b")


def normalize_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    s = _SQL_FENCE.sub(lambda m: m.group(1), sql)
    s = re.sub(r"(?m)^\s*--.*$", " ", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(";").strip()
    return s


def extract_final_sql(text: str) -> Optional[str]:
    """Get the answer out of a student generation.

    Robust to partially-internalised output: after RL the model may drop the
    debate tags entirely and emit only a fenced query (that is the *point* of
    internalisation), so we fall back progressively.
    """
    if not text:
        return None
    m = _CONSENSUS.search(text)
    if m:
        inner = m.group(1)
        fences = _SQL_FENCE.findall(inner)
        if fences:
            return normalize_sql(fences[-1])
        s = _SQL_START.search(inner)
        if s:
            return normalize_sql(inner[s.start():])
    fences = _SQL_FENCE.findall(text)
    if fences:
        return normalize_sql(fences[-1])
    s = _SQL_START.search(text)
    if s:
        return normalize_sql(text[s.start():])
    return None


def parse_trace(text: str) -> Dict[str, Any]:
    """Structural summary of a generated trace (for rewards and diagnostics)."""
    rounds = _ROUND_BLOCK.findall(text or "")
    agents: List[Tuple[str, Optional[str]]] = []
    for name, body in _AGENT_BLOCK.findall(text or ""):
        fences = _SQL_FENCE.findall(body)
        agents.append((name.lower(), normalize_sql(fences[-1]) if fences else None))
    return {
        "n_rounds": len(rounds),
        "agents": agents,
        "agent_names": sorted({a for a, _ in agents}),
        "final_sql": extract_final_sql(text or ""),
        "has_end": T_END in (text or ""),
        "has_consensus": T_CONSENSUS_OPEN in (text or ""),
        "has_schema_link": T_SCHEMA_OPEN in (text or ""),
    }


# --------------------------------------------------------------------------
# reward-facing helpers (kept here so tags live in exactly one place)
# --------------------------------------------------------------------------
def format_score(text: str, required: Sequence[str] = REQUIRED_TAGS) -> float:
    """IMAD's R_fmt: fraction of the required structure tags that are present."""
    if not required:
        return 0.0
    hit = sum(1 for t in required if t in (text or ""))
    return hit / len(required)


def tag_order_score(text: str, required: Sequence[str] = REQUIRED_TAGS) -> float:
    """Fraction of consecutive required-tag pairs appearing in canonical order."""
    text = text or ""
    positions = [text.find(t) for t in required]
    pairs = [
        (positions[i], positions[i + 1])
        for i in range(len(positions) - 1)
        if positions[i] >= 0 and positions[i + 1] >= 0
    ]
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a < b) / len(pairs)


# --------------------------------------------------------------------------
# jsonl io
# --------------------------------------------------------------------------
def save_traces(traces: Sequence[DebateTrace], path: str) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def load_traces(path: str) -> List[DebateTrace]:
    out: List[DebateTrace] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(DebateTrace.from_dict(json.loads(line)))
    return out
