"""Deterministic offline stand-in for the teacher LLM.

WHAT THIS IS: a *test double*. It reads the gold SQL from an oracle table and
returns per-agent perturbations of it, so the full teacher pipeline - keyword
extraction, schema linking, classification, three debate agents over two
rounds, the execution-guided fixer, and the selector - can be exercised with no
API key, no GPU and no network.

WHAT THIS IS NOT: a language model. Accuracy numbers produced in ``mock`` mode
measure orchestration plumbing only and must never be reported as results. Use
``backend: openai`` (or ``hf``) for real teacher runs.

The perturbations are chosen so the debate is non-trivial: each agent has a
distinct, deterministic failure mode, so agents disagree, the fixer has broken
SQL to repair, and the selector has to choose.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .base import LLMBackend, Usage


def _seed(*parts: str) -> int:
    h = hashlib.blake2b("||".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


# --------------------------------------------------------------------------
# SQL perturbations = the "mistakes" different agents make
# --------------------------------------------------------------------------
def _drop_where(sql: str) -> Optional[str]:
    m = re.search(r"(?is)\bWHERE\b.*?(?=\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)", sql)
    return (sql[: m.start()] + sql[m.end():]).strip() if m else None


def _swap_aggregate(sql: str) -> Optional[str]:
    swaps = {"avg(": "sum(", "sum(": "avg(", "max(": "min(", "min(": "max("}
    low = sql.lower()
    for src, dst in swaps.items():
        i = low.find(src)
        if i >= 0:
            return sql[:i] + dst + sql[i + len(src):]
    return None


def _flip_order(sql: str) -> Optional[str]:
    if re.search(r"(?i)\bDESC\b", sql):
        return re.sub(r"(?i)\bDESC\b", "ASC", sql, count=1)
    if re.search(r"(?i)\bORDER\s+BY\b", sql):
        return re.sub(r"(?i)(\bORDER\s+BY\b\s+[\w\.\(\)]+)", r"\1 DESC", sql, count=1)
    return None


def _break_syntax(sql: str) -> str:
    """Produce something the fixer has to repair (unknown column)."""
    return re.sub(r"(?i)\bFROM\b", "FROM ", sql, count=1).replace("SELECT", "SELECT DISTINCTT", 1)


def _drop_limit(sql: str) -> Optional[str]:
    m = re.search(r"(?is)\bLIMIT\b\s+\d+\s*$", sql)
    return sql[: m.start()].strip() if m else None


_PERTURBATIONS = (_drop_where, _swap_aggregate, _flip_order, _drop_limit)


# --------------------------------------------------------------------------
# backend
# --------------------------------------------------------------------------
class MockBackend(LLMBackend):
    """Oracle-driven stub. Call :meth:`set_oracle` before running the MAS."""

    #: per-agent probability (out of 100) of returning a *wrong* query in round 1
    ERROR_RATE = {
        "divide_conquer": 25,
        "query_plan": 35,
        "dail_icl": 45,
        "single": 40,
        "din_sql": 35,
        "dail_sql": 40,
        "chess": 30,
        "chase_sql": 25,
    }
    #: how often a wrong query is *syntactically* broken rather than merely wrong
    SYNTAX_ERROR_RATE = 30

    def __init__(self, cfg: Any = None):
        super().__init__()
        self.cfg = cfg
        self._oracle: Dict[str, str] = {}
        self._evidence: Dict[str, str] = {}

    # -- oracle -----------------------------------------------------------
    def set_oracle(self, examples: Iterable[Any]) -> "MockBackend":
        for ex in examples:
            q = getattr(ex, "question", None) or ex["question"]
            gold = getattr(ex, "gold_sql", None) or ex.get("gold_sql", "")
            self._oracle[self._key(q)] = gold
        return self

    @staticmethod
    def _key(question: str) -> str:
        return re.sub(r"\s+", " ", (question or "").strip().lower())

    def _gold_for(self, user: str) -> Optional[str]:
        """Find the gold SQL for the *target* question in the prompt.

        A prompt can mention several known questions: the DAIL-SQL agent embeds
        few-shot examples, and the debate rounds quote earlier turns. The
        target is always the last one, because every template ends with the
        question it is asking about - so we match on latest occurrence, with
        the longest match winning ties.
        """
        low = self._key(user)
        best, best_pos, best_len = None, -1, 0
        for key, sql in self._oracle.items():
            if not key:
                continue
            pos = low.rfind(key)
            if pos < 0:
                continue
            if pos > best_pos or (pos == best_pos and len(key) > best_len):
                best, best_pos, best_len = sql, pos, len(key)
        return best

    # -- chat -------------------------------------------------------------
    def chat(self, messages: Sequence[Dict[str, str]], **kw: Any) -> str:
        tag = str(kw.get("tag", ""))
        user = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
        system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
        out = self._respond(tag, system, user, kw)
        self.usage.add(
            Usage(
                prompt_tokens=self.approx_tokens(system + user),
                completion_tokens=self.approx_tokens(out),
                calls=1,
            )
        )
        if self.record_log:
            self.log.append({"tag": tag, "user": user[:400], "output": out[:400]})
        return out

    # -- routing ----------------------------------------------------------
    def _respond(self, tag: str, system: str, user: str, kw: Dict[str, Any]) -> str:
        gold = self._gold_for(user)
        sample = int(kw.get("sample_index", 0))
        rnd = int(kw.get("round_index", 0))

        if tag.startswith("keywords"):
            return self._keywords(user, gold)
        if tag.startswith("schema_link"):
            return self._schema_links(gold)
        if tag.startswith("classify"):
            return self._classify(gold)
        if tag.startswith("decompose"):
            return self._decompose(user, gold)
        if tag.startswith("unit_test"):
            return self._unit_tests(user, gold)
        if tag.startswith("select_pairwise"):
            return self._pairwise(user, gold)
        if tag.startswith("fix"):
            return self._fix(user, gold)
        if tag.startswith("critique"):
            return self._critique(user, gold)
        return self._generate(tag, user, gold, sample, rnd)

    # -- individual behaviours -------------------------------------------
    def _keywords(self, user: str, gold: Optional[str]) -> str:
        lits = re.findall(r"'([^']+)'", gold or "")
        words = [w for w in re.findall(r"\b[A-Z][a-z]+\b", user) if len(w) > 2]
        return ", ".join(dict.fromkeys(lits + words[:5])) or "none"

    def _schema_links(self, gold: Optional[str]) -> str:
        if not gold:
            return "Tables: unknown\nColumns: unknown"
        tables = re.findall(r"(?i)\b(?:FROM|JOIN)\s+([A-Za-z_][\w]*)", gold)
        cols = re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)", gold)
        bare = re.findall(r"(?i)SELECT\s+(?:DISTINCT\s+)?([\w\*,\s\(\)]+?)\s+FROM", gold)
        flat = []
        for b in bare:
            flat += [c.strip() for c in b.split(",") if c.strip() and c.strip() != "*"]
        colnames = [c for _t, c in cols] + [re.sub(r".*\(|\).*", "", f) for f in flat]
        return (
            "Tables: " + ", ".join(dict.fromkeys(tables)) + "\n"
            "Columns: " + ", ".join(dict.fromkeys(c for c in colnames if c and c.isidentifier()))
        )

    def _classify(self, gold: Optional[str]) -> str:
        g = (gold or "").upper()
        if "SELECT" in g[6:] or "INTERSECT" in g or "EXCEPT" in g or "UNION" in g:
            return "NESTED"
        if "JOIN" in g:
            return "NON-NESTED"
        return "EASY"

    def _decompose(self, user: str, gold: Optional[str]) -> str:
        g = gold or ""
        steps = ["Identify the tables involved."]
        if re.search(r"(?i)\bJOIN\b", g):
            steps.append("Join the tables on their foreign keys.")
        if re.search(r"(?i)\bWHERE\b", g):
            steps.append("Apply the filter conditions from the question.")
        if re.search(r"(?i)\bGROUP\s+BY\b", g):
            steps.append("Group the rows and aggregate.")
        if re.search(r"(?i)\bORDER\s+BY\b", g):
            steps.append("Sort the result and take the requested rows.")
        steps.append("Project the requested columns.")
        return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))

    def _unit_tests(self, user: str, gold: Optional[str]) -> str:
        g = gold or ""
        tests = ["The query must run without error."]
        if re.search(r"(?i)\bcount\(", g):
            tests.append("The result must be a single count value.")
        if re.search(r"(?i)\bWHERE\b", g):
            tests.append("The result must be filtered, not the full table.")
        if re.search(r"(?i)\bLIMIT\s+1\b", g):
            tests.append("The result must contain exactly one row.")
        tests.append("The result must not be empty unless the question implies it.")
        return "\n".join(f"- {t}" for t in tests[:4])

    def _pairwise(self, user: str, gold: Optional[str]) -> str:
        cands = re.findall(r"(?is)Candidate\s+([AB])\s*:\s*(.+?)(?=Candidate\s+[AB]\s*:|$)", user)
        if not (gold and len(cands) == 2):
            return "A"
        norm_gold = _norm(gold)
        for label, body in cands:
            if _norm(body) == norm_gold:
                return label
        return "A"

    def _critique(self, user: str, gold: Optional[str]) -> str:
        return (
            "The other agents' queries differ on the filter and the projection. "
            "I will re-check the schema links and keep the version whose "
            "execution result matches the question."
        )

    def _fix(self, user: str, gold: Optional[str]) -> str:
        # the fixer is competent: it repairs to the oracle query
        return f"```sql\n{gold or 'SELECT 1'}\n```"

    def _generate(
        self, tag: str, user: str, gold: Optional[str], sample: int, rnd: int
    ) -> str:
        if not gold:
            return "```sql\nSELECT 1\n```"
        agent = tag.split(":")[0] or "single"
        rate = self.ERROR_RATE.get(agent, 35)
        r = _seed(agent, self._key(user)[-160:], str(sample), str(rnd)) % 100
        # round 2+ of the debate converges: the error rate is halved, which is
        # exactly the effect explicit multi-agent debate is supposed to have
        if rnd >= 1:
            rate = rate // 3
        sql = gold
        if r < rate:
            if (r * 7) % 100 < self.SYNTAX_ERROR_RATE:
                sql = _break_syntax(gold)
            else:
                pert = _PERTURBATIONS[r % len(_PERTURBATIONS)]
                sql = pert(gold) or _swap_aggregate(gold) or _break_syntax(gold)
        reasoning = self._reasoning_for(agent, gold)
        return f"{reasoning}\n\n```sql\n{sql}\n```"

    @staticmethod
    def _reasoning_for(agent: str, gold: str) -> str:
        if agent == "divide_conquer":
            return (
                "Divide: the question breaks into (a) which rows to keep and "
                "(b) what to report.\nConquer: solve each part as a pseudo-SQL "
                "fragment.\nAssemble: merge the fragments and drop redundant clauses."
            )
        if agent == "query_plan":
            return (
                "Query plan: 1) locate the relevant tables, 2) scan/filter and "
                "join on the foreign keys, 3) aggregate if required, "
                "4) project the requested columns."
            )
        if agent == "dail_icl":
            return (
                "The most similar solved examples use the same "
                "question-to-SQL mapping, so I follow their shape."
            )
        return "Reasoning step by step over the schema."


def _norm(sql: str) -> str:
    sql = re.sub(r"```[a-z]*", " ", sql or "")
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").lower()
