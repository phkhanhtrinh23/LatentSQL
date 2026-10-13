"""Prompts and role tags.

The student input ``x = (q, S, K, H)`` (Sec. 3.1) is serialized into one prompt
together with the retrieved database values (Sec. 4.1). The same function
builds it for training, RL and inference, so the three phases condition on
exactly the same text.

A teacher trace (Eq. 9) is ``<r1> T1 <r2> T2 <r3> T3 <sql> Y``. The role tags
are plain text; the student reads them as ordinary tokens.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

#: role tags <r_j> and the SQL tag <sql> (Eq. 9, 11, 22)
ROLE_TAGS: Dict[str, str] = {
    "schema_selector": "<|schema_selector|>",
    "candidate_generator": "<|candidate_generator|>",
    "reviser": "<|reviser|>",
}
SQL_TAG = "<|sql|>"


def role_tag(role: str) -> str:
    return ROLE_TAGS.get(role, f"<|{role}|>")


DIALECT_NOTE = "The target dialect is SQLite."

# --------------------------------------------------------------------------
# student
# --------------------------------------------------------------------------
STUDENT_SYSTEM = (
    "You are a Text-to-SQL model. Given a database schema, retrieved database "
    "values, optional external knowledge and dialogue history, reason through "
    "schema selection, candidate generation and revision, then write the final "
    f"SQLite query after {SQL_TAG} inside a ```sql block."
)

STUDENT_USER = """{dialect}

Database schema:
{schema}
{values}{knowledge}{history}Question: {question}"""


def _values_block(values: Sequence[str]) -> str:
    values = [v for v in values or [] if v]
    return ("\nRetrieved values:\n" + "\n".join(f"- {v}" for v in values) + "\n") if values else ""


def _knowledge_block(knowledge: str) -> str:
    knowledge = (knowledge or "").strip()
    return f"\nExternal knowledge: {knowledge}\n" if knowledge else ""


def _history_block(history: Sequence[str]) -> str:
    history = [h for h in history or [] if h and h.strip()]
    if not history:
        return ""
    lines = "\n".join(f"{i}. {h.strip()}" for i, h in enumerate(history, 1))
    return f"\nEarlier questions in this dialogue:\n{lines}\n"


def student_prompt(schema: str, question: str, knowledge: str = "",
                   history: Sequence[str] = (), values: Sequence[str] = ()) -> Dict[str, str]:
    """The (system, user) pair used wherever the student is invoked."""
    return {
        "system": STUDENT_SYSTEM,
        "user": STUDENT_USER.format(
            dialect=DIALECT_NOTE, schema=schema, values=_values_block(values),
            knowledge=_knowledge_block(knowledge), history=_history_block(history),
            question=question.strip(),
        ),
    }


def sql_target(sql: str) -> str:
    """What the student decodes after ``<sql>``."""
    return f"\n```sql\n{sql.strip()}\n```"


# --------------------------------------------------------------------------
# teacher roles (Sec. 4.1)
# --------------------------------------------------------------------------
TEACHER_CONTEXT = """{dialect}

Database schema:
{schema}
{values}{knowledge}{history}Question: {question}"""

SCHEMA_SELECTOR_SYSTEM = (
    "You are the schema selector of a Text-to-SQL system. Identify the tables "
    "and columns the question needs, the join paths between those tables "
    "(through foreign keys), and the literal conditions. Reason briefly, then "
    "answer in exactly this form:\n"
    "Tables: <comma-separated table names>\n"
    "Columns: <comma-separated table.column names>\n"
    "Joins: <table.column = table.column pairs, or none>\n"
    "Conditions: <column operator value, or none>\n"
    "Keep it under 150 words. Do not write SQL."
)

CANDIDATE_GENERATOR_SYSTEM = (
    "You are the candidate generator of a Text-to-SQL system. Using the selected "
    "schema, write the query step by step: state the sub-problems, build the "
    "clauses that solve them, and combine them into one query. Keep the reasoning "
    "under 150 words and end with the query in a single ```sql block."
)

REVISER_SYSTEM = (
    "You are the reviser of a Text-to-SQL system. You receive a draft query and "
    "the result or error of executing it on the database. Check the draft "
    "against the question. If it is correct, say so and repeat it. Otherwise "
    "name the error (syntax error, missing column, wrong join, empty or wrong "
    "result, ...) and fix it. Keep it under 120 words and end with the final "
    "query in a single ```sql block."
)

SELECTED_SCHEMA = "\n\nSelected schema:\n{t1}"
DRAFT_BLOCK = "\n\nDraft query:\n```sql\n{draft}\n```\nExecution: {report}"


def teacher_context(schema: str, question: str, knowledge: str = "",
                    history: Sequence[str] = (), values: Sequence[str] = ()) -> str:
    return TEACHER_CONTEXT.format(
        dialect=DIALECT_NOTE, schema=schema, values=_values_block(values),
        knowledge=_knowledge_block(knowledge), history=_history_block(history),
        question=question.strip(),
    )


# --------------------------------------------------------------------------
# SQL extraction
# --------------------------------------------------------------------------
_SQL_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START = re.compile(r"(?is)\b(WITH|SELECT)\b")


def normalize_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    s = _SQL_FENCE.sub(lambda m: m.group(1), sql)
    s = re.sub(r"(?m)^\s*--.*$", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.split(";")[0].strip() if ";" in s else s
    return s


def extract_sql(text: Optional[str]) -> Optional[str]:
    """The last fenced SQL block, else the text from the first SELECT/WITH."""
    if not text:
        return None
    if SQL_TAG in text:
        text = text[text.rfind(SQL_TAG) + len(SQL_TAG):]
    fences = _SQL_FENCE.findall(text)
    if fences:
        return normalize_sql(fences[-1]) or None
    # an unterminated fence (generation cut at the length limit)
    open_fence = re.search(r"```(?:sql|sqlite)?\s*\n?(.*)$", text, re.DOTALL | re.IGNORECASE)
    if open_fence:
        return normalize_sql(open_fence.group(1)) or None
    m = _SQL_START.search(text)
    return normalize_sql(text[m.start():]) or None if m else None


def sql_is_closed(text: str) -> bool:
    """True once a decoded completion contains a finished ```sql block."""
    i = text.find("```")
    return i >= 0 and text.find("```", i + 3) >= 0
