"""Prompts for the teacher agents, one section per source paper.

Every prompt is a plain ``str.format`` template so it can be inspected, edited
and logged. The provenance comment on each one names the paper section it
implements, because the whole point of the teacher is to be a faithful (if
compact) reconstruction of the published systems.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------
DIALECT_NOTE = (
    "The target dialect is SQLite. Use only SQLite-supported syntax "
    "(no FULL OUTER JOIN, no window functions unless necessary)."
)

SQL_OUTPUT_RULE = (
    "End your answer with the final query inside a single ```sql fenced block. "
    "Output exactly one statement, no trailing semicolon commentary."
)


def evidence_block(evidence: str) -> str:
    """BIRD ships an 'external knowledge' hint with each question."""
    ev = (evidence or "").strip()
    return f"\nExternal knowledge: {ev}\n" if ev else ""


# --------------------------------------------------------------------------
# CHESS - Information Retriever (section 3.1) / CHASE-SQL - value retrieval (3.2)
# --------------------------------------------------------------------------
KEYWORD_SYSTEM = (
    "You extract search keywords for a Text-to-SQL system. Given a question, "
    "list the entities, proper nouns, and literal cell values that are likely "
    "to appear verbatim in the database. Do not list column or table names. "
    "Answer with a comma-separated list and nothing else."
)

KEYWORD_USER = """Question: {question}{evidence}

Keywords:"""


# --------------------------------------------------------------------------
# DIN-SQL - schema linking (section 4.1) + CHESS - Schema Selector (3.1)
# --------------------------------------------------------------------------
SCHEMA_LINK_SYSTEM = (
    "You are the schema-linking module of a Text-to-SQL system. "
    "For every mention in the question, identify the database tables and "
    "columns it refers to, plus any literal condition values. "
    "Let's think step by step, then give the final answer in exactly this form:\n"
    "Tables: <comma-separated table names>\n"
    "Columns: <comma-separated table.column names>\n"
    "Conditions: <comma-separated column = value pairs, or 'none'>\n"
    "Be exhaustive about what is needed and strict about what is not. "
    "Always include the columns needed for any JOIN."
)

SCHEMA_LINK_USER = """Database schema:
{schema}
{value_hints}
Question: {question}{evidence}

Schema links:"""


# --------------------------------------------------------------------------
# DIN-SQL - query classification & decomposition (section 4.2)
# --------------------------------------------------------------------------
CLASSIFY_SYSTEM = (
    "You are the query-classification module of a Text-to-SQL system. "
    "Classify the question into exactly one class:\n"
    "EASY - answerable from a single table with no join and no sub-query.\n"
    "NON-NESTED - requires one or more joins but no sub-query or set operation.\n"
    "NESTED - requires a sub-query, a set operation (INTERSECT/UNION/EXCEPT), "
    "or an aggregate compared against another aggregate.\n"
    "Answer with the class name only."
)

CLASSIFY_USER = """Schema links:
{schema_links}

Question: {question}{evidence}

Class:"""

DECOMPOSE_SYSTEM = (
    "You are the query-decomposition module of a Text-to-SQL system. "
    "Break the question into the minimal ordered list of sub-questions whose "
    "answers combine into the final query. One sub-question per line, numbered. "
    "Do not write SQL yet."
)

DECOMPOSE_USER = """Schema:
{schema}

Schema links:
{schema_links}

Question: {question}{evidence}

Sub-questions:"""


# --------------------------------------------------------------------------
# CHASE-SQL - candidate generator 1: Divide-and-Conquer CoT (section 3.3, Alg. 1)
# --------------------------------------------------------------------------
DIVIDE_CONQUER_SYSTEM = (
    "You are a Text-to-SQL agent that reasons by divide-and-conquer.\n"
    "DIVIDE: decompose the question into sub-problems, writing each as a "
    "pseudo-SQL fragment with placeholders.\n"
    "CONQUER: solve each sub-problem in order, reusing earlier fragments.\n"
    "ASSEMBLE: combine the fragments into one final query, then OPTIMIZE it by "
    "removing redundant clauses, sub-queries and conditions.\n"
    "Keep the reasoning under 200 words. " + SQL_OUTPUT_RULE
)

QUERY_PLAN_SYSTEM = (
    "You are a Text-to-SQL agent that reasons like a database query planner.\n"
    "Write the plan the engine would follow, in three steps:\n"
    "1. Locate the tables that hold the requested information.\n"
    "2. Describe the scans, filters and joins, naming the join keys explicitly.\n"
    "3. Describe the aggregation, ordering and the columns to return.\n"
    "Then translate the plan into SQL.\n"
    "Keep the reasoning under 200 words. " + SQL_OUTPUT_RULE
)

# --------------------------------------------------------------------------
# DAIL-SQL - CR^p question representation + DAIL example organisation (3.1-3.3)
# --------------------------------------------------------------------------
DAIL_ICL_SYSTEM = (
    "You are a Text-to-SQL agent that reasons from analogy. "
    "You are shown solved question/SQL pairs that are similar to the target "
    "question. Identify which mapping pattern applies, state it in one or two "
    "sentences, then adapt it to the target schema.\n"
    "Keep the reasoning under 150 words. " + SQL_OUTPUT_RULE
)

GENERATOR_USER = """{dialect}

Database schema:
{schema}
{value_hints}
Schema links:
{schema_links}

Query class: {query_class}
{decomposition}{examples}
Question: {question}{evidence}

Your answer:"""


DAIL_EXAMPLES_HEADER = (
    "\n/* Some example questions and corresponding SQL queries are provided "
    "based on similar problems: */\n"
)


def render_dail_examples(pairs) -> str:
    """DAIL Organization (DAIL-SQL Listing 8): keep the question->SQL mapping,
    drop the per-example schema to save tokens."""
    if not pairs:
        return ""
    out = [DAIL_EXAMPLES_HEADER]
    for q, sql in pairs:
        out.append(f"/* Answer the following: {q} */\n{sql}\n")
    return "\n".join(out)


# --------------------------------------------------------------------------
# debate round 2+ - Du et al. (2023) style critique, the structure IMAD distils
# --------------------------------------------------------------------------
DEBATE_SYSTEM = (
    "You are one agent in a Text-to-SQL debate. You previously proposed a "
    "query; the other agents proposed theirs, and each query's execution "
    "result is shown. Using their reasoning and the execution feedback as "
    "additional advice, decide whether your query was wrong and produce your "
    "updated answer.\n"
    "State in one or two sentences what you changed and why (or why you are "
    "keeping your query). " + SQL_OUTPUT_RULE
)

DEBATE_USER = """{dialect}

Database schema:
{schema}
{value_hints}
Schema links:
{schema_links}

Question: {question}{evidence}

Your previous answer:
{own_answer}

Other agents' answers this round:
{other_answers}

Your updated answer:"""


# --------------------------------------------------------------------------
# CHASE-SQL - query fixer (3.4) / CHESS - revise (3.1) / DIN-SQL - self-correct (4.4)
# --------------------------------------------------------------------------
FIXER_SYSTEM = (
    "You are the query-fixer module of a Text-to-SQL system. You are given a "
    "query and the feedback from executing it (a syntax error, an empty "
    "result, or a runtime error). Reflect on the cause in one or two "
    "sentences, then output the corrected query. If the query is already "
    "correct, return it unchanged. " + SQL_OUTPUT_RULE
)

FIXER_USER = """{dialect}

Database schema:
{schema}
{value_hints}
Question: {question}{evidence}

Query:
{sql}

Execution feedback: {feedback}

Corrected query:"""


# --------------------------------------------------------------------------
# CHESS - Unit Tester (3.1) and CHASE-SQL - selection agent (3.5)
# --------------------------------------------------------------------------
UNIT_TEST_SYSTEM = (
    "You are the unit-test generator of a Text-to-SQL system. Given a question "
    "and several candidate queries that disagree, write {k} short natural-language "
    "assertions that only the CORRECT query can satisfy. Each assertion must "
    "discriminate between the candidates - do not write assertions all "
    "candidates trivially pass. One assertion per line, prefixed with '- '."
)

UNIT_TEST_USER = """Question: {question}{evidence}

Candidates and their execution results:
{candidates}

Assertions:"""

UNIT_EVAL_SYSTEM = (
    "You are the unit-test evaluator of a Text-to-SQL system. Given one "
    "assertion and several candidate queries with their execution results, "
    "answer with the labels of every candidate that PASSES the assertion, "
    "comma-separated (e.g. 'A, C'). If none pass, answer 'none'."
)

UNIT_EVAL_USER = """Question: {question}{evidence}

Assertion: {assertion}

Candidates:
{candidates}

Passing candidates:"""

PAIRWISE_SYSTEM = (
    "You are the selection agent of a Text-to-SQL system. You are given a "
    "question and two candidate queries with their execution results. Decide "
    "which one correctly answers the question. Answer with a single letter, "
    "A or B, and nothing else."
)

PAIRWISE_USER = """Database schema:
{schema}

Question: {question}{evidence}

Candidate A: {sql_a}
Result A: {res_a}

Candidate B: {sql_b}
Result B: {res_b}

Better candidate:"""


# --------------------------------------------------------------------------
# single-agent baseline + the internalised student
# --------------------------------------------------------------------------
SINGLE_AGENT_SYSTEM = (
    "You are a Text-to-SQL model. Read the schema and the question, reason "
    "briefly, and write one SQL query that answers it. " + SQL_OUTPUT_RULE
)

SINGLE_AGENT_USER = """{dialect}

Database schema:
{schema}
{value_hints}
Question: {question}{evidence}

Your answer:"""

#: The student is trained with this exact instruction, so the same string is
#: used for SFT, GRPO rollouts and evaluation. Changing it invalidates a
#: checkpoint, so it is versioned.
STUDENT_SYSTEM_V1 = (
    "You are a Text-to-SQL model with an internalised multi-agent debate. "
    "Given a database schema and a question, deliberate over the schema links, "
    "several solution strategies and their execution behaviour, then commit to "
    "one query. Emit your final answer inside "
    "<|consensus|> ```sql ... ``` <|/consensus|> followed by <|endofdebate|>."
)

STUDENT_USER_V1 = """{dialect}

Database schema:
{schema}
{value_hints}
Question: {question}{evidence}"""


def student_prompt(schema: str, question: str, evidence: str = "", value_hints: str = "") -> dict:
    """The (system, user) pair used everywhere the student is invoked."""
    return {
        "system": STUDENT_SYSTEM_V1,
        "user": STUDENT_USER_V1.format(
            dialect=DIALECT_NOTE,
            schema=schema,
            value_hints=(f"\nRetrieved values:\n{value_hints}\n" if value_hints else ""),
            question=question,
            evidence=evidence_block(evidence),
        ),
    }
