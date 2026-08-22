"""The teacher agents.

Each agent is a small object with a single ``run``-style method that takes the
shared :class:`Context` and returns its contribution. Composition happens in
``orchestrator.py``; keeping the agents dumb makes it easy to ablate any one of
them (which is how the ablation table in the README is produced).

Provenance:

==========================  =========================================================
agent                       paper
==========================  =========================================================
``InformationRetriever``    CHESS 3.1 "retrieve entity"/"retrieve context"; CHASE 3.2
``SchemaLinker``            DIN-SQL 4.1 schema linking; CHESS 3.1 Schema Selector
``QueryClassifier``         DIN-SQL 4.2 classification & decomposition
``DivideConquerAgent``      CHASE-SQL 3.3, Algorithm 1
``QueryPlanAgent``          CHASE-SQL 3.3, "Query Plan CoT"
``DailICLAgent``            DAIL-SQL 3.1-3.3 (CR^p + DAIL selection + organisation)
``QueryFixer``              CHASE-SQL 3.4; CHESS "revise"; DIN-SQL 4.4
``UnitTestSelector``        CHESS "Unit Tester"; CHASE-SQL 3.5 selection agent
==========================  =========================================================
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..db import schema as schema_mod
from ..db.execute import ExecResult, execute_sql
from ..db.value_index import ValueHit, get_value_index
from ..llm.base import LLMBackend, extract_list, extract_sql
from . import prompts as P
from .trace import AgentTurn, normalize_sql


# --------------------------------------------------------------------------
# shared state
# --------------------------------------------------------------------------
@dataclass
class Context:
    """Everything the agents read and write while solving one question."""

    question: str
    db_path: str
    db_id: str = ""
    evidence: str = ""
    gold_sql: str = ""              # only used to *label* traces, never shown to agents
    schema: Optional[schema_mod.DBSchema] = None
    pruned_schema: Optional[schema_mod.DBSchema] = None
    schema_text: str = ""
    keywords: List[str] = field(default_factory=list)
    value_hits: List[ValueHit] = field(default_factory=list)
    schema_links: str = ""
    query_class: str = ""
    decomposition: str = ""
    examples: List[Tuple[str, str]] = field(default_factory=list)
    exec_timeout: float = 30.0
    _exec_cache: Dict[str, ExecResult] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------
    @property
    def value_hint_text(self) -> str:
        return "\n".join(f"- {h.as_text()}" for h in self.value_hits)

    def execute(self, sql: Optional[str]) -> ExecResult:
        """Execute with memoisation - the fixer and the selector re-run a lot."""
        key = normalize_sql(sql)
        if not key:
            return ExecResult(ok=False, error="empty query")
        if key not in self._exec_cache:
            self._exec_cache[key] = execute_sql(
                self.db_path, sql or "", timeout=self.exec_timeout
            )
        return self._exec_cache[key]

    def feedback_for(self, sql: Optional[str]) -> str:
        res = self.execute(sql)
        if not res.ok:
            return res.error
        if res.empty:
            return "the query executed but returned 0 rows"
        return ""


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------
class Agent:
    id: str = "agent"

    def __init__(self, llm: LLMBackend, cfg: Any):
        self.llm = llm
        self.cfg = cfg


# --------------------------------------------------------------------------
# CHESS Information Retriever / CHASE-SQL value retrieval
# --------------------------------------------------------------------------
class InformationRetriever(Agent):
    id = "information_retriever"

    def run(
        self, ctx: Context, cache_dir: Optional[str] = None, force: bool = False
    ) -> None:
        # idempotent: when a whole published system sits on the panel it calls
        # this again on the ctx the orchestrator already populated, and paying
        # for the same retrieval once per panellist is pure waste
        if ctx.keywords and not force:
            return
        raw = self.llm.ask(
            P.KEYWORD_SYSTEM,
            P.KEYWORD_USER.format(
                question=ctx.question, evidence=P.evidence_block(ctx.evidence)
            ),
            tag="keywords",
            max_tokens=160,
            temperature=0.0,
        )
        kws = [k for k in extract_list(raw, max_items=12) if len(k) > 1]
        # the question's own capitalised spans are reliable extra probes
        kws += re.findall(r"'([^']{2,40})'", ctx.question)
        ctx.keywords = list(dict.fromkeys(kws))[:12]

        if not ctx.keywords:
            return
        try:
            index = get_value_index(
                ctx.db_path,
                db_id=ctx.db_id,
                cache_dir=cache_dir,
                max_values_per_column=self.cfg.max_values_indexed_per_column,
                num_perm=self.cfg.lsh_signature_size,
                bands=self.cfg.lsh_bands,
            )
        except Exception:
            return
        ctx.value_hits = index.search_many(
            ctx.keywords,
            top_k=self.cfg.value_top_k,
            min_similarity=self.cfg.value_min_similarity,
        )[: self.cfg.value_top_k * 2]


# --------------------------------------------------------------------------
# DIN-SQL schema linking + CHESS Schema Selector
# --------------------------------------------------------------------------
_TABLES_RE = re.compile(r"(?im)^\s*tables?\s*:\s*(.+)$")
_COLUMNS_RE = re.compile(r"(?im)^\s*columns?\s*:\s*(.+)$")


class SchemaLinker(Agent):
    id = "schema_linker"

    def run(self, ctx: Context, force: bool = False) -> None:
        assert ctx.schema is not None
        if ctx.schema_links and not force:   # see InformationRetriever.run
            return
        raw = self.llm.ask(
            P.SCHEMA_LINK_SYSTEM,
            P.SCHEMA_LINK_USER.format(
                schema=ctx.schema_text,
                value_hints=(
                    f"\nRetrieved values:\n{ctx.value_hint_text}\n" if ctx.value_hits else ""
                ),
                question=ctx.question,
                evidence=P.evidence_block(ctx.evidence),
            ),
            tag="schema_link",
            max_tokens=512,
            temperature=0.0,
        )
        ctx.schema_links = (raw or "").strip()

        if not self.cfg.use_schema_selector:
            return
        # tiny schemas: CHESS 3.3 reports that pruning *hurts* when the schema
        # already fits comfortably, so skip it
        if ctx.schema.n_columns <= self.cfg.schema_keep_all_below_columns:
            ctx.pruned_schema = ctx.schema
            return
        links = self._parse_links(raw, ctx.schema)
        ctx.pruned_schema = ctx.schema.prune(links) if links else ctx.schema
        ctx.schema_text = schema_mod.render_schema(
            ctx.pruned_schema,
            style="code",
            max_columns_per_table=self.cfg.schema_max_columns_per_table,
        )

    @staticmethod
    def _parse_links(raw: str, schema: schema_mod.DBSchema) -> Dict[str, List[str]]:
        """Map the model's free-text answer onto real schema identifiers."""
        raw = raw or ""
        table_names = {t.name.lower(): t.name for t in schema.tables}
        wanted: Dict[str, List[str]] = {}

        m = _TABLES_RE.search(raw)
        for tok in extract_list(m.group(1) if m else "", max_items=32):
            real = table_names.get(tok.strip().lower())
            if real:
                wanted.setdefault(real, [])

        m = _COLUMNS_RE.search(raw)
        for tok in extract_list(m.group(1) if m else "", max_items=64):
            tok = tok.strip()
            if "." in tok:
                t, _, c = tok.partition(".")
                real_t = table_names.get(t.strip().lower())
                if real_t:
                    wanted.setdefault(real_t, []).append(c.strip())
            else:
                for tbl in schema.tables:
                    if tbl.column(tok):
                        wanted.setdefault(tbl.name, []).append(tok)
        # a table named with no columns keeps all of its columns
        return {k: v for k, v in wanted.items()}


# --------------------------------------------------------------------------
# DIN-SQL classification & decomposition
# --------------------------------------------------------------------------
class QueryClassifier(Agent):
    id = "query_classifier"

    def run(self, ctx: Context, force: bool = False) -> None:
        if ctx.query_class and not force:   # see InformationRetriever.run
            return
        raw = self.llm.ask(
            P.CLASSIFY_SYSTEM,
            P.CLASSIFY_USER.format(
                schema_links=ctx.schema_links,
                question=ctx.question,
                evidence=P.evidence_block(ctx.evidence),
            ),
            tag="classify",
            max_tokens=16,
            temperature=0.0,
        )
        up = (raw or "").upper()
        if "NEST" in up and "NON" not in up:
            ctx.query_class = "NESTED"
        elif "NON" in up:
            ctx.query_class = "NON-NESTED"
        elif "EASY" in up:
            ctx.query_class = "EASY"
        else:
            ctx.query_class = "NON-NESTED"

        if ctx.query_class == "NESTED":
            ctx.decomposition = (
                self.llm.ask(
                    P.DECOMPOSE_SYSTEM,
                    P.DECOMPOSE_USER.format(
                        schema=ctx.schema_text,
                        schema_links=ctx.schema_links,
                        question=ctx.question,
                        evidence=P.evidence_block(ctx.evidence),
                    ),
                    tag="decompose",
                    max_tokens=400,
                    temperature=0.0,
                )
                or ""
            ).strip()


# --------------------------------------------------------------------------
# candidate generators = the debate panel
# --------------------------------------------------------------------------
class CandidateAgent(Agent):
    """One member of the debate panel (CHASE-SQL 3.3 / DAIL-SQL)."""

    system_prompt: str = P.SINGLE_AGENT_SYSTEM

    def _user(self, ctx: Context) -> str:
        return P.GENERATOR_USER.format(
            dialect=P.DIALECT_NOTE,
            schema=ctx.schema_text,
            value_hints=(
                f"\nRetrieved values:\n{ctx.value_hint_text}\n" if ctx.value_hits else ""
            ),
            schema_links=ctx.schema_links or "(none)",
            query_class=ctx.query_class or "UNKNOWN",
            decomposition=(
                f"\nSub-questions:\n{ctx.decomposition}\n" if ctx.decomposition else ""
            ),
            examples=self._examples(ctx),
            question=ctx.question,
            evidence=P.evidence_block(ctx.evidence),
        )

    def _examples(self, ctx: Context) -> str:
        return ""

    def propose(self, ctx: Context, round_index: int = 0, sample_index: int = 0) -> AgentTurn:
        raw = self.llm.ask(
            self.system_prompt,
            self._user(ctx),
            tag=f"{self.id}:generate",
            max_tokens=self.cfg_max_tokens,
            temperature=(0.0 if sample_index == 0 else 0.8),
            sample_index=sample_index,
            round_index=round_index,
        )
        return self._turn(ctx, raw, round_index)

    def revise(
        self,
        ctx: Context,
        own: AgentTurn,
        others: Sequence[AgentTurn],
        round_index: int,
        sample_index: int = 0,
    ) -> AgentTurn:
        """Round 2+ of the debate: respond to the other agents (Du et al. 2023)."""
        raw = self.llm.ask(
            P.DEBATE_SYSTEM,
            P.DEBATE_USER.format(
                dialect=P.DIALECT_NOTE,
                schema=ctx.schema_text,
                value_hints=(
                    f"\nRetrieved values:\n{ctx.value_hint_text}\n" if ctx.value_hits else ""
                ),
                schema_links=ctx.schema_links or "(none)",
                question=ctx.question,
                evidence=P.evidence_block(ctx.evidence),
                own_answer=_render_answer(ctx, own),
                other_answers="\n\n".join(_render_answer(ctx, o) for o in others) or "(none)",
            ),
            tag=f"{self.id}:generate",
            max_tokens=self.cfg_max_tokens,
            temperature=(0.0 if sample_index == 0 else 0.8),
            sample_index=sample_index,
            round_index=round_index,
        )
        return self._turn(ctx, raw, round_index)

    # -- internals --------------------------------------------------------
    @property
    def cfg_max_tokens(self) -> int:
        return 1024

    def _turn(self, ctx: Context, raw: str, round_index: int) -> AgentTurn:
        sql = extract_sql(raw)
        reasoning = _strip_code(raw)
        res = ctx.execute(sql)
        return AgentTurn(
            agent=self.id,
            round=round_index,
            reasoning=reasoning,
            sql=normalize_sql(sql) or None,
            exec_ok=res.ok,
            exec_error="" if res.ok else res.error,
            exec_preview=res.preview() if res.ok else "",
        )


class DivideConquerAgent(CandidateAgent):
    id = "divide_conquer"
    system_prompt = P.DIVIDE_CONQUER_SYSTEM


class QueryPlanAgent(CandidateAgent):
    id = "query_plan"
    system_prompt = P.QUERY_PLAN_SYSTEM


class DailICLAgent(CandidateAgent):
    """DAIL-SQL: pick few-shot examples by *masked question* + *query* similarity."""

    id = "dail_icl"
    system_prompt = P.DAIL_ICL_SYSTEM

    def _examples(self, ctx: Context) -> str:
        return P.render_dail_examples(ctx.examples)


# --------------------------------------------------------------------------
# whole published systems as panellists
# --------------------------------------------------------------------------
class SystemAgent(CandidateAgent):
    """A panellist that *is* one of the four published systems.

    Round 1 runs that system's own published pipeline end to end (DIN-SQL's
    link/classify/self-correct, CHESS's IR + unit tester, and so on) and takes
    its winning candidate as the panellist's answer. Round 2+ falls back to the
    inherited debate prompt, so the later rounds are a genuine argument
    *between the four systems* over their round-1 answers plus execution
    feedback - which is the transcript the student then internalises.

    Selected with ``mas.agents: [din_sql, dail_sql, chess, chase_sql]``. Note
    this is several times more expensive per question than the default
    three-persona panel, because each panellist is itself multi-call.
    """

    #: name of the ``*_turn`` entry point in ``baselines``
    system_fn: str = ""

    def propose(self, ctx: Context, round_index: int = 0, sample_index: int = 0) -> AgentTurn:
        from . import baselines

        fn = getattr(baselines, self.system_fn)
        # ctx is passed through rather than an example: it already carries the
        # schema, the retrieved values, the links and the DAIL few-shot pairs
        # the orchestrator selected. The context stages are idempotent, so a
        # panellist re-invoking them is a no-op rather than a repeated bill;
        # an empty ``pool`` likewise skips re-running example selection.
        return fn(self.llm, self.cfg, None, (), ctx=ctx, round_index=round_index)


class DinSQLAgent(SystemAgent):
    id = "din_sql"
    system_fn = "din_sql_turn"
    system_prompt = P.QUERY_PLAN_SYSTEM


class DailSQLAgent(SystemAgent):
    id = "dail_sql"
    system_fn = "dail_sql_turn"
    system_prompt = P.DAIL_ICL_SYSTEM


class ChessAgent(SystemAgent):
    id = "chess"
    system_fn = "chess_turn"
    system_prompt = P.QUERY_PLAN_SYSTEM


class ChaseSQLAgent(SystemAgent):
    id = "chase_sql"
    system_fn = "chase_sql_turn"
    system_prompt = P.DIVIDE_CONQUER_SYSTEM


AGENT_REGISTRY = {
    a.id: a
    for a in (
        DivideConquerAgent, QueryPlanAgent, DailICLAgent,
        DinSQLAgent, DailSQLAgent, ChessAgent, ChaseSQLAgent,
    )
}


# --------------------------------------------------------------------------
# DAIL-SQL example selection (section 3.3)
# --------------------------------------------------------------------------
_MASK_TOKEN = "<mask>"


def mask_question(question: str, schema: Optional[schema_mod.DBSchema]) -> str:
    """MQS: replace domain-specific words so similarity is cross-domain."""
    q = (question or "").lower()
    if schema is not None:
        idents = sorted(
            {t.name.lower() for t in schema.tables}
            | {c.name.lower() for t in schema.tables for c in t.columns},
            key=len,
            reverse=True,
        )
        for ident in idents:
            if len(ident) >= 3:
                q = q.replace(ident, _MASK_TOKEN)
    q = re.sub(r"'[^']*'", _MASK_TOKEN, q)
    q = re.sub(r"\b\d+\b", _MASK_TOKEN, q)
    return re.sub(r"\s+", " ", q).strip()


_SQL_KEYWORDS = (
    "select", "from", "where", "join", "group by", "order by", "having",
    "limit", "distinct", "count", "sum", "avg", "min", "max", "not in",
    "intersect", "union", "except", "between", "like", "desc", "asc",
)


def query_skeleton(sql: str) -> frozenset:
    """QRS: a binary discrete syntax vector, as a set of present keywords."""
    low = " " + re.sub(r"\s+", " ", (sql or "").lower()) + " "
    return frozenset(k for k in _SQL_KEYWORDS if k in low)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def select_dail_examples(
    ctx: Context,
    pool: Sequence[Any],
    k: int = 5,
    query_similarity_threshold: float = 0.85,
    preliminary_sql: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """DAIL Selection: rank by masked-question similarity, gated on query similarity.

    ``pool`` holds ``Example``-like records (``question``/``gold_sql``) drawn
    from the *training* split - never from the split being evaluated.
    """
    if not pool or k <= 0:
        return []
    target_mask = mask_question(ctx.question, ctx.schema)
    target_skel = query_skeleton(preliminary_sql or "")

    scored: List[Tuple[float, float, Any]] = []
    for ex in pool:
        q = getattr(ex, "question", "") or ""
        sql = getattr(ex, "gold_sql", "") or ""
        if not (q and sql):
            continue
        qs = SequenceMatcher(None, target_mask, mask_question(q, None)).ratio()
        rs = _jaccard(target_skel, query_skeleton(sql)) if target_skel else 1.0
        scored.append((qs, rs, ex))

    gated = [s for s in scored if s[1] >= query_similarity_threshold]
    chosen = gated or scored
    chosen.sort(key=lambda s: (-s[0], -s[1]))
    return [(getattr(e, "question"), getattr(e, "gold_sql")) for _q, _r, e in chosen[:k]]


# --------------------------------------------------------------------------
# CHASE-SQL query fixer
# --------------------------------------------------------------------------
class QueryFixer(Agent):
    id = "query_fixer"

    def run(self, ctx: Context, turn: AgentTurn) -> AgentTurn:
        """Iteratively repair a query that errors out or returns nothing."""
        sql = turn.sql
        for _ in range(max(0, self.cfg.max_fix_attempts)):
            feedback = ctx.feedback_for(sql)
            if not feedback:
                break
            raw = self.llm.ask(
                P.FIXER_SYSTEM,
                P.FIXER_USER.format(
                    dialect=P.DIALECT_NOTE,
                    schema=ctx.schema_text,
                    value_hints=(
                        f"\nRetrieved values:\n{ctx.value_hint_text}\n" if ctx.value_hits else ""
                    ),
                    question=ctx.question,
                    evidence=P.evidence_block(ctx.evidence),
                    sql=sql or "(no query produced)",
                    feedback=feedback,
                ),
                tag="fix",
                max_tokens=800,
                temperature=0.0,
            )
            fixed = normalize_sql(extract_sql(raw))
            if not fixed or fixed == normalize_sql(sql):
                break
            sql = fixed

        res = ctx.execute(sql)
        turn.sql = normalize_sql(sql) or None
        turn.exec_ok = res.ok
        turn.exec_error = "" if res.ok else res.error
        turn.exec_preview = res.preview() if res.ok else ""
        return turn


# --------------------------------------------------------------------------
# selection: CHESS Unit Tester + CHASE-SQL pairwise selection agent
# --------------------------------------------------------------------------
@dataclass
class Selection:
    sql: Optional[str]
    method: str
    scores: Dict[str, float] = field(default_factory=dict)
    consensus: bool = False
    n_groups: int = 0


class Selector(Agent):
    id = "selector"

    def run(self, ctx: Context, turns: Sequence[AgentTurn]) -> Selection:
        cands = [t for t in turns if t.sql]
        if not cands:
            return Selection(sql=None, method="none")

        groups = _group_by_execution(ctx, cands)
        # a group whose members all executed and agree is already a consensus
        if len(groups) == 1:
            return Selection(
                sql=groups[0][0].sql, method="unanimous", consensus=True, n_groups=1
            )

        method = self.cfg.selector
        if method == "exec_consistency":
            sel = self._exec_consistency(groups)
        elif method == "pairwise":
            sel = self._pairwise(ctx, groups)
        else:
            sel = self._unit_test(ctx, groups)
        sel.n_groups = len(groups)
        return sel

    # -- strategies -------------------------------------------------------
    @staticmethod
    def _exec_consistency(groups: List[List[AgentTurn]]) -> Selection:
        """Self-consistency over *execution results*, the DAIL-SQL/CHASE baseline."""
        best = max(groups, key=lambda g: (g[0].exec_ok, len(g)))
        return Selection(
            sql=best[0].sql,
            method="exec_consistency",
            scores={str(i): float(len(g)) for i, g in enumerate(groups)},
            consensus=len(best) > 1,
        )

    def _pairwise(self, ctx: Context, groups: List[List[AgentTurn]]) -> Selection:
        """CHASE-SQL 3.5 with k = 2: a round-robin of pairwise comparisons."""
        reps = [g[0] for g in groups]
        wins = {i: 0.0 for i in range(len(reps))}
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                winner = self._compare(ctx, reps[i], reps[j])
                wins[i if winner == "A" else j] += 1.0
                # weight by group size, i.e. how many agents produced it
                wins[i] += 0.25 * (len(groups[i]) - 1)
                wins[j] += 0.25 * (len(groups[j]) - 1)
        best = max(wins, key=lambda i: (reps[i].exec_ok, wins[i]))
        return Selection(
            sql=reps[best].sql,
            method="pairwise",
            scores={reps[i].sql or str(i): v for i, v in wins.items()},
            consensus=len(groups[best]) > 1,
        )

    def _compare(self, ctx: Context, a: AgentTurn, b: AgentTurn) -> str:
        raw = self.llm.ask(
            P.PAIRWISE_SYSTEM,
            P.PAIRWISE_USER.format(
                schema=ctx.schema_text,
                question=ctx.question,
                evidence=P.evidence_block(ctx.evidence),
                sql_a=a.sql,
                res_a=ctx.execute(a.sql).preview(),
                sql_b=b.sql,
                res_b=ctx.execute(b.sql).preview(),
            ),
            tag="select_pairwise",
            max_tokens=8,
            temperature=0.0,
        )
        return "B" if (raw or "").strip().upper().startswith("B") else "A"

    def _unit_test(self, ctx: Context, groups: List[List[AgentTurn]]) -> Selection:
        """CHESS Unit Tester: generate discriminating assertions, then vote."""
        reps = [g[0] for g in groups]
        labels = [chr(ord("A") + i) for i in range(len(reps))]
        rendered = "\n\n".join(
            f"Candidate {lab}: {t.sql}\nResult {lab}: {ctx.execute(t.sql).preview()}"
            for lab, t in zip(labels, reps)
        )
        raw = self.llm.ask(
            P.UNIT_TEST_SYSTEM.format(k=self.cfg.n_unit_tests),
            P.UNIT_TEST_USER.format(
                question=ctx.question,
                evidence=P.evidence_block(ctx.evidence),
                candidates=rendered,
            ),
            tag="unit_test",
            max_tokens=400,
            temperature=0.0,
        )
        assertions = [
            a.strip("- ").strip()
            for a in (raw or "").splitlines()
            if a.strip().startswith("-") or a.strip().startswith(("1", "2", "3", "4"))
        ][: self.cfg.n_unit_tests]

        scores = {lab: 0.0 for lab in labels}
        for assertion in assertions:
            verdict = self.llm.ask(
                P.UNIT_EVAL_SYSTEM,
                P.UNIT_EVAL_USER.format(
                    question=ctx.question,
                    evidence=P.evidence_block(ctx.evidence),
                    assertion=assertion,
                    candidates=rendered,
                ),
                tag="unit_test_eval",
                max_tokens=24,
                temperature=0.0,
            )
            for lab in labels:
                if re.search(rf"\b{lab}\b", (verdict or "").upper()):
                    scores[lab] += 1.0
        # tie-breaks: executability first, then how many agents proposed it
        for lab, g in zip(labels, groups):
            scores[lab] += 0.5 * (len(g) - 1) + (0.25 if g[0].exec_ok else -2.0)

        best_lab = max(labels, key=lambda lab: scores[lab])
        best_i = labels.index(best_lab)
        return Selection(
            sql=reps[best_i].sql,
            method="unit_test",
            scores={reps[i].sql or labels[i]: scores[labels[i]] for i in range(len(reps))},
            consensus=len(groups[best_i]) > 1,
        )


def _group_by_execution(ctx: Context, turns: Sequence[AgentTurn]) -> List[List[AgentTurn]]:
    """Bucket candidates by execution-result signature (CHASE-SQL 3.5 baseline)."""
    buckets: Dict[str, List[AgentTurn]] = {}
    for t in turns:
        res = ctx.execute(t.sql)
        if res.ok:
            sig = "OK:" + repr(sorted(map(str, (tuple(r) for r in res.rows[:50]))))
        else:
            sig = "ERR:" + normalize_sql(t.sql)
        buckets.setdefault(sig, []).append(t)
    groups = list(buckets.values())
    groups.sort(key=lambda g: (-int(g[0].exec_ok), -len(g)))
    return groups


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _strip_code(text: Optional[str]) -> str:
    body = _FENCE_RE.sub("", text or "").strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body[:1500]


def _render_answer(ctx: Context, turn: AgentTurn) -> str:
    res = ctx.execute(turn.sql)
    status = "OK" if res.ok else f"ERROR ({res.error})"
    return (
        f"[{turn.agent}] {turn.reasoning[:400]}\n"
        f"SQL: {turn.sql}\n"
        f"Execution: {status} -> {res.preview(2, 200) if res.ok else ''}"
    )
