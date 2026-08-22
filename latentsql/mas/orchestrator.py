"""FusionSQL: the teacher multi-agent system whose behaviour we internalise.

It is a *fusion* of the four systems rather than a re-run of any one of them,
picking the component each paper is strongest on:

    value retrieval    <- CHESS (LSH + edit distance) / CHASE-SQL
    schema linking     <- DIN-SQL, narrowed by the CHESS Schema Selector
    classification     <- DIN-SQL (easy / non-nested / nested)
    candidate panel    <- CHASE-SQL divide-and-conquer + query-plan CoT,
                          plus DAIL-SQL's masked-similarity ICL agent
    debate rounds      <- Du et al. 2023, the protocol IMAD distils
    fixer              <- CHASE-SQL / CHESS revise / DIN-SQL self-correction
    selection          <- CHESS Unit Tester (default) or CHASE-SQL pairwise

The panel is deliberately *diverse* rather than an ensemble of one prompt:
CHASE-SQL 3.3 shows diversity is what makes candidate generation pay off, and
IMAD section 3 shows that distinct agent personas are what create separable
agent subspaces after internalisation.

Output is a :class:`DebateTrace` - the training signal for the student.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..db import schema as schema_mod
from ..db.execute import execution_accuracy
from ..llm.base import LLMBackend
from .agents import (
    AGENT_REGISTRY,
    Context,
    InformationRetriever,
    QueryClassifier,
    QueryFixer,
    SchemaLinker,
    Selector,
    select_dail_examples,
)
from .trace import AgentTurn, DebateTrace


class FusionSQL:
    """Run the teacher on one example and return its debate transcript."""

    def __init__(
        self,
        llm: LLMBackend,
        cfg: Any,
        example_pool: Sequence[Any] = (),
        cache_dir: Optional[str] = None,
    ):
        self.llm = llm
        self.cfg = cfg
        self.example_pool = list(example_pool)
        self.cache_dir = cache_dir

        self.retriever = InformationRetriever(llm, cfg)
        self.linker = SchemaLinker(llm, cfg)
        self.classifier = QueryClassifier(llm, cfg)
        self.fixer = QueryFixer(llm, cfg)
        self.selector = Selector(llm, cfg)
        self.panel = [AGENT_REGISTRY[a](llm, cfg) for a in cfg.agents if a in AGENT_REGISTRY]
        if not self.panel:
            raise ValueError(f"no known agents in mas.agents={cfg.agents!r}")

    # ----------------------------------------------------------------
    def solve(self, example: Any, label_with_gold: bool = True) -> DebateTrace:
        t0 = time.time()
        usage_before = (self.llm.usage.total_tokens, self.llm.usage.calls)

        ctx = self._build_context(example)
        trace = DebateTrace(
            qid=getattr(example, "qid", ""),
            db_id=ctx.db_id,
            question=ctx.question,
            evidence=ctx.evidence,
            gold_sql=getattr(example, "gold_sql", ""),
        )

        # -- stage 1: context (CHESS IR + DIN-SQL linking + classification) --
        if self.cfg.use_value_retrieval:
            self.retriever.run(ctx, cache_dir=self.cache_dir)
        self.linker.run(ctx)
        if self.cfg.use_classifier:
            self.classifier.run(ctx)

        trace.schema_prompt = ctx.schema_text
        trace.schema_links = ctx.schema_links
        trace.value_hints = [h.as_text() for h in ctx.value_hits]
        trace.difficulty_class = ctx.query_class

        # -- stage 2: DAIL-SQL few-shot selection for the ICL panellist -----
        if self.example_pool:
            ctx.examples = select_dail_examples(
                ctx,
                self.example_pool,
                k=self.cfg.dail_shots,
                query_similarity_threshold=self.cfg.dail_query_similarity_threshold,
            )

        # -- stage 3: the debate --------------------------------------------
        rounds: List[List[AgentTurn]] = []
        for r in range(max(1, self.cfg.rounds)):
            turns: List[AgentTurn] = []
            for agent in self.panel:
                for s in range(max(1, self.cfg.candidates_per_agent)):
                    if r == 0:
                        turn = agent.propose(ctx, round_index=r, sample_index=s)
                    else:
                        prev = _last_turn_of(rounds[-1], agent.id)
                        others = [t for t in rounds[-1] if t.agent != agent.id]
                        turn = agent.revise(
                            ctx, prev, others, round_index=r, sample_index=s
                        )
                    if self.cfg.use_fixer:
                        turn = self.fixer.run(ctx, turn)
                    turns.append(turn)
            rounds.append(turns)
            trace.turns.extend(turns)
            # early exit: every agent already agrees and executes cleanly
            if _all_agree(ctx, turns) and r + 1 >= min(2, self.cfg.rounds):
                break

        trace.n_rounds = len(rounds)

        # -- stage 4: selection (CHESS Unit Tester / CHASE-SQL selection agent) --
        selection = self.selector.run(ctx, rounds[-1])
        trace.consensus_sql = selection.sql
        trace.consensus_reached = selection.consensus or selection.method == "unanimous"
        trace.meta["selector"] = selection.method
        trace.meta["selector_scores"] = selection.scores
        trace.meta["n_result_groups"] = selection.n_groups
        trace.meta["wall_time_s"] = round(time.time() - t0, 3)

        # -- stage 5: label the trace (never shown to the agents) ------------
        if label_with_gold and trace.gold_sql:
            ok, _pred, _gold = execution_accuracy(
                ctx.db_path, trace.consensus_sql, trace.gold_sql,
                timeout=self.cfg.execution_timeout,
            )
            trace.correct = ok
            for turn in trace.turns:
                turn.correct = execution_accuracy(
                    ctx.db_path, turn.sql, trace.gold_sql,
                    timeout=self.cfg.execution_timeout,
                )[0]

        trace.teacher_tokens = self.llm.usage.total_tokens - usage_before[0]
        trace.teacher_calls = self.llm.usage.calls - usage_before[1]
        return trace

    # ----------------------------------------------------------------
    def _build_context(self, example: Any) -> Context:
        db_path = getattr(example, "db_path", "")
        schema = schema_mod.load_schema_cached(db_path, getattr(example, "db_id", ""))
        ctx = Context(
            question=getattr(example, "question", ""),
            db_path=db_path,
            db_id=getattr(example, "db_id", ""),
            evidence=getattr(example, "evidence", ""),
            gold_sql=getattr(example, "gold_sql", ""),
            schema=schema,
            exec_timeout=self.cfg.execution_timeout,
        )
        ctx.schema_text = schema_mod.render_schema(
            schema, style="code", max_columns_per_table=None
        )
        return ctx


# --------------------------------------------------------------------------
def _last_turn_of(turns: Sequence[AgentTurn], agent_id: str) -> AgentTurn:
    for t in reversed(turns):
        if t.agent == agent_id:
            return t
    return AgentTurn(agent=agent_id, round=0, reasoning="(no previous answer)", sql=None)


def _all_agree(ctx: Context, turns: Sequence[AgentTurn]) -> bool:
    sqls = [t for t in turns if t.sql]
    if len(sqls) < 2:
        return False
    sigs = set()
    for t in sqls:
        res = ctx.execute(t.sql)
        if not res.ok:
            return False
        sigs.add(repr(sorted(map(str, (tuple(r) for r in res.rows[:50])))))
    return len(sigs) == 1
