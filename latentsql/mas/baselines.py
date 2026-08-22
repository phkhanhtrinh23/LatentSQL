"""Standalone re-implementations of the four published systems.

These exist so the thesis claim - "the internalised 3B student matches or beats
the explicit multi-agent system" - is measured against each source system on
the same schema rendering, the same executor and the same scorer, not against
numbers copied from four different papers.

Each entry point has the signature ``fn(llm, cfg, example, pool=()) -> str|None``
and returns one predicted SQL string. Each also has a ``*_turn`` twin returning
the full :class:`AgentTurn` - reasoning, SQL and execution feedback - which is
what lets a whole published system sit on the teacher's debate panel
(``mas.agents: [din_sql, dail_sql, chess, chase_sql]``) instead of only serving
as a comparison baseline. They share the agents in ``agents.py``
wherever the papers genuinely share a component (e.g. DIN-SQL's schema linking
is the same module the fusion teacher uses).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from ..db import schema as schema_mod
from ..llm.base import LLMBackend, extract_sql
from . import prompts as P
from .agents import (
    Context,
    DailICLAgent,
    DivideConquerAgent,
    InformationRetriever,
    QueryClassifier,
    QueryFixer,
    QueryPlanAgent,
    SchemaLinker,
    Selector,
    select_dail_examples,
)
from .trace import AgentTurn, normalize_sql


def _as_turn(
    ctx: Context, agent_id: str, sql: Optional[str], round_index: int = 0,
    reasoning: str = "",
) -> AgentTurn:
    """Wrap a bare SQL string in an ``AgentTurn`` with execution feedback."""
    sql = normalize_sql(sql) or None
    res = ctx.execute(sql)
    return AgentTurn(
        agent=agent_id,
        round=round_index,
        reasoning=reasoning,
        sql=sql,
        exec_ok=res.ok,
        exec_error="" if res.ok else res.error,
        exec_preview=res.preview() if res.ok else "",
    )


def _relabel(turn: AgentTurn, agent_id: str, round_index: int) -> AgentTurn:
    """Re-tag an inner agent's turn as the *system* that produced it."""
    turn.agent = agent_id
    turn.round = round_index
    return turn


def _context(cfg: Any, example: Any) -> Context:
    db_path = getattr(example, "db_path", "")
    schema = schema_mod.load_schema_cached(db_path, getattr(example, "db_id", ""))
    ctx = Context(
        question=getattr(example, "question", ""),
        db_path=db_path,
        db_id=getattr(example, "db_id", ""),
        evidence=getattr(example, "evidence", ""),
        gold_sql=getattr(example, "gold_sql", ""),
        schema=schema,
        exec_timeout=cfg.execution_timeout,
    )
    ctx.schema_text = schema_mod.render_schema(schema, style="code")
    return ctx


# --------------------------------------------------------------------------
# Single agent (zero-shot) - the "Single" row in IMAD Table 1
# --------------------------------------------------------------------------
def single_agent(llm: LLMBackend, cfg: Any, example: Any, pool: Sequence[Any] = ()) -> Optional[str]:
    ctx = _context(cfg, example)
    raw = llm.ask(
        P.SINGLE_AGENT_SYSTEM,
        P.SINGLE_AGENT_USER.format(
            dialect=P.DIALECT_NOTE,
            schema=ctx.schema_text,
            value_hints="",
            question=ctx.question,
            evidence=P.evidence_block(ctx.evidence),
        ),
        tag="single:generate",
        max_tokens=1024,
        temperature=0.0,
    )
    return normalize_sql(extract_sql(raw)) or None


# --------------------------------------------------------------------------
# DIN-SQL (Pourreza & Rafiei 2023)
#   schema linking -> classify & decompose -> class-specific generation -> self-correct
# --------------------------------------------------------------------------
def din_sql(llm: LLMBackend, cfg: Any, example: Any, pool: Sequence[Any] = ()) -> Optional[str]:
    return din_sql_turn(llm, cfg, example, pool).sql


def din_sql_turn(
    llm: LLMBackend, cfg: Any, example: Any, pool: Sequence[Any] = (),
    ctx: Optional[Context] = None, round_index: int = 0,
) -> AgentTurn:
    ctx = ctx or _context(cfg, example)
    SchemaLinker(llm, cfg).run(ctx)
    QueryClassifier(llm, cfg).run(ctx)

    # DIN-SQL 4.3 uses a different prompt per class; the divide-and-conquer
    # agent is the closest match for NESTED, plain generation for EASY.
    agent = DivideConquerAgent(llm, cfg) if ctx.query_class == "NESTED" else QueryPlanAgent(llm, cfg)
    turn = agent.propose(ctx, round_index=0)
    turn = QueryFixer(llm, cfg).run(ctx, turn)     # 4.4 self-correction
    return _relabel(turn, "din_sql", round_index)


# --------------------------------------------------------------------------
# DAIL-SQL (Gao et al. 2023)
#   CR^p representation + DAIL example selection/organisation + self-consistency
# --------------------------------------------------------------------------
def dail_sql(
    llm: LLMBackend,
    cfg: Any,
    example: Any,
    pool: Sequence[Any] = (),
    self_consistency: int = 3,
) -> Optional[str]:
    return dail_sql_turn(llm, cfg, example, pool, self_consistency=self_consistency).sql


def dail_sql_turn(
    llm: LLMBackend,
    cfg: Any,
    example: Any,
    pool: Sequence[Any] = (),
    self_consistency: int = 3,
    ctx: Optional[Context] = None,
    round_index: int = 0,
) -> AgentTurn:
    ctx = ctx or _context(cfg, example)
    if pool:
        ctx.examples = select_dail_examples(
            ctx, pool, k=cfg.dail_shots,
            query_similarity_threshold=cfg.dail_query_similarity_threshold,
        )
    agent = DailICLAgent(llm, cfg)
    turns = [agent.propose(ctx, round_index=0, sample_index=i) for i in range(max(1, self_consistency))]
    cands = [t for t in turns if t.sql]
    if not cands:
        return _as_turn(ctx, "dail_sql", None, round_index)
    if len(cands) == 1:
        return _relabel(cands[0], "dail_sql", round_index)
    # self-consistency over execution results
    buckets: Dict[str, List[Any]] = {}
    for t in cands:
        res = ctx.execute(t.sql)
        sig = ("OK:" + repr([tuple(r) for r in res.rows[:20]])) if res.ok else "ERR"
        buckets.setdefault(sig, []).append(t)
    good = {k: v for k, v in buckets.items() if k != "ERR"} or buckets
    best = max(good.values(), key=len)
    return _relabel(best[0], "dail_sql", round_index)


# --------------------------------------------------------------------------
# CHESS (Talaei et al. 2024)
#   IR -> Schema Selector -> Candidate Generator (+revise) -> Unit Tester
# --------------------------------------------------------------------------
def chess(
    llm: LLMBackend,
    cfg: Any,
    example: Any,
    pool: Sequence[Any] = (),
    n_candidates: int = 3,
) -> Optional[str]:
    return chess_turn(llm, cfg, example, pool, n_candidates=n_candidates).sql


def chess_turn(
    llm: LLMBackend,
    cfg: Any,
    example: Any,
    pool: Sequence[Any] = (),
    n_candidates: int = 3,
    ctx: Optional[Context] = None,
    round_index: int = 0,
) -> AgentTurn:
    ctx = ctx or _context(cfg, example)
    InformationRetriever(llm, cfg).run(ctx)            # IR agent
    SchemaLinker(llm, cfg).run(ctx)                    # SS agent
    agent = QueryPlanAgent(llm, cfg)                   # CG agent
    fixer = QueryFixer(llm, cfg)
    turns = []
    for i in range(max(1, n_candidates)):
        t = agent.propose(ctx, round_index=0, sample_index=i)
        turns.append(fixer.run(ctx, t))                # CG "revise" loop
    sel_cfg = _with(cfg, selector="unit_test")         # UT agent
    return _pick(ctx, "chess", Selector(llm, sel_cfg).run(ctx, turns), turns, round_index)


# --------------------------------------------------------------------------
# CHASE-SQL (Pourreza et al. 2024)
#   value retrieval -> multi-path candidates -> fixer -> pairwise selection agent
# --------------------------------------------------------------------------
def chase_sql(
    llm: LLMBackend,
    cfg: Any,
    example: Any,
    pool: Sequence[Any] = (),
) -> Optional[str]:
    return chase_sql_turn(llm, cfg, example, pool).sql


def chase_sql_turn(
    llm: LLMBackend,
    cfg: Any,
    example: Any,
    pool: Sequence[Any] = (),
    ctx: Optional[Context] = None,
    round_index: int = 0,
) -> AgentTurn:
    ctx = ctx or _context(cfg, example)
    InformationRetriever(llm, cfg).run(ctx)
    SchemaLinker(llm, cfg).run(ctx)
    if pool:
        ctx.examples = select_dail_examples(
            ctx, pool, k=cfg.dail_shots,
            query_similarity_threshold=cfg.dail_query_similarity_threshold,
        )
    fixer = QueryFixer(llm, cfg)
    turns = []
    # the three CHASE-SQL generators: divide-and-conquer, query plan, and
    # (in place of online synthetic examples) the retrieved-example generator
    for agent in (DivideConquerAgent(llm, cfg), QueryPlanAgent(llm, cfg), DailICLAgent(llm, cfg)):
        t = agent.propose(ctx, round_index=0)
        turns.append(fixer.run(ctx, t))
    sel_cfg = _with(cfg, selector="pairwise")
    return _pick(ctx, "chase_sql", Selector(llm, sel_cfg).run(ctx, turns), turns, round_index)


# --------------------------------------------------------------------------
# the fusion teacher, exposed with the same signature
# --------------------------------------------------------------------------
def fusion_mas(
    llm: LLMBackend,
    cfg: Any,
    example: Any,
    pool: Sequence[Any] = (),
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    from .orchestrator import FusionSQL

    trace = FusionSQL(llm, cfg, example_pool=pool, cache_dir=cache_dir).solve(
        example, label_with_gold=False
    )
    return trace.consensus_sql


BASELINES: Dict[str, Callable[..., Optional[str]]] = {
    "single": single_agent,
    "din_sql": din_sql,
    "dail_sql": dail_sql,
    "chess": chess,
    "chase_sql": chase_sql,
    "fusion_mas": fusion_mas,
}


def _pick(
    ctx: Context, agent_id: str, selection: Any, turns: Sequence[AgentTurn],
    round_index: int,
) -> AgentTurn:
    """Recover the ``AgentTurn`` behind a ``Selection`` so its reasoning survives.

    The selector returns a SQL string; the reasoning that produced it lives on
    the candidate turn. Matching them back up is what lets a whole system act
    as one panellist without losing the text the student has to learn.
    """
    if selection.sql:
        want = (normalize_sql(selection.sql) or "").lower()
        for t in turns:
            if t.sql and (normalize_sql(t.sql) or "").lower() == want:
                return _relabel(t, agent_id, round_index)
    return _as_turn(ctx, agent_id, selection.sql, round_index)


def _with(cfg: Any, **overrides: Any) -> Any:
    """Shallow copy of a config dataclass with a few fields replaced."""
    import copy

    new = copy.copy(cfg)
    for k, v in overrides.items():
        setattr(new, k, v)
    return new
