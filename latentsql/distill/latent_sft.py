"""Stage 1.5b (optional) - latent SFT: give the model a worked example of
success *inside* the exact conditioning GRPO and eval will later ask of it.

The root cause found for the four-arm factorial's 0% EX (RESULTS.md 5c): SFT's
only training target is the written debate transcript, generated from a plain
prompt with no latent prefix at all. GRPO's latent rollouts (and eval's
``LatentReasoner.generate``) instead prime the model through
``build_prefix`` - the same role loop, m latent steps per role, no tokens
decoded - and then ask for a bare SQL completion continuing from that primed
KV-cache. SFT never once demonstrated *that* transition, so GRPO had to
bootstrap it from a 0%-correct starting point with no imitation head start,
which is exactly why every logged step died (adv_nonzero_frac stuck at 0.0).

Follow-up finding (RESULTS.md 5d): plain latent-SFT (feed the full 96-step
latent prefix cold, every step, from step one) works on a 63-example corpus
only in the sense that it does not crash - it either under-learns (too few
updates: reverts to the original garbage) or overfits to near-total
memorisation (enough updates to learn anything: collapses to 1-2 answers
across 90 questions). Batching the gradient did not fix this; the corpus was
the bottleneck, not the optimiser.

This module now supports a COCONUT-style CURRICULUM (Deng et al. 2023/2024,
credited in COCONUT sec. 2, "Training Large Language Models to Reason in a
Continuous Latent Space") on top of the same worked-example mechanism: instead
of demanding the full jump from "plain text prompt" to "96 latent steps" on
step one, the number of roles run in latent space ramps from 0 (pure text -
exactly what ordinary SFT already trains successfully) up to all of them
(the fully-latent regime GRPO/eval need) over the course of training. Early
steps are close to free (text-mode SFT the model already knows how to do);
only late steps ask for the genuinely novel latent-conditioned completion,
by which point the model has had many intermediate steps to adapt gradually
rather than being asked to learn to read a totally alien input from scratch
in one leap.
"""
from __future__ import annotations

import json
import hashlib
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

from ..mas.prompts import student_prompt
from ..llm.hf_local import render_chat
from ..mas.trace import T_CONSENSUS_OPEN, T_CONSENSUS_CLOSE, T_END, T_AGENT_CLOSE, agent_tag


def _identifier_error_mutant(sql: str) -> Optional[str]:
    """Make one deterministic unknown-column mutant when sqlglot can parse SQL."""
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, read="sqlite")
        column = next(tree.find_all(exp.Column), None)
        if column is None or not column.name:
            return None
        column.set("this", exp.Identifier(this=f"{column.name}__invalid"))
        return tree.sql(dialect="sqlite")
    except Exception:
        return None


def _empty_result_mutant(sql: str) -> Optional[str]:
    """Replace one string literal with an impossible value (one AST edit)."""
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, read="sqlite")
        literal = next(
            (node for node in tree.find_all(exp.Literal) if node.is_string), None
        )
        if literal is None:
            return None
        literal.replace(exp.Literal.string("__latentsql_missing_value__"))
        return tree.sql(dialect="sqlite")
    except Exception:
        return None


def _offline_execution_negatives(trace: Any, correct_sql: str,
                                 max_negatives: int = 2) -> List[Dict[str, str]]:
    """Build deterministic training-only outcome examples for one correct SQL.

    These labels are produced offline and embedded in the teacher trace.  They
    are never computed or supplied during latent inference.  A mutant is kept
    only when execution proves it differs from the gold result.
    """
    from ..db.execute import execute_sql, gold_is_ordered, result_match
    from ..db.spurious import mutate_query

    db_path = str(getattr(trace, "meta", {}).get("db_path", "") or "")
    gold_sql = str(getattr(trace, "gold_sql", "") or "")
    if not (db_path and os.path.exists(db_path) and gold_sql and correct_sql):
        return []
    gold_result = execute_sql(db_path, gold_sql, timeout=30.0)
    if not gold_result.ok:
        return []

    seed_text = f"{getattr(trace, 'qid', '')}\0{correct_sql}"
    seed = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    candidates: List[str] = []
    for mutant in (
        _identifier_error_mutant(correct_sql),
        _empty_result_mutant(correct_sql),
        *mutate_query(correct_sql, rng, max_mutants=12),
    ):
        if mutant and mutant != correct_sql and mutant not in candidates:
            candidates.append(mutant)

    labelled: List[Dict[str, str]] = []
    seen_labels: set[str] = set()
    for mutant in candidates:
        result = execute_sql(db_path, mutant, timeout=30.0)
        if not result.ok:
            lower = result.error.lower()
            label = "syntax_error" if "syntax" in lower or "incomplete" in lower else "runtime_error"
            detail = result.error[:240]
        elif result.empty and not gold_result.empty:
            label, detail = "empty_result", "query executed but returned no rows"
        elif not result_match(result, gold_result, ordered=gold_is_ordered(gold_sql)):
            label, detail = "result_mismatch", "query executed but disagreed with the gold result"
        else:
            continue
        # Prefer diverse failure modes over two nearly identical mutations.
        if label in seen_labels and len(candidates) > max_negatives:
            continue
        seen_labels.add(label)
        labelled.append({"sql": mutant, "label": label, "detail": detail})
        if len(labelled) >= max_negatives:
            break
    return labelled

#: sequence positions upcast to fp32 at once when scoring log-probabilities -
#: same chunking trick as grpo.py/dpo.py, same OOM this avoids.
_LOGP_CHUNK = 512


def _target_ids(tok, gold_sql: str) -> List[int]:
    text = f"```sql\n{gold_sql.strip()}\n```"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        ids = ids + [tok.eos_token_id]
    return ids


def _consensus_target_ids(tok, gold_sql: str) -> List[int]:
    """SQL-only target after the consensus marker."""
    text = f"\n```sql\n{gold_sql.strip()}\n```\n{T_CONSENSUS_CLOSE}\n{T_END}"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        ids = ids + [tok.eos_token_id]
    return ids


def _explicit_curriculum_turns(role_turns: Sequence[str], roles: Sequence[str],
                               n_latent_roles: int) -> Dict[int, str]:
    """Validated explicit suffix for a COCONUT curriculum stage.

    The first ``n_latent_roles`` are replaced by continuous thoughts. Every
    later non-consensus role must remain as supervised text until its own
    replacement stage; otherwise trace-only supervision such as offline
    execution outcomes would never enter the loss.
    """
    out: Dict[int, str] = {}
    for idx, role in enumerate(roles):
        if role == "consensus":
            break
        if idx < n_latent_roles:
            continue
        if idx >= len(role_turns) or not str(role_turns[idx]).strip():
            raise ValueError(
                f"missing explicit role turn {idx} ({role}) for COCONUT curriculum"
            )
        out[idx] = str(role_turns[idx]).rstrip()
    return out


def _coconut_sequence_loss(reasoner: Any, rec: Dict[str, Any], n_latent_roles: int,
                           target_ids: Sequence[int],
                           sql_ready_role: Optional[int] = None,
                           sql_ready_tokens: int = 16) -> tuple:
    """COCONUT curriculum loss, without the inference-time KV cache.

    COCONUT trains by repeatedly running the model on the current sequence to
    obtain the next continuous thought, appending that hidden state as an input
    embedding, while roles not yet internalised remain explicit supervised
    text.  A final teacher-forced forward supervises that explicit suffix and
    the answer, with the prompt and latent thoughts masked. This is deliberately
    separate from ``LatentReasoner.build_prefix``: that method builds the
    inference/GRPO KV prefix, while this method follows the paper's memory-safer
    training shape using ``use_cache=False``.
    """
    import torch

    tok, model = reasoner.bundle.tokenizer, reasoner.bundle.model
    device = reasoner.bundle.device
    cfg = reasoner.cfg
    embed = model.get_input_embeddings()
    chat = render_chat(
        tok,
        [{"role": "system", "content": rec["system"]}, {"role": "user", "content": rec["prompt"]}],
        add_generation_prompt=True,
    )

    pieces = []
    labels = []
    sql_ready_terms = []

    def add_token_ids(ids: Sequence[int], supervise: bool):
        if not ids:
            return
        ids_t = torch.tensor([list(ids)], device=device)
        pieces.append(embed(ids_t))
        labels.extend(list(ids) if supervise else [-100] * len(ids))

    def add_text(text: str, supervise: bool):
        add_token_ids(tok(text, add_special_tokens=False)["input_ids"], supervise)

    add_text(chat, supervise=False)
    explicit_turns = _explicit_curriculum_turns(
        list(rec.get("role_turns") or []), cfg.roles, n_latent_roles
    )
    for idx, role in enumerate(cfg.roles):
        if role == "consensus":
            break
        if idx < n_latent_roles:
            add_text(agent_tag(role) + "\n", supervise=False)
            for _ in range(cfg.latent_steps):
                cur = torch.cat(pieces, dim=1)
                attn = torch.ones(cur.shape[:2], dtype=torch.long, device=device)
                out = model(
                    inputs_embeds=cur,
                    attention_mask=attn,
                    use_cache=False,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1][:, -1:, :]
                pieces.append(reasoner.align(hidden))
                labels.append(-100)
                del out, cur, attn
        else:
            # COCONUT curriculum: roles not yet replaced by continuous
            # thoughts remain explicit and supervised.  In the two-role run,
            # stage 1 therefore learns the candidate-generation trace,
            # including its training-only offline execution outcomes; stage 2
            # replaces that whole turn with latent states.  Dropping this
            # suffix would reduce the objective to SQL-only supervision and
            # the outcome trace would never influence training.
            # role_turns_for returns a complete tagged block, so do not add a
            # second role tag here.
            add_text(explicit_turns[idx] + "\n", supervise=True)

        if idx < n_latent_roles and sql_ready_role and idx + 1 == sql_ready_role and target_ids:
            probe_target = list(target_ids[: max(1, int(sql_ready_tokens))])
            consensus_ids = tok(T_CONSENSUS_OPEN, add_special_tokens=False)["input_ids"]
            probe_pieces = list(pieces)
            probe_labels = list(labels)
            if consensus_ids:
                ids_t = torch.tensor([consensus_ids], device=device)
                probe_pieces.append(embed(ids_t))
                probe_labels.extend([-100] * len(consensus_ids))
            ids_t = torch.tensor([probe_target], device=device)
            probe_pieces.append(embed(ids_t))
            probe_labels.extend(probe_target)
            probe_inputs = torch.cat(probe_pieces, dim=1)
            probe_labels_t = torch.tensor([probe_labels], device=device)
            probe_attn = torch.ones(probe_inputs.shape[:2], dtype=torch.long, device=device)
            probe_out = model(
                inputs_embeds=probe_inputs,
                attention_mask=probe_attn,
                labels=probe_labels_t,
                use_cache=False,
            )
            sql_ready_terms.append(probe_out.loss)
            del probe_out, probe_inputs, probe_labels_t, probe_attn, probe_pieces, probe_labels

    add_text(T_CONSENSUS_OPEN, supervise=False)
    add_token_ids(target_ids, supervise=True)

    inputs_embeds = torch.cat(pieces, dim=1)
    labels_t = torch.tensor([labels], device=device)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        labels=labels_t,
        use_cache=False,
    )
    n_supervised = sum(1 for x in labels if x != -100)
    sql_ready_loss = (
        torch.stack(sql_ready_terms).mean()
        if sql_ready_terms
        else torch.zeros((), device=device, dtype=torch.float32)
    )
    return out.loss, n_supervised, len(cfg.roles) - 1, sql_ready_loss


def role_turns_for(trace: Any, roles: Sequence[str]) -> Optional[List[str]]:
    """Ordered, rendered per-role text matching ``roles`` (excluding the
    trailing "consensus"), or ``None`` if the trace is missing any of the
    required (agent, round) turns - e.g. consensus was reached before a
    second round happened for every agent.
    """
    counts: Dict[str, int] = {}
    by_key = {(t.agent, t.round): t for t in trace.turns}
    out: List[str] = []
    for role in roles:
        if role == "consensus":
            break
        r = counts.get(role, 0)
        counts[role] = r + 1
        turn = by_key.get((role, r))
        if turn is None:
            return None
        out.append(turn.render(with_exec=True))
    return out


def chess_step_turns_for(trace: Any, roles: Sequence[str]) -> Optional[List[str]]:
    """Render two internalisable stages from a correct first-pass CHESS run.

    Value retrieval is deliberately absent: it is an external database/index
    tool whose results belong in the input prompt, not a reasoning role the
    student can internalise.  Execution-driven revision and verification are
    also absent because latent steps cannot invoke SQLite; those signals are
    applied later as GRPO rewards.  Only schema linking and the round-0 CHESS
    candidate are supervised here, and the candidate must already have EX=1.
    """
    chess_turns = [t for t in trace.turns if t.agent == "chess"]
    first_chess = sorted(chess_turns, key=lambda t: t.round)[0] if chess_turns else None
    if not (
        first_chess
        and int(getattr(first_chess, "round", -1)) == 0
        and getattr(first_chess, "correct", None) is True
        and getattr(first_chess, "sql", None)
    ):
        return None

    negative_examples = _offline_execution_negatives(trace, first_chess.sql)

    def block(role: str, body: str) -> Optional[str]:
        body = (body or "").strip()
        if not body:
            return None
        return f"{agent_tag(role)}\n{body}\n{T_AGENT_CLOSE}"

    step_map: Dict[str, Optional[str]] = {
        "schema_linker": block("schema_linker", getattr(trace, "schema_links", "")),
        "candidate_generator": block(
            "candidate_generator",
            (
                "Training-only execution outcomes (not available at inference):\n"
                + "\n\n".join(
                    f"Rejected one-edit mutant {idx}:\n```sql\n{negative['sql']}\n```\n"
                    f"Outcome: {negative['label']} — {negative['detail']}"
                    for idx, negative in enumerate(negative_examples, 1)
                )
                + ("\n\n" if negative_examples else "")
                + "Correct first-pass CHESS candidate:\n"
                + f"{first_chess.reasoning.strip()}\n```sql\n{first_chess.sql}\n```\n"
                + "Outcome: valid — EX=1 on the training database"
                if first_chess and first_chess.sql else ""
            ),
        ),
    }

    out: List[str] = []
    for role in roles:
        if role == "consensus":
            break
        text = step_map.get(role)
        if text is None:
            return None
        out.append(text)
    return out


def build_records_from_traces(traces: Sequence[Any], cfg: Any,
                              max_prompt_chars: Optional[int] = None,
                              max_role_chars: Optional[int] = None,
                              tokenizer: Any = None,
                              max_total_tokens: Optional[int] = None,
                              chain_format: str = "debate_roles") -> List[Dict[str, Any]]:
    """Turn ``DebateTrace`` objects into latent-SFT records, with ``role_turns``
    populated for the curriculum. Traces missing a full set of role turns keep
    ``role_turns=None`` (curriculum falls back to a bare tag for any role it
    cannot render as text - see ``LatentReasoner.build_prefix``).

    ``max_role_chars`` applies the same treatment to any individual role's
    *rendered* turn (``with_exec=True``, so it can carry a large execution
    result table, unlike the prompt): a record with one oversized turn also
    falls back to ``role_turns=None`` rather than being dropped outright,
    since low-curriculum steps buffer every non-latent role's full text into
    one KV cache and an outlier turn there OOMs regardless of GPU headroom.

    ``tokenizer`` + ``max_total_tokens`` add a SECOND, more accurate pass on
    top of the char-based caps: SQL/schema-heavy debate text can tokenize far
    less efficiently than natural language of the same char length (dotted
    ``table.column`` references, quoted literals, comparison operators each
    splitting into multiple sub-word tokens), so a record can slip under both
    char caps and still be the single most expensive one in the whole corpus
    at low curriculum depth - this is what actually happened in practice.
    When given, records whose (prompt + all role turns) TOKEN count exceeds
    ``max_total_tokens`` also fall back to ``role_turns=None``.
    """
    out: List[Dict[str, Any]] = []
    n_missing_turns = 0
    n_oversized_turns = 0
    n_oversized_tokens = 0
    n_truncated_schema = 0
    n_dropped_overflow = 0
    for tr in traces:
        if not tr.gold_sql:
            continue
        from ..db.value_index import retrieve_value_hints

        db_path = str(getattr(tr, "meta", {}).get("db_path", "") or "")
        retrieval_cache = os.path.join(cfg.paths.cache_dir, "value_indexes")
        retrieved_values = retrieve_value_hints(
            db_path, tr.db_id, tr.question, tr.evidence,
            cfg=cfg.mas, cache_dir=retrieval_cache,
        )
        value_hint_text = "\n".join(f"- {value}" for value in retrieved_values)
        sp = student_prompt(
            schema=tr.schema_prompt, question=tr.question, evidence=tr.evidence,
            value_hints=value_hint_text,
        )
        if max_prompt_chars and len(sp["user"]) > max_prompt_chars:
            # Truncate the schema (almost always the largest, most cuttable
            # part - a full CREATE TABLE dump) rather than dropping the
            # record outright, so an otherwise-fine question doesn't get
            # thrown away just because its schema is long. Question/evidence/
            # value hints are short and load-bearing - never touched.
            overage = len(sp["user"]) - max_prompt_chars
            schema_budget = max(0, len(tr.schema_prompt) - overage - 20)
            truncated_schema = tr.schema_prompt[:schema_budget] + "\n-- (schema truncated)"
            sp = student_prompt(
                schema=truncated_schema, question=tr.question, evidence=tr.evidence,
                value_hints=value_hint_text,
            )
            n_truncated_schema += 1
            if len(sp["user"]) > max_prompt_chars:
                # question/evidence/value hints alone already overflow - no
                # amount of schema truncation fixes that, so this one really
                # does have to be dropped.
                n_dropped_overflow += 1
                continue
        if chain_format == "chess_steps":
            rt = chess_step_turns_for(tr, cfg.latent.roles)
        else:
            rt = role_turns_for(tr, cfg.latent.roles)
        if rt is None:
            n_missing_turns += 1
        elif max_role_chars and max(len(t) for t in rt) > max_role_chars:
            rt = None
            n_oversized_turns += 1
        elif tokenizer is not None and max_total_tokens and rt is not None:
            n_prompt_tok = len(tokenizer(sp["user"], add_special_tokens=False)["input_ids"])
            n_role_tok = sum(
                len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in rt
            )
            if n_prompt_tok + n_role_tok > max_total_tokens:
                rt = None
                n_oversized_tokens += 1
        out.append({
            "qid": tr.qid, "system": sp["system"], "prompt": sp["user"],
            "gold_sql": tr.gold_sql, "role_turns": rt,
            "db_id": tr.db_id, "db_path": db_path,
            "question": tr.question, "evidence": tr.evidence,
            "retrieved_values": retrieved_values,
        })
    if n_truncated_schema:
        print(f"[latent_sft] {n_truncated_schema}/{len(out) + n_dropped_overflow} traces had "
              f"their schema truncated to fit max_prompt_chars={max_prompt_chars} "
              f"({n_dropped_overflow} still overflowed on question/evidence alone and were "
              f"dropped)", flush=True)
    if n_missing_turns:
        print(f"[latent_sft] {n_missing_turns}/{len(out)} traces missing a full role-turn set "
              f"(curriculum falls back to a bare tag for those roles)", flush=True)
    if n_oversized_tokens:
        print(f"[latent_sft] {n_oversized_tokens}/{len(out)} traces had prompt+role-turn "
              f"TOKEN count over max_total_tokens={max_total_tokens} (curriculum falls back "
              f"to a bare tag for those roles)", flush=True)
    if n_oversized_turns:
        print(f"[latent_sft] {n_oversized_turns}/{len(out)} traces had a role turn over "
              f"max_role_chars={max_role_chars} (curriculum falls back to a bare tag for those roles)",
              flush=True)
    return out


def train_latent_sft(
    bundle: Any,
    records: Sequence[Dict[str, Any]],
    cfg: Any,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train on (latent-primed state -> correct SQL) pairs.

    ``records`` need ``system``, ``prompt``, ``gold_sql``, and optionally
    ``role_turns`` (see :func:`build_records_from_traces`) for curriculum
    mode. ``cfg`` needs ``latent`` (a LatentConfig), ``lr``, ``epochs``,
    ``grad_accum``, ``logging_steps``, ``save_steps``, ``output_dir``,
    ``seed``, and optionally ``curriculum`` (bool, default False - when True,
    ramps the number of latent roles linearly from 0 to all of them across
    training instead of always running fully latent).
    """
    import torch
    import torch.nn.functional as F

    from .latent import LatentReasoner
    from ..utils.telemetry import RunLog, gpu_stats, reset_gpu_peak, run_environment

    tok, model = bundle.tokenizer, bundle.model
    reasoner = LatentReasoner(bundle, cfg.latent)
    n_roles = len(cfg.latent.roles) - 1  # excludes "consensus"
    curriculum = bool(getattr(cfg, "curriculum", False))

    coconut_objective = bool(getattr(cfg, "coconut_objective", False))
    require_role_turns = bool(getattr(cfg, "require_role_turns", False))
    usable = [r for r in records if r.get("gold_sql")]
    if require_role_turns:
        before = len(usable)
        usable = [r for r in usable if r.get("role_turns") is not None]
        dropped = before - len(usable)
        if dropped:
            print(f"[latent_sft] dropped {dropped}/{before} records without full role-turn text "
                  f"(strict role-turn training)", flush=True)
    if not usable:
        raise ValueError("no records with a gold_sql - nothing to train on")
    if coconut_objective and not curriculum:
        raise ValueError("--coconut-objective requires curriculum=True")
    if coconut_objective:
        cfg.supervise_text = False

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr)

    # Chain mode (scripts/run_curriculum_chain*.sh) trains one fresh process
    # per record/batch to avoid the CUDA/checkpoint state corruption that a
    # long-running single process hit (see the OOM except-clause below) -
    # but a brand-new AdamW every process means zero momentum carries
    # between steps, which is a much noisier optimiser than any real
    # training run (COCONUT included) actually uses. Restoring the previous
    # process' exp_avg/exp_avg_sq/step here makes momentum continuous across
    # the chain the same way it would be within one un-interrupted process,
    # without needing the corruption-prone long-lived process itself.
    optim_state_path = getattr(cfg, "optim_state_path", None)
    if optim_state_path and os.path.exists(optim_state_path):
        optim.load_state_dict(torch.load(optim_state_path, map_location=bundle.device))
        print(f"[latent_sft] restored optimizer momentum from {optim_state_path}", flush=True)

    steplog = None
    if log_path:
        steplog = RunLog(
            os.path.splitext(log_path)[0] + "_steps.jsonl",
            header=run_environment(
                stage="latent_sft", n_records=len(usable), lr=cfg.lr, epochs=cfg.epochs,
                latent_steps=cfg.latent.latent_steps, roles=list(cfg.latent.roles),
                curriculum=curriculum, coconut_objective=coconut_objective,
            ),
        )
    reset_gpu_peak()
    if os.environ.get("LATENTSQL_DEBUG_CTX"):
        print(f"[latent_sft DEBUG] baseline after model+optim load: "
              f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    order = list(range(len(usable)))
    rng = random.Random(getattr(cfg, "seed", 0))
    rng.shuffle(order)
    n_steps = max(1, int(len(order) * cfg.epochs))
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()
    optim.zero_grad(set_to_none=True)
    print(f"[latent_sft] {len(usable)} records | {n_steps} steps | "
          f"latent_steps={cfg.latent.latent_steps} roles={list(cfg.latent.roles)} "
          f"curriculum={curriculum}", flush=True)

    # build_prefix carries the KV cache across roles, which needs use_cache=True
    # and conflicts with HF's own per-layer checkpointing exactly as in GRPO -
    # suspend it for the whole prefix+completion computation, not just the prefix.
    ckpt = bool(getattr(model, "is_gradient_checkpointing", False))
    prev_use_cache = getattr(model.config, "use_cache", True)
    if ckpt and not coconut_objective:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    n_acc = 0
    try:
        for step in range(n_steps):
            if step % len(order) == 0 and step > 0:
                rng.shuffle(order)
            rec = usable[order[step % len(order)]]
            target_ids = (
                _consensus_target_ids(tok, rec["gold_sql"])
                if coconut_objective else _target_ids(tok, rec["gold_sql"])
            )
            if not target_ids:
                continue

            if curriculum:
                forced = getattr(cfg, "force_n_latent_roles", None)
                if forced is not None:
                    # Chain mode (one record per fresh subprocess - see
                    # scripts/run_curriculum_chain.sh): n_steps here is
                    # always 1, so the frac-from-step-count formula below
                    # would collapse to frac=0 for every single record. The
                    # orchestrator computes the record's position in the
                    # GLOBAL curriculum ramp itself and passes the resulting
                    # depth directly instead.
                    n_latent_roles = forced
                else:
                    frac = step / max(1, n_steps - 1)
                    n_latent_roles = round(frac * n_roles)
                role_turns = rec.get("role_turns")
            else:
                n_latent_roles = None  # fully latent throughout - original behaviour
                role_turns = None

            import gc

            decode_compare = bool(getattr(cfg, "decode_compare", False))
            aux_weight = float(getattr(cfg, "aux_weight", 0.5))
            supervise_text = bool(getattr(cfg, "supervise_text", False))
            text_weight = float(getattr(cfg, "text_weight", 1.0))
            sql_ready_role = getattr(cfg, "sql_ready_role", None)
            sql_ready_weight = float(getattr(cfg, "sql_ready_weight", 0.0) or 0.0)
            sql_ready_tokens = int(getattr(cfg, "sql_ready_tokens", 16) or 16)
            sql_ready_active = (
                coconut_objective
                and sql_ready_role is not None
                and sql_ready_weight > 0
                and n_latent_roles is not None
                and n_latent_roles >= int(sql_ready_role)
            )

            try:
                if coconut_objective:
                    base_loss, n_tok, _n_roles, sql_ready_loss = _coconut_sequence_loss(
                        reasoner, rec, n_latent_roles, target_ids,
                        sql_ready_role=(int(sql_ready_role) if sql_ready_active else None),
                        sql_ready_tokens=sql_ready_tokens,
                    )
                    sql_loss = base_loss
                    loss = sql_loss
                    if sql_ready_active:
                        loss = loss + sql_ready_weight * sql_ready_loss
                    aux_loss = torch.zeros((), device=bundle.device, dtype=torch.float32)
                    text_gen_loss = sql_loss
                    past = attn = first_logits = terms = token_logp = ids = None
                else:
                    past, attn, first_logits, aux_loss, text_gen_loss = reasoner.build_prefix(
                        rec["system"], rec["prompt"], grad=True,
                        role_turns=role_turns, n_latent_roles=n_latent_roles,
                        decode_compare=decode_compare, supervise_text=supervise_text,
                    )
                    ids = torch.tensor([target_ids], device=bundle.device)
                    first_lp = F.log_softmax(first_logits.float(), dim=-1)[0, target_ids[0]]
                    terms = [first_lp]
                    if len(target_ids) > 1:
                        attn2 = torch.cat(
                            [attn, torch.ones(1, ids.shape[1] - 1, dtype=attn.dtype, device=attn.device)],
                            dim=-1,
                        )
                        out = model(input_ids=ids[:, :-1], past_key_values=past,
                                    attention_mask=attn2, use_cache=False)
                        logits, tgt = out.logits, ids[:, 1:]
                        parts = []
                        for i in range(0, logits.shape[1], _LOGP_CHUNK):
                            block = F.log_softmax(logits[:, i:i + _LOGP_CHUNK, :].float(), dim=-1)
                            tb = tgt[:, i:i + _LOGP_CHUNK]
                            parts.append(block.gather(-1, tb.unsqueeze(-1)).squeeze(-1))
                        terms.append(torch.cat(parts, dim=1).sum())
                        del out, logits
                    token_logp = torch.stack(
                        [x.reshape(()) if x.dim() == 0 else x.sum() for x in terms]
                    ).sum()
                    n_tok = len(target_ids)
                    sql_loss = -token_logp / n_tok  # mean NLL per token
                    loss = sql_loss
                    if decode_compare:
                        loss = loss + aux_weight * aux_loss
                    if supervise_text:
                        loss = loss + text_weight * text_gen_loss
                    sql_ready_loss = torch.zeros((), device=bundle.device, dtype=torch.float32)

                (loss / cfg.grad_accum).backward()
            except torch.cuda.OutOfMemoryError as e:
                # Empirically NOT a recoverable-and-continue situation: two
                # full runs, one that only stashed str(e) here vs. one that
                # additionally gc.collect()/empty_cache()'d right in this
                # except block, both left the SAME ~16.5GB permanently
                # resident from this step onward - every remaining step then
                # OOMs identically regardless of any Python-side cleanup. So
                # this is not a live-reference leak reachable from Python at
                # all; something inside PyTorch/CUDA's own state (most
                # likely the checkpoint's non-reentrant recompute machinery,
                # or the caching allocator's fragmentation bookkeeping)
                # is left corrupted once a backward pass OOMs mid-flight
                # through a checkpointed region, for the rest of THIS
                # process's life. Silently skipping just burns the rest of
                # the run producing a checkpoint trained on a handful of
                # steps while looking like it "completed" (exit code 0) -
                # worse than failing loudly. Surface it and stop; the real
                # fix is preventing the OOM (tighter, token-based record
                # filtering), not surviving it.
                print(f"[latent_sft] step {step + 1} OOM on qid={rec.get('qid', '')} "
                      f"- NOT recoverable in-process, aborting run: {e}", flush=True)
                raise
            n_acc += 1
            loss_value = float(loss.detach())
            sql_loss_value = float(sql_loss.detach())
            aux_loss_value = float(aux_loss.detach())
            sql_ready_loss_value = float(sql_ready_loss.detach())
            text_loss_value = float(text_gen_loss.detach())
            del past, attn, first_logits, terms, token_logp, loss, sql_loss, aux_loss, sql_ready_loss, text_gen_loss, ids
            # empty_cache() alone does not reclaim tensors still held by a
            # Python reference cycle (autograd graphs commonly form one via
            # grad_fn <-> saved tensors) - only a real gc pass breaks those.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if n_acc % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if (step + 1) % cfg.logging_steps == 0 or step + 1 == n_steps:
                entry = {
                    "step": step + 1, "loss": loss_value,
                    "sql_loss": sql_loss_value,
                    "aux_loss": aux_loss_value if decode_compare else None,
                    "sql_ready_loss": sql_ready_loss_value if sql_ready_active else None,
                    "text_loss": text_loss_value if supervise_text else None,
                    "qid": rec.get("qid", ""), "n_target_tokens": n_tok,
                    "n_latent_roles": n_latent_roles if curriculum else n_roles,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                entry.update(gpu_stats())
                history.append(entry)
                if steplog:
                    steplog.write({"record": "step", **entry})
                print(f"[latent_sft] {entry}", flush=True)

            if cfg.save_steps and (step + 1) % cfg.save_steps == 0:
                from .sft import save_student

                save_student(bundle, cfg.output_dir)
    finally:
        if ckpt and not coconut_objective:
            model.gradient_checkpointing_enable()
        model.config.use_cache = prev_use_cache

    from .sft import save_student

    save_student(bundle, cfg.output_dir)
    if optim_state_path:
        os.makedirs(os.path.dirname(os.path.abspath(optim_state_path)) or ".", exist_ok=True)
        torch.save(optim.state_dict(), optim_state_path)
    result = {
        "output_dir": cfg.output_dir,
        "n_records": len(usable),
        "steps": n_steps,
        "curriculum": curriculum,
        "coconut_objective": coconut_objective,
        "elapsed_s": round(time.time() - t0, 1),
        "final_loss": history[-1]["loss"] if history else None,
        **gpu_stats(),
        "history": history,
    }
    if steplog:
        steplog.write({"record": "summary", **{k: v for k, v in result.items() if k != "history"}})
        steplog.close()
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result
