#!/usr/bin/env python3
"""Probe intermediate latent roles without changing training or evaluation.

For each non-consensus role this script saves:
  * the LM-head top-token projection of the role's final latent state;
  * SQL decoded after forcing consensus at that depth;
  * referenced tables/columns and execution status/correctness;
  * a diff from the preceding role's SQL; and
  * free text decoded after an EOT marker instead of consensus.

The projection and EOT continuation are diagnostics, not faithful decoders of
all information contained in a continuous latent state.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.data.loaders import load_examples
from latentsql.db.execute import execute_sql, execution_accuracy
from latentsql.db.schema import load_schema_cached, render_schema
from latentsql.db.value_index import retrieve_value_hints
from latentsql.distill.latent import LatentReasoner
from latentsql.llm.hf_local import load_model, render_chat, resolve_model_path
from latentsql.mas.prompts import student_prompt
from latentsql.mas.trace import T_CONSENSUS_OPEN, agent_tag, extract_final_sql
from latentsql.utils.common import resolve_examples, set_seed


def _greedy_decode(reasoner: LatentReasoner, past: Any, attention_mask: Any,
                   next_logits: Any, max_new_tokens: int) -> tuple[str, int]:
    import torch

    model, tok = reasoner.bundle.model, reasoner.bundle.tokenizer
    out_ids: list[int] = []
    inputs_embeds = None
    for _ in range(max(0, max_new_tokens)):
        logits = next_logits
        if logits is None:
            out = model(
                inputs_embeds=inputs_embeds,
                past_key_values=past,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        token_id = int(nxt[0, 0])
        if token_id == tok.eos_token_id:
            break
        out_ids.append(token_id)
        inputs_embeds = model.get_input_embeddings()(nxt)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
        )
        next_logits = None
    return tok.decode(out_ids, skip_special_tokens=True), len(out_ids)


def _state_after_role(reasoner: LatentReasoner, system: str, prompt: str,
                      role_depth: int) -> tuple[Any, Any, Any]:
    """Rebuild the untouched KV state immediately after ``role_depth`` roles."""
    import torch

    tok = reasoner.bundle.tokenizer
    chat = render_chat(
        tok,
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    enc = tok(chat, return_tensors="pt", add_special_tokens=False)
    attention_mask = torch.ones(
        1, 0, dtype=enc["attention_mask"].dtype, device=reasoner.bundle.device
    )
    past = None
    with torch.no_grad():
        past, attention_mask, _hidden, _loss, next_logits = reasoner._text_steps(
            [chat], past, attention_mask, checkpoint=False, supervise=False,
            max_total_tokens=getattr(reasoner.cfg, "max_context_tokens", None),
            need_hidden=False,
        )
        for role in [r for r in reasoner.cfg.roles if r != "consensus"][:role_depth]:
            past, attention_mask, hidden, _loss, next_logits = reasoner._text_steps(
                [agent_tag(role) + "\n"], past, attention_mask, checkpoint=False,
                supervise=False, prev_logits=next_logits,
                max_total_tokens=getattr(reasoner.cfg, "max_context_tokens", None),
                need_hidden=True,
            )
            latent_input = reasoner.align(hidden[:, -1:, :])
            _next_embed, past, attention_mask, next_logits, _aux = reasoner._latent_steps(
                latent_input, past, attention_mask, reasoner.cfg.latent_steps
            )
            if not reasoner.cfg.share_working_memory:
                past = None
                attention_mask = torch.ones(
                    1, 1, dtype=attention_mask.dtype, device=attention_mask.device
                )
    return past, attention_mask, next_logits


def _top_projection(logits: Any, tokenizer: Any, top_k: int) -> list[dict[str, Any]]:
    import torch

    vector = logits[0].float()
    probs = torch.softmax(vector, dim=-1)
    values, indices = probs.topk(min(top_k, probs.shape[-1]))
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "probability": float(probability),
            "logit": float(vector[int(token_id)]),
        }
        for probability, token_id in zip(values, indices)
    ]


def _sql_references(sql: Optional[str]) -> tuple[list[str], list[str], Optional[str]]:
    if not sql:
        return [], [], None
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, read="sqlite")
        tables = sorted({node.name for node in tree.find_all(exp.Table) if node.name})
        columns = sorted({node.sql(dialect="sqlite") for node in tree.find_all(exp.Column)})
        return tables, columns, None
    except Exception as exc:
        return [], [], f"{type(exc).__name__}: {exc}"


def _diff(before: Optional[str], after: Optional[str]) -> list[str]:
    if before is None:
        return []
    return list(
        difflib.unified_diff(
            before.splitlines(), (after or "").splitlines(),
            fromfile="previous_role", tofile="current_role", lineterm="",
        )
    )


def _probe_one_role(reasoner: LatentReasoner, system: str, prompt: str,
                    db_path: str, gold_sql: str, role: str, depth: int,
                    previous_sql: Optional[str], top_k: int, sql_tokens: int,
                    verbalize_tokens: int, eot_text: str) -> dict[str, Any]:
    import torch

    # Branch 1: projection and SQL decode after forcing consensus.
    past, mask, latent_logits = _state_after_role(reasoner, system, prompt, depth)
    projection = _top_projection(latent_logits, reasoner.bundle.tokenizer, top_k)
    with torch.no_grad():
        past, mask, _hidden, _loss, consensus_logits = reasoner._text_steps(
            [T_CONSENSUS_OPEN], past, mask, checkpoint=False, supervise=False,
            prev_logits=latent_logits,
            max_total_tokens=getattr(reasoner.cfg, "max_context_tokens", None),
            need_hidden=False,
        )
        sql_text, sql_n_tokens = _greedy_decode(
            reasoner, past, mask, consensus_logits, sql_tokens
        )
    sql = extract_final_sql(T_CONSENSUS_OPEN + sql_text)
    tables, columns, parse_error = _sql_references(sql)
    execution = execute_sql(db_path, sql or "", timeout=30.0)
    correct, _pred_result, _gold_result = execution_accuracy(
        db_path, sql, gold_sql, timeout=30.0
    )

    # Branch 2: rebuild the same state so EOT probing cannot alter branch 1.
    past, mask, latent_logits = _state_after_role(reasoner, system, prompt, depth)
    with torch.no_grad():
        past, mask, _hidden, _loss, eot_logits = reasoner._text_steps(
            [eot_text], past, mask, checkpoint=False, supervise=False,
            prev_logits=latent_logits,
            max_total_tokens=getattr(reasoner.cfg, "max_context_tokens", None),
            need_hidden=False,
        )
        verbalization, verbalization_n_tokens = _greedy_decode(
            reasoner, past, mask, eot_logits, verbalize_tokens
        )

    return {
        "role_index": depth,
        "role": role,
        "latent_steps_per_role": int(reasoner.cfg.latent_steps),
        "cumulative_latent_steps": depth * int(reasoner.cfg.latent_steps),
        "top_vocabulary": projection,
        "decoded_sql": sql,
        "decoded_sql_tokens": sql_n_tokens,
        "tables": tables,
        "columns": columns,
        "sql_parse_error": parse_error,
        "execution_ok": execution.ok,
        "execution_error": execution.error or None,
        "execution_preview": execution.preview(),
        "execution_correct": bool(correct),
        "changed_from_previous_role": previous_sql is not None and sql != previous_sql,
        "sql_diff": _diff(previous_sql, sql),
        "eot_marker": eot_text,
        "eot_verbalization": verbalization,
        "eot_verbalization_tokens": verbalization_n_tokens,
        "interpretation_warning": (
            "Projection and verbalization are probes, not faithful decoders of all "
            "information or alternatives represented by the latent state."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    ap.add_argument("--latent-steps", type=int, default=None)
    ap.add_argument("--roles", default=None, help="comma-separated roles including consensus")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--sql-tokens", type=int, default=200)
    ap.add_argument("--verbalize-tokens", type=int, default=128)
    ap.add_argument("--eot-text", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    if args.latent_steps is not None:
        cfg.latent.latent_steps = args.latent_steps
    if args.roles:
        cfg.latent.roles = [role.strip() for role in args.roles.split(",") if role.strip()]

    if args.heldout:
        examples = load_examples(args.heldout)[:args.limit]
    else:
        examples, _used = resolve_examples(
            cfg, dataset=args.dataset, limit=args.limit, split="dev"
        )

    base = args.base or cfg.sft.model_name_or_path
    base_path = resolve_model_path(base, hub_dir=cfg.paths.hf_hub, local_only=False)
    adapter = args.student if os.path.exists(
        os.path.join(args.student, "adapter_config.json")
    ) else None
    if adapter is None:
        raise SystemExit(f"no adapter_config.json under {args.student}")
    bundle = load_model(
        base_path, device="auto", hub_dir=cfg.paths.hf_hub,
        adapter_path=adapter, quantization=args.quantization,
    )
    reasoner = LatentReasoner(bundle, cfg.latent)
    eot_text = args.eot_text
    if eot_text is None:
        eot_text = (bundle.tokenizer.eos_token or "<|im_end|>") + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    non_consensus_roles = [role for role in cfg.latent.roles if role != "consensus"]
    # Stream one line per role while running (crash-safe), then write the
    # human-readable indented JSON to --out and drop the stream.
    partial = args.out + ".partial.jsonl"
    records: list[dict] = []
    with open(partial, "w", encoding="utf-8") as fh:
        for example_index, ex in enumerate(examples, 1):
            schema = render_schema(load_schema_cached(ex.db_path, ex.db_id), style="code")
            retrieved_values = retrieve_value_hints(
                ex.db_path, ex.db_id, ex.question, ex.evidence,
                cfg=cfg.mas,
                cache_dir=os.path.join(cfg.paths.cache_dir, "value_indexes"),
            )
            prompt = student_prompt(
                schema=schema, question=ex.question, evidence=ex.evidence,
                value_hints="\n".join(f"- {value}" for value in retrieved_values),
            )
            previous_sql = None
            for depth, role in enumerate(non_consensus_roles, 1):
                record = _probe_one_role(
                    reasoner, prompt["system"], prompt["user"], ex.db_path,
                    ex.gold_sql, role, depth, previous_sql, args.top_k,
                    args.sql_tokens, args.verbalize_tokens, eot_text,
                )
                record.update({
                    "qid": ex.qid,
                    "dataset": getattr(ex, "dataset", ""),
                    "db_id": ex.db_id,
                    "question": ex.question,
                    "gold_sql": ex.gold_sql,
                    "retrieved_values": retrieved_values,
                })
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(record)
                previous_sql = record["decoded_sql"]
                print(
                    f"[latent-probe] {example_index}/{len(examples)} {ex.qid} "
                    f"role={depth}/{len(non_consensus_roles)} {role} "
                    f"exec={record['execution_ok']} correct={record['execution_correct']}",
                    flush=True,
                )
    write_readable(records, args.out)
    os.remove(partial)
    print(f"wrote {args.out}")


_QUESTION_KEYS = ("qid", "dataset", "db_id", "question", "gold_sql", "retrieved_values")


def write_readable(records: list[dict], path: str) -> None:
    """Group per-role records by question and write them as indent=4 JSON."""
    questions: list[dict] = []
    by_qid: dict[str, dict] = {}
    for rec in records:
        qid = rec["qid"]
        if qid not in by_qid:
            by_qid[qid] = {k: rec.get(k) for k in _QUESTION_KEYS}
            by_qid[qid]["roles"] = []
            questions.append(by_qid[qid])
        by_qid[qid]["roles"].append({k: v for k, v in rec.items() if k not in _QUESTION_KEYS})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(questions, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
