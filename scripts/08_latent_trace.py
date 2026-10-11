#!/usr/bin/env python3
"""Step-by-step inspection of the latent debate, COCONUT-style, plus interventions.

The latent sequence for the two-role student is

    prompt || <schema_linker> z1 z2 || <candidate_generator> z3 z4 || <|consensus|> SQL

For every continuous thought z_t this script branches off the *untouched*
sequence and records:

  * candidates  - a fixed candidate set of SQL queries (gold, every model's
    eval prediction, every step's own decode), each scored by
    log P(candidate | prefix up to z_t, <|consensus|>) and renormalised over the
    set, with execution correctness. This is the analogue of COCONUT's
    ProsQA analysis (Hao et al., 2024, Sec. 5): which candidates a latent state
    supports, and whether that support concentrates on correct ones as steps
    advance.
  * sql         - greedy SQL decoded after forcing <|consensus|> at z_t, executed
    against gold (intermediate SQL execution).
  * language    - text decoded after forcing the next role's tag, i.e. switching
    back to language mode (the explicit turns earlier curriculum stages trained).
  * top_tokens  - LM-head projection of the state right after z_t.

Step 0 is the prompt plus the schema-linker tag with no latent yet; its
language branch is what the model writes for the schema linker in words.

Decodable information is not proof of use, so each z_t is also intervened on
(the remaining thoughts are recomputed from the edited state):

  * swap   - z_t replaced by the same step's thought from another question;
  * random - z_t replaced by a random vector with the same norm.

If the final SQL changes or turns wrong, the model's answer depended on z_t.

``--mode verbalize`` skips candidates and interventions and instead appends a
natural-language cue right after each z_t (no role tag) and decodes greedily:

  * plain        - " Relevant tables are" (the model may close the turn early);
  * plain_forced - the same cue with closing tags blocked for the first
    ``--min-text-tokens`` tokens, so the thought is not cut off by the end tag;
  * reviser      - "Execute the draft once.", the opening of the reviser turn
    in the r3 corpus, also with closing tags blocked. Its continuation shows
    whether the thought forecasts an error ("Error:", "Diagnosis and fix") or
    keeps the draft ("keep it").

The forecast is compared with what really happens: the draft is the SQL forced
after the last candidate-generator thought, executed against the database.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.data.loaders import load_examples
from latentsql.db.execute import execution_accuracy
from latentsql.db.schema import load_schema_cached, render_schema
from latentsql.db.value_index import retrieve_value_hints
from latentsql.distill.latent import LatentReasoner
from latentsql.llm.hf_local import load_model, render_chat, resolve_model_path
from latentsql.mas.prompts import student_prompt
from latentsql.mas.trace import (T_AGENT_CLOSE, T_CONSENSUS_CLOSE, T_CONSENSUS_OPEN, T_END,
                                  agent_tag, extract_final_sql)
from latentsql.utils.common import resolve_examples, set_seed


class LatentTracer:
    """Builds the latent sequence as input embeddings so any z_t can be edited."""

    def __init__(self, reasoner: LatentReasoner, sql_tokens: int, text_tokens: int):
        self.r = reasoner
        self.model = reasoner.bundle.model
        self.tok = reasoner.bundle.tokenizer
        self.device = reasoner.bundle.device
        self.embed = self.model.get_input_embeddings()
        self.roles = [x for x in reasoner.cfg.roles if x != "consensus"]
        self.m = int(reasoner.cfg.latent_steps)
        self.sql_tokens = sql_tokens
        self.text_tokens = text_tokens

    # -- sequence construction ---------------------------------------------
    def _embed_text(self, text: str):
        import torch

        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return self.embed(torch.tensor([ids], device=self.device))

    def _last_hidden(self, pieces: List[Any]):
        import torch

        cur = torch.cat(pieces, dim=1)
        out = self.model(inputs_embeds=cur, use_cache=False, output_hidden_states=True)
        return out.hidden_states[-1][:, -1:, :], out.logits[:, -1, :]

    def build(self, system: str, prompt: str,
              replace: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
        """Run every latent step; ``replace[t]`` overrides thought z_t (1-based).

        Returns the embedding pieces after each step (``prefixes[t]`` ends with
        z_t; ``prefixes[0]`` ends with the first role tag), the thoughts, and the
        next-token logits after each step.
        """
        replace = replace or {}
        chat = render_chat(
            self.tok,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        pieces = [self._embed_text(chat)]
        prefixes: Dict[int, List[Any]] = {}
        thoughts: Dict[int, Any] = {}
        roles_at: Dict[int, str] = {}
        t = 0
        for ri, role in enumerate(self.roles):
            pieces.append(self._embed_text(agent_tag(role) + "\n"))
            if ri == 0:
                prefixes[0] = list(pieces)
                roles_at[0] = role
            for _ in range(self.m):
                t += 1
                hidden, _logits = self._last_hidden(pieces)
                z = self.r.align(hidden)
                if t in replace:
                    z = replace[t].to(z.dtype)
                pieces.append(z)
                thoughts[t] = z
                prefixes[t] = list(pieces)
                roles_at[t] = role
        return {"prefixes": prefixes, "thoughts": thoughts, "roles_at": roles_at}

    def next_role_after(self, t: int) -> Optional[str]:
        """Role whose tag follows z_t in the normal sequence (None = consensus)."""
        if t == 0:
            return self.roles[0]
        role_idx = (t - 1) // self.m
        if t % self.m:            # mid-role: the next explicit step is this role
            return self.roles[role_idx]
        return self.roles[role_idx + 1] if role_idx + 1 < len(self.roles) else None

    # -- branches ------------------------------------------------------------
    def _end_ids(self) -> List[int]:
        """Single-token closing/role tags and EOS, blocked while min_tokens applies."""
        if not hasattr(self, "_end_cache"):
            tags = [T_AGENT_CLOSE, T_CONSENSUS_OPEN, T_CONSENSUS_CLOSE, T_END]
            tags += [agent_tag(r) for r in self.r.cfg.roles]
            ids = {self.tok.eos_token_id}
            for tag in tags:
                enc = self.tok(tag, add_special_tokens=False)["input_ids"]
                if len(enc) == 1:
                    ids.add(enc[0])
            self._end_cache = sorted(i for i in ids if i is not None)
        return self._end_cache

    def greedy(self, pieces: List[Any], suffix: str, max_tokens: int,
               stop: Sequence[str] = (), min_tokens: int = 0) -> str:
        import torch

        seq = list(pieces) + ([self._embed_text(suffix)] if suffix else [])
        cur = torch.cat(seq, dim=1)
        out = self.model(inputs_embeds=cur, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        ids: List[int] = []
        for _ in range(max_tokens):
            if len(ids) < min_tokens:
                logits = logits.clone()
                logits[:, self._end_ids()] = float("-inf")
            nxt = int(logits.argmax(-1))
            if nxt == self.tok.eos_token_id:
                break
            ids.append(nxt)
            if stop and any(s in self.tok.decode(ids[-12:], skip_special_tokens=False)
                            for s in stop):
                break
            out = self.model(input_ids=torch.tensor([[nxt]], device=self.device),
                             past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
        return self.tok.decode(ids, skip_special_tokens=False)

    def sql_after(self, pieces: List[Any]) -> Optional[str]:
        text = self.greedy(pieces, T_CONSENSUS_OPEN, self.sql_tokens,
                           stop=("<|/consensus|>", "<|endofdebate|>"))
        return extract_final_sql(T_CONSENSUS_OPEN + text)

    def language_after(self, pieces: List[Any], t: int) -> Dict[str, Any]:
        role = self.next_role_after(t)
        if role is None:
            return {"forced_tag": None, "text": None}
        tag = agent_tag(role) + "\n"
        # step 0 already ends with the schema-linker tag
        suffix = "" if t == 0 else tag
        text = self.greedy(pieces, suffix, self.text_tokens,
                           stop=("<|/agent|>", "<|consensus|>", "<|endofdebate|>"))
        return {"forced_tag": tag.strip(), "text": text}

    def logp(self, pieces: List[Any], sql: str) -> float:
        """log P("```sql ... ```" | pieces, <|consensus|>), summed over tokens."""
        import torch
        import torch.nn.functional as F

        head = self._embed_text(T_CONSENSUS_OPEN)
        target = self.tok(f"\n```sql\n{sql.strip()}\n```", add_special_tokens=False)["input_ids"]
        tgt = torch.tensor([target], device=self.device)
        cur = torch.cat(list(pieces) + [head, self.embed(tgt)], dim=1)
        logits = self.model(inputs_embeds=cur, use_cache=False).logits
        n = len(target)
        pred = logits[0, -n - 1:-1, :].float()
        return float(F.log_softmax(pred, -1).gather(-1, tgt[0].unsqueeze(-1)).sum())

    def top_tokens(self, pieces: List[Any], k: int) -> List[Dict[str, Any]]:
        import torch

        _h, logits = self._last_hidden(pieces)
        probs = torch.softmax(logits[0].float(), -1)
        vals, idx = probs.topk(k)
        return [{"token": self.tok.decode([int(i)]), "probability": round(float(v), 4)}
                for v, i in zip(vals, idx)]


def _softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def _entropy(ps: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in ps if p > 0)


def _correct(db_path: str, sql: Optional[str], gold: str) -> bool:
    if not sql:
        return False
    ok, _p, _g = execution_accuracy(db_path, sql, gold, timeout=30.0)
    return bool(ok)


def _load_preds(spec: Sequence[str]) -> Dict[str, Dict[str, str]]:
    preds: Dict[str, Dict[str, str]] = {}
    for item in spec:
        name, path = item.split("=", 1)
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("pred_sql"):
                preds.setdefault(rec["qid"], {})[name] = rec["pred_sql"]
    return preds


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--dataset", default="mix")
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    ap.add_argument("--latent-steps", type=int, default=None)
    ap.add_argument("--roles", default=None, help="comma-separated roles including consensus")
    ap.add_argument("--pred", action="append", default=[],
                    help="name=preds_latent.jsonl; predictions added to the candidate set")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--sql-tokens", type=int, default=200)
    ap.add_argument("--text-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("full", "verbalize"), default="full")
    ap.add_argument("--min-text-tokens", type=int, default=30,
                    help="verbalize: closing tags blocked for this many tokens")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    if args.latent_steps is not None:
        cfg.latent.latent_steps = args.latent_steps
    if args.roles:
        cfg.latent.roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    if args.heldout:
        examples = load_examples(args.heldout)[:args.limit]
    else:
        examples, _ = resolve_examples(cfg, dataset=args.dataset, limit=args.limit, split="dev")

    base_path = resolve_model_path(args.base or cfg.sft.model_name_or_path,
                                   hub_dir=cfg.paths.hf_hub, local_only=False)
    if not os.path.exists(os.path.join(args.student, "adapter_config.json")):
        raise SystemExit(f"no adapter_config.json under {args.student}")
    bundle = load_model(base_path, device="auto", hub_dir=cfg.paths.hf_hub,
                        adapter_path=args.student, quantization=args.quantization)
    bundle.model.eval()
    tracer = LatentTracer(LatentReasoner(bundle, cfg.latent), args.sql_tokens, args.text_tokens)
    if args.mode == "verbalize":
        run_verbalize(args, cfg, tracer, examples)
        return
    preds = _load_preds(args.pred)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    partial = os.path.join(args.out, "trace.partial.jsonl")
    records: List[Dict[str, Any]] = []
    prev_thoughts: Optional[Dict[int, Any]] = None

    with open(partial, "w", encoding="utf-8") as fh, torch.no_grad():
        for qi, ex in enumerate(examples, 1):
            schema = render_schema(load_schema_cached(ex.db_path, ex.db_id), style="code")
            values = retrieve_value_hints(
                ex.db_path, ex.db_id, ex.question, ex.evidence, cfg=cfg.mas,
                cache_dir=os.path.join(cfg.paths.cache_dir, "value_indexes"),
            )
            prompt = student_prompt(schema=schema, question=ex.question, evidence=ex.evidence,
                                    value_hints="\n".join(f"- {v}" for v in values))
            system, user = prompt["system"], prompt["user"]
            run = tracer.build(system, user)
            steps = sorted(run["prefixes"])
            final_t = steps[-1]

            # branches on the untouched sequence
            step_recs: List[Dict[str, Any]] = []
            for t in steps:
                pieces = run["prefixes"][t]
                sql = tracer.sql_after(pieces) if t > 0 else None
                step_recs.append({
                    "step": t,
                    "after": "role tag, no latent yet" if t == 0
                             else f"z{t} ({run['roles_at'][t]}, thought {(t - 1) % tracer.m + 1}/{tracer.m})",
                    "top_tokens": tracer.top_tokens(pieces, args.top_k),
                    "sql": sql,
                    "sql_correct": _correct(ex.db_path, sql, ex.gold_sql) if t > 0 else None,
                    "language": tracer.language_after(pieces, t),
                })
            final_sql = step_recs[-1]["sql"]

            # candidate set: gold, model predictions, every step's decode
            cands: Dict[str, Dict[str, Any]] = {}

            def add(sql: Optional[str], source: str) -> None:
                if not sql:
                    return
                key = " ".join(sql.split())
                if key not in cands:
                    cands[key] = {"sql": sql, "sources": [],
                                  "correct": _correct(ex.db_path, sql, ex.gold_sql)}
                if source not in cands[key]["sources"]:
                    cands[key]["sources"].append(source)

            add(ex.gold_sql, "gold")
            for name, sql in sorted(preds.get(ex.qid, {}).items()):
                add(sql, f"model:{name}")
            for rec in step_recs:
                add(rec["sql"], f"decode@z{rec['step']}")
            cand_list = list(cands.values())
            for rec in step_recs:
                if rec["step"] == 0:
                    continue
                lps = [tracer.logp(run["prefixes"][rec["step"]], c["sql"]) for c in cand_list]
                ps = _softmax(lps)
                rec["candidates"] = sorted(
                    [{"probability": round(p, 4), "logp": round(lp, 3), "correct": c["correct"],
                      "sources": c["sources"], "sql": c["sql"]}
                     for p, lp, c in zip(ps, lps, cand_list)],
                    key=lambda d: -d["probability"],
                )
                rec["p_correct"] = round(sum(p for p, c in zip(ps, cand_list) if c["correct"]), 4)
                rec["entropy"] = round(_entropy(ps), 4)

            # interventions: edit z_t, recompute later thoughts, decode final SQL
            final_lp = tracer.logp(run["prefixes"][final_t], final_sql) if final_sql else None
            interventions: List[Dict[str, Any]] = []
            for t in steps:
                if t == 0:
                    continue
                z = run["thoughts"][t]
                edits = {}
                if prev_thoughts is not None:
                    edits["swap"] = prev_thoughts[t]
                noise = torch.randn(z.shape, generator=gen).to(z.device, torch.float32)
                edits["random"] = (noise / noise.norm() * z.float().norm()).to(z.dtype)
                for kind, z_new in edits.items():
                    alt = tracer.build(system, user, replace={t: z_new})
                    alt_sql = tracer.sql_after(alt["prefixes"][final_t])
                    alt_lp = (tracer.logp(alt["prefixes"][final_t], final_sql)
                              if final_sql else None)
                    interventions.append({
                        "step": t,
                        "kind": kind,
                        "final_sql": alt_sql,
                        "changed": " ".join((alt_sql or "").split())
                                   != " ".join((final_sql or "").split()),
                        "correct": _correct(ex.db_path, alt_sql, ex.gold_sql),
                        "delta_logp_original_answer": (round(alt_lp - final_lp, 3)
                                                       if final_lp is not None else None),
                    })
            prev_thoughts = {t: z.detach().clone() for t, z in run["thoughts"].items()}

            record = {
                "qid": ex.qid, "dataset": getattr(ex, "dataset", ""), "db_id": ex.db_id,
                "question": ex.question, "gold_sql": ex.gold_sql,
                "final_sql": final_sql, "final_correct": step_recs[-1]["sql_correct"],
                "steps": step_recs, "interventions": interventions,
            }
            records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[latent-trace] {qi}/{len(examples)} {ex.qid} final_correct="
                  f"{record['final_correct']} p_correct="
                  f"{[s.get('p_correct') for s in step_recs if s['step']]}", flush=True)

    summary = summarise(records)
    with open(os.path.join(args.out, "trace.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(render_md(summary))
    os.remove(partial)
    print(render_md(summary))
    print(f"wrote {args.out}/trace.json, summary.json, summary.md")


CUES = {
    "plain": (" Relevant tables are", False),
    "plain_forced": (" Relevant tables are", True),
    "reviser": ("\nExecute the draft once.\n", True),
}
TEXT_STOP = ("<|/agent|>", "<|consensus|>", "<|endofdebate|>")


def _exec_status(db_path: str, sql: Optional[str]) -> str:
    from latentsql.db.execute import execute_sql

    if not sql:
        return "no_sql"
    res = execute_sql(db_path, sql, timeout=30.0)
    if not res.ok:
        return "error"
    return "empty" if not res.rows else "rows"


def _forecast(text: Optional[str]) -> Dict[str, str]:
    """What the reviser-cued text predicts about the draft."""
    t = text or ""
    if "Error" in t:
        exec_pred = "error"
    elif "Result: empty" in t:
        exec_pred = "empty"
    elif "Result preview" in t:
        exec_pred = "rows"
    else:
        exec_pred = "none"
    if "Diagnosis and fix" in t:
        verdict = "fix"
    elif "keep it" in t:
        verdict = "keep"
    else:
        verdict = "none"
    return {"exec": exec_pred, "verdict": verdict}


def _prompt_for(cfg, ex) -> Dict[str, str]:
    schema = render_schema(load_schema_cached(ex.db_path, ex.db_id), style="code")
    values = retrieve_value_hints(
        ex.db_path, ex.db_id, ex.question, ex.evidence, cfg=cfg.mas,
        cache_dir=os.path.join(cfg.paths.cache_dir, "value_indexes"),
    )
    return student_prompt(schema=schema, question=ex.question, evidence=ex.evidence,
                          value_hints="\n".join(f"- {v}" for v in values))


def run_verbalize(args, cfg, tracer: "LatentTracer", examples) -> None:
    import torch

    roles = tracer.roles
    draft_role = ("candidate_generator" if "candidate_generator" in roles
                  else roles[max(0, len(roles) - 2)])
    os.makedirs(args.out, exist_ok=True)
    partial = os.path.join(args.out, "verbalize.partial.jsonl")
    records: List[Dict[str, Any]] = []
    with open(partial, "w", encoding="utf-8") as fh, torch.no_grad():
        for qi, ex in enumerate(examples, 1):
            prompt = _prompt_for(cfg, ex)
            run = tracer.build(prompt["system"], prompt["user"])
            steps = [t for t in sorted(run["prefixes"]) if t > 0]
            draft_t = max(t for t in steps if run["roles_at"][t] == draft_role)
            draft_sql = tracer.sql_after(run["prefixes"][draft_t])
            final_sql = tracer.sql_after(run["prefixes"][steps[-1]])
            step_recs = []
            for t in steps:
                pieces = run["prefixes"][t]
                cues, raw = {}, {}
                for name, (cue, forced) in CUES.items():
                    out = tracer.greedy(pieces, cue, args.text_tokens, stop=TEXT_STOP,
                                        min_tokens=args.min_text_tokens if forced else 0)
                    raw[name] = cue + out
                    # cut at the first tag-like piece: with closing tags blocked
                    # the model may spell fake ones ("<|end|>") and ramble on
                    cues[name] = (cue + out.split("<|", 1)[0]).rstrip()
                step_recs.append({
                    "step": t,
                    "after": f"z{t} ({run['roles_at'][t]}, thought "
                             f"{(t - 1) % tracer.m + 1}/{tracer.m})",
                    "role": run["roles_at"][t],
                    "text": cues,
                    "raw_text": raw,
                    "reviser_forecast": _forecast(cues["reviser"]),
                })
            record = {
                "qid": ex.qid, "dataset": getattr(ex, "dataset", ""), "db_id": ex.db_id,
                "question": ex.question, "evidence": ex.evidence, "gold_sql": ex.gold_sql,
                "draft": {"after": f"z{draft_t}", "sql": draft_sql,
                          "exec": _exec_status(ex.db_path, draft_sql),
                          "correct": _correct(ex.db_path, draft_sql, ex.gold_sql)},
                "final": {"after": f"z{steps[-1]}", "sql": final_sql,
                          "correct": _correct(ex.db_path, final_sql, ex.gold_sql)},
                "steps": step_recs,
            }
            records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            last = step_recs[-1]["reviser_forecast"]
            print(f"[verbalize] {qi}/{len(examples)} {ex.qid} draft={record['draft']['exec']}/"
                  f"{record['draft']['correct']} final={record['final']['correct']} "
                  f"forecast={last}", flush=True)

    summary = summarise_verbalize(records)
    with open(os.path.join(args.out, "verbalize.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    with open(os.path.join(args.out, "verbalize_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    md = render_verbalize_md(summary, records)
    with open(os.path.join(args.out, "verbalize_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    os.remove(partial)
    print(md)
    print(f"wrote {args.out}/verbalize.json, verbalize_summary.json, verbalize_summary.md")


def summarise_verbalize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    per_step: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        for s in rec["steps"]:
            d = per_step.setdefault(f"z{s['step']}", {
                "after": s["after"],
                "mentions_error": 0, "says_fix": 0, "says_keep": 0, "no_verdict": 0,
            })
            f = s["reviser_forecast"]
            d["mentions_error"] += int(f["exec"] == "error")
            d["says_fix"] += int(f["verdict"] == "fix")
            d["says_keep"] += int(f["verdict"] == "keep")
            d["no_verdict"] += int(f["verdict"] == "none")

    # forecast of the last thought vs. what the draft really did
    pairs: Dict[str, int] = {}
    hits = {"error_flagged": 0, "error_total": 0, "wrong_flagged": 0, "wrong_total": 0,
            "right_kept": 0, "right_total": 0}
    for rec in records:
        f = rec["steps"][-1]["reviser_forecast"]
        key = (f"draft {'correct' if rec['draft']['correct'] else 'wrong'}"
               f" ({rec['draft']['exec']}) -> forecast {f['exec']}/{f['verdict']}")
        pairs[key] = pairs.get(key, 0) + 1
        if rec["draft"]["exec"] == "error":
            hits["error_total"] += 1
            hits["error_flagged"] += int(f["exec"] == "error")
        if rec["draft"]["correct"]:
            hits["right_total"] += 1
            hits["right_kept"] += int(f["verdict"] == "keep")
        else:
            hits["wrong_total"] += 1
            hits["wrong_flagged"] += int(f["verdict"] == "fix" or f["exec"] == "error")
    return {
        "n_questions": n,
        "draft_ex": round(100 * sum(r["draft"]["correct"] for r in records) / max(1, n), 2),
        "final_ex": round(100 * sum(r["final"]["correct"] for r in records) / max(1, n), 2),
        "draft_changed_by_reviser_thoughts": sum(
            " ".join((r["draft"]["sql"] or "").split())
            != " ".join((r["final"]["sql"] or "").split()) for r in records),
        "reviser_cue_per_step": per_step,
        "last_thought_forecast_vs_draft": dict(sorted(pairs.items())),
        "last_thought_hits": hits,
    }


def render_verbalize_md(s: Dict[str, Any], records: List[Dict[str, Any]]) -> str:
    h = s["last_thought_hits"]
    lines = [f"# Latent verbalization (n={s['n_questions']})", "",
             f"Draft EX (SQL forced after the candidate generator): {s['draft_ex']}%  ",
             f"Final EX (after all thoughts): {s['final_ex']}%  ",
             f"Drafts changed by the reviser thoughts: {s['draft_changed_by_reviser_thoughts']}", "",
             "## Reviser cue after each thought", "",
             "| thought | mentions Error | says fix | says keep | no verdict |",
             "|---|---:|---:|---:|---:|"]
    for k, d in s["reviser_cue_per_step"].items():
        lines.append(f"| {d['after']} | {d['mentions_error']} | {d['says_fix']} | "
                     f"{d['says_keep']} | {d['no_verdict']} |")
    lines += ["", "## Last thought's forecast vs. the real draft", "",
              f"- drafts that crash, flagged as Error: {h['error_flagged']}/{h['error_total']}",
              f"- wrong drafts, flagged (Error or fix): {h['wrong_flagged']}/{h['wrong_total']}",
              f"- correct drafts, kept: {h['right_kept']}/{h['right_total']}", "",
              "| draft -> forecast | count |", "|---|---:|"]
    for k, v in s["last_thought_forecast_vs_draft"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Examples (wrong drafts)", ""]
    for rec in [r for r in records if not r["draft"]["correct"]][:5]:
        lines += [f"### {rec['qid']}: {rec['question']}", "",
                  f"- gold: `{rec['gold_sql']}`",
                  f"- draft ({rec['draft']['exec']}): `{rec['draft']['sql']}`",
                  f"- final ({'correct' if rec['final']['correct'] else 'wrong'}): "
                  f"`{rec['final']['sql']}`", ""]
        for st in rec["steps"]:
            txt = " ".join(st["text"]["reviser"].split())[:300]
            lines.append(f"- {st['after']} reviser cue: {txt}")
        lines.append("")
    return "\n".join(lines) + "\n"


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    per_step: Dict[int, Dict[str, Any]] = {}
    for rec in records:
        for s in rec["steps"]:
            if s["step"] == 0:
                continue
            d = per_step.setdefault(s["step"], {"after": s["after"], "sql_ex": 0,
                                                "p_correct": 0.0, "entropy": 0.0})
            d["sql_ex"] += int(bool(s["sql_correct"]))
            d["p_correct"] += s["p_correct"]
            d["entropy"] += s["entropy"]
    for d in per_step.values():
        d["sql_ex"] = round(100 * d["sql_ex"] / n, 2)
        d["p_correct"] = round(d["p_correct"] / n, 4)
        d["entropy"] = round(d["entropy"] / n, 4)

    inter: Dict[str, Dict[str, Any]] = {}
    base_correct = sum(int(bool(r["final_correct"])) for r in records)
    for rec in records:
        for it in rec["interventions"]:
            key = f"z{it['step']}:{it['kind']}"
            d = inter.setdefault(key, {"n": 0, "changed": 0, "correct": 0,
                                       "broke": 0, "fixed": 0, "delta_logp_sum": 0.0})
            d["n"] += 1
            d["changed"] += int(it["changed"])
            d["correct"] += int(it["correct"])
            d["broke"] += int(bool(rec["final_correct"]) and not it["correct"])
            d["fixed"] += int(not rec["final_correct"] and it["correct"])
            d["delta_logp_sum"] += it["delta_logp_original_answer"] or 0.0
    for d in inter.values():
        d["changed_frac"] = round(d["changed"] / d["n"], 4)
        d["ex"] = round(100 * d["correct"] / d["n"], 2)
        d["mean_delta_logp_original_answer"] = round(d.pop("delta_logp_sum") / d["n"], 3)
    return {"n_questions": n, "final_ex": round(100 * base_correct / max(1, n), 2),
            "per_step": per_step, "interventions": inter}


def render_md(s: Dict[str, Any]) -> str:
    lines = [f"# Latent trace (n={s['n_questions']}, final EX {s['final_ex']}%)", "",
             "## What each thought supports", "",
             "| step | after | EX of SQL forced here | P(correct candidates) | entropy |",
             "|---|---|---:|---:|---:|"]
    for t, d in sorted(s["per_step"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {t} | {d['after']} | {d['sql_ex']}% | {d['p_correct']} | {d['entropy']} |")
    lines += ["", "## Interventions (later thoughts recomputed)", "",
              "| edit | n | answer changed | EX after | broke | fixed | mean Δ log P(original answer) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k, d in sorted(s["interventions"].items()):
        lines.append(f"| {k} | {d['n']} | {100 * d['changed_frac']:.1f}% | {d['ex']}% | "
                     f"{d['broke']} | {d['fixed']} | {d['mean_delta_logp_original_answer']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
