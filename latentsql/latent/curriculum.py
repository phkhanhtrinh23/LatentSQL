"""Warm-up and the text-to-latent curriculum (Sec. 4.2, Figure 2-2).

Stage 0 (warm-up) trains on the full verified trace
``x <r1> T1 <r2> T2 <r3> T3 <sql> Y*``, which teaches the three roles and their
order. At stage k = 1..J the first k roles become latent and the remaining
written roles are dropped (Eq. 11)::

    Phi_k = (x, <r1> .^m, ..., <rk> .^m),   followed by <sql>

and the objective is (Eq. 17-19)::

    L_k = L_main + lambda L_probe + lambda_S L_schema
    L_main   = -1/N     sum_n log p(y*_n | Phi_k,  <sql>, y*_<n)
    L_probe  = -1/kappa sum_{n<=kappa} log p(y*_n | Phi_j*, <sql>, y*_<n),  j* ~ U{0..k-1}
    L_schema = -1/(m|C*|) sum_t sum_{c in C*} log pihat_c(z_{1,t})

The forward pass rebuilds the sequence step by step without a KV cache, so
gradients flow back through every latent step. LoRA adapters are trained with
a smaller learning rate in stages 2..J (Table 3).
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

from ..llm.hf_local import render_chat
from ..prompts import SQL_TAG, role_tag, sql_target, student_prompt
from .reasoner import LatentReasoner
from .schema_align import gold_schema_items


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
def build_records(traces: Sequence[Any], cfg: Any, tokenizer: Any = None) -> List[Dict[str, Any]]:
    """One training record per verified trace (EX = 1)."""
    from ..data.inputs import schema_text

    roles = list(cfg.latent.roles)
    out, n_long = [], 0
    for tr in traces:
        if not tr.correct:
            continue
        turns = tr.role_turns(roles)
        if turns is None:
            continue
        sp = student_prompt(schema=schema_text(tr, cfg.data.max_schema_chars), question=tr.question,
                            knowledge=tr.evidence, history=tr.history, values=tr.values)
        rec = {
            "qid": tr.qid, "dataset": tr.dataset, "db_id": tr.db_id, "db_path": tr.db_path,
            "system": sp["system"], "prompt": sp["user"], "gold_sql": tr.gold_sql,
            "role_turns": turns,
            "schema_gold": gold_schema_items(tr.gold_sql, tr.db_path, tr.db_id),
        }
        if tokenizer is not None and cfg.curriculum.max_record_tokens:
            n = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"])
                    for t in [rec["prompt"], *turns, rec["gold_sql"]])
            if n > cfg.curriculum.max_record_tokens:
                n_long += 1
                continue
        out.append(rec)
    if n_long:
        print(f"[curriculum] dropped {n_long} records over "
              f"{cfg.curriculum.max_record_tokens} tokens", flush=True)
    return out


def stage_schedule(cfg: Any) -> List[int]:
    """Stages to train: 0 = warm-up, k = first k roles latent."""
    J = len(cfg.latent.roles)
    stages = [0] if cfg.curriculum.use_role_traces else []
    stages += list(range(1, J + 1)) if cfg.curriculum.curriculum else [J]
    return stages


# --------------------------------------------------------------------------
# loss of one record at one stage
# --------------------------------------------------------------------------
def sql_target_ids(tok: Any, sql: str) -> List[int]:
    ids = tok(sql_target(sql), add_special_tokens=False)["input_ids"]
    return ids + ([tok.eos_token_id] if tok.eos_token_id is not None else [])


def stage_loss(reasoner: LatentReasoner, rec: Dict[str, Any], k: int, target_ids: Sequence[int],
               probe_j: Optional[int] = None, probe_tokens: int = 24) -> Dict[str, Any]:
    """``{"main", "probe" (opt.), "schema" (opt.), "n_tokens"}`` for stage ``k``.

    Stage 0 supervises the written role turns and the SQL; stage k >= 1 runs
    the first k roles as latent steps and supervises the SQL only. RL reuses
    this with ``target_ids`` = a rollout (``main`` is then its mean NLL).
    """
    import torch

    bundle = reasoner.bundle
    tok, model, device = bundle.tokenizer, bundle.model, bundle.device
    embed = model.get_input_embeddings()
    reasoner.reset()
    if rec.get("db_path"):
        reasoner.set_schema(rec["db_path"], rec.get("db_id"))
    pieces: List[Any] = []
    labels: List[int] = []

    def add_ids(ids: Sequence[int], supervise: bool) -> None:
        if ids:
            pieces.append(embed(torch.tensor([list(ids)], device=device)))
            labels.extend(list(ids) if supervise else [-100] * len(ids))

    def add_text(text: str, supervise: bool) -> None:
        add_ids(tok(text, add_special_tokens=False)["input_ids"], supervise)

    def prefix_loss(extra_ids: Sequence[int]):
        """Loss of ``<sql> extra_ids`` after the current prefix (the probe)."""
        sql_ids = tok(SQL_TAG, add_special_tokens=False)["input_ids"]
        ids = list(sql_ids) + list(extra_ids)
        inp = torch.cat(pieces + [embed(torch.tensor([ids], device=device))], dim=1)
        lab = labels + [-100] * len(sql_ids) + list(extra_ids)
        return model(inputs_embeds=inp, labels=torch.tensor([lab], device=device),
                     use_cache=False).loss

    out: Dict[str, Any] = {}
    chat = render_chat(tok, [{"role": "system", "content": rec["system"]},
                             {"role": "user", "content": rec["prompt"]}])
    add_text(chat, supervise=False)
    probe_target = list(target_ids[: max(1, int(probe_tokens))])
    if probe_j == 0:
        out["probe"] = prefix_loss(probe_target)

    roles = reasoner.roles
    if k == 0:
        for role, turn in zip(roles, rec["role_turns"]):
            add_text(role_tag(role) + "\n", supervise=False)
            add_text(turn.strip() + "\n", supervise=True)
    else:
        for j, role in enumerate(roles[:k], 1):
            add_text(role_tag(role) + "\n", supervise=False)
            reasoner.set_role(role)
            for _ in range(reasoner.cfg.latent_steps):
                cur = torch.cat(pieces, dim=1)
                h = model(inputs_embeds=cur, use_cache=False,
                          output_hidden_states=True).hidden_states[-1][:, -1:, :]
                pieces.append(reasoner.align(h))
                labels.append(-100)
            reasoner.set_role(None)
            if probe_j == j and j < k:
                out["probe"] = prefix_loss(probe_target)

    add_text(SQL_TAG, supervise=False)
    add_ids(target_ids, supervise=True)
    out["main"] = model(inputs_embeds=torch.cat(pieces, dim=1),
                        labels=torch.tensor([labels], device=device), use_cache=False).loss
    out["n_tokens"] = sum(1 for x in labels if x != -100)

    gold = rec.get("schema_gold") or []
    if reasoner.schema_pis and gold:
        keys = [key.lower() for key in reasoner.schema_keys()]
        idx = [keys.index(g.lower()) for g in gold if g.lower() in keys]
        if idx:
            g = torch.tensor(idx, device=reasoner.schema_pis[0].device)
            out["schema"] = torch.stack([
                -pi.reshape(-1)[g].clamp(min=1e-12).log().mean() for pi in reasoner.schema_pis
            ]).mean()
    reasoner.reset()
    return out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _load_stage(bundle: Any, reasoner: LatentReasoner, stage_dir: str) -> None:
    """Restore a finished stage (adapter + W_S) to resume a run."""
    from peft import set_peft_model_state_dict

    path = os.path.join(stage_dir, "adapter_model.safetensors")
    if os.path.exists(path):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        import torch

        state = torch.load(os.path.join(stage_dir, "adapter_model.bin"), map_location="cpu")
    set_peft_model_state_dict(bundle.model, state)
    if reasoner.schema is not None:
        from .schema_align import SchemaAligner

        saved = SchemaAligner.load(stage_dir, bundle.model.config.hidden_size, bundle.device)
        if saved is not None:
            reasoner.schema.load_state_dict(saved.state_dict())


def train_curriculum(bundle: Any, records: Sequence[Dict[str, Any]], cfg: Any,
                     out_dir: str) -> Dict[str, Any]:
    """Run every stage of ``stage_schedule(cfg)``; checkpoint each to ``out_dir/stage_k``."""
    import torch

    from .student import save_student
    from ..utils.telemetry import RunLog, gpu_stats

    ccfg = cfg.curriculum
    reasoner = LatentReasoner(bundle, cfg.latent)
    model, tok = bundle.model, bundle.tokenizer
    stages = stage_schedule(cfg)
    log = RunLog(os.path.join(out_dir, "train_log.jsonl"))
    rng = random.Random(ccfg.seed)
    summary: Dict[str, Any] = {"stages": {}, "n_records": len(records)}
    model.train()

    for k in stages:
        stage_dir = os.path.join(out_dir, f"stage_{k}")
        if os.path.exists(os.path.join(stage_dir, "DONE")):
            print(f"[curriculum] stage {k} already trained; loading {stage_dir}", flush=True)
            _load_stage(bundle, reasoner, stage_dir)
            continue
        lr = ccfg.lr if k <= 1 else ccfg.lr_late
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999))
        schema_params = reasoner.schema.parameters() if (reasoner.schema is not None and k >= 1) else []
        schema_optim = torch.optim.AdamW(schema_params, lr=ccfg.schema_lr) if schema_params else None
        n_steps = ccfg.epochs_per_stage * len(records)
        print(f"[curriculum] stage {k}/{len(cfg.latent.roles)}: {len(records)} records x "
              f"{ccfg.epochs_per_stage} epochs, lr={lr}", flush=True)
        t0, step, acc = time.time(), 0, 0
        optim.zero_grad(set_to_none=True)
        running: Dict[str, float] = {}
        for epoch in range(ccfg.epochs_per_stage):
            order = list(range(len(records)))
            rng.shuffle(order)
            for i in order:
                rec = records[i]
                target = sql_target_ids(tok, rec["gold_sql"])
                probe_j = rng.randrange(0, k) if (k >= 1 and ccfg.probe_weight > 0) else None
                parts = stage_loss(reasoner, rec, k, target, probe_j=probe_j,
                                   probe_tokens=ccfg.probe_tokens)
                loss = parts["main"]
                if "probe" in parts:
                    loss = loss + ccfg.probe_weight * parts["probe"]
                if "schema" in parts:
                    loss = loss + ccfg.schema_weight * parts["schema"]
                (loss / ccfg.batch_size).backward()
                for name in ("main", "probe", "schema"):
                    if name in parts:
                        running[name] = running.get(name, 0.0) + float(parts[name].detach())
                        running[name + "_n"] = running.get(name + "_n", 0) + 1
                del parts, loss
                acc += 1
                step += 1
                if acc == ccfg.batch_size or step == n_steps:
                    torch.nn.utils.clip_grad_norm_(params, ccfg.max_grad_norm)
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    if schema_optim is not None:
                        torch.nn.utils.clip_grad_norm_(schema_params, ccfg.max_grad_norm)
                        schema_optim.step()
                        schema_optim.zero_grad(set_to_none=True)
                    acc = 0
                if step % ccfg.logging_steps == 0 or step == n_steps:
                    entry = {"stage": k, "epoch": epoch + 1, "step": step, "of": n_steps,
                             "elapsed_s": round(time.time() - t0, 1), **gpu_stats()}
                    for name in ("main", "probe", "schema"):
                        if running.get(name + "_n"):
                            entry[f"loss_{name}"] = round(running[name] / running[name + "_n"], 4)
                    running = {}
                    log.write(entry)
                    print(f"[curriculum] {entry}", flush=True)
        save_student(bundle, stage_dir)
        reasoner.save(stage_dir)
        open(os.path.join(stage_dir, "DONE"), "w").close()
        summary["stages"][k] = {"steps": step, "elapsed_s": round(time.time() - t0, 1)}

    final = os.path.join(out_dir, "final")
    save_student(bundle, final)
    reasoner.save(final)
    log.close()
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4)
    return summary
