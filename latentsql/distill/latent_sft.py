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

This module closes that gap the direct way: for each teacher trace with a
known-correct ``gold_sql``, run the *same* ``build_prefix`` priming GRPO/eval
use, then train an ordinary next-token cross-entropy loss whose target is the
correct SQL completion. Gradient flows back through the latent steps too (via
``build_prefix(grad=True)``, which reuses the Option A per-role checkpointing
trick), so the latent computation itself gets shaped toward "leads to correct
SQL", not just the final decode layer.

This is deliberately a light, short warm-up stage - a few dozen records, a
couple of epochs - not a full training run. Its only job is to give the model
one worked example per question of what a good answer looks like *from this
exact starting condition*, so GRPO afterward has something non-degenerate to
build on instead of 0% everywhere.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

#: sequence positions upcast to fp32 at once when scoring log-probabilities -
#: same chunking trick as grpo.py/dpo.py, same OOM this avoids.
_LOGP_CHUNK = 512


def _target_ids(tok, gold_sql: str) -> List[int]:
    text = f"```sql\n{gold_sql.strip()}\n```"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        ids = ids + [tok.eos_token_id]
    return ids


def train_latent_sft(
    bundle: Any,
    records: Sequence[Dict[str, Any]],
    cfg: Any,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train on (latent-primed state -> correct SQL) pairs.

    ``records`` need only ``system``, ``prompt`` and ``gold_sql``. ``cfg``
    needs ``latent`` (a LatentConfig - role list and step count), ``lr``,
    ``epochs``, ``grad_accum``, ``logging_steps``, ``save_steps``,
    ``output_dir``, ``seed``.
    """
    import torch
    import torch.nn.functional as F

    from .latent import LatentReasoner
    from ..utils.telemetry import RunLog, gpu_stats, reset_gpu_peak, run_environment

    tok, model = bundle.tokenizer, bundle.model
    reasoner = LatentReasoner(bundle, cfg.latent)

    usable = [r for r in records if r.get("gold_sql")]
    if not usable:
        raise ValueError("no records with a gold_sql - nothing to train on")

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr)

    steplog = None
    if log_path:
        steplog = RunLog(
            os.path.splitext(log_path)[0] + "_steps.jsonl",
            header=run_environment(
                stage="latent_sft", n_records=len(usable), lr=cfg.lr, epochs=cfg.epochs,
                latent_steps=cfg.latent.latent_steps, roles=list(cfg.latent.roles),
            ),
        )
    reset_gpu_peak()

    order = list(range(len(usable)))
    rng = random.Random(getattr(cfg, "seed", 0))
    rng.shuffle(order)
    n_steps = max(1, int(len(order) * cfg.epochs))
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()
    optim.zero_grad(set_to_none=True)
    print(f"[latent_sft] {len(usable)} records | {n_steps} steps | "
          f"latent_steps={cfg.latent.latent_steps} roles={list(cfg.latent.roles)}", flush=True)

    # build_prefix carries the KV cache across roles, which needs use_cache=True
    # and conflicts with HF's own per-layer checkpointing exactly as in GRPO -
    # suspend it for the whole prefix+completion computation, not just the prefix.
    ckpt = bool(getattr(model, "is_gradient_checkpointing", False))
    prev_use_cache = getattr(model.config, "use_cache", True)
    if ckpt:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    n_acc = 0
    try:
        for step in range(n_steps):
            if step % len(order) == 0 and step > 0:
                rng.shuffle(order)
            rec = usable[order[step % len(order)]]
            target_ids = _target_ids(tok, rec["gold_sql"])
            if not target_ids:
                continue

            past, attn, first_logits = reasoner.build_prefix(rec["system"], rec["prompt"], grad=True)
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
            loss = -token_logp / n_tok  # mean NLL per token

            (loss / cfg.grad_accum).backward()
            n_acc += 1
            loss_value = float(loss.detach())
            del past, attn, first_logits, terms, token_logp, loss, ids
            # empty_cache() alone does not reclaim tensors still held by a
            # Python reference cycle (autograd graphs commonly form one via
            # grad_fn <-> saved tensors) - only a real gc pass breaks those,
            # which is why peak memory climbed step over step without this.
            import gc

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
                    "qid": rec.get("qid", ""), "n_target_tokens": n_tok,
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
        if ckpt:
            model.gradient_checkpointing_enable()
        model.config.use_cache = prev_use_cache

    from .sft import save_student

    save_student(bundle, cfg.output_dir)
    result = {
        "output_dir": cfg.output_dir,
        "n_records": len(usable),
        "steps": n_steps,
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
