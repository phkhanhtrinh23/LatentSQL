"""Stage 1.5 - preference optimisation on the teacher's own failures.

SFT can only learn from debates that went *right*: it maximises
``log pi(y)``, so showing it a wrong trace raises the probability of that
wrong trace. That leaves the most instructive data on the floor - on run-3's
traces, 36% of questions had one panellist right and another wrong on the very
same question, giving 325 ready-made preference pairs against only 48 SFT
traces.

DPO (Rafailov et al., 2023) is the loss that can use them, because it optimises
a *margin* rather than a likelihood::

    L = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                             - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

so the rejected continuation is actively pushed **down**. That is the
difference between "here is what a good debate looks like" and "here is what
to avoid", and it is why the wrong pipelines are training data here and not in
stage 1.

Pairs come free from the panel: same question, same context, one system's turn
correct and another's wrong. Round-1 turns are preferred because the panellists
have not yet seen each other, so the two continuations differ only in the
pipeline that produced them.

The reference policy is the SFT checkpoint, recovered by disabling the LoRA
adapters rather than holding a second copy of the weights.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..mas.prompts import student_prompt
from ..mas.trace import DebateTrace


@dataclass
class PreferencePair:
    """One (chosen, rejected) continuation for the same question."""

    qid: str
    prompt: str
    system: str
    chosen: str
    rejected: str
    chosen_agent: str = ""
    rejected_agent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def build_pairs(
    traces: Sequence[DebateTrace],
    round_one_only: bool = True,
    max_per_question: int = 4,
    seed: int = 0,
) -> List[PreferencePair]:
    """Pair each correct panellist turn against a wrong one on the same question.

    ``round_one_only`` keeps the comparison clean: from round 2 the panellists
    have read each other, so a "wrong" turn there may be wrong *because* it was
    talked out of a good answer, which is not the preference we want to teach.
    """
    rng = random.Random(seed)
    out: List[PreferencePair] = []
    for tr in traces:
        turns = [t for t in tr.turns if t.sql]
        if round_one_only:
            turns = [t for t in turns if t.round == 0]
        good = [t for t in turns if t.correct]
        bad = [t for t in turns if t.correct is False]
        if not good or not bad:
            continue
        # identical prompt construction to the SFT corpus, so DPO optimises
        # the same conditional the student was trained on
        sp = student_prompt(
            schema=tr.schema_prompt,
            question=tr.question,
            evidence=tr.evidence,
            value_hints="\n".join(f"- {v}" for v in tr.value_hints),
        )
        cand: List[Tuple[Any, Any]] = [(g, b) for g in good for b in bad]
        rng.shuffle(cand)
        for g, b in cand[:max_per_question]:
            out.append(
                PreferencePair(
                    qid=tr.qid,
                    system=sp["system"],
                    prompt=sp["user"],
                    chosen=g.render(with_exec=True),
                    rejected=b.render(with_exec=True),
                    chosen_agent=g.agent,
                    rejected_agent=b.agent,
                )
            )
    return out


def save_pairs(pairs: Sequence[PreferencePair], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_pairs(path: str) -> List[PreferencePair]:
    return [PreferencePair(**json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


#: sequence positions upcast to fp32 at once when scoring log-probabilities
_LOGP_CHUNK = 512


# --------------------------------------------------------------------------
def _seq_logprob(model, tok, system: str, prompt: str, completion: str,
                 device: str, max_len: int, length_normalize: bool = True):
    """Log p of the completion, prompt masked out.

    ``length_normalize`` divides by the completion length. Vanilla DPO sums,
    which is fine when both continuations come from the same policy and have
    similar lengths. Ours do not: they come from *different pipelines* with
    systematically different verbosity - measured on our pairs, DAIL-SQL turns
    average 427 characters against CHESS's 824, and the chosen continuation is
    the shorter one in 84% of pairs. A summed margin would therefore rank
    "writes less" rather than "is right", and since DAIL-SQL happens to be both
    the tersest and the most accurate panellist, that confound would look like
    success. Normalising removes the length term.
    """
    import torch
    import torch.nn.functional as F

    from ..llm.hf_local import render_chat

    head = render_chat(tok, [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}])
    h_ids = tok(head, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    ids = (h_ids + c_ids)[-max_len:]
    n_comp = min(len(c_ids), len(ids) - 1)
    if n_comp <= 0:
        return None
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, attention_mask=torch.ones_like(x))
    logits = out.logits[:, :-1, :]
    tgt = x[:, 1:]

    # Upcasting the whole logits block to fp32 costs ~1.9GB at a 152k vocab and
    # a 3k sequence, and DPO needs four of these per pair (policy and reference,
    # chosen and rejected). Walk the time axis in blocks instead; autograd
    # stitches them back together so the gradient is unchanged.
    parts = []
    for i in range(0, logits.shape[1], _LOGP_CHUNK):
        parts.append(
            -F.cross_entropy(
                logits[:, i : i + _LOGP_CHUNK, :].transpose(1, 2).float(),
                tgt[:, i : i + _LOGP_CHUNK],
                reduction="none",
            )
        )
    lp = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    del out, logits
    comp = lp[0, -n_comp:]
    return comp.mean() if length_normalize else comp.sum()


def train_dpo(
    bundle: Any,
    pairs: Sequence[PreferencePair],
    cfg: Any,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run DPO against the SFT policy as reference (adapters disabled)."""
    import torch

    from ..utils.telemetry import RunLog, gpu_stats, reset_gpu_peak, run_environment

    tok, model = bundle.tokenizer, bundle.model
    device = bundle.device
    # DPO is pure training - no KV cache is carried between calls - so
    # checkpointing is free here (unlike the latent prefix, which needs cache)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr)

    steplog = None
    if log_path:
        steplog = RunLog(os.path.splitext(log_path)[0] + "_steps.jsonl",
                         header=run_environment(stage="dpo", n_pairs=len(pairs),
                                                beta=cfg.beta, lr=cfg.lr))
    reset_gpu_peak()

    order = list(range(len(pairs)))
    random.Random(cfg.seed).shuffle(order)
    n_steps = max(1, int(len(order) * cfg.epochs))
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()
    optim.zero_grad(set_to_none=True)
    print(f"[dpo] {len(pairs)} pairs | {n_steps} steps | beta={cfg.beta}", flush=True)

    n_acc = 0
    for step in range(n_steps):
        p = pairs[order[step % len(order)]]

        # reference = the SFT policy, i.e. this model with LoRA switched off
        with torch.no_grad():
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    ref_w = _seq_logprob(model, tok, p.system, p.prompt, p.chosen,
                                         device, cfg.max_seq_len,
                                         cfg.length_normalize)
                    ref_l = _seq_logprob(model, tok, p.system, p.prompt, p.rejected,
                                         device, cfg.max_seq_len,
                                         cfg.length_normalize)
            else:
                ref_w = ref_l = None
        if ref_w is None or ref_l is None:
            continue

        pol_w = _seq_logprob(model, tok, p.system, p.prompt, p.chosen, device, cfg.max_seq_len, cfg.length_normalize)
        pol_l = _seq_logprob(model, tok, p.system, p.prompt, p.rejected, device, cfg.max_seq_len, cfg.length_normalize)
        if pol_w is None or pol_l is None:
            continue

        margin = (pol_w - ref_w.detach()) - (pol_l - ref_l.detach())
        loss = -torch.nn.functional.logsigmoid(cfg.beta * margin)
        (loss / cfg.grad_accum).backward()
        n_acc += 1

        if n_acc % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

        if (step + 1) % cfg.logging_steps == 0 or step + 1 == n_steps:
            rec = {
                "step": step + 1,
                "loss": float(loss.detach()),
                #: >0 means the policy already prefers the correct panellist
                "margin": float(margin.detach()),
                "acc": float((margin > 0).float().detach()),
                "elapsed_s": round(time.time() - t0, 1),
            }
            rec.update(gpu_stats())
            history.append(rec)
            if steplog:
                steplog.write({"record": "step", **rec})
            print(f"[dpo] {rec}", flush=True)

        del pol_w, pol_l, margin, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from .sft import save_student

    save_student(bundle, cfg.output_dir)
    result = {
        "output_dir": cfg.output_dir,
        "n_pairs": len(pairs),
        "steps": n_steps,
        "beta": cfg.beta,
        "final_margin": history[-1]["margin"] if history else None,
        "mean_acc": (sum(h["acc"] for h in history) / len(history)) if history else None,
        "elapsed_s": round(time.time() - t0, 1),
        **gpu_stats(),
        "history": history,
    }
    if steplog:
        steplog.write({"record": "summary",
                       **{k: v for k, v in result.items() if k != "history"}})
        steplog.close()
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result
