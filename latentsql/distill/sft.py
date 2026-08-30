"""Stage 1: debate structure learning (IMAD section 2.2).

Standard next-token prediction over the *whole* debate transcript, with the
prompt masked out of the loss. LoRA by default, so a 3B student trains on one
consumer GPU.

Written as a plain PyTorch loop rather than ``Trainer`` so the exact same code
path runs on CPU for the smoke test and on GPU for the real run, and so stage 2
(GRPO) can reuse the same checkpoint plumbing.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..llm.hf_local import ModelBundle, load_model, render_chat, resolve_model_path
from ..mas.trace import special_tokens
from .build_dataset import SFTRecord


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class Batch:
    input_ids: Any
    attention_mask: Any
    labels: Any


def build_example(
    tokenizer: Any,
    record: SFTRecord,
    max_seq_len: int,
    train_on_full_trace: bool = True,
) -> Optional[Dict[str, List[int]]]:
    """Tokenise one record, masking the prompt tokens with ``-100``.

    ``train_on_full_trace=False`` reproduces the DebateGPT ablation: supervise
    only the final ``<|consensus|>`` block, discarding the debate itself.
    """
    prompt_text = render_chat(
        tokenizer,
        [{"role": "system", "content": record.system}, {"role": "user", "content": record.prompt}],
        add_generation_prompt=True,
    )
    target_text = record.completion
    if not train_on_full_trace:
        from ..mas.trace import T_CONSENSUS_OPEN

        idx = target_text.find(T_CONSENSUS_OPEN)
        if idx >= 0:
            target_text = target_text[idx:]

    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text + eos, add_special_tokens=False)["input_ids"]

    if len(prompt_ids) >= max_seq_len - 8:
        return None  # schema alone overflows the window; skip rather than truncate the answer
    target_ids = target_ids[: max_seq_len - len(prompt_ids)]
    if not target_ids:
        return None

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    return {"input_ids": input_ids, "labels": labels}


def collate(batch: Sequence[Dict[str, List[int]]], pad_id: int):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return Batch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(mask, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
    )


# --------------------------------------------------------------------------
# model setup
# --------------------------------------------------------------------------
def prepare_student(
    cfg: Any,
    hub_dir: Optional[str] = None,
    add_tag_tokens: bool = False,
    trainable: bool = True,
) -> ModelBundle:
    """Load the student and attach LoRA adapters."""
    path = resolve_model_path(
        cfg.model_name_or_path,
        hub_dir=hub_dir,
        fallbacks=getattr(cfg, "model_fallbacks", ()) or (),
        local_only=getattr(cfg, "local_only", False),
    ) if hub_dir else cfg.model_name_or_path

    bundle = load_model(
        path,
        device=cfg.device,
        dtype=("bfloat16" if cfg.bf16 else "auto"),
        hub_dir=hub_dir,
        trainable=trainable,
        quantization=getattr(cfg, "quantization", "none"),
    )
    if add_tag_tokens:
        added = bundle.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens()}
        )
        if added:
            bundle.model.resize_token_embeddings(len(bundle.tokenizer))

    if cfg.lora and trainable:
        from peft import LoraConfig, get_peft_model

        targets = [
            m for m in cfg.lora_target_modules
            if any(m in n for n, _ in bundle.model.named_modules())
        ] or None
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=targets,
            task_type="CAUSAL_LM",
            bias="none",
        )
        bundle.model = get_peft_model(bundle.model, lora_cfg)
        bundle.model.print_trainable_parameters()
    return bundle


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------
def train_sft(
    records: Sequence[SFTRecord],
    cfg: Any,
    hub_dir: Optional[str] = None,
    bundle: Optional[ModelBundle] = None,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    import gc
    import torch

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    bundle = bundle or prepare_student(cfg, hub_dir=hub_dir)
    model, tok = bundle.model, bundle.tokenizer
    model.train()
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    encoded = []
    n_pruned = 0
    for r in records:
        ex = build_example(tok, r, cfg.max_seq_len, cfg.train_on_full_trace)
        if ex:
            encoded.append(ex)
        else:
            n_pruned += 1
    if n_pruned:
        print(f"[sft] {n_pruned}/{len(records)} records pruned - their prompt alone "
              f"exceeds max_seq_len={cfg.max_seq_len} (kept rather than truncating the "
              f"answer, which would train on a cut-off target)", flush=True)
    if not encoded:
        raise ValueError("no trainable examples after tokenisation (max_seq_len too small?)")

    steps_per_epoch = max(1, math.ceil(len(encoded) / cfg.batch_size))
    opt_steps = max(1, int(steps_per_epoch * cfg.epochs / cfg.grad_accum))
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = max(1, int(opt_steps * cfg.warmup_ratio))

    def lr_at(s: int) -> float:
        if s < warmup:
            return cfg.lr * (s + 1) / warmup
        p = (s - warmup) / max(1, opt_steps - warmup)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    from ..utils.telemetry import RunLog, gpu_stats, reset_gpu_peak, run_environment

    steplog = None
    if log_path:
        steplog = RunLog(
            os.path.splitext(log_path)[0] + "_steps.jsonl",
            header=run_environment(
                stage="sft", model=cfg.model_name_or_path, lora=cfg.lora,
                quantization=getattr(cfg, "quantization", "none"),
                n_examples=len(encoded), optimizer_steps=opt_steps,
            ),
        )
    reset_gpu_peak()

    history: List[Dict[str, Any]] = []
    tokens_seen = 0
    micro, opt_step, t0 = 0, 0, time.time()
    n_micro = max(1, int(steps_per_epoch * cfg.epochs))
    order: List[int] = []

    print(
        f"[sft] {len(encoded)} examples | {opt_steps} optimizer steps "
        f"| device={bundle.device} | lora={cfg.lora}",
        flush=True,
    )
    optim.zero_grad(set_to_none=True)
    while opt_step < opt_steps:
        if not order:
            order = list(range(len(encoded)))
            random.shuffle(order)
        take = [order.pop() for _ in range(min(cfg.batch_size, len(order)))]
        batch = collate([encoded[i] for i in take], tok.pad_token_id)

        out = model(
            input_ids=batch.input_ids.to(bundle.device),
            attention_mask=batch.attention_mask.to(bundle.device),
            labels=batch.labels.to(bundle.device),
        )
        loss = out.loss / cfg.grad_accum
        loss.backward()
        micro += 1
        tokens_seen += int(batch.attention_mask.sum())
        loss_value = float(out.loss.detach().float())
        # This loop had no per-step cleanup at all - unlike train_latent_sft's
        # loop (see its own comment on why gc.collect() is needed, not just
        # empty_cache()), so held references from one micro-step's forward
        # graph could accumulate into the next. Observed in practice as a
        # steady per-step GPU climb (21.68 -> 21.96 -> 22.10GB across three
        # logged steps) that then OOM'd on an unremarkable-sized record - not
        # obviously a single huge outlier, consistent with accumulation.
        del out, loss, batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if micro % cfg.grad_accum == 0:
            for g in optim.param_groups:
                g["lr"] = lr_at(opt_step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step % cfg.logging_steps == 0 or opt_step == opt_steps:
                rec = {
                    "step": opt_step,
                    "loss": loss_value,
                    "lr": lr_at(opt_step),
                    "tokens_seen": tokens_seen,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                rec.update(gpu_stats())
                history.append(rec)
                if steplog:
                    steplog.write({"record": "step", **rec})
                print(f"[sft] {rec}", flush=True)
            if cfg.save_steps and opt_step % cfg.save_steps == 0:
                save_student(bundle, cfg.output_dir)

    save_student(bundle, cfg.output_dir)
    result = {
        "output_dir": cfg.output_dir,
        "model": cfg.model_name_or_path,
        "quantization": getattr(cfg, "quantization", "none"),
        "n_examples": len(encoded),
        "optimizer_steps": opt_step,
        "tokens_seen": tokens_seen,
        "final_loss": history[-1]["loss"] if history else None,
        "elapsed_s": round(time.time() - t0, 1),
        **gpu_stats(),
        "history": history,
    }
    if steplog:
        steplog.write({"record": "summary", **{k: v for k, v in result.items()
                                               if k != "history"}})
        steplog.close()
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result


def save_student(bundle: ModelBundle, output_dir: str) -> None:
    """Save adapters (LoRA) or the full model, plus the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    bundle.model.save_pretrained(output_dir)
    bundle.tokenizer.save_pretrained(output_dir)
    base = getattr(getattr(bundle.model, "peft_config", {}).get("default", None),
                   "base_model_name_or_path", None)
    with open(os.path.join(output_dir, "latentsql_student.json"), "w", encoding="utf-8") as fh:
        json.dump({"base_model": base or bundle.name, "is_peft": base is not None}, fh, indent=2)
