"""Load the student SLM with LoRA adapters, and save it."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..llm.hf_local import ModelBundle, load_model, resolve_model_path

STUDENT_FILE = "latentsql_student.json"


def base_model_of(adapter_dir: str) -> Optional[str]:
    path = os.path.join(adapter_dir, STUDENT_FILE)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("base_model")
    return None


def load_student(cfg: Any, adapter_dir: Optional[str] = None, trainable: bool = True,
                 hub_dir: Optional[str] = None) -> ModelBundle:
    """Base model + a new LoRA adapter, or + the adapter saved in ``adapter_dir``."""
    scfg = cfg.student
    name = (base_model_of(adapter_dir) if adapter_dir else None) or scfg.model
    path = resolve_model_path(name, hub_dir=hub_dir, local_only=False)
    has_adapter = bool(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    bundle = load_model(
        path, device=scfg.device, dtype=("bfloat16" if scfg.bf16 else "auto"), hub_dir=hub_dir,
        adapter_path=adapter_dir if has_adapter else None, trainable=trainable,
        quantization=scfg.quantization,
    )
    bundle.name = name
    if trainable and not has_adapter:
        from peft import LoraConfig, get_peft_model

        names = [n for n, _ in bundle.model.named_modules()]
        targets = [m for m in scfg.lora_target_modules if any(n.endswith(m) for n in names)] or None
        bundle.model = get_peft_model(bundle.model, LoraConfig(
            r=scfg.lora_r, lora_alpha=scfg.lora_alpha, lora_dropout=scfg.lora_dropout,
            target_modules=targets, task_type="CAUSAL_LM", bias="none",
        ))
    if trainable:
        for n, p in bundle.model.named_parameters():
            p.requires_grad_("lora_" in n)
        if scfg.gradient_checkpointing and hasattr(bundle.model, "gradient_checkpointing_enable"):
            bundle.model.gradient_checkpointing_enable()
            if hasattr(bundle.model, "enable_input_require_grads"):
                bundle.model.enable_input_require_grads()
    return bundle


def save_student(bundle: ModelBundle, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    bundle.model.save_pretrained(out_dir)
    bundle.tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, STUDENT_FILE), "w", encoding="utf-8") as fh:
        json.dump({"base_model": bundle.name}, fh, indent=4)
