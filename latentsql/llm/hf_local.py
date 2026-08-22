"""Local HuggingFace ``transformers`` backend.

Used two ways:

1. as a *teacher* driver when the teacher itself is an open-weight model, and
2. as the *student* at evaluation time, where ``ModelBundle`` is shared with
   the SFT / GRPO / latent modules so nothing loads the weights twice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .base import LLMBackend, Usage


def resolve_device(spec: str = "auto") -> str:
    import torch

    if spec and spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(spec: str, device: str):
    import torch

    if spec and spec != "auto":
        return getattr(torch, spec)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _hub_roots(hub_dir: Optional[str]) -> List[str]:
    if not hub_dir:
        return []
    return [hub_dir, os.path.join(hub_dir, "hub"), os.path.join(hub_dir, "models")]


def find_in_hub(name: str, hub_dir: Optional[str]) -> Optional[str]:
    """Resolve ``org/model`` to a local snapshot directory, if one exists.

    Handles both the HF cache layout (``models--org--model/snapshots/<rev>``)
    and a plain mirror (``<hub>/org/model`` or ``<hub>/model``).
    """
    import glob

    if not name:
        return None
    if os.path.isdir(name) and os.path.exists(os.path.join(name, "config.json")):
        return name

    slug = name.replace("/", "--")
    tail = name.split("/")[-1]
    for root in _hub_roots(hub_dir):
        if not os.path.isdir(root):
            continue
        candidates = []
        for snap in glob.glob(os.path.join(root, f"models--{slug}", "snapshots", "*")):
            candidates.append(snap)
        candidates += [
            os.path.join(root, *name.split("/")),
            os.path.join(root, tail),
            os.path.join(root, slug),
        ]
        # case-insensitive sweep as a last resort (Windows mirrors, renames)
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "config.json")):
                return cand
        for entry in glob.glob(os.path.join(root, "*")):
            if os.path.basename(entry).lower() in (tail.lower(), slug.lower()) and os.path.exists(
                os.path.join(entry, "config.json")
            ):
                return entry
    return None


def resolve_model_path(
    name: str,
    hub_dir: Optional[str] = None,
    fallbacks: Sequence[str] = (),
    local_only: bool = True,
) -> str:
    """Pick a model path, preferring the local hub over the network.

    Order: ``name`` in the local hub -> each fallback in the local hub ->
    ``name`` as-is (network) unless ``local_only``.
    """
    for cand in [name, *fallbacks]:
        hit = find_in_hub(cand, hub_dir)
        if hit:
            return hit
    if local_only:
        listing = ""
        try:
            found = scan_hub(hub_dir or "", max_params_b=0)
            listing = "\n  ".join(sorted({m["name"] for m in found})[:20]) or "(none found)"
        except Exception:
            listing = "(hub not readable)"
        raise FileNotFoundError(
            f"no local snapshot for {name!r} (or its fallbacks) under hub {hub_dir!r}.\n"
            f"Causal LMs visible in that hub:\n  {listing}\n"
            f"Fix by pointing paths.hf_hub at the right directory, setting "
            f"llm.model to one of the names above, or setting llm.local_only: false."
        )
    return name


@dataclass
class ModelBundle:
    """A loaded tokenizer + causal LM, plus the device it lives on."""

    model: Any
    tokenizer: Any
    device: str
    name: str = ""

    @property
    def config(self):
        return self.model.config


def load_model(
    name_or_path: str,
    device: str = "auto",
    dtype: str = "auto",
    hub_dir: Optional[str] = None,
    adapter_path: Optional[str] = None,
    trainable: bool = False,
    attn_implementation: Optional[str] = None,
) -> ModelBundle:
    """Load a causal LM, optionally attaching a PEFT adapter.

    ``hub_dir`` is set as ``HF_HOME``/``HF_HUB_CACHE`` before import so that an
    offline hub mirror (e.g. ``/mnt/win_d/hub``) is picked up.
    """
    if hub_dir and os.path.isdir(hub_dir):
        os.environ.setdefault("HF_HOME", hub_dir)
        os.environ.setdefault("HF_HUB_CACHE", hub_dir)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, dev)

    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # correct for batched generation

    kwargs: Dict[str, Any] = {"trust_remote_code": True, "dtype": torch_dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    model.to(dev)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return ModelBundle(model=model, tokenizer=tok, device=dev, name=name_or_path)


def render_chat(tokenizer: Any, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Apply the model's chat template, falling back to a plain format."""
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
        )
    parts = []
    for m in messages:
        parts.append(f"### {m['role'].capitalize()}\n{m['content']}")
    if add_generation_prompt:
        parts.append("### Assistant\n")
    return "\n\n".join(parts)


class HFLocalBackend(LLMBackend):
    """Runs the teacher agents on a local open-weight model (the default)."""

    def __init__(self, cfg: Any, bundle: Optional[ModelBundle] = None):
        super().__init__()
        self.cfg = cfg
        if bundle is None:
            hub = getattr(cfg, "hub_dir", None)
            path = resolve_model_path(
                cfg.model,
                hub_dir=hub,
                fallbacks=getattr(cfg, "model_fallbacks", ()) or (),
                local_only=getattr(cfg, "local_only", True),
            )
            bundle = load_model(
                path,
                device=getattr(cfg, "device", "auto"),
                dtype=getattr(cfg, "dtype", "auto"),
                hub_dir=hub,
            )
        self.bundle = bundle

    def chat(self, messages: Sequence[Dict[str, str]], **kw: Any) -> str:
        import torch

        tok, model = self.bundle.tokenizer, self.bundle.model
        prompt = render_chat(tok, messages)
        enc = tok(prompt, return_tensors="pt").to(self.bundle.device)
        temperature = float(kw.get("temperature", self.cfg.temperature))
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(kw.get("max_tokens", self.cfg.max_tokens)),
            "pad_token_id": tok.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs.update(
                do_sample=True, temperature=temperature,
                top_p=float(kw.get("top_p", self.cfg.top_p)),
            )
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        self.usage.add(
            Usage(
                prompt_tokens=int(enc["input_ids"].shape[1]),
                completion_tokens=int(new_tokens.shape[0]),
                calls=1,
            )
        )
        if self.record_log:
            self.log.append({"tag": kw.get("tag", ""), "output": text})
        return text


# --------------------------------------------------------------------------
# hub inspection - "which of my local models are under 7B?"
# --------------------------------------------------------------------------
def scan_hub(hub_dir: str, max_params_b: float = 7.0) -> List[Dict[str, Any]]:
    """List model snapshots in a HuggingFace cache, with parameter counts.

    Parameter counts are read from ``config.json`` / safetensors metadata
    without loading any weights.
    """
    import glob
    import json
    import re

    results: List[Dict[str, Any]] = []
    roots = [hub_dir, os.path.join(hub_dir, "hub"), os.path.join(hub_dir, "models")]
    seen = set()
    patterns = []
    for r in roots:
        patterns += [
            os.path.join(r, "models--*", "snapshots", "*"),
            os.path.join(r, "*", "config.json"),
            os.path.join(r, "*", "*", "config.json"),
        ]
    for pat in patterns:
        for hit in glob.glob(pat):
            folder = hit if os.path.isdir(hit) else os.path.dirname(hit)
            cfg_path = os.path.join(folder, "config.json")
            if not os.path.exists(cfg_path) or folder in seen:
                continue
            seen.add(folder)
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
            except Exception:
                continue
            name = folder
            m = re.search(r"models--(.+?)[/\\]snapshots", folder.replace("\\", "/"))
            if m:
                name = m.group(1).replace("--", "/")
            else:
                name = os.path.basename(folder.rstrip("/\\"))
            n_params = _estimate_params(cfg, folder)
            results.append(
                {
                    "name": name,
                    "path": folder,
                    "architecture": (cfg.get("architectures") or ["?"])[0],
                    "hidden_size": cfg.get("hidden_size"),
                    "layers": cfg.get("num_hidden_layers"),
                    "params_b": round(n_params / 1e9, 2) if n_params else None,
                    "is_causal_lm": "ForCausalLM" in (cfg.get("architectures") or [""])[0],
                }
            )
    results = [r for r in results if r["is_causal_lm"]]
    if max_params_b:
        results = [r for r in results if not r["params_b"] or r["params_b"] <= max_params_b]
    results.sort(key=lambda r: (r["params_b"] or 99, r["name"]))
    return results


def _estimate_params(cfg: Dict[str, Any], folder: str) -> int:
    """Prefer exact counts from safetensors headers; else estimate from config."""
    import glob
    import json
    import struct

    total = 0
    for shard in glob.glob(os.path.join(folder, "*.safetensors")):
        try:
            with open(shard, "rb") as fh:
                (hdr_len,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(hdr_len).decode("utf-8"))
            for key, meta in header.items():
                if key == "__metadata__" or "shape" not in meta:
                    continue
                n = 1
                for d in meta["shape"]:
                    n *= d
                total += n
        except Exception:
            continue
    if total:
        return total

    h = cfg.get("hidden_size")
    L = cfg.get("num_hidden_layers")
    V = cfg.get("vocab_size")
    inter = cfg.get("intermediate_size") or (4 * h if h else None)
    if not (h and L and V and inter):
        return 0
    return int(L * (4 * h * h + 3 * h * inter) + 2 * V * h)
