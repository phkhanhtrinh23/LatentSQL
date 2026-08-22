"""One evaluation entry point for every method in the paper's comparison table.

Modes:

``single``        zero-shot single agent on the teacher backend
``din_sql`` ...   the four published systems, re-implemented on shared infra
``fusion_mas``    the hybrid teacher
``student``       the internalised student (SFT or SFT+GRPO checkpoint)
``latent``        the student with LatentMAS-style latent roles before decoding
``steered``       the student with an agent steering vector applied

All modes produce the same :class:`Prediction` records, so accuracy and token
consumption are directly comparable - which is the whole point.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..db import schema as schema_mod
from ..llm.base import LLMBackend, extract_sql
from ..mas.baselines import BASELINES
from ..mas.prompts import student_prompt
from ..mas.trace import extract_final_sql, normalize_sql
from .metrics import EvalReport, Prediction, build_report, score_predictions


def _render_schema_for(example: Any) -> str:
    schema = schema_mod.load_schema_cached(example.db_path, example.db_id)
    return schema_mod.render_schema(schema, style="code")


# --------------------------------------------------------------------------
# teacher-side methods
# --------------------------------------------------------------------------
def eval_baseline(
    method: str,
    examples: Sequence[Any],
    llm: LLMBackend,
    cfg: Any,
    pool: Sequence[Any] = (),
    cache_dir: Optional[str] = None,
    progress: bool = True,
) -> List[Prediction]:
    fn = BASELINES[method]
    preds: List[Prediction] = []
    for i, ex in enumerate(examples):
        before = (llm.usage.prompt_tokens, llm.usage.completion_tokens)
        t0 = time.time()
        try:
            kw = {"cache_dir": cache_dir} if method == "fusion_mas" else {}
            sql = fn(llm, cfg.mas, ex, pool=pool, **kw)
        except Exception as exc:
            sql = None
            print(f"  [{method}] {ex.qid} failed: {type(exc).__name__}: {exc}", flush=True)
        preds.append(
            Prediction(
                qid=ex.qid, db_id=ex.db_id, db_path=ex.db_path, question=ex.question,
                gold_sql=ex.gold_sql, pred_sql=normalize_sql(sql) or None,
                difficulty=ex.difficulty,
                prompt_tokens=llm.usage.prompt_tokens - before[0],
                completion_tokens=llm.usage.completion_tokens - before[1],
                latency_s=round(time.time() - t0, 3),
            )
        )
        if progress and (i + 1) % 10 == 0:
            print(f"  [{method}] {i + 1}/{len(examples)}", flush=True)
    return preds


# --------------------------------------------------------------------------
# student-side methods
# --------------------------------------------------------------------------
def eval_student(
    examples: Sequence[Any],
    bundle: Any,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    mode: str = "student",
    latent_cfg: Any = None,
    steering: Any = None,
    steering_agent: str = "",
    steering_alpha: float = 1.0,
    progress: bool = True,
) -> List[Prediction]:
    """Run the internalised student over ``examples``."""
    import torch

    from ..llm.hf_local import render_chat

    tok, model = bundle.tokenizer, bundle.model
    reasoner = None
    if mode == "latent":
        from ..distill.latent import LatentReasoner

        reasoner = LatentReasoner(bundle, latent_cfg)

    preds: List[Prediction] = []
    for i, ex in enumerate(examples):
        sp = student_prompt(
            schema=_render_schema_for(ex), question=ex.question, evidence=ex.evidence
        )
        t0 = time.time()

        if reasoner is not None:
            run = reasoner.generate(
                sp["system"], sp["user"], max_new_tokens=max_new_tokens, temperature=temperature
            )
            text, n_out = run.text, run.n_decoded_tokens
            n_in = len(tok(sp["system"] + sp["user"])["input_ids"])
            extra = {"latent_steps": run.n_latent_steps, "roles": run.roles}
        else:
            chat = render_chat(
                tok,
                [{"role": "system", "content": sp["system"]},
                 {"role": "user", "content": sp["user"]}],
                add_generation_prompt=True,
            )
            enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(bundle.device)
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens, "pad_token_id": tok.pad_token_id,
            }
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            else:
                gen_kwargs.update(do_sample=False)

            ctx = _steer_ctx(bundle, steering, steering_agent, steering_alpha)
            with ctx, torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            n_in = int(enc["input_ids"].shape[1])
            new_ids = out[0, n_in:]
            text = tok.decode(new_ids, skip_special_tokens=True)
            n_out = int(new_ids.shape[0])
            extra = {}

        preds.append(
            Prediction(
                qid=ex.qid, db_id=ex.db_id, db_path=ex.db_path, question=ex.question,
                gold_sql=ex.gold_sql,
                pred_sql=extract_final_sql(text) or extract_sql(text),
                difficulty=ex.difficulty,
                prompt_tokens=n_in, completion_tokens=n_out,
                latency_s=round(time.time() - t0, 3),
                raw_output=text[:4000],
                extra=extra,
            )
        )
        if progress and (i + 1) % 10 == 0:
            print(f"  [{mode}] {i + 1}/{len(examples)}", flush=True)
    return preds


def _steer_ctx(bundle, steering, agent, alpha):
    from contextlib import nullcontext

    if steering is None or not agent:
        return nullcontext()
    from ..steering.caa import steer_layer

    v = steering.vectors.get(agent)
    if v is None:
        return nullcontext()
    return steer_layer(bundle.model, steering.layer, v, alpha)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run(
    method: str,
    examples: Sequence[Any],
    cfg: Any,
    llm: Optional[LLMBackend] = None,
    bundle: Any = None,
    pool: Sequence[Any] = (),
    cache_dir: Optional[str] = None,
    compute_ves: bool = False,
    **kw: Any,
) -> tuple[List[Prediction], EvalReport]:
    if method in BASELINES:
        if llm is None:
            raise ValueError(f"method {method!r} needs an LLM backend")
        preds = eval_baseline(method, examples, llm, cfg, pool=pool, cache_dir=cache_dir)
    else:
        if bundle is None:
            raise ValueError(f"method {method!r} needs a loaded student model")
        preds = eval_student(
            examples, bundle,
            max_new_tokens=cfg.eval.max_new_tokens,
            temperature=cfg.eval.temperature,
            mode=("latent" if method == "latent" else "student"),
            latent_cfg=cfg.latent,
            **kw,
        )
    score_predictions(preds, timeout=cfg.mas.execution_timeout, compute_ves=compute_ves)
    dataset = getattr(examples[0], "dataset", cfg.eval.dataset) if examples else cfg.eval.dataset
    return preds, build_report(preds, method=method, dataset=dataset, compute_ves=compute_ves)
