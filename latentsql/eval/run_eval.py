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
                difficulty=ex.difficulty, dataset=getattr(ex, "dataset", ""),
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
    latent_roles: Optional[int] = None,
    progress: bool = True,
    spec: Any = None,
    stream_path: Optional[str] = None,
    retrieval_cfg: Any = None,
    retrieval_cache_dir: Optional[str] = None,
) -> List[Prediction]:
    """Run the internalised student over ``examples``.

    ``spec``, if given, is a ``SpeculativeSampler`` (see
    ``distill/speculative.py``) - previously only used to accelerate GRPO's
    training-time rollout sampling. Wired in here so it can also accelerate
    real inference/eval, on the ``student`` (non-latent) path. Greedy
    (``temperature=0``) decoding through it is provably lossless, so EX
    should be unchanged versus plain ``model.generate()``; what should change
    is wall-clock.

    ``stream_path``, if given, appends each ``Prediction`` (as JSON, one per
    line, flushed immediately) as soon as it's produced - the normal
    ``save_predictions`` write only happens after the WHOLE method finishes
    (n=90 takes minutes), so there was previously no way to inspect what the
    model is actually generating while a long run is still in progress.
    ``correct``/``ves`` are not populated yet at stream time (scoring happens
    once the full batch returns in ``run()``) - this is for reading raw
    generated text mid-run, not for a live EX count.
    """
    import json as _json
    import torch

    from ..llm.hf_local import render_chat

    tok, model = bundle.tokenizer, bundle.model
    reasoner = None
    if mode == "latent":
        from ..distill.latent import LatentReasoner

        reasoner = LatentReasoner(bundle, latent_cfg)

    stream_fh = None
    if stream_path:
        os.makedirs(os.path.dirname(os.path.abspath(stream_path)) or ".", exist_ok=True)
        stream_fh = open(stream_path, "w", encoding="utf-8")

    def _stream(pred: Prediction) -> None:
        if stream_fh is not None:
            stream_fh.write(_json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
            stream_fh.flush()

    preds: List[Prediction] = []
    for i, ex in enumerate(examples):
        retrieval_t0 = time.time()
        retrieved_values: List[str] = []
        if retrieval_cfg is not None:
            from ..db.value_index import retrieve_value_hints

            retrieved_values = retrieve_value_hints(
                ex.db_path, ex.db_id, ex.question, ex.evidence,
                cfg=retrieval_cfg, cache_dir=retrieval_cache_dir,
            )
        retrieval_latency = time.time() - retrieval_t0
        sp = student_prompt(
            schema=_render_schema_for(ex), question=ex.question, evidence=ex.evidence,
            value_hints="\n".join(f"- {value}" for value in retrieved_values),
        )
        t0 = time.time()
        oom_msg = None
        pred = None
        try:
            if reasoner is not None:
                run = reasoner.generate(
                    sp["system"], sp["user"], max_new_tokens=max_new_tokens,
                    temperature=temperature, n_latent_roles=latent_roles,
                    spec=spec,
                )
                text, n_out = run.text, run.n_decoded_tokens
                n_in = len(tok(sp["system"] + sp["user"])["input_ids"])
                extra = {
                    "latent_steps": run.n_latent_steps,
                    "latent_roles": latent_roles,
                    "roles": run.roles,
                }
                if run.spec is not None:
                    extra["spec"] = run.spec
            else:
                chat = render_chat(
                    tok,
                    [{"role": "system", "content": sp["system"]},
                     {"role": "user", "content": sp["user"]}],
                    add_generation_prompt=True,
                )
                enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(bundle.device)
                n_in = int(enc["input_ids"].shape[1])

                if spec is not None:
                    prompt_ids = enc["input_ids"][0].tolist()
                    new_ids_list = spec.generate(
                        prompt_ids, max_new_tokens, eos_token_id=tok.eos_token_id,
                    )
                    text = tok.decode(new_ids_list, skip_special_tokens=True)
                    n_out = len(new_ids_list)
                    extra = {"spec": spec.stats.take().as_dict()}
                else:
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
                    new_ids = out[0, n_in:]
                    text = tok.decode(new_ids, skip_special_tokens=True)
                    n_out = int(new_ids.shape[0])
                    extra = {}

            model_latency = time.time() - t0
            extra.update({
                "retrieved_values": retrieved_values,
                "retrieval_latency_s": round(retrieval_latency, 6),
                "model_latency_s": round(model_latency, 6),
            })

            pred = Prediction(
                qid=ex.qid, db_id=ex.db_id, db_path=ex.db_path, question=ex.question,
                gold_sql=ex.gold_sql,
                pred_sql=extract_final_sql(text) or extract_sql(text),
                difficulty=ex.difficulty, dataset=getattr(ex, "dataset", ""),
                prompt_tokens=n_in, completion_tokens=n_out,
                latency_s=round(retrieval_latency + model_latency, 3),
                raw_output=text[:4000],
                extra=extra,
            )
        except torch.cuda.OutOfMemoryError as e:
            # Eval is forward-only (no backward, no torch.utils.checkpoint
            # recompute) - unlike training-time OOM (see latent_sft.py's own
            # except clause), there is no known corrupted-CUDA-state risk
            # here, so skip-and-continue is safe: a single runaway/rambling
            # sample (long prompt + long latent conditioning + up to
            # max_new_tokens of output, all in one KV cache) can overshoot
            # the memory cap without derailing the rest of the run.
            oom_msg = str(e)

        if oom_msg is not None:
            # Cleanup lives here, OUTSIDE the except block - see
            # latent_sft.py's own except clause for why: Python's "currently
            # handled exception" keeps the exception's traceback (and every
            # frame/tensor it references) alive for the WHOLE except block,
            # so gc.collect()/empty_cache() inside it would be a no-op.
            print(f"  [{mode}] {i + 1}/{len(examples)} OOM on qid={ex.qid} - skipping: {oom_msg}",
                  flush=True)
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pred = Prediction(
                qid=ex.qid, db_id=ex.db_id, db_path=ex.db_path, question=ex.question,
                gold_sql=ex.gold_sql, pred_sql=None, difficulty=ex.difficulty,
                dataset=getattr(ex, "dataset", ""),
                latency_s=round(retrieval_latency + time.time() - t0, 3),
                raw_output="", extra={
                    "oom": True,
                    "retrieved_values": retrieved_values,
                    "retrieval_latency_s": round(retrieval_latency, 6),
                },
            )

        preds.append(pred)
        _stream(pred)
        if progress and (i + 1) % 10 == 0:
            print(f"  [{mode}] {i + 1}/{len(examples)}", flush=True)
    if stream_fh is not None:
        stream_fh.close()
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
            retrieval_cfg=cfg.mas,
            retrieval_cache_dir=(os.path.join(cache_dir, "value_indexes")
                                 if cache_dir else None),
            **kw,
        )
    score_predictions(preds, timeout=cfg.mas.execution_timeout, compute_ves=compute_ves)
    # label the report by what was actually evaluated: a combined set must not
    # be reported as whichever benchmark happened to come first
    sources = {getattr(e, "dataset", "") for e in examples} - {""}
    if len(sources) > 1:
        dataset = "mix"
    elif sources:
        dataset = sources.pop()
    else:
        dataset = cfg.eval.dataset
    return preds, build_report(preds, method=method, dataset=dataset, compute_ves=compute_ves)
