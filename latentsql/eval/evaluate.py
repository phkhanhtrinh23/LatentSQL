"""Inference and scoring of the latent student (Sec. 4.4).

Retrieved values are appended to x, the J roles run as m latent steps each,
and only the final SQL is decoded (greedy). No teacher and no query execution
are involved at inference; execution is used afterwards to score EX.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, List, Optional, Sequence

from ..data.inputs import student_input
from ..latent.reasoner import LatentReasoner
from .metrics import Prediction, score


def evaluate(examples: Sequence[Any], reasoner: LatentReasoner, cfg: Any,
             n_latent_roles: Optional[int] = None, stream_path: Optional[str] = None,
             progress_every: int = 50) -> List[Prediction]:
    import torch

    tok = reasoner.bundle.tokenizer
    preds: List[Prediction] = []
    fh = open(stream_path, "w", encoding="utf-8") if stream_path else None
    try:
        for i, ex in enumerate(examples, 1):
            t0 = time.time()
            inp = student_input(ex, cfg)       # retrieval time counts towards latency
            reasoner.set_schema(ex.db_path, ex.db_id)
            try:
                run = reasoner.generate(inp["system"], inp["prompt"],
                                        max_new_tokens=cfg.eval.max_new_tokens,
                                        temperature=cfg.eval.temperature,
                                        n_latent_roles=n_latent_roles)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                run = None
            latency = time.time() - t0
            p = Prediction(
                qid=ex.qid, dataset=ex.dataset, db_id=ex.db_id, db_path=ex.db_path,
                question=ex.question, gold_sql=ex.gold_sql,
                pred_sql=run.sql if run else None, difficulty=ex.difficulty,
                prompt_tokens=len(tok(inp["system"] + inp["prompt"], add_special_tokens=False)["input_ids"]),
                completion_tokens=run.n_decoded_tokens if run else 0,
                latent_steps=run.n_latent_steps if run else 0,
                reasoning_positions=run.n_reasoning_positions if run else 0,
                latency_s=round(latency, 4), raw_output=(run.text if run else "")[:4000],
                extra={"schema_top10": (run.schema_distribution or [])[:10] if run else [],
                       "oom": run is None},
            )
            score([p], timeout=cfg.eval.execution_timeout)
            preds.append(p)
            if fh:
                fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
            if i % progress_every == 0 or i == len(examples):
                ex_now = 100.0 * sum(bool(q.correct) for q in preds) / len(preds)
                print(f"[eval] {i}/{len(examples)} EX={ex_now:.2f}%", flush=True)
    finally:
        if fh:
            fh.close()
    return preds
