#!/usr/bin/env python3
"""Execution-property heads on held-out questions (main_latentsql.pdf Sec. 4.2).

Runs the latent roles exactly as at inference, applies the trained heads to the
latent states of the roles after r1, and reports

* per-property accuracy against labels from executing the gold query, next to
  the majority-class baseline;
* how head accuracy relates to the final answer's correctness (mean accuracy
  on correct vs wrong predictions, and their point-biserial correlation),
  using ``--preds`` from ``04_eval.py`` for the same student.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.db.schema import load_schema_cached, render_schema
from latentsql.db.value_index import retrieve_value_hints
from latentsql.distill.latent import LatentReasoner
from latentsql.distill.schema_latent import EXEC_PROPERTIES, IGNORE, exec_properties
from latentsql.llm.hf_local import load_model, render_chat, resolve_model_path
from latentsql.mas.prompts import student_prompt
from latentsql.mas.trace import agent_tag
from latentsql.utils.common import resolve_examples, set_seed


def _pbis(xs: List[float], ys: List[int]) -> float:
    """Point-biserial correlation (Pearson with a binary variable)."""
    n = len(xs)
    if n < 3 or len(set(ys)) < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx and vy else float("nan")


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--preds", default=None, help="preds_latent.jsonl of the same student")
    ap.add_argument("--dataset", default="mix")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--latent-steps", type=int, default=2)
    ap.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(0)
    cfg.latent.latent_steps = args.latent_steps
    cfg.latent.roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    examples, _ = resolve_examples(cfg, dataset=args.dataset, limit=args.limit, split="dev")
    base_path = resolve_model_path(args.base or cfg.sft.model_name_or_path,
                                   hub_dir=cfg.paths.hf_hub, local_only=False)
    bundle = load_model(base_path, device="auto", hub_dir=cfg.paths.hf_hub,
                        adapter_path=args.student, quantization=args.quantization)
    bundle.model.eval()
    reasoner = LatentReasoner(bundle, cfg.latent)
    heads = getattr(reasoner.extras, "exec_heads", None) if reasoner.extras else None
    if heads is None:
        raise SystemExit(f"{args.student} has no execution-property heads")
    correct: Dict[str, bool] = {}
    if args.preds:
        for line in open(args.preds, encoding="utf-8"):
            r = json.loads(line)
            correct[r["qid"]] = bool(r.get("correct"))

    tok, model = bundle.tokenizer, bundle.model
    embed = model.get_input_embeddings()
    roles = [r for r in cfg.latent.roles if r != "consensus"]
    rows: List[Dict[str, Any]] = []
    hit = collections.Counter()
    tot = collections.Counter()
    label_hist: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        for qi, ex in enumerate(examples, 1):
            labels = exec_properties(ex.db_path, ex.gold_sql)
            if labels is None:
                continue
            schema = render_schema(load_schema_cached(ex.db_path, ex.db_id), style="code")
            values = retrieve_value_hints(
                ex.db_path, ex.db_id, ex.question, ex.evidence, cfg=cfg.mas,
                cache_dir=os.path.join(cfg.paths.cache_dir, "value_indexes"))
            sp = student_prompt(schema=schema, question=ex.question, evidence=ex.evidence,
                                value_hints="\n".join(f"- {v}" for v in values))
            chat = render_chat(tok, [{"role": "system", "content": sp["system"]},
                                     {"role": "user", "content": sp["user"]}],
                               add_generation_prompt=True)
            reasoner.reset_records()
            reasoner.set_schema(ex.db_path, ex.db_id)
            reasoner.record_states = True
            pieces = [embed(tok(chat, return_tensors="pt", add_special_tokens=False)
                            ["input_ids"].to(bundle.device))]
            for role in roles:
                pieces.append(embed(tok(agent_tag(role) + "\n", return_tensors="pt",
                                        add_special_tokens=False)["input_ids"].to(bundle.device)))
                reasoner.set_role(role)
                for _ in range(cfg.latent.latent_steps):
                    out = model(inputs_embeds=torch.cat(pieces, dim=1), use_cache=False,
                                output_hidden_states=True)
                    pieces.append(reasoner.align(out.hidden_states[-1][:, -1:, :]))
            reasoner.set_role(None)
            reasoner.record_states = False
            states = reasoner.exec_states
            if not states:
                continue
            pred = heads.predict(states[-1])        # last latent state before consensus
            n_ok = n_lab = 0
            per = {}
            for name in EXEC_PROPERTIES:
                y = labels.get(name, IGNORE)
                if y == IGNORE:
                    continue
                ok = int(pred[name] == y)
                per[name] = {"label": y, "pred": pred[name], "correct": bool(ok)}
                hit[name] += ok
                tot[name] += 1
                label_hist[name][y] += 1
                n_ok += ok
                n_lab += 1
            rows.append({"qid": ex.qid, "question": ex.question,
                         "final_correct": correct.get(ex.qid),
                         "head_accuracy": n_ok / max(1, n_lab), "properties": per})
            if qi % 25 == 0:
                print(f"[exec-probe] {qi}/{len(examples)}", flush=True)

    acc = {k: round(hit[k] / tot[k], 4) for k in EXEC_PROPERTIES if tot[k]}
    majority = {k: round(max(label_hist[k].values()) / tot[k], 4) for k in EXEC_PROPERTIES if tot[k]}
    scored = [r for r in rows if r["final_correct"] is not None]
    xs = [r["head_accuracy"] for r in scored]
    ys = [int(r["final_correct"]) for r in scored]
    right = [x for x, y in zip(xs, ys) if y]
    wrong = [x for x, y in zip(xs, ys) if not y]
    summary = {
        "student": args.student,
        "n_questions": len(rows),
        "accuracy": acc,
        "majority_baseline": majority,
        "mean_head_accuracy": round(sum(r["head_accuracy"] for r in rows) / max(1, len(rows)), 4),
        "vs_final_correctness": {
            "n": len(scored),
            "mean_head_accuracy_when_correct": round(sum(right) / len(right), 4) if right else None,
            "mean_head_accuracy_when_wrong": round(sum(wrong) / len(wrong), 4) if wrong else None,
            "point_biserial_r": round(_pbis(xs, ys), 4) if scored else None,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "questions": rows}, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(summary, indent=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
