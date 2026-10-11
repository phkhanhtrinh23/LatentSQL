#!/usr/bin/env python3
"""Schema recall of r1's latent thoughts (main_latentsql.pdf Sec. 4.2, Component 1).

For every latent step z_{1,t} of the schema-selection role this reads off a
distribution over the database's tables and columns and checks how many of
the gold query's items (C*) are in its top-k:

* ``schema``  - the trained schema-space distribution pihat (Eq. 11), only for
  a student trained with ``--align-r1 schema``;
* ``lexical`` - the same state read through the LM head: an item's score is
  the mean log-probability of its name's tokens. This needs no training, so it
  measures every aligner (vocab mixture, hard token, raw hidden state, schema)
  on equal terms.

The thoughts are produced exactly as at inference (each step's aligned vector
is fed back), so the numbers describe what the student actually carries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.db.schema import load_schema_cached, render_schema
from latentsql.db.value_index import retrieve_value_hints
from latentsql.distill.latent import LatentReasoner
from latentsql.distill.schema_latent import gold_schema_items
from latentsql.llm.hf_local import load_model, render_chat, resolve_model_path
from latentsql.mas.prompts import student_prompt
from latentsql.mas.trace import agent_tag
from latentsql.utils.common import resolve_examples, set_seed

KS = (1, 3, 5, 10)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--dataset", default="mix")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--latent-steps", type=int, default=2)
    ap.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    ap.add_argument("--out", required=True, help="output .json")
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
    tok, model = bundle.tokenizer, bundle.model
    embed = model.get_input_embeddings()
    W_out = model.get_output_embeddings().weight
    r1 = cfg.latent.roles[0]
    print(f"align_r1={reasoner.align_r1}", flush=True)

    per_q: List[Dict[str, Any]] = []
    hits = {kind: {k: 0.0 for k in KS} for kind in ("schema", "lexical")}
    n_scored = 0
    with torch.no_grad():
        for qi, ex in enumerate(examples, 1):
            gold = gold_schema_items(ex.gold_sql, ex.db_path, ex.db_id)
            if not gold:
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
            if reasoner.align_r1 == "schema":
                reasoner.set_schema(ex.db_path, ex.db_id)
            else:
                # the lexical read-out still needs the item list
                from latentsql.distill.schema_latent import build_schema_index
                reasoner._schema_ctx = (None, None, build_schema_index(ex.db_path, ex.db_id))
            keys = [k for k in reasoner._schema_ctx[2].keys]
            low = [k.lower() for k in keys]
            gold_idx = {low.index(g.lower()) for g in gold if g.lower() in low}
            # LM-head row for each item: mean over its name tokens
            name_ids = [tok(k.split(".")[-1], add_special_tokens=False)["input_ids"] or [0]
                        for k in keys]

            ids = tok(chat + agent_tag(r1) + "\n", return_tensors="pt",
                      add_special_tokens=False)["input_ids"].to(bundle.device)
            pieces = [embed(ids)]
            reasoner.set_role(r1)
            steps = []
            for t in range(cfg.latent.latent_steps):
                out = model(inputs_embeds=torch.cat(pieces, dim=1), use_cache=False,
                            output_hidden_states=True)
                h = out.hidden_states[-1][:, -1:, :]
                logp = torch.log_softmax(out.logits[0, -1].float(), -1)
                lex = torch.stack([logp[torch.tensor(i, device=logp.device)].mean()
                                   for i in name_ids])
                rec = {"step": t + 1, "lexical_top": [keys[i] for i in lex.topk(min(10, len(keys))).indices.tolist()]}
                for k in KS:
                    top = set(lex.topk(min(k, len(keys))).indices.tolist())
                    rec[f"lexical_recall@{k}"] = len(top & gold_idx) / len(gold_idx)
                pieces.append(reasoner.align(h))
                if reasoner.align_r1 == "schema":
                    pi = reasoner.schema_pis[-1].reshape(-1).float()
                    rec["schema_top"] = [[keys[i], round(float(pi[i]), 4)]
                                         for i in pi.topk(min(10, len(keys))).indices.tolist()]
                    for k in KS:
                        top = set(pi.topk(min(k, len(keys))).indices.tolist())
                        rec[f"schema_recall@{k}"] = len(top & gold_idx) / len(gold_idx)
                steps.append(rec)
            reasoner.set_role(None)
            last = steps[-1]
            n_scored += 1
            for k in KS:
                hits["lexical"][k] += last[f"lexical_recall@{k}"]
                if reasoner.align_r1 == "schema":
                    hits["schema"][k] += last[f"schema_recall@{k}"]
            per_q.append({"qid": ex.qid, "question": ex.question, "gold_items": gold,
                          "n_items": len(keys), "steps": steps})
            if qi % 25 == 0:
                print(f"[schema-recall] {qi}/{len(examples)} lexical@10="
                      f"{hits['lexical'][10] / n_scored:.3f}", flush=True)

    summary = {
        "student": args.student,
        "align_r1": reasoner.align_r1,
        "n_questions": n_scored,
        "recall_at_last_r1_step": {
            kind: {f"@{k}": round(v / max(1, n_scored), 4) for k, v in d.items()}
            for kind, d in hits.items()
            if kind == "lexical" or reasoner.align_r1 == "schema"
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "questions": per_q}, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(summary, indent=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
