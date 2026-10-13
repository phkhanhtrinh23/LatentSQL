#!/usr/bin/env python
"""Appendix C.3, Table 5: precision and recall of the schema items used by the final SQL.

    python scripts/analysis/schema_linking.py --predictions outputs/.../eval/predictions.jsonl

For each question, the tables and columns referenced by the predicted query
are compared with those of the gold query; precision and recall are averaged
over questions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from latentsql.eval.metrics import load_predictions
from latentsql.latent.schema_align import gold_schema_items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--dataset", default=None, help="restrict to one benchmark, e.g. bird")
    args = ap.parse_args()

    preds = [p for p in load_predictions(args.predictions)
             if args.dataset is None or p.dataset == args.dataset]
    prec, rec = [], []
    for p in preds:
        gold = {x.lower() for x in gold_schema_items(p.gold_sql, p.db_path, p.db_id)}
        pred = {x.lower() for x in gold_schema_items(p.pred_sql or "", p.db_path, p.db_id)}
        if not gold:
            continue
        rec.append(len(gold & pred) / len(gold))
        prec.append(len(gold & pred) / len(pred) if pred else 0.0)
    out = {"n": len(rec), "precision": round(100 * sum(prec) / max(1, len(prec)), 1),
           "recall": round(100 * sum(rec) / max(1, len(rec)), 1)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
