#!/usr/bin/env python3
"""Add the latent-state targets of main_latentsql.pdf Sec. 4.2 to prepared records.

* ``schema_gold``: tables and columns the gold query references (C*, Eq. 13).
* ``exec_props``:  execution-property labels from running the gold query
  (Eq. 14): column count, row-count bucket, first column types, aggregation,
  ordering.

Both are derived from the gold query and its database only, so they cost no
annotation. Records keep every other field unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.distill.schema_latent import build_schema_index, exec_properties, gold_schema_items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, help="prepared_records.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.records, encoding="utf-8") if line.strip()]
    n_items = n_props = 0
    n_schema = []
    label_counts = collections.defaultdict(collections.Counter)
    for i, rec in enumerate(rows, 1):
        db_path, db_id, gold = rec.get("db_path"), rec.get("db_id"), rec.get("gold_sql")
        items = gold_schema_items(gold, db_path, db_id) if db_path and gold else []
        rec["schema_gold"] = items
        n_items += bool(items)
        if db_path:
            n_schema.append(len(build_schema_index(db_path, db_id).keys))
        props = exec_properties(db_path, gold) if db_path and gold else None
        rec["exec_props"] = props
        n_props += props is not None
        for k, v in (props or {}).items():
            label_counts[k][v] += 1
        if i % 100 == 0:
            print(f"{i}/{len(rows)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records to {args.out}")
    print(f"schema_gold found for {n_items}/{len(rows)}; exec_props for {n_props}/{len(rows)}")
    if n_schema:
        print(f"schema items per database: mean {sum(n_schema) / len(n_schema):.1f}, "
              f"max {max(n_schema)}")
    print(json.dumps({k: dict(sorted(v.items())) for k, v in label_counts.items()}, indent=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
