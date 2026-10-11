#!/usr/bin/env python3
"""Run VeriEQL (He et al., OOPSLA 2024) on one query pair; JSON in, JSON out.

Runs inside VeriEQL's own Python environment (it pins old parser versions),
started by ``latentsql/db/counterexample.py`` as a subprocess:

    stdin : {"sql1", "sql2", "schema": {table: {column: TYPE}},
             "constraints": [...], "bound": N, "verieql_dir": path}
    stdout: {"state": "EQU" | "NEQ" | "UNK" | "ERR", "error": str | null,
             "tables": {table: [{column: value-literal}]}}   # NEQ only

``tables`` is VeriEQL's counterexample database: at most ``bound`` rows per
table on which the two queries return different results. Values are SQL
literals as VeriEQL prints them ('NULL', numbers, quoted strings).
"""
import io
import json
import os
import sys
import contextlib


def main() -> int:
    req = json.load(sys.stdin)
    sys.path.insert(0, req["verieql_dir"])
    os.chdir(req["verieql_dir"])
    out = {"state": "ERR", "error": None, "tables": {}}
    try:
        from constants import DIALECT
        from environment import Environment
        from errors import NotEquivalenceError  # noqa: F401  (import check)

        with contextlib.redirect_stdout(io.StringIO()):
            with Environment(generate_code=True, show_counterexample=False, timer=True,
                             dialect=DIALECT.ALL) as env:
                for name, cols in req["schema"].items():
                    env.create_database(attributes=cols, bound_size=int(req.get("bound", 2)),
                                        name=name)
                if req.get("constraints"):
                    env.add_constraints(req["constraints"])
                env.save_checkpoints()
                result = env.analyze(req["sql1"], req["sql2"])
        if result is True:
            out["state"] = "EQU"
        elif result is False:
            out["state"] = "NEQ"
            out["tables"] = {k: list(v) for k, v in env.counterexample_dict.items()}
        else:
            out["state"] = "UNK"
    except BaseException as exc:      # unsupported SQL, parser errors, solver timeouts
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
