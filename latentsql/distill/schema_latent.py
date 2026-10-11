"""Schema-space alignment and execution-property heads for latent steps.

main_latentsql.pdf Sec. 4.2:

* **Schema-space alignment** (Eq. 10-13). For role r1 (schema selection) a
  latent state z is mapped to a distribution over the tables and columns of
  the question's database instead of over subword tokens::

      g_c(z)   = s * cos(q(z), e_S(c)),       q(z) = A(z) + W_S z
      pi(z)    = softmax_c(g_c(z) / tau)
      pihat(z) = (1 - beta) pi(z) + beta P^T pi(z)        (FK propagation)
      A_S(z)   = sum_{c in top-n_S(pihat)} pibar_c e_S(c)

  ``e_S(c)`` is the mean input-token embedding of the item's serialized name,
  type and sampled values, rescaled to the average token-embedding norm so the
  fed-back vector has the scale of a real token. ``P`` is the row-normalized
  foreign-key adjacency over items (an item with no FK neighbour keeps its own
  mass). The paper writes ``g_c = <W_S z, e_S(c)>``; here ``W_S`` is a residual
  on top of the vocabulary soft embedding ``A(z)`` and starts at zero, so at
  initialization the score is the similarity between the token the state
  predicts and the item's name - a learned projection from a random start
  would give a uniform, uninformative pi for the first thousand steps.
  ``s`` is a learned logit scale (CLIP-style) because cosine similarities live
  in [-1, 1] and ``tau = 0.5`` alone would leave pi nearly flat.

  ``L_schema = -1/(m |C*|) sum_t sum_{c in C*} log pihat_c(z_{1,t})`` with C*
  the tables and columns the gold query references.

* **Execution-property heads** (Eq. 14). Small classifiers on the latent
  states of r2 (and r3) predict properties of the gold result: number of
  columns, row-count bucket, type of the first columns, aggregation, ordering.
  Labels come from executing the gold query, so they cost no annotation.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXTRAS_FILE = "latent_extras.pt"

# -- execution properties ----------------------------------------------------
N_TYPE_COLUMNS = 3
TYPE_CLASSES = ("int", "real", "text", "null", "absent")
EXEC_PROPERTIES: Dict[str, int] = {
    "n_columns": 5,        # 1, 2, 3, 4, 5+
    "row_bucket": 4,       # 0, 1, 2-10, >10
    **{f"type_{i}": len(TYPE_CLASSES) for i in range(N_TYPE_COLUMNS)},
    "aggregates": 2,
    "orders": 2,
}
IGNORE = -100


def _row_bucket(n: int) -> int:
    return 0 if n == 0 else 1 if n == 1 else 2 if n <= 10 else 3


def _value_type(v: Any) -> int:
    if v is None:
        return TYPE_CLASSES.index("null")
    if isinstance(v, bool) or isinstance(v, int):
        return TYPE_CLASSES.index("int")
    if isinstance(v, float):
        return TYPE_CLASSES.index("real")
    return TYPE_CLASSES.index("text")


def exec_properties(db_path: str, gold_sql: str, timeout: float = 30.0) -> Optional[Dict[str, int]]:
    """Labels for the execution-property heads, or None if the gold query fails."""
    from ..db.execute import execute_sql

    res = execute_sql(db_path, gold_sql, timeout=timeout, max_rows=100_000)
    if not res.ok:
        return None
    rows = list(res.rows or [])
    n_cols = len(res.columns or (rows[0] if rows else ()))
    labels: Dict[str, int] = {
        "n_columns": max(0, min(n_cols, 5) - 1) if n_cols else IGNORE,
        "row_bucket": _row_bucket(len(rows)),
    }
    for i in range(N_TYPE_COLUMNS):
        if i >= n_cols:
            labels[f"type_{i}"] = TYPE_CLASSES.index("absent")
        elif not rows:
            labels[f"type_{i}"] = IGNORE          # no row to read a type from
        else:
            first = next((r[i] for r in rows if r[i] is not None), None)
            labels[f"type_{i}"] = _value_type(first)
    agg = order = 0
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(gold_sql, read="sqlite")
        agg = int(tree.find(exp.AggFunc) is not None)
        order = int(tree.find(exp.Order) is not None)
    except Exception:
        up = gold_sql.upper()
        agg = int(any(f + "(" in up.replace(" (", "(")
                      for f in ("COUNT", "SUM", "AVG", "MIN", "MAX")))
        order = int("ORDER BY" in up)
    labels["aggregates"] = agg
    labels["orders"] = order
    return labels


# -- schema items --------------------------------------------------------------
@dataclass
class SchemaIndex:
    """Tables and columns of one database, with FK adjacency."""

    db_path: str
    keys: List[str] = field(default_factory=list)       # "table" or "table.column"
    texts: List[str] = field(default_factory=list)      # serialization embedded as e_S
    edges: List[Tuple[int, int]] = field(default_factory=list)  # undirected FK edges
    _pos: Dict[str, int] = field(default_factory=dict)

    def index(self, key: str) -> Optional[int]:
        return self._pos.get(key.lower())


def build_schema_index(db_path: str, db_id: Optional[str] = None,
                       n_values: int = 3, max_value_chars: int = 30) -> SchemaIndex:
    from ..db.schema import load_schema

    schema = load_schema(db_path, db_id, n_examples=n_values, max_example_chars=max_value_chars)
    idx = SchemaIndex(db_path=db_path)

    def add(key: str, text: str) -> int:
        idx._pos[key.lower()] = len(idx.keys)
        idx.keys.append(key)
        idx.texts.append(text)
        return len(idx.keys) - 1

    for t in schema.tables:
        add(t.name, f"table {t.name}")
        for c in t.columns:
            vals = ", ".join(str(v) for v in (c.examples or [])[:n_values])
            text = f"{t.name}.{c.name} {c.type or ''}".strip()
            if vals:
                text += f" values: {vals}"
            add(f"{t.name}.{c.name}", text)
    for fk in schema.foreign_keys:
        a = idx.index(f"{fk.from_table}.{fk.from_column}")
        b = idx.index(f"{fk.to_table}.{fk.to_column}")
        if a is not None and b is not None and a != b:
            idx.edges.append((a, b))
        ta, tb = idx.index(fk.from_table), idx.index(fk.to_table)
        if ta is not None and tb is not None and ta != tb:
            idx.edges.append((ta, tb))
    return idx


def gold_schema_items(sql: str, db_path: str, db_id: Optional[str] = None) -> List[str]:
    """Tables and columns referenced by ``sql``, as SchemaIndex keys."""
    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return []
    from ..db.schema import load_schema_cached
    from ..db.spurious import _resolve_table, _table_alias_map

    schema = load_schema_cached(db_path, db_id)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    alias = _table_alias_map(tree)
    out: List[str] = []

    def push(key: str) -> None:
        if key.lower() not in {k.lower() for k in out}:
            out.append(key)

    for t in tree.find_all(exp.Table):
        real = schema.table(t.name)
        if real is not None:
            push(real.name)
    for c in tree.find_all(exp.Column):
        if not c.name or c.name == "*":
            continue
        tname = _resolve_table(c, alias, schema)
        table = schema.table(tname) if tname else None
        col = table.column(c.name) if table is not None else None
        if col is not None:
            push(f"{table.name}.{col.name}")
    return out


# -- modules ---------------------------------------------------------------------
def _torch():
    import torch

    return torch


class SchemaAligner:
    """A_S(z) and pihat(z); holds W_S and the logit scale (Eq. 10-12)."""

    def __init__(self, d: int, device: Any, top_k: int = 10, tau: float = 0.5,
                 beta: float = 0.3, init_scale: float = 10.0):
        torch = _torch()
        self.top_k, self.tau, self.beta = int(top_k), float(tau), float(beta)
        self.W_S = torch.nn.Parameter(torch.zeros(d, d, device=device, dtype=torch.float32))
        self.log_scale = torch.nn.Parameter(
            torch.tensor(float(init_scale), device=device).log())
        self._cache: Dict[str, Tuple[Any, Any, SchemaIndex]] = {}

    def parameters(self) -> List[Any]:
        return [self.W_S, self.log_scale]

    def state_dict(self) -> Dict[str, Any]:
        return {"W_S": self.W_S.detach().cpu(), "log_scale": self.log_scale.detach().cpu(),
                "top_k": self.top_k, "tau": self.tau, "beta": self.beta}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        with _torch().no_grad():
            self.W_S.copy_(sd["W_S"].to(self.W_S.device))
            self.log_scale.copy_(sd["log_scale"].to(self.log_scale.device))
        self.top_k, self.tau, self.beta = int(sd["top_k"]), float(sd["tau"]), float(sd["beta"])

    def items(self, bundle: Any, db_path: str, db_id: Optional[str] = None):
        """``(E, P, index)``: item embeddings (n, d) and FK propagation (n, n)."""
        torch = _torch()
        key = os.path.abspath(db_path)
        if key not in self._cache:
            index = build_schema_index(db_path, db_id)
            tok = bundle.tokenizer
            embed = bundle.model.get_input_embeddings()
            W_in = embed.weight
            with torch.no_grad():
                target = W_in.float().norm(dim=-1).mean()
                rows = []
                for text in index.texts:
                    ids = tok(text, add_special_tokens=False)["input_ids"][:48] or [0]
                    v = W_in[torch.tensor(ids, device=W_in.device)].float().mean(0)
                    rows.append(v / v.norm().clamp(min=1e-6) * target)
                E = torch.stack(rows).to(W_in.dtype)
                n = len(index.keys)
                A = torch.zeros(n, n, device=W_in.device, dtype=torch.float32)
                for a, b in index.edges:
                    A[a, b] = 1.0
                    A[b, a] = 1.0
                lonely = A.sum(-1) == 0
                A[lonely, lonely] = 1.0      # no FK neighbour: keep own mass
                P = A / A.sum(-1, keepdim=True)
            self._cache[key] = (E, P, index)
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def distribution(self, h, vocab_soft, E, P):
        """pihat over items for hidden state(s) ``h`` (..., d)."""
        torch = _torch()
        q = vocab_soft.float() + h.float() @ self.W_S
        sim = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(
            E.float(), dim=-1).t()
        g = self.log_scale.exp() * sim
        pi = torch.softmax(g / max(1e-6, self.tau), dim=-1)
        return (1.0 - self.beta) * pi + self.beta * (pi @ P)

    def align(self, h, vocab_soft, E, P):
        """``(A_S(z), pihat)``."""
        torch = _torch()
        pihat = self.distribution(h, vocab_soft, E, P)
        k = min(self.top_k, pihat.shape[-1])
        vals, idx = pihat.topk(k, dim=-1)
        pibar = vals / vals.sum(-1, keepdim=True).clamp(min=1e-12)
        vecs = E[idx].float()                                   # (..., k, d)
        a = (pibar.unsqueeze(-2) @ vecs).squeeze(-2)
        return a.to(E.dtype), pihat


class ExecHeads:
    """One small classifier per execution property (Eq. 14)."""

    def __init__(self, d: int, device: Any, hidden: int = 256):
        torch = _torch()
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Sequential(
                torch.nn.LayerNorm(d), torch.nn.Linear(d, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, n_cls),
            )
            for name, n_cls in EXEC_PROPERTIES.items()
        }).to(device=device, dtype=torch.float32)

    def parameters(self) -> List[Any]:
        return list(self.heads.parameters())

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.detach().cpu() for k, v in self.heads.state_dict().items()}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.heads.load_state_dict(sd)

    def logits(self, z) -> Dict[str, Any]:
        return {name: head(z.float()) for name, head in self.heads.items()}

    def loss(self, zs: Sequence[Any], labels: Dict[str, int]):
        """Mean cross-entropy over latent states and labelled properties."""
        torch = _torch()
        F = torch.nn.functional
        terms = []
        for z in zs:
            out = self.logits(z.reshape(1, -1))
            for name, logit in out.items():
                y = labels.get(name, IGNORE)
                if y is None or y == IGNORE:
                    continue
                terms.append(F.cross_entropy(logit, torch.tensor([int(y)], device=logit.device)))
        if not terms:
            return None
        return torch.stack(terms).mean()

    def predict(self, z) -> Dict[str, int]:
        return {k: int(v.argmax(-1)) for k, v in self.logits(z.reshape(1, -1)).items()}


class LatentExtras:
    """Trainable pieces that live outside the LoRA adapter."""

    def __init__(self, schema: Optional[SchemaAligner] = None,
                 exec_heads: Optional[ExecHeads] = None,
                 align_r1: str = "vocab"):
        self.schema = schema
        self.exec_heads = exec_heads
        self.align_r1 = align_r1

    def parameters(self) -> List[Any]:
        out: List[Any] = []
        if self.schema is not None:
            out += self.schema.parameters()
        if self.exec_heads is not None:
            out += self.exec_heads.parameters()
        return out

    def save(self, out_dir: str) -> None:
        torch = _torch()
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "align_r1": self.align_r1,
            "schema": self.schema.state_dict() if self.schema is not None else None,
            "exec_heads": self.exec_heads.state_dict() if self.exec_heads is not None else None,
        }, os.path.join(out_dir, EXTRAS_FILE))

    @staticmethod
    def load(path: str, d: int, device: Any) -> "LatentExtras":
        torch = _torch()
        sd = torch.load(path, map_location="cpu")
        schema = heads = None
        if sd.get("schema") is not None:
            schema = SchemaAligner(d, device)
            schema.load_state_dict(sd["schema"])
        if sd.get("exec_heads") is not None:
            heads = ExecHeads(d, device)
            heads.load_state_dict(sd["exec_heads"])
        return LatentExtras(schema, heads, sd.get("align_r1", "vocab"))
