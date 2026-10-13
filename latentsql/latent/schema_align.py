"""Schema-space alignment A_S for the schema-selection role r1 (Sec. 4.2).

Each latent state z of r1 is matched against the tables and columns C(S) of
the question's database (Eq. 13), the probabilities are propagated through
foreign keys (Eq. 14), and the top-n_S items are mixed back into an input
vector (Eq. 15)::

    pi(z)    = softmax_c( g_c(z) / tau )
    pihat(z) = (1 - beta) pi(z) + beta P^T pi(z)
    A_S(z)   = sum_{c in top-n_S(pihat)} pibar_c e_S(c)

``e_S(c)`` is the mean input embedding of the item's serialized name, type and
sampled values, rescaled to the average token-embedding norm. ``P`` is the
row-normalized foreign-key adjacency; an item without FK neighbour keeps its
own mass.

Implementation note on the score ``g_c``. The paper writes
``g_c = <W_S z, e_S(c)>``. Here the projection is applied as a residual on the
token-space soft embedding, ``q(z) = A(z) + W_S z`` with ``W_S`` initialised at
zero, and the score is a cosine similarity with a learned scale ``s``::

    g_c(z) = s * cos(q(z), e_S(c))

At initialisation the score is then the similarity between the tokens the
state predicts and the item name, instead of a uniform distribution from a
random projection, and the cosine keeps the scale of ``g`` independent of the
hidden-state norm. ``W_S`` and ``s`` are trained by L_schema (Eq. 16).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_ALIGNER_FILE = "schema_aligner.pt"


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
    from ..db.instances import _resolve_table, _table_alias_map

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
    """A_S(z) and pihat(z); holds W_S and the logit scale ``s`` (Eq. 13-15)."""

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

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        _torch().save(self.state_dict(), os.path.join(out_dir, SCHEMA_ALIGNER_FILE))

    @staticmethod
    def load(out_dir: str, d: int, device: Any) -> Optional["SchemaAligner"]:
        path = os.path.join(out_dir, SCHEMA_ALIGNER_FILE)
        if not os.path.exists(path):
            return None
        al = SchemaAligner(d, device)
        al.load_state_dict(_torch().load(path, map_location="cpu"))
        return al
