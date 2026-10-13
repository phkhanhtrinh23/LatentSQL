"""Recurrent latent reasoning (Sec. 3.2 and 4.4, Figure 2-4 and 2-5).

For each role r_j the model reads the tag <r_j> and then runs m latent steps:
the last-layer hidden state z is mapped back to an input vector a = A_j(z) and
appended to the sequence, with no token emitted (Eq. 22-23)::

    z_{j,1}   = F(Phi_{j-1}, <r_j>)
    a_{j,t}   = A_j(z_{j,t})
    z_{j,t+1} = F(Phi_{j-1}, <r_j>, a_{j,1:t})
    Phi_j     = (Phi_{j-1}, <r_j>, a_{j,1}, ..., a_{j,m}),     Phi_0 = x

``A_1 = A_S`` (schema-space alignment, ``schema_align.py``) and
``A_2 = A_3 = A``, the token-space soft alignment of Eq. 12::

    A(z) = sum_{v in top-n_top(o)} softmax(o_v / tau) e(v),     o = W_LM z

After J roles the model reads <sql> and decodes only the final SQL. Every
position attends to all earlier latent states through the KV cache.

This module holds the alignment and the cached inference path (evaluation,
RL rollouts, analysis). The training forward pass, which back-propagates
through every latent step, is in ``curriculum.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..llm.hf_local import ModelBundle, render_chat
from ..prompts import SQL_TAG, extract_sql, role_tag, sql_is_closed
from .schema_align import SchemaAligner


def token_aligner(bundle: ModelBundle, top_k: int = 10, temperature: float = 0.5) -> Callable:
    """``A(z)`` of Eq. 12: mixture of the top-n_top token embeddings."""
    import torch

    W_in = bundle.model.get_input_embeddings().weight          # (V, d)
    W_out = bundle.model.get_output_embeddings().weight        # (V, d), o = W_LM z

    def align(h):
        logits = h.to(W_out.dtype) @ W_out.t()
        k = min(int(top_k), logits.shape[-1])
        vals, idx = logits.topk(k, dim=-1)
        alpha = torch.softmax(vals.float() / max(1e-6, temperature), dim=-1).to(W_in.dtype)
        return (alpha.unsqueeze(-2) @ W_in[idx]).squeeze(-2)

    return align


@dataclass
class LatentRun:
    text: str
    sql: Optional[str]
    #: L = J m latent steps actually run
    n_latent_steps: int
    n_decoded_tokens: int
    #: reasoning positions held in the KV cache (role tags + latent steps + <sql>)
    n_reasoning_positions: int = 0
    #: pihat at the last latent step of r1 (schema keys -> probability), for analysis
    schema_distribution: Optional[List[Any]] = None


class LatentReasoner:
    """Holds A, A_S and the role order; runs latent inference."""

    def __init__(self, bundle: ModelBundle, cfg: Any, adapter_dir: Optional[str] = None):
        self.bundle = bundle
        self.cfg = cfg
        self.roles = list(cfg.roles)
        self._token_align = token_aligner(bundle, cfg.top_k, cfg.temperature)
        self.schema: Optional[SchemaAligner] = None
        if cfg.schema_alignment:
            d = bundle.model.config.hidden_size
            adapter_dir = adapter_dir or getattr(bundle, "adapter_path", None)
            self.schema = SchemaAligner.load(adapter_dir, d, bundle.device) if adapter_dir else None
            if self.schema is None:
                self.schema = SchemaAligner(d, bundle.device, top_k=cfg.schema_top_k,
                                            tau=cfg.schema_temperature, beta=cfg.schema_beta)
            # beta, n_S and tau always follow the run's config (ablation "beta = 0")
            self.schema.top_k = int(cfg.schema_top_k)
            self.schema.tau = float(cfg.schema_temperature)
            self.schema.beta = float(cfg.schema_beta)
        self._schema_ctx = None
        self._role: Optional[str] = None
        #: pihat of every r1 latent step since the last ``reset`` (for L_schema)
        self.schema_pis: List[Any] = []

    # -- alignment ----------------------------------------------------------
    def set_schema(self, db_path: str, db_id: Optional[str] = None) -> None:
        """The database whose tables and columns r1 ranges over."""
        if self.schema is not None:
            self._schema_ctx = self.schema.items(self.bundle, db_path, db_id)

    def set_role(self, role: Optional[str]) -> None:
        self._role = role

    def reset(self) -> None:
        self.schema_pis = []

    def schema_keys(self) -> List[str]:
        return list(self._schema_ctx[2].keys) if self._schema_ctx is not None else []

    def align(self, h):
        """``A_j(z)`` for the active role."""
        if self.schema is not None and self._role == self.roles[0]:
            if self._schema_ctx is None:
                raise RuntimeError("schema alignment needs set_schema(db_path) first")
            E, P, _ = self._schema_ctx
            a, pihat = self.schema.align(h, self._token_align(h), E, P)
            self.schema_pis.append(pihat)
            return a.to(h.dtype)
        return self._token_align(h)

    def save(self, out_dir: str) -> None:
        if self.schema is not None:
            self.schema.save(out_dir)

    # -- cached inference -----------------------------------------------------
    def _forward(self, past, ids=None, embeds=None, hidden: bool = False):
        import torch

        model = self.bundle.model
        if ids is not None:
            embeds = model.get_input_embeddings()(
                torch.tensor([ids], device=self.bundle.device))
        out = model(inputs_embeds=embeds, past_key_values=past, use_cache=True,
                    output_hidden_states=hidden)
        h = out.hidden_states[-1][:, -1:, :] if hidden else None
        return out.past_key_values, out.logits[:, -1, :], h

    def prefix(self, system: str, prompt: str, n_latent_roles: Optional[int] = None,
               add_sql_tag: bool = True):
        """Run ``Phi_k`` (k = ``n_latent_roles``, default J) and optionally <sql>.

        Returns ``(past_key_values, next_token_logits, n_reasoning_positions)``.
        No gradient; used for evaluation and RL rollouts.
        """
        import torch

        tok = self.bundle.tokenizer
        k = len(self.roles) if n_latent_roles is None else int(n_latent_roles)
        chat = render_chat(tok, [{"role": "system", "content": system},
                                 {"role": "user", "content": prompt}])
        self.reset()
        n_pos = 0
        with torch.no_grad():
            past, logits, _ = self._forward(None, ids=tok(chat, add_special_tokens=False)["input_ids"])
            for role in self.roles[:k]:
                tag_ids = tok(role_tag(role) + "\n", add_special_tokens=False)["input_ids"]
                past, logits, h = self._forward(past, ids=tag_ids, hidden=True)
                n_pos += len(tag_ids)
                self.set_role(role)
                for _ in range(self.cfg.latent_steps):
                    past, logits, h = self._forward(past, embeds=self.align(h), hidden=True)
                    n_pos += 1
                self.set_role(None)
            if add_sql_tag:
                sql_ids = tok(SQL_TAG, add_special_tokens=False)["input_ids"]
                past, logits, _ = self._forward(past, ids=sql_ids)
                n_pos += len(sql_ids)
        return past, logits, n_pos

    def decode(self, past, logits, max_new_tokens: int = 512, temperature: float = 0.0,
               stop_at_sql: bool = True) -> List[int]:
        """Greedy (or sampled) decoding from a prefix; stops at EOS or a closed SQL block."""
        import torch

        tok = self.bundle.tokenizer
        ids: List[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if temperature and temperature > 0:
                    probs = torch.softmax(logits.float() / temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, 1).item())
                else:
                    nxt = int(logits.argmax(-1).item())
                if nxt == tok.eos_token_id:
                    break
                ids.append(nxt)
                if stop_at_sql and sql_is_closed(tok.decode(ids)):
                    break
                past, logits, _ = self._forward(past, ids=[nxt])
        return ids

    def generate(self, system: str, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0, n_latent_roles: Optional[int] = None) -> LatentRun:
        k = len(self.roles) if n_latent_roles is None else int(n_latent_roles)
        past, logits, n_pos = self.prefix(system, prompt, n_latent_roles=k)
        pis = list(self.schema_pis)
        ids = self.decode(past, logits, max_new_tokens, temperature)
        text = self.bundle.tokenizer.decode(ids, skip_special_tokens=True)
        dist = None
        if pis:
            keys = self.schema_keys()
            last = pis[-1].reshape(-1).float().tolist()
            dist = sorted(zip(keys, last), key=lambda kv: -kv[1])
        return LatentRun(text=text, sql=extract_sql(text), n_latent_steps=k * self.cfg.latent_steps,
                         n_decoded_tokens=len(ids), n_reasoning_positions=n_pos,
                         schema_distribution=dist)

    def language_mode(self, system: str, prompt: str, n_latent_roles: int,
                      max_new_tokens: int = 256) -> str:
        """Stop the latent chain after ``n_latent_roles`` roles and let the model
        continue as text (the probe of Appendix C.4, Figure 9). Diagnostic only."""
        past, logits, _ = self.prefix(system, prompt, n_latent_roles=n_latent_roles,
                                      add_sql_tag=False)
        ids = self.decode(past, logits, max_new_tokens, 0.0, stop_at_sql=True)
        return self.bundle.tokenizer.decode(ids, skip_special_tokens=True)
