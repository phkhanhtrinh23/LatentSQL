"""Latent agent collaboration inside a single model (LatentMAS, sections 3.1-3.2).

LatentMAS is training-free and runs N *separate* models. We apply the same two
mechanisms to the *one* internalised student, so the roles the student absorbed
during IMAD get explicit latent compute before it commits to a query:

* **Latent thoughts** (3.1): instead of decoding a token, feed the last-layer
  hidden state back in as the next input embedding, for ``m`` steps. No text is
  produced, so these steps cost one forward pass each and emit zero tokens.

* **Input-output alignment** (3.1): a raw last-layer hidden state is out of
  distribution as an *input* embedding, so it is mapped back with
  ``e = h W_a``. Two constructions are provided - see :func:`build_aligner`.

* **Latent working memory** (3.2): after a role's latent steps, its layer-wise
  KV cache is kept and the next role generates conditioned on it, which is
  lossless in the sense of their Theorem 3.3 (the successor sees exactly the
  predecessor's representations, with no re-encoding through text).

Concretely, generation becomes::

    prompt -> [role 1 tag] -> m latent steps -> ... -> [role k tag]
           -> m latent steps -> "<|consensus|>" -> decode SQL normally

so the debate happens in activation space and only the final query is decoded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm.hf_local import ModelBundle, render_chat
from ..mas.trace import T_CONSENSUS_OPEN, agent_tag, extract_final_sql


# --------------------------------------------------------------------------
# input-output alignment  (LatentMAS eq. 3)
# --------------------------------------------------------------------------
def build_aligner(bundle: ModelBundle, mode: str = "soft_embed", top_k: int = 32,
                  temperature: float = 1.0):
    """Return ``align(h) -> e`` mapping last-layer hidden states to input space.

    ``soft_embed`` (default): ``e = softmax(h W_out / T) W_in`` restricted to
    the top-k logits. This is the exact soft-embedding of the token
    distribution the hidden state encodes, is cheap, and needs no |V| x d_h
    matrix product up front.

    ``linear``: precompute the single matrix ``W_a = W_out W_in`` (d_h x d_h)
    and use ``e = h W_a``, i.e. the first-order expansion of the above. This is
    the form LatentMAS eq. (3) describes. Note the paper writes ``W_a ~
    W_out^+ W_in`` with a pseudo-inverse; with the shape convention
    ``logits = h W_out`` the composition that maps hidden -> vocab -> embedding
    is ``W_out W_in``, which is what we implement. The two agree up to a
    scaling of the logit temperature, and ``soft_embed`` sidesteps the question
    entirely, which is why it is the default.

    ``identity``: feed hidden states straight back (COCONUT-style); kept for
    ablation, since LatentMAS argues this is the out-of-distribution case.
    """
    import torch

    model = bundle.model
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)

    lm_head = model.get_output_embeddings()
    embed = model.get_input_embeddings()
    if lm_head is None or embed is None:
        mode = "identity"

    if mode == "identity":
        return lambda h: h

    W_in = embed.weight            # (V, d)
    W_out = lm_head.weight         # (V, d) - logits = h @ W_out.T

    if mode == "linear":
        with torch.no_grad():
            # W_a = W_out^T W_in  ->  h W_a = (h W_out^T) W_in = logits @ W_in
            W_a = W_out.t().to(torch.float32) @ W_in.to(torch.float32)
            # The composition sums over the whole vocabulary, so its raw scale
            # is off by orders of magnitude from a real token embedding.
            # Rescale once so that a typical hidden state maps to a vector of
            # the same norm as an average input embedding - without this the
            # latent steps are fed near-zero (or exploding) inputs.
            target = W_in.to(torch.float32).norm(dim=-1).mean()
            probe = torch.randn(64, W_a.shape[0]) @ W_a
            scale = target / probe.norm(dim=-1).mean().clamp(min=1e-12)
            W_a = (W_a * scale).to(W_in.dtype)

        def align_linear(h):
            return h.to(W_a.dtype) @ W_a

        return align_linear

    def align_soft(h):
        logits = h.to(W_out.dtype) @ W_out.t()
        k = min(top_k, logits.shape[-1])
        vals, idx = logits.topk(k, dim=-1)
        probs = torch.softmax(vals.float() / max(1e-6, temperature), dim=-1).to(W_in.dtype)
        vecs = W_in[idx]                                  # (..., k, d)
        return (probs.unsqueeze(-2) @ vecs).squeeze(-2)   # (..., d)

    return align_soft


# --------------------------------------------------------------------------
# latent generation
# --------------------------------------------------------------------------
@dataclass
class LatentRun:
    text: str
    sql: Optional[str]
    n_latent_steps: int
    n_decoded_tokens: int
    roles: List[str] = field(default_factory=list)
    #: positions in the KV working memory when decoding started. With
    #: ``share_working_memory`` this accumulates every role's latent thoughts;
    #: without it, each role starts from an empty cache.
    n_memory_positions: int = 0


class LatentReasoner:
    """Runs latent-thought roles, then decodes only the final query."""

    def __init__(self, bundle: ModelBundle, cfg: Any):
        self.bundle = bundle
        self.cfg = cfg
        self.align = build_aligner(
            bundle, mode=cfg.align, top_k=cfg.soft_embed_top_k,
            temperature=cfg.soft_embed_temperature,
        )

    # -- low level --------------------------------------------------------
    def _latent_steps(self, inputs_embeds, past_key_values, attention_mask, m: int):
        """Auto-regress ``m`` times on hidden states (LatentMAS 3.1)."""
        import torch

        model = self.bundle.model
        hidden = None
        for _ in range(max(0, m)):
            out = model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = out.past_key_values
            hidden = out.hidden_states[-1][:, -1:, :]      # last-layer, last position
            inputs_embeds = self.align(hidden)             # back into input space
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
            )
        return inputs_embeds, past_key_values, attention_mask, hidden

    def _append_text(self, text: str, past_key_values, attention_mask):
        import torch

        tok = self.bundle.tokenizer
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = ids.to(self.bundle.device)
        embeds = self.bundle.model.get_input_embeddings()(ids)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, ids.shape[1], dtype=attention_mask.dtype,
                                        device=attention_mask.device)],
            dim=-1,
        )
        return embeds, past_key_values, attention_mask

    # -- public -----------------------------------------------------------
    def build_prefix(self, system: str, prompt: str, grad: bool = False):
        """Run the role tags and their latent steps; return the KV prefix.

        This is the whole "debate in continuous space" step: for each role we
        append its tag, then auto-regress ``latent_steps`` times on hidden
        states **emitting no tokens**, keeping the KV cache so the next role
        sees the predecessor's actual representations. What comes back is the
        state the model is in immediately after ``<|consensus|>`` - i.e. ready
        to decode the query and nothing else.

        ``grad=True`` keeps the graph, so GRPO can backpropagate *through* the
        latent steps into the policy. That is what makes this trained rather
        than training-free: LatentMAS applies these mechanisms at inference to
        a frozen model, and we optimise in the same regime we infer in.

        Returns ``(past_key_values, attention_mask, next_token_logits)`` where
        the logits predict the first decoded token.
        """
        import torch

        tok, model = self.bundle.tokenizer, self.bundle.model
        cfg = self.cfg
        chat = render_chat(
            tok,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(self.bundle.device)
        attention_mask = enc["attention_mask"]
        inputs_embeds = model.get_input_embeddings()(enc["input_ids"])
        past = None

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for role in cfg.roles:
                tag = T_CONSENSUS_OPEN if role == "consensus" else agent_tag(role) + "\n"
                tag_embeds, past, attention_mask = self._append_text(tag, past, attention_mask)
                inputs_embeds = (
                    torch.cat([inputs_embeds, tag_embeds], dim=1)
                    if past is None else tag_embeds
                )
                if role == "consensus":
                    break
                inputs_embeds, past, attention_mask, _h = self._latent_steps(
                    inputs_embeds, past, attention_mask, cfg.latent_steps
                )
                if not cfg.share_working_memory:
                    past = None
                    attention_mask = torch.ones(
                        1, 1, dtype=attention_mask.dtype, device=attention_mask.device
                    )
            out = model(inputs_embeds=inputs_embeds, past_key_values=past,
                        attention_mask=attention_mask, use_cache=True)
        return out.past_key_values, attention_mask, out.logits[:, -1, :]

    def generate(
        self,
        system: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LatentRun:
        import torch

        tok, model = self.bundle.tokenizer, self.bundle.model
        cfg = self.cfg
        chat = render_chat(
            tok,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(self.bundle.device)
        attention_mask = enc["attention_mask"]
        inputs_embeds = model.get_input_embeddings()(enc["input_ids"])
        past = None
        n_latent = 0

        with torch.no_grad():
            for role in cfg.roles:
                # the role tag anchors the latent block to the subspace the
                # student learned for that agent during SFT
                tag = T_CONSENSUS_OPEN if role == "consensus" else agent_tag(role) + "\n"
                tag_embeds, past, attention_mask = self._append_text(tag, past, attention_mask)
                inputs_embeds = (
                    torch.cat([inputs_embeds, tag_embeds], dim=1)
                    if past is None else tag_embeds
                )
                if role == "consensus":
                    break
                inputs_embeds, past, attention_mask, _h = self._latent_steps(
                    inputs_embeds, past, attention_mask, cfg.latent_steps
                )
                n_latent += cfg.latent_steps
                if not cfg.share_working_memory:
                    # ablation: drop the predecessor's KV working memory
                    past = None
                    attention_mask = torch.ones(
                        1, 1, dtype=attention_mask.dtype, device=attention_mask.device
                    )

            n_memory = int(attention_mask.shape[-1])

            # decode the final query normally
            out_ids: List[int] = []
            for _ in range(max_new_tokens):
                out = model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                if temperature and temperature > 0:
                    probs = torch.softmax(logits.float() / temperature, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)
                else:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                tid = int(nxt[0, 0])
                if tid == tok.eos_token_id:
                    break
                out_ids.append(tid)
                inputs_embeds = model.get_input_embeddings()(nxt)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
                )

        text = tok.decode(out_ids, skip_special_tokens=True)
        full = T_CONSENSUS_OPEN + text
        return LatentRun(
            text=full,
            sql=extract_final_sql(full),
            n_latent_steps=n_latent,
            n_decoded_tokens=len(out_ids),
            roles=list(cfg.roles),
            n_memory_positions=n_memory,
        )


# --------------------------------------------------------------------------
# latent-slot SFT curriculum (optional stage 1.5)
# --------------------------------------------------------------------------
def latent_slot_curriculum(
    completion: str,
    roles: Sequence[str],
    stage: int,
    latent_token: str = "<|latent|>",
    latent_steps: int = 8,
) -> str:
    """Replace the first ``stage`` agent blocks with fixed-width latent slots.

    A COCONUT-style curriculum: at stage 0 the student sees the full text
    debate; at stage k the first k agent blocks are gone, replaced by
    placeholder slots that carry no text. Combined with the GRPO length clip
    this gives a second, discrete route to internalisation - useful as an
    ablation against IMAD's purely reward-driven route.
    """
    import re

    if stage <= 0:
        return completion
    pattern = re.compile(r"<\|agent\s+([a-z_0-9]+)\|>.*?<\|/agent\|>", re.DOTALL)
    replaced = 0

    def _sub(m):
        nonlocal replaced
        if replaced >= stage:
            return m.group(0)
        replaced += 1
        return f"<|agent {m.group(1)}|>" + latent_token * latent_steps + "<|/agent|>"

    return pattern.sub(_sub, completion)
