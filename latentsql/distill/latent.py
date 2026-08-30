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
    #: ``SpecStats.as_dict()`` for the SQL decode when a ``SpeculativeSampler``
    #: was used, else ``None``. The latent prefix is never drafted - only the
    #: decoded query is - so these counts cover the query alone.
    spec: Optional[Dict[str, Any]] = None


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
    def _latent_steps(self, inputs_embeds, past_key_values, attention_mask, m: int,
                      checkpoint: bool = False, aux_target_ids=None):
        """Auto-regress ``m`` times on hidden states (LatentMAS 3.1).

        ``checkpoint=True`` wraps the whole ``m``-step block in
        ``torch.utils.checkpoint`` so none of its per-layer activations are
        retained for backward - only recomputed on demand. This is what makes
        GRPO-through-the-latent-debate feasible: a 12-step role otherwise holds
        12 forward passes' worth of every layer's activations simultaneously,
        which is what OOMed a 24GB card once the debate was deepened to mirror
        the teacher's two rounds.

        The KV cache makes this non-trivial. ``past_key_values`` is one
        object, mutated **in place** by every step's ``model(..., use_cache=
        True)`` call (``DynamicLayer.update`` does
        ``self.keys = torch.cat([self.keys, key_states], dim=-2)``). Naively
        checkpointing a function with that side effect is unsafe: non-reentrant
        checkpoint calls the function once now and once more, later, during
        backward - and a second blind call would append the same ``m`` steps
        again, corrupting the cache's length for every later role that already
        ran on top of it.

        The fix is ``past_key_values.crop(start_len)`` at the top of the
        checkpointed function. ``DynamicLayer.crop`` is a plain tensor slice
        (``self.keys = self.keys[..., :max_length, :]``, fully differentiable)
        and is a true no-op when the cache is already at or below that length.
        So: the first (real, in-order) call finds the cache already at
        ``start_len`` and cropping does nothing; the second (backward-time)
        call finds the cache extended by this role's own ``m`` steps - because
        forward already ran start-to-finish before any backward begins - and
        crop removes exactly that extension before redoing it identically.
        Each role's recompute is therefore idempotent with respect to how many
        times it runs, which is the property non-reentrant checkpoint requires.

        Verified empirically in ``tests/test_core.py`` by comparing LoRA
        gradients computed with and without this checkpointing on the same
        rollout: they agree to float32 tolerance.

        ``aux_target_ids`` (optional token-id sequence): a "decode-and-compare"
        auxiliary loss - at latent step ``t`` (``t < len(aux_target_ids)``),
        the model's own LM head has already produced ``out.logits`` for that
        step's hidden state as a side effect of the forward call above (no
        extra matmul against the vocab needed); this adds
        ``-log p(aux_target_ids[t])`` under that distribution to an
        accumulated aux loss. The intent (see RESULTS.md / this session's
        discussion): nothing currently forces a latent slot to correspond to
        what the teacher's actual agent turn would say at that position -
        this is a lightweight way to add that pressure without a second
        teacher forward pass, by reusing logits the step already computes.
        ``aux_target_ids`` is truncated to the first ``m`` tokens elsewhere
        (by the caller) since there are only ``m`` steps to compare against.
        Returns 0.0 (as a tensor) when ``aux_target_ids`` is ``None``/empty,
        so the checkpointed function's output structure - and therefore what
        non-reentrant checkpoint recomputes - stays identical whether or not
        the aux loss is active for a given call.
        """
        import torch
        import torch.nn.functional as F
        from transformers import DynamicCache

        model = self.bundle.model
        mask_dtype, mask_device = attention_mask.dtype, attention_mask.device
        if past_key_values is None:
            # _text_steps already does this (see its own docstring: "so both
            # the real call and the backward-time recompute close over the
            # exact same cache object") - this function was missing it. A
            # bare `None` can't be mutated in place, so every one of the m
            # internal model() calls below was silently getting its OWN
            # ephemeral HF-internal cache (created and discarded inside
            # model(), never captured via out.past_key_values) instead of
            # one real cache growing across steps - breaking
            # share_working_memory's whole premise (nothing was actually
            # being carried forward) and feeding attention_mask lengths that
            # didn't match the real (absent) cache state, which is the
            # actual source of this record set's erratic, content-
            # independent OOMs - not sequence length.
            past_key_values = DynamicCache()
        start_len = int(past_key_values.get_seq_length())
        n_aux = len(aux_target_ids) if aux_target_ids else 0

        def _run(embeds):
            past_key_values.crop(start_len)   # no-op first call; undoes 2nd call's own append
            e, hidden, length = embeds, None, start_len
            aux_terms = []
            next_logits = None
            for t in range(max(0, m)):
                attn = torch.ones(1, length + 1, dtype=mask_dtype, device=mask_device)
                out = model(
                    inputs_embeds=e,
                    past_key_values=past_key_values,
                    attention_mask=attn,
                    use_cache=True,
                    output_hidden_states=True,
                )
                length += 1
                hidden = out.hidden_states[-1][:, -1:, :]  # last-layer, last position
                next_logits = out.logits[:, -1, :]         # this step's own next-token prediction
                if t < n_aux:
                    step_logits = next_logits.float()  # single position - cheap upcast
                    step_logp = F.log_softmax(step_logits, dim=-1)[0, aux_target_ids[t]]
                    aux_terms.append(-step_logp)
                e = self.align(hidden)                     # back into input space
            aux_loss = (torch.stack(aux_terms).mean() if aux_terms
                       else torch.zeros((), device=embeds.device, dtype=torch.float32))
            return e, aux_loss, next_logits

        if checkpoint and m > 0 and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as _ckpt

            new_embeds, aux_loss, next_logits = _ckpt(_run, inputs_embeds, use_reentrant=False)
        else:
            new_embeds, aux_loss, next_logits = _run(inputs_embeds)

        new_len = start_len + max(0, m)
        new_attention_mask = torch.ones(1, new_len, dtype=mask_dtype, device=mask_device)
        # `next_logits` (this role's own last step's next-token prediction) lets
        # a caller bridge a following supervised-text role's first token back to
        # whatever this latent role actually predicted - see build_prefix's
        # COCONUT-fidelity text-generation supervision.
        return new_embeds, past_key_values, new_attention_mask, next_logits, aux_loss

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

    def _text_steps(self, pieces, past_key_values, attention_mask, checkpoint=False,
                    prev_logits=None, supervise=False, max_total_tokens=None,
                    need_hidden=False):
        """Forward a consecutive run of explicit-text (curriculum) roles'
        real text through the model, extending the shared KV cache.

        Mirrors ``_latent_steps``'s shape exactly: ONE checkpoint wraps an
        internal loop over multiple ``model()`` calls, rather than one
        checkpoint per role. An earlier version checkpointed each role
        separately while sharing one mutating cache object across those
        independent checkpoint calls; backward could then recompute one
        role while the cache still carried a LATER role's extension
        (``CheckpointError``: recomputed KV shapes came out longer than
        the ones saved during the real forward). Grouping the whole run
        into a single checkpoint keeps its cache mutation inside one
        recompute - the same guarantee ``_latent_steps`` already relies on
        for its own internal ``m``-step loop.

        ``past_key_values=None`` is replaced with a fresh empty
        ``DynamicCache`` up front (rather than left ``None``) so both the
        real call and the backward-time recompute close over the exact
        same cache object - no separate path is needed for "this is the
        very first forward in the whole prefix".

        ``supervise=True`` adds a real next-token generation loss over
        every piece's own tokens - COCONUT-fidelity fix: the reference
        method keeps training the model to actually PRODUCE whatever
        reasoning text has not yet been replaced by latent thoughts, not
        just read it as conditioning. Two components, both from logits the
        forward pass already computes (no extra matmul):
          * within a piece, position i predicts token i+1 (ordinary
            teacher-forced LM loss over that piece's own span);
          * a piece's OWN first token is bridged from ``prev_logits`` (this
            role run's own prediction before its first piece) or, from the
            second piece onward, the previous piece's own last-position
            logits - chained via ``cur_logits`` below. The very first
            piece of the very first (fully-fresh-cache) call has no
            ``prev_logits`` available and simply skips that one bridge
            term, since there is nothing upstream to bridge from.
        """
        import torch
        import torch.nn.functional as F
        from transformers import DynamicCache

        model, tok = self.bundle.model, self.bundle.tokenizer
        mask_dtype, mask_device = attention_mask.dtype, attention_mask.device
        if past_key_values is None:
            past_key_values = DynamicCache()
        start_len = int(past_key_values.get_seq_length())
        piece_ids = [
            tok(piece, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
            for piece in pieces
        ]

        if max_total_tokens:
            # Hard token-level safety cap: char-based --max-prompt-chars/
            # --max-role-chars are only a proxy for token count and have
            # already proven insufficient on their own (an OOM inside
            # self_attn mid-forward, not a clean pre-flight rejection).
            # Truncate here instead, at the actual point the sequence is
            # about to grow, so a batch never gets aborted for a runaway
            # prefix - later pieces are shortened (or dropped entirely once
            # the budget is spent) rather than letting the shared KV cache
            # grow unboundedly.
            budget = max(0, max_total_tokens - start_len)
            used, capped = 0, []
            for ids in piece_ids:
                room = max(0, budget - used)   # clamp: a negative room would slice
                                                # ids[:room] from the END (Python
                                                # semantics), NOT truncate to empty -
                                                # without this clamp, once the budget
                                                # was spent, later pieces kept almost
                                                # all of their tokens instead of none
                capped.append(ids[:room])
                used += len(capped[-1])
            piece_ids = capped

        import os as _os
        if _os.environ.get("LATENTSQL_DEBUG_CTX"):
            print(f"[_text_steps DEBUG] start_len={start_len} n_pieces={len(piece_ids)} "
                  f"piece_lens={[len(x) for x in piece_ids]} "
                  f"total={start_len + sum(len(x) for x in piece_ids)} "
                  f"max_total_tokens={max_total_tokens}", flush=True)

        def _run(dummy):
            past_key_values.crop(start_len)   # no-op first call; undoes 2nd call's own append
            length = start_len
            last_hidden = dummy
            gen_terms = []
            cur_logits = prev_logits
            for ids in piece_ids:
                if not ids:
                    continue
                ids_t = torch.tensor([ids], device=self.bundle.device)
                embeds = model.get_input_embeddings()(ids_t)
                length += len(ids)
                attn = torch.ones(1, length, dtype=mask_dtype, device=mask_device)
                out = model(inputs_embeds=embeds, past_key_values=past_key_values,
                           attention_mask=attn, use_cache=True,
                           output_hidden_states=need_hidden)
                if need_hidden:
                    last_hidden = out.hidden_states[-1]
                if supervise and ids:
                    if cur_logits is not None:
                        lp0 = F.log_softmax(cur_logits.float(), dim=-1)[0, ids[0]]
                        gen_terms.append(-lp0)
                    if len(ids) > 1:
                        step_logits = out.logits[0, :-1, :].float()
                        step_logp = F.log_softmax(step_logits, dim=-1)
                        tgt = torch.tensor(ids[1:], device=step_logits.device)
                        lp = step_logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                        gen_terms.append(-lp.mean())
                    cur_logits = out.logits[:, -1, :]
            gen_loss = (torch.stack(gen_terms).mean() if gen_terms
                       else torch.zeros((), device=dummy.device, dtype=torch.float32))
            return last_hidden, gen_loss, cur_logits

        dummy = torch.zeros(1, 1, model.config.hidden_size, device=self.bundle.device,
                            dtype=next(model.parameters()).dtype)
        if checkpoint and pieces and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as _ckpt

            last_hidden, gen_loss, cur_logits = _ckpt(_run, dummy, use_reentrant=False)
        else:
            last_hidden, gen_loss, cur_logits = _run(dummy)

        total_len = start_len + sum(len(ids) for ids in piece_ids)
        new_attention_mask = torch.ones(1, total_len, dtype=mask_dtype, device=mask_device)
        return past_key_values, new_attention_mask, last_hidden, gen_loss, cur_logits

    # -- public -----------------------------------------------------------
    def build_prefix(self, system: str, prompt: str, grad: bool = False,
                     role_turns: Optional[Sequence[Optional[str]]] = None,
                     n_latent_roles: Optional[int] = None,
                     decode_compare: bool = False,
                     supervise_text: bool = False):
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

        ``n_latent_roles`` controls how many non-consensus role markers are
        converted into latent thoughts before jumping directly to
        ``<|consensus|>``. Later agent debate text is deliberately not fed or
        generated on this path; the supervised continuation is SQL only.
        ``role_turns`` is only used by optional ``decode_compare``.

        ``decode_compare`` (default ``False``, so GRPO/eval are unaffected):
        adds a "decode-and-compare" auxiliary loss on every latent role that
        still has real teacher text available (``role_turns[idx]`` not
        ``None``) - nothing else currently forces a latent slot to
        correspond to what that agent would actually have said, since a
        latent role's ``role_turns[idx]`` is otherwise simply discarded (only
        its bare tag is used to seed the m latent steps). See
        ``_latent_steps``'s docstring for the mechanism: reuses each latent
        step's own already-computed ``out.logits`` (no extra teacher forward
        pass, no extra vocab matmul) to cross-entropy against the first
        ``latent_steps`` tokens of that role's real turn.

        ``supervise_text`` is retained for compatibility with older callers,
        but this SQL-direct path does not feed later explicit debate turns.

        Returns ``(past_key_values, attention_mask, next_token_logits,
        aux_loss, text_gen_loss)`` where the logits predict the first
        decoded token, ``aux_loss`` is the mean decode-and-compare loss
        across all latent roles that had real text available, and
        ``text_gen_loss`` is the mean next-token generation loss across all
        supervised explicit-text roles (each 0.0 if its flag is off or no
        role qualified).
        """
        import torch

        tok, model = self.bundle.tokenizer, self.bundle.model
        cfg = self.cfg
        n_roles = len(cfg.roles) - 1  # excludes "consensus"
        chat = render_chat(
            tok,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        # Gradient checkpointing forces use_cache=False, which returns a null
        # past_key_values - and the latent prefix is built by carrying that
        # cache from one role to the next, so the attention mask would then
        # describe a cache that does not exist. Suspend it for this pass.
        ckpt = bool(getattr(model, "is_gradient_checkpointing", False))
        prev_use_cache = getattr(model.config, "use_cache", True)
        if grad and ckpt:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True

        ctx = torch.enable_grad() if grad else torch.no_grad()
        try:
          with ctx:
            enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(self.bundle.device)
            max_ctx = getattr(cfg, "max_context_tokens", None)
            attention_mask = torch.ones(1, 0, dtype=enc["attention_mask"].dtype, device=self.bundle.device)
            past = None
            # Prefill the chat prompt into the KV cache before any latent step.
            # A latent step is one hidden-state position; feeding the whole prompt
            # as that "one" position gives wrong cache/attention accounting.
            past, attention_mask, prompt_hidden, _prompt_loss, prompt_logits = self._text_steps(
                [chat], past, attention_mask, checkpoint=grad, supervise=False,
                max_total_tokens=max_ctx, need_hidden=False,
            )
            import os as _os
            if _os.environ.get("LATENTSQL_DEBUG_CTX"):
                print(f"[build_prefix DEBUG] prompt_len={enc['input_ids'].shape[1]} "
                      f"max_ctx={max_ctx}", flush=True)

            # Always real tensors (never a bare 0.0 float), even when the
            # corresponding flag is off or no role qualified - callers (e.g.
            # latent_sft.py) unconditionally call .detach()/.float() on
            # these return values.
            aux_loss_total = torch.zeros((), device=self.bundle.device, dtype=torch.float32)
            text_gen_loss_total = torch.zeros((), device=self.bundle.device, dtype=torch.float32)
            n_aux_roles = 0
            n_text_roles = 0
            last_logits = [prompt_logits]  # this run's own most recent next-token prediction
            last_hidden = [prompt_hidden]

            consensus_fed = False
            for idx, role in enumerate(cfg.roles):
                is_consensus = role == "consensus"
                if is_consensus:
                    piece = T_CONSENSUS_OPEN
                else:
                    if n_latent_roles is not None and idx >= n_latent_roles:
                        break
                    piece = agent_tag(role) + "\n"

                past, attention_mask, h, _tag_loss, logits = self._text_steps(
                    [piece], past, attention_mask, checkpoint=grad,
                    prev_logits=last_logits[0], supervise=False,
                    max_total_tokens=getattr(cfg, "max_context_tokens", None),
                    need_hidden=not is_consensus,
                )
                last_hidden[0] = h
                last_logits[0] = logits
                if is_consensus:
                    consensus_fed = True
                    break

                aux_target_ids = None
                if decode_compare and role_turns and role_turns[idx] is not None:
                    # Real text was discarded above (only agent_tag(role) was
                    # used to seed this latent role) - reuse it here as the
                    # decode-and-compare target instead of throwing it away.
                    full_ids = tok(role_turns[idx], add_special_tokens=False)["input_ids"]
                    aux_target_ids = full_ids[: cfg.latent_steps]

                inputs_embeds = self.align(last_hidden[0][:, -1:, :])
                inputs_embeds, past, attention_mask, next_logits, aux_loss = self._latent_steps(
                    inputs_embeds, past, attention_mask, cfg.latent_steps,
                    checkpoint=grad, aux_target_ids=aux_target_ids,
                )
                last_logits[0] = next_logits  # bridges into a later supervised-text role, if any
                if aux_target_ids:
                    aux_loss_total = aux_loss_total + aux_loss
                    n_aux_roles += 1
                if not cfg.share_working_memory:
                    past = None
                    attention_mask = torch.ones(
                        1, 1, dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    last_logits[0] = None  # working memory dropped - nothing to bridge from
            if not consensus_fed:
                past, attention_mask, h, _tag_loss, logits = self._text_steps(
                    [T_CONSENSUS_OPEN], past, attention_mask, checkpoint=grad,
                    prev_logits=last_logits[0], supervise=False,
                    max_total_tokens=getattr(cfg, "max_context_tokens", None),
                    need_hidden=False,
                )
                last_hidden[0] = h
                last_logits[0] = logits
            first_logits = last_logits[0]
            if first_logits is None:
                raise RuntimeError("latent prefix ended without next-token logits")
        finally:
            if grad and ckpt:
                model.gradient_checkpointing_enable()
            model.config.use_cache = prev_use_cache
        if n_aux_roles:
            aux_loss_total = aux_loss_total / n_aux_roles
        if n_text_roles:
            text_gen_loss_total = text_gen_loss_total / n_text_roles
        return (past, attention_mask, first_logits,
                aux_loss_total, text_gen_loss_total)

    def generate(
        self,
        system: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        n_latent_roles: Optional[int] = None,
        spec: Optional[Any] = None,
    ) -> LatentRun:
        """``spec``, if given, is a ``SpeculativeSampler`` used for the SQL
        decode only. The latent prefix is built exactly as on the greedy path
        and handed to the sampler as an already-processed target prefix; the
        draft model cannot see latent thoughts, so it conditions on the plain
        text prompt and rejection sampling absorbs the mismatch. Greedy
        verification is lossless, so EX should match the non-spec run."""
        import torch

        tok, model = self.bundle.tokenizer, self.bundle.model
        cfg = self.cfg
        chat = render_chat(
            tok,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(self.bundle.device)
        attention_mask = torch.ones(1, 0, dtype=enc["attention_mask"].dtype, device=self.bundle.device)
        past = None
        n_latent = 0
        next_logits = None

        with torch.no_grad():
            past, attention_mask, last_hidden, _loss, next_logits = self._text_steps(
                [chat], past, attention_mask, checkpoint=False, supervise=False,
                max_total_tokens=getattr(cfg, "max_context_tokens", None),
                need_hidden=False,
            )
            total_budget = max(0, int(max_new_tokens))
            consensus_fed = False
            for idx, role in enumerate(cfg.roles):
                if role != "consensus" and n_latent_roles is not None and idx >= n_latent_roles:
                    break
                # the role tag anchors the latent block to the subspace the
                # student learned for that agent during SFT
                tag = T_CONSENSUS_OPEN if role == "consensus" else agent_tag(role) + "\n"
                past, attention_mask, last_hidden, _loss, next_logits = self._text_steps(
                    [tag], past, attention_mask, checkpoint=False, supervise=False,
                    prev_logits=next_logits,
                    max_total_tokens=getattr(cfg, "max_context_tokens", None),
                    need_hidden=role != "consensus",
                )
                if role == "consensus":
                    consensus_fed = True
                    break
                inputs_embeds = self.align(last_hidden[:, -1:, :])
                inputs_embeds, past, attention_mask, _h, _aux = self._latent_steps(
                    inputs_embeds, past, attention_mask, cfg.latent_steps
                )
                next_logits = _h
                n_latent += cfg.latent_steps
                if not cfg.share_working_memory:
                    # ablation: drop the predecessor's KV working memory
                    past = None
                    attention_mask = torch.ones(
                        1, 1, dtype=attention_mask.dtype, device=attention_mask.device
                    )

            n_memory = int(attention_mask.shape[-1])

            if not consensus_fed:
                past, attention_mask, _h, _loss, next_logits = self._text_steps(
                    [T_CONSENSUS_OPEN], past, attention_mask, checkpoint=False,
                    supervise=False, prev_logits=next_logits,
                    max_total_tokens=getattr(cfg, "max_context_tokens", None),
                    need_hidden=False,
                )

            # Decode SQL directly. No remaining agent debate is generated.
            sql_budget = total_budget
            out_ids: List[int] = []
            spec_stats = None
            if spec is not None:
                # The draft has no access to the latent prefix, so it conditions
                # on the plain text prompt; the target continues from the real
                # latent KV cache. next_logits already predicts the first
                # decoded token, which is what target_first_logits expects.
                draft_prompt_ids = tok(
                    prompt, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0].tolist()
                out_ids = spec.generate(
                    [], sql_budget, eos_token_id=tok.eos_token_id,
                    target_past=past, target_attn=attention_mask,
                    target_first_logits=next_logits,
                    draft_prompt_ids=draft_prompt_ids,
                )
                spec_stats = spec.stats.take().as_dict()
                sql_budget = 0
            for _ in range(sql_budget):
                logits = next_logits
                if logits is None:
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
                next_logits = None

        text = tok.decode(out_ids, skip_special_tokens=True)
        full = T_CONSENSUS_OPEN + text
        return LatentRun(
            text=full,
            sql=extract_final_sql(full),
            n_latent_steps=n_latent,
            n_decoded_tokens=len(out_ids),
            roles=list(cfg.roles),
            n_memory_positions=n_memory,
            spec=spec_stats,
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
