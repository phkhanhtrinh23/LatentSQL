"""Agent subspaces via difference-in-means / CAA (IMAD section 3).

The claim being tested: after internalisation the student holds *separable*
directions for each teacher agent, rather than collapsing them into one
undifferentiated policy. If true, steering along an agent's direction should
make the student's output look more like that agent's, and the effect should be
stronger after IMAD than on the base model.

Method, following Rimsky et al. (2024):

1. For agent i, build contrastive pairs sharing the same context (question +
   debate history up to ``<|agent i|>``). Positive continuation = agent i's real
   response; negative = another agent's response in the same slot.
2. Record residual-stream activations at a chosen layer and take the
   difference in means: ``v_i = mean(pos) - mean(neg)``.
3. At inference add ``alpha * v_i`` to that layer's output.

For Text-to-SQL this has a concrete use beyond interpretability: steering
toward ``divide_conquer`` vs ``query_plan`` at generation time is a cheap
diversity knob, recovering some of CHASE-SQL's multi-path candidate generation
from a *single* forward pass per candidate.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..llm.hf_local import ModelBundle, render_chat
from ..mas.trace import AgentTurn, DebateTrace, agent_tag


# --------------------------------------------------------------------------
# layer access
# --------------------------------------------------------------------------
def get_layers(model) -> Any:
    """Find the ``ModuleList`` of transformer blocks across architectures."""
    for path in (
        ("model", "layers"), ("model", "model", "layers"),
        ("transformer", "h"), ("gpt_neox", "layers"), ("model", "decoder", "layers"),
    ):
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "__len__"):
            return obj
    base = getattr(model, "base_model", None)
    if base is not None and base is not model:
        return get_layers(base)
    raise AttributeError("could not locate the transformer layer list on this model")


def resolve_layer(model, index: int) -> int:
    layers = get_layers(model)
    n = len(layers)
    return index if index >= 0 else max(0, n + index)


# --------------------------------------------------------------------------
# activation capture
# --------------------------------------------------------------------------
@contextmanager
def capture_layer(model, layer_index: int, store: List[Any]):
    layers = get_layers(model)
    idx = resolve_layer(model, layer_index)

    def hook(_mod, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        store.append(hidden.detach())
        return out

    handle = layers[idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def steer_layer(model, layer_index: int, vector, alpha: float = 1.0):
    """Add ``alpha * vector`` to a layer's residual output during generation."""
    layers = get_layers(model)
    idx = resolve_layer(model, layer_index)

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            hidden = out[0]
            return (hidden + alpha * vector.to(hidden.dtype).to(hidden.device),) + out[1:]
        return out + alpha * vector.to(out.dtype).to(out.device)

    handle = layers[idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------
# contrastive pair construction
# --------------------------------------------------------------------------
@dataclass
class ContrastPair:
    context: str
    positive: str
    negative: str
    agent: str


def build_pairs(
    traces: Sequence[DebateTrace],
    agent: str,
    max_pairs: int = 128,
    max_chars: int = 1200,
) -> List[ContrastPair]:
    """Same context, target agent's response vs. another agent's response."""
    pairs: List[ContrastPair] = []
    for tr in traces:
        by_round: Dict[int, List[AgentTurn]] = {}
        for t in tr.turns:
            by_round.setdefault(t.round, []).append(t)
        for r, turns in sorted(by_round.items()):
            target = next((t for t in turns if t.agent == agent), None)
            others = [t for t in turns if t.agent != agent and t.sql]
            if target is None or not others:
                continue
            prefix = _context_prefix(tr, r, agent)
            pairs.append(
                ContrastPair(
                    context=prefix[-max_chars:],
                    positive=_response_text(target)[:max_chars],
                    negative=_response_text(others[0])[:max_chars],
                    agent=agent,
                )
            )
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _response_text(turn: AgentTurn) -> str:
    body = turn.reasoning.strip()
    if turn.sql:
        body += f"\n```sql\n{turn.sql}\n```"
    return body


def _context_prefix(trace: DebateTrace, round_index: int, agent: str) -> str:
    """Everything up to and including ``<|agent X|>`` for the target slot."""
    rendered = trace.render()
    marker = agent_tag(agent)
    # find the occurrence belonging to this round
    pos, seen = -1, -1
    start = 0
    while True:
        i = rendered.find(marker, start)
        if i < 0:
            break
        seen += 1
        if seen == round_index:
            pos = i
            break
        start = i + len(marker)
    if pos < 0:
        pos = rendered.find(marker)
    if pos < 0:
        return trace.question
    return rendered[: pos + len(marker)] + "\n"


# --------------------------------------------------------------------------
# steering vectors
# --------------------------------------------------------------------------
@dataclass
class SteeringVectors:
    layer: int
    vectors: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        import torch

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {"layer": self.layer,
             "vectors": {k: v.cpu() for k, v in self.vectors.items()},
             "meta": self.meta},
            path,
        )

    @staticmethod
    def load(path: str) -> "SteeringVectors":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        return SteeringVectors(layer=blob["layer"], vectors=blob["vectors"],
                               meta=blob.get("meta", {}))


def _pooled_activation(bundle: ModelBundle, layer: int, context: str, response: str,
                       positions: str = "response_mean"):
    """Mean residual activation over the response span (or its last token)."""
    import torch

    tok = bundle.tokenizer
    ctx_ids = tok(context, add_special_tokens=False)["input_ids"]
    full_ids = tok(context + response, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(ctx_ids):
        return None
    ids = torch.tensor([full_ids], device=bundle.device)

    store: List[Any] = []
    with torch.no_grad(), capture_layer(bundle.model, layer, store):
        bundle.model(input_ids=ids, use_cache=False)
    if not store:
        return None
    hidden = store[-1][0]  # (T, d)
    span = hidden[len(ctx_ids):]
    if span.shape[0] == 0:
        return None
    return span[-1].float() if positions == "last_token" else span.mean(dim=0).float()


def extract_steering_vectors(
    bundle: ModelBundle,
    traces: Sequence[DebateTrace],
    agents: Sequence[str],
    cfg: Any,
) -> SteeringVectors:
    """Difference-in-means vector per agent (Marks & Tegmark 2023)."""
    import torch

    out = SteeringVectors(layer=resolve_layer(bundle.model, cfg.layer))
    for agent in agents:
        pairs = build_pairs(traces, agent, max_pairs=cfg.n_pairs)
        pos_acts, neg_acts = [], []
        for p in pairs:
            a = _pooled_activation(bundle, cfg.layer, p.context, p.positive, cfg.positions)
            b = _pooled_activation(bundle, cfg.layer, p.context, p.negative, cfg.positions)
            if a is not None and b is not None:
                pos_acts.append(a)
                neg_acts.append(b)
        if not pos_acts:
            continue
        v = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
        if cfg.normalize:
            n = v.norm()
            if n > 0:
                v = v / n
        out.vectors[agent] = v
        out.meta[agent] = {"n_pairs": len(pos_acts)}
    return out


def cosine_separation(vectors: SteeringVectors) -> Dict[str, float]:
    """Pairwise cosine between agent directions - low values => separable subspaces."""
    import torch

    names = sorted(vectors.vectors)
    out: Dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = vectors.vectors[a], vectors.vectors[b]
            out[f"{a}|{b}"] = float(
                torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0))
            )
    return out


def generate_steered(
    bundle: ModelBundle,
    system: str,
    prompt: str,
    vectors: SteeringVectors,
    agent: str,
    alpha: float = 1.0,
    max_new_tokens: int = 256,
) -> str:
    """Generate with agent-specific steering applied."""
    import torch

    tok = bundle.tokenizer
    text = render_chat(
        tok,
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(bundle.device)
    v = vectors.vectors.get(agent)
    ctx = (
        steer_layer(bundle.model, vectors.layer, v, alpha)
        if v is not None else _null_context()
    )
    with ctx, torch.no_grad():
        out = bundle.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


@contextmanager
def _null_context():
    yield
