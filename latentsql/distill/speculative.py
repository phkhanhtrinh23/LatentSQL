"""Lossless speculative decoding for GRPO rollouts.

After Iso et al., *Accelerating RL Post-Training Rollouts via System-Integrated
Speculative Decoding* (NVIDIA, 2026): RL post-training is dominated by rollout
generation, and speculative decoding attacks exactly that without perturbing
the sampling distribution. That last property is what makes it usable in RL at
all - methods that trade distribution fidelity for throughput (off-policy
replay, low-precision rollouts) buy speed with *effectiveness*, whereas the
rejection procedure here is distribution-preserving by construction, so the
policy gradient is unchanged.

The algorithm is Leviathan et al. (2023):

    1. the draft proposes k tokens autoregressively from distributions q_i
    2. the target scores all k in ONE forward pass, giving p_1..p_{k+1}
    3. token i is accepted with probability min(1, p_i(x_i) / q_i(x_i));
       on the first rejection we resample from norm(max(0, p_i - q_i))
    4. if all k are accepted, a bonus token is drawn from p_{k+1}

Crucially the accepted sequence is distributed **exactly** as if sampled from
the target alone, for *any* draft q. A poor draft costs acceptance rate, never
correctness - which is why we can draft from the plain prompt even when the
target is continuing from a latent debate prefix the draft cannot see.

Two cautions carried over from the paper:

* Speedup is bounded by Amdahl over the non-generation stages,
  ``S <= 1 / (R_gen/alpha + (1 - R_gen))`` with ``alpha`` the mean accepted
  length. Verification is not free: they measure n-gram drafting as *slower*
  than plain autoregression despite alpha ~2.5, because the verify pass eats
  the gain. We therefore measure alpha and wall-clock, and log both.
* Log-probabilities, KL and the policy loss must be computed against the
  **target**, never the draft. Nothing here touches the loss path; this module
  only produces token ids.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class SpecStats:
    """Telemetry for one generation call."""

    n_tokens: int = 0
    n_proposed: int = 0
    n_accepted: int = 0
    n_rounds: int = 0
    seconds: float = 0.0

    @property
    def acceptance_length(self) -> float:
        """Mean tokens produced per speculation round (the paper's ``alpha``)."""
        return (self.n_tokens / self.n_rounds) if self.n_rounds else 0.0

    @property
    def acceptance_rate(self) -> float:
        return (self.n_accepted / self.n_proposed) if self.n_proposed else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spec_tokens": self.n_tokens,
            "spec_rounds": self.n_rounds,
            "spec_acceptance_length": round(self.acceptance_length, 3),
            "spec_acceptance_rate": round(self.acceptance_rate, 3),
            "spec_seconds": round(self.seconds, 3),
        }

    def take(self) -> "SpecStats":
        """Return the stats accumulated since the last call, and reset.

        Logging the cumulative average would hide drift: the draft is fixed
        while GRPO moves the policy, so acceptance is expected to decay, and a
        running mean smears that out.
        """
        out = SpecStats(self.n_tokens, self.n_proposed, self.n_accepted,
                        self.n_rounds, self.seconds)
        self.n_tokens = self.n_proposed = self.n_accepted = self.n_rounds = 0
        self.seconds = 0.0
        return out

    def merge(self, other: "SpecStats") -> None:
        self.n_tokens += other.n_tokens
        self.n_proposed += other.n_proposed
        self.n_accepted += other.n_accepted
        self.n_rounds += other.n_rounds
        self.seconds += other.seconds


def _copy_cache(past):
    """Deep-copy a KV cache so a verification pass cannot mutate the real one."""
    import copy

    return copy.deepcopy(past) if past is not None else None


def _sample(probs, temperature: float):
    """Draw one token id from a [1, V] probability row."""
    import torch

    if temperature and temperature > 0:
        return int(torch.multinomial(probs, 1).item())
    return int(probs.argmax(-1).item())


def _to_probs(logits, temperature: float):
    import torch

    if temperature and temperature > 0:
        return torch.softmax(logits.float() / temperature, dim=-1)
    # greedy: a point mass keeps the accept test well defined (ratio 0 or >=1)
    out = torch.zeros_like(logits, dtype=torch.float32)
    out[0, int(logits.argmax(-1).item())] = 1.0
    return out


class SpeculativeSampler:
    """Draft-then-verify decoding that preserves the target's distribution.

    ``draft`` may be any causal LM sharing the target's tokenizer. It is only a
    proposal distribution: correctness never depends on its quality.
    """

    def __init__(
        self,
        target_bundle: Any,
        draft_bundle: Any,
        k: int = 4,
        temperature: float = 1.0,
    ):
        self.target = target_bundle
        self.draft = draft_bundle
        self.k = max(1, int(k))
        self.temperature = temperature
        self.stats = SpecStats()

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        target_past: Any = None,
        target_attn: Any = None,
        target_first_logits: Any = None,
        draft_prompt_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Generate token ids for the target model.

        ``target_past`` / ``target_attn`` / ``target_first_logits`` let the
        caller hand in a prefix the target has already processed - a latent
        debate prefix, for example. The draft cannot see such a prefix, so it
        conditions on ``draft_prompt_ids`` (the plain text prompt) instead;
        rejection sampling absorbs the mismatch, at the cost of a lower
        acceptance rate.
        """
        import torch

        t0 = time.time()
        stats = SpecStats()
        tgt, drf = self.target.model, self.draft.model
        dev = self.target.device

        # -- prime the target -------------------------------------------
        if target_past is None:
            ids = torch.tensor([list(prompt_ids)], device=dev)
            attn = torch.ones_like(ids)
            out = tgt(input_ids=ids, attention_mask=attn, use_cache=True)
            t_past, t_attn = out.past_key_values, attn
            t_logits = out.logits[:, -1, :]
        else:
            t_past, t_attn, t_logits = target_past, target_attn, target_first_logits

        # -- prime the draft --------------------------------------------
        d_ids = list(draft_prompt_ids if draft_prompt_ids is not None else prompt_ids)
        d_in = torch.tensor([d_ids or [eos_token_id or 0]], device=self.draft.device)
        d_out = drf(input_ids=d_in, attention_mask=torch.ones_like(d_in), use_cache=True)
        d_past, d_logits = d_out.past_key_values, d_out.logits[:, -1, :]
        d_attn = torch.ones_like(d_in)

        emitted: List[int] = []
        while len(emitted) < max_new_tokens:
            stats.n_rounds += 1
            budget = min(self.k, max_new_tokens - len(emitted))

            # 1. draft proposes `budget` tokens, remembering its own q_i
            props: List[int] = []
            qs = []
            # the proposal loop mutates the cache in place, so branch off a copy:
            # only the ACCEPTED tokens may advance the draft's real state
            dp, dl, da = _copy_cache(d_past), d_logits, d_attn.clone()
            for _ in range(budget):
                q = _to_probs(dl, self.temperature)
                x = _sample(q, self.temperature)
                qs.append(q)
                props.append(x)
                step = torch.tensor([[x]], device=self.draft.device)
                da = torch.cat([da, torch.ones(1, 1, dtype=da.dtype, device=da.device)], -1)
                o = drf(input_ids=step, past_key_values=dp, attention_mask=da, use_cache=True)
                dp, dl = o.past_key_values, o.logits[:, -1, :]
            stats.n_proposed += len(props)

            # 2. the target scores all proposals in ONE pass
            #    t_logits already holds p_1; the forward gives p_2..p_{k+1}
            p_rows = [_to_probs(t_logits, self.temperature)]
            prop_t = torch.tensor([props], device=dev)
            t_attn_ext = torch.cat(
                [t_attn, torch.ones(1, len(props), dtype=t_attn.dtype, device=dev)], -1
            )
            # verify on a COPY: this pass advances the cache by all k proposals,
            # but only the accepted prefix may be committed, so the real cache
            # must stay where it was until we know how many survived
            v = tgt(input_ids=prop_t, past_key_values=_copy_cache(t_past),
                    attention_mask=t_attn_ext, use_cache=True)
            for i in range(len(props)):
                p_rows.append(_to_probs(v.logits[:, i, :], self.temperature))

            # 3. accept / reject, preserving the target distribution
            n_acc = 0
            resampled: Optional[int] = None
            for i, x in enumerate(props):
                p_x = float(p_rows[i][0, x])
                q_x = float(qs[i][0, x])
                ratio = 1.0 if q_x <= 0 else min(1.0, p_x / q_x)
                if float(torch.rand(1).item()) < ratio:
                    n_acc += 1
                    continue
                resid = torch.clamp(p_rows[i] - qs[i], min=0.0)
                s = float(resid.sum())
                resampled = (_sample(resid / s, self.temperature) if s > 0
                             else _sample(p_rows[i], self.temperature))
                break
            stats.n_accepted += n_acc

            new = props[:n_acc] + ([resampled] if resampled is not None else [])
            if resampled is None and len(new) == len(props):
                # every proposal accepted -> free bonus token from p_{k+1}
                new.append(_sample(p_rows[len(props)], self.temperature))

            # 4. commit: re-advance the target over exactly what we kept
            keep = new[: max_new_tokens - len(emitted)]
            ids = torch.tensor([keep], device=dev)
            t_attn = torch.cat(
                [t_attn, torch.ones(1, len(keep), dtype=t_attn.dtype, device=dev)], -1
            )
            o = tgt(input_ids=ids, past_key_values=t_past,
                    attention_mask=t_attn, use_cache=True)
            t_past, t_logits = o.past_key_values, o.logits[:, -1, :]

            # the draft must follow the *accepted* path, not its own guesses
            d_ids_new = torch.tensor([keep], device=self.draft.device)
            d_attn = torch.cat(
                [d_attn, torch.ones(1, len(keep), dtype=d_attn.dtype,
                                    device=self.draft.device)], -1
            )
            od = drf(input_ids=d_ids_new, past_key_values=d_past,
                     attention_mask=d_attn, use_cache=True)
            d_past, d_logits = od.past_key_values, od.logits[:, -1, :]

            emitted.extend(keep)
            stats.n_tokens += len(keep)
            if eos_token_id is not None and eos_token_id in keep:
                emitted = emitted[: emitted.index(eos_token_id)]
                break

        stats.seconds = time.time() - t0
        self.stats.merge(stats)
        return emitted


def load_draft(name: str, hub_dir: Optional[str] = None, device: str = "auto",
               quantization: str = "none") -> Any:
    """Load the drafter. It only has to share the target's tokenizer."""
    from ..llm.hf_local import load_model, resolve_model_path

    path = resolve_model_path(name, hub_dir=hub_dir, local_only=False)
    return load_model(path, device=device, hub_dir=hub_dir, quantization=quantization)
