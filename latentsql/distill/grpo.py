"""Stage 2: internalisation via GRPO (IMAD section 2.3).

A self-contained GRPO implementation (Shao et al. 2024) rather than a TRL
dependency, because the whole method lives in two things TRL does not expose
cleanly: a reward whose *weights* change every step, and a length clip that
shrinks every step. Keeping the loop here makes the schedule auditable.

Per optimizer step:

1. sample ``group_size`` completions for each prompt in the batch
2. score each with :class:`~latentsql.distill.rewards.DebateReward` at the
   current ``(w_fmt, l)``
3. advantage = (r - group mean) / group std
4. maximise the clipped policy-gradient objective with a KL penalty to the
   frozen SFT reference

Because the reward is computed on the *token-clipped* prefix, and the clip
shrinks toward ``len_end``, the only way to keep earning reward is to stop
verbalising the debate and put the answer up front. That is internalisation.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm.hf_local import ModelBundle, render_chat
from ..mas.prompts import student_prompt
from .rewards import DebateReward, ExecutionOracle, RewardBreakdown, RewardSchedule, group_advantages


@dataclass
class Rollout:
    prompt_ids: List[int]
    completion_ids: List[int]
    text: str
    reward: RewardBreakdown
    advantage: float = 0.0
    logprob_old: Optional[Any] = None


@dataclass
class GRPOTask:
    """One training prompt plus everything needed to score its rollouts."""

    system: str
    prompt: str
    db_path: str
    gold_sql: str
    qid: str = ""


def tasks_from_records(records: Sequence[Any]) -> List[GRPOTask]:
    out = []
    for r in records:
        if not (getattr(r, "db_path", "") and getattr(r, "gold_sql", "")):
            continue
        out.append(
            GRPOTask(
                system=r.system, prompt=r.prompt, db_path=r.db_path,
                gold_sql=r.gold_sql, qid=r.qid,
            )
        )
    return out


def tasks_from_examples(examples: Sequence[Any], schema_render) -> List[GRPOTask]:
    """Build GRPO tasks straight from benchmark examples (no teacher needed)."""
    out = []
    for ex in examples:
        sp = student_prompt(
            schema=schema_render(ex), question=ex.question, evidence=ex.evidence
        )
        out.append(
            GRPOTask(
                system=sp["system"], prompt=sp["user"],
                db_path=ex.db_path, gold_sql=ex.gold_sql, qid=ex.qid,
            )
        )
    return out


# --------------------------------------------------------------------------
#: how many sequence positions are upcast to float32 at once when scoring
#: log-probabilities. 512 x 152k vocab is ~310MB per block.
_LOGP_CHUNK = 512


def _seq_logprobs(model, input_ids, attention_mask, completion_start: int):
    """Sum of log p(token) over the completion span, per sequence.

    The per-token log-probability comes from ``cross_entropy``, not from an
    explicit ``log_softmax``: the latter materialises a float32
    ``[batch, seq, vocab]`` tensor, which for a Qwen vocabulary (~152k) is
    ~2.4GB *per 4k-token sequence* and is what puts a GRPO group over the
    memory of a 24GB card. ``cross_entropy`` fuses the same computation.
    """
    import torch
    import torch.nn.functional as F

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    # Upcasting the whole logits block to float32 is what actually blows the
    # memory budget: [1, 4000, 152k] in fp32 is ~2.4GB, and the transpose
    # forces a second contiguous copy. Walk the time axis in blocks so the
    # fp32 buffer stays bounded; autograd stitches the blocks back together,
    # so the gradient is unchanged.
    parts = []
    for i in range(0, logits.shape[1], _LOGP_CHUNK):
        lg = logits[:, i : i + _LOGP_CHUNK, :]
        tg = targets[:, i : i + _LOGP_CHUNK]
        parts.append(
            -F.cross_entropy(lg.transpose(1, 2).float(), tg, reduction="none")
        )
    token_logp = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    del out, logits

    mask = attention_mask[:, 1:].clone().float()
    mask[:, : max(0, completion_start - 1)] = 0.0
    return (token_logp * mask).sum(dim=-1), token_logp, mask


def _clone_cache(past):
    """Copy a KV cache so k rollouts can branch from one latent prefix."""
    try:
        import copy

        return copy.deepcopy(past)
    except Exception:
        return past


class GRPOTrainer:
    def __init__(
        self,
        bundle: ModelBundle,
        cfg: Any,
        oracle: ExecutionOracle,
        total_steps: Optional[int] = None,
        memory: Optional[Any] = None,
        latent: Optional[Any] = None,
        spec: Optional[Any] = None,
    ):
        self.bundle = bundle
        #: when set, rollouts are sampled from a latent debate prefix instead of
        #: verbalised text, and the loss backpropagates through the latent steps
        self.latent = latent
        #: SpeculativeSampler; lossless, so it changes throughput only
        self.spec = spec
        self.cfg = cfg
        self.oracle = oracle
        self.schedule = RewardSchedule(cfg.reward, total_steps=total_steps or 100)
        self.memory = memory
        self.reward_fn = DebateReward(
            cfg.reward,
            oracle,
            self.schedule,
            decode=lambda ids: bundle.tokenizer.decode(list(ids), skip_special_tokens=True),
            memory=memory,
        )
        self.history: List[Dict[str, Any]] = []

    # -- rollouts ---------------------------------------------------------
    def sample_group(self, task: GRPOTask, step: int) -> List[Rollout]:
        if self.latent is not None:
            return self._sample_group_latent(task, step)
        import torch

        tok, model = self.bundle.tokenizer, self.bundle.model
        text = render_chat(
            tok,
            [{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}],
            add_generation_prompt=True,
        )
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(self.bundle.device)
        prompt_len = int(enc["input_ids"].shape[1])

        was_training = model.training
        model.eval()
        if self.spec is not None:
            prompt_ids = enc["input_ids"][0].tolist()
            rollouts: List[Rollout] = []
            with torch.no_grad():
                for _ in range(max(1, self.cfg.group_size)):
                    ids = self.spec.generate(
                        prompt_ids, self.cfg.max_new_tokens,
                        eos_token_id=tok.eos_token_id,
                    )
                    text = tok.decode(ids, skip_special_tokens=True)
                    rb = self.reward_fn(text, ids, task.db_path, task.gold_sql, step)
                    rollouts.append(Rollout(prompt_ids=prompt_ids, completion_ids=ids,
                                            text=text, reward=rb))
            if was_training:
                model.train()
            advs = group_advantages([r.reward.total for r in rollouts])
            for r, a in zip(rollouts, advs):
                r.advantage = a
            return rollouts
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                num_return_sequences=self.cfg.group_size,
                pad_token_id=tok.pad_token_id,
            )
        if was_training:
            model.train()

        rollouts: List[Rollout] = []
        prompt_ids = enc["input_ids"][0].tolist()
        for row in gen:
            comp_ids = row[prompt_len:].tolist()
            while comp_ids and comp_ids[-1] == tok.pad_token_id:
                comp_ids.pop()
            comp_text = tok.decode(comp_ids, skip_special_tokens=True)
            rb = self.reward_fn(comp_text, comp_ids, task.db_path, task.gold_sql, step)
            rollouts.append(
                Rollout(prompt_ids=prompt_ids, completion_ids=comp_ids, text=comp_text, reward=rb)
            )

        advs = group_advantages([r.reward.total for r in rollouts])
        for r, a in zip(rollouts, advs):
            r.advantage = a
        return rollouts

    # -- latent rollouts --------------------------------------------------
    def _sample_group_latent(self, task: GRPOTask, step: int) -> List[Rollout]:
        """Sample k completions that follow a *latent* debate, not a written one.

        The roles run as latent steps (zero tokens emitted); only the query is
        decoded. So a rollout's completion is just the SQL, which is why the
        length clip and the format reward have nothing left to do here - the
        debate costs no tokens by construction.
        """
        import torch

        tok, model = self.bundle.tokenizer, self.bundle.model
        was_training = model.training
        model.eval()
        rollouts: List[Rollout] = []
        with torch.no_grad():
            past0, attn0, logits0 = self.latent.build_prefix(task.system, task.prompt)
            if self.spec is not None:
                dp = tok(task.prompt, return_tensors="pt",
                         add_special_tokens=False)["input_ids"][0].tolist()
                for _ in range(max(1, self.cfg.group_size)):
                    ids = self.spec.generate(
                        [], self.cfg.max_new_tokens, eos_token_id=tok.eos_token_id,
                        target_past=_clone_cache(past0), target_attn=attn0.clone(),
                        target_first_logits=logits0.clone(), draft_prompt_ids=dp,
                    )
                    text = tok.decode(ids, skip_special_tokens=True)
                    rb = self.reward_fn(text, ids, task.db_path, task.gold_sql, step)
                    rollouts.append(Rollout(prompt_ids=[], completion_ids=ids,
                                            text=text, reward=rb))
                if was_training:
                    model.train()
                advs = group_advantages([r.reward.total for r in rollouts])
                for r, a in zip(rollouts, advs):
                    r.advantage = a
                return rollouts
            for _ in range(max(1, self.cfg.group_size)):
                past = _clone_cache(past0)
                attn = attn0.clone()
                logits = logits0.clone()
                ids: List[int] = []
                for _t in range(self.cfg.max_new_tokens):
                    if self.cfg.temperature and self.cfg.temperature > 0:
                        probs = torch.softmax(logits.float() / self.cfg.temperature, dim=-1)
                        nxt = int(torch.multinomial(probs, 1).item())
                    else:
                        nxt = int(logits.argmax(-1).item())
                    if nxt == tok.eos_token_id:
                        break
                    ids.append(nxt)
                    step_ids = torch.tensor([[nxt]], device=self.bundle.device)
                    attn = torch.cat(
                        [attn, torch.ones(1, 1, dtype=attn.dtype, device=attn.device)], dim=-1
                    )
                    out = model(input_ids=step_ids, past_key_values=past,
                                attention_mask=attn, use_cache=True)
                    past, logits = out.past_key_values, out.logits[:, -1, :]
                text = tok.decode(ids, skip_special_tokens=True)
                rb = self.reward_fn(text, ids, task.db_path, task.gold_sql, step)
                rollouts.append(Rollout(prompt_ids=[], completion_ids=ids,
                                        text=text, reward=rb))
        if was_training:
            model.train()
        advs = group_advantages([r.reward.total for r in rollouts])
        for r, a in zip(rollouts, advs):
            r.advantage = a
        return rollouts

    def _loss_for_latent(self, rollouts: Sequence[Rollout], ref_model, scale: float):
        """GRPO loss whose gradient flows through the latent debate steps."""
        import torch
        import torch.nn.functional as F

        active = [r for r in rollouts if abs(r.advantage) > 1e-8 and r.completion_ids]
        if not active:
            return None
        task = self._cur_task
        denom = float(max(1, sum(len(r.completion_ids) for r in active)))
        total = 0.0

        for r in active:
            # rebuild the latent prefix WITH grad: this is the path the policy
            # is actually optimised through
            past, attn, first_logits = self.latent.build_prefix(
                task.system, task.prompt, grad=True
            )
            ids = torch.tensor([r.completion_ids], device=self.bundle.device)
            lp = [F.log_softmax(first_logits.float(), dim=-1)[0, r.completion_ids[0]]]
            if len(r.completion_ids) > 1:
                attn2 = torch.cat(
                    [attn, torch.ones(1, ids.shape[1] - 1, dtype=attn.dtype,
                                      device=attn.device)], dim=-1
                )
                out = self.bundle.model(input_ids=ids[:, :-1], past_key_values=past,
                                        attention_mask=attn2, use_cache=False)
                logp = F.log_softmax(out.logits.float(), dim=-1)
                tgt = ids[:, 1:]
                lp.append(logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())
            token_logp = torch.stack([x.reshape(()) if x.dim() == 0 else x.sum()
                                      for x in lp]).sum()
            loss = -(r.advantage * token_logp) / denom
            (loss * scale).backward()
            total += float(loss.detach())
            del past, attn, first_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return total

    # -- loss -------------------------------------------------------------
    def _loss_for(self, rollouts: Sequence[Rollout], ref_model, scale: float = 1.0) -> Any:
        """Accumulate the GRPO gradient for one rollout group.

        The group is walked in chunks of ``cfg.micro_batch`` and each chunk's
        contribution is backwarded as it is computed, rather than holding one
        activation graph over the whole group. A group is
        ``group_size`` x (prompt + completion) tokens long - on BIRD, where the
        prompt alone is thousands of tokens, materialising that at once OOMs a
        24GB card. Chunking changes only peak memory: every chunk is divided by
        the *group-wide* token count, so the accumulated gradient is identical
        to the single-batch form.

        Returns the (unscaled) group loss as a float, or ``None`` when every
        advantage is zero and there is nothing to learn from.
        """
        import torch

        tok = self.bundle.tokenizer
        active = [r for r in rollouts if abs(r.advantage) > 1e-8 and r.completion_ids]
        if not active:
            return None

        start = len(active[0].prompt_ids)
        # mask counts exactly the completion tokens of each rollout, so the
        # group-wide denominator needs no forward pass
        denom = float(max(1, sum(len(r.completion_ids) for r in active)))
        pad = tok.pad_token_id
        chunk = max(1, int(getattr(self.cfg, "micro_batch", 1)))
        total = 0.0

        for i in range(0, len(active), chunk):
            part = active[i : i + chunk]
            seqs = [r.prompt_ids + r.completion_ids for r in part]
            n = max(len(s) for s in seqs)
            input_ids = torch.tensor(
                [s + [pad] * (n - len(s)) for s in seqs], dtype=torch.long
            ).to(self.bundle.device)
            attn = torch.tensor(
                [[1] * len(s) + [0] * (n - len(s)) for s in seqs], dtype=torch.long
            ).to(self.bundle.device)

            _, token_logp, mask = _seq_logprobs(self.bundle.model, input_ids, attn, start)
            with torch.no_grad():
                _, ref_logp, _ = _seq_logprobs(ref_model, input_ids, attn, start)

            adv = torch.tensor(
                [r.advantage for r in part], dtype=torch.float32, device=self.bundle.device
            ).unsqueeze(-1)

            # We take exactly one gradient update per rollout group, so the
            # policy that generated the group *is* the policy being scored: the
            # importance ratio is identically 1 and the clipped surrogate
            # reduces to -advantage * logp. The clipped form is written out
            # anyway so that raising the number of inner epochs stays a
            # one-line change.
            ratio = torch.exp(token_logp - token_logp.detach())
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
            pg = -torch.min(unclipped, clipped)

            # k3 estimator of KL(pi || pi_ref), the low-variance form used by GRPO
            delta = ref_logp - token_logp
            kl = torch.exp(delta) - delta - 1.0

            per_token = pg + self.cfg.kl_beta * kl
            loss = (per_token * mask).sum() / denom
            (loss * scale).backward()
            total += float(loss.detach())

            del input_ids, attn, token_logp, mask, ref_logp, per_token, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total

    # -- loop -------------------------------------------------------------
    def train(
        self,
        tasks: Sequence[GRPOTask],
        ref_model,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        import torch

        cfg = self.cfg
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        model = self.bundle.model
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=cfg.lr)

        n_batches = max(1, math.ceil(len(tasks) / cfg.batch_prompts))
        total_opt = max(1, int(n_batches * cfg.epochs / cfg.grad_accum))
        self.schedule.total_steps = total_opt
        print(
            f"[grpo] {len(tasks)} prompts | group={cfg.group_size} | "
            f"{total_opt} optimizer steps | w_fmt {cfg.reward.w_fmt_start}->{cfg.reward.w_fmt_end} "
            f"| l {cfg.reward.len_start}->{cfg.reward.len_end}",
            flush=True,
        )

        from ..utils.telemetry import RunLog, gpu_stats, reset_gpu_peak, run_environment

        steplog = None
        if log_path:
            steplog = RunLog(
                os.path.splitext(log_path)[0] + "_steps.jsonl",
                header=run_environment(
                    stage="grpo", optimizer_steps=total_opt,
                    group_size=cfg.group_size, n_tasks=len(tasks),
                    quantization=getattr(cfg, "quantization", "none"),
                ),
            )
        reset_gpu_peak()

        order: List[int] = []
        gen_tokens_total = 0
        micro, opt_step, t0 = 0, 0, time.time()
        optim.zero_grad(set_to_none=True)

        while opt_step < total_opt:
            if not order:
                order = list(range(len(tasks)))
                random.shuffle(order)
            take = [order.pop() for _ in range(min(cfg.batch_prompts, len(order)))]

            batch_rewards, batch_lens, batch_fmt, batch_clip = [], [], [], []
            batch_std, batch_live = [], []
            for ti in take:
                self._cur_task = tasks[ti]
                rollouts = self.sample_group(tasks[ti], opt_step)
                # A group whose rewards are all equal produces all-zero
                # advantages and therefore NO gradient at all. Tracking how
                # often that happens is the difference between "training is
                # slow" and "training is not happening" - log it, do not
                # infer it from the reward mean.
                _rs = [r.reward.total for r in rollouts]
                _m = sum(_rs) / max(1, len(_rs))
                batch_std.append(
                    (sum((x - _m) ** 2 for x in _rs) / max(1, len(_rs))) ** 0.5
                )
                batch_live += [1.0 if abs(r.advantage) > 1e-8 else 0.0 for r in rollouts]
                batch_rewards += [r.reward.total for r in rollouts]
                batch_lens += [r.reward.n_tokens for r in rollouts]
                batch_fmt += [r.reward.r_fmt for r in rollouts]
                batch_clip += [r.reward.r_clip for r in rollouts]
                gen_tokens_total += sum(len(r.completion_ids) for r in rollouts)

                # the loss backwards chunk by chunk inside _loss_for, so a
                # long BIRD group never has to fit in memory all at once
                if self.latent is not None:
                    self._loss_for_latent(rollouts, ref_model, scale=1.0 / cfg.grad_accum)
                else:
                    self._loss_for(rollouts, ref_model, scale=1.0 / cfg.grad_accum)
            micro += 1

            if micro % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % cfg.logging_steps == 0 or opt_step == total_opt:
                    snap = self.schedule.snapshot(opt_step)
                    rec = {
                        **snap,
                        "reward_mean": _mean(batch_rewards),
                        "r_fmt_mean": _mean(batch_fmt),
                        "r_clip_mean": _mean(batch_clip),
                        "gen_tokens_mean": _mean(batch_lens),
                        #: within-group reward spread; 0 => no learning signal
                        "reward_std_mean": _mean(batch_std),
                        **(self.spec.stats.as_dict() if self.spec is not None else {}),
                        #: fraction of rollouts that actually contributed a
                        #: gradient this step
                        "adv_nonzero_frac": _mean(batch_live),
                        "gen_tokens_total": gen_tokens_total,
                        "n_executions": self.oracle.n_exec,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    rec.update(gpu_stats())
                    self.history.append(rec)
                    if steplog:
                        steplog.write({"record": "step", **rec})
                    print(f"[grpo] {rec}", flush=True)
                if cfg.save_steps and opt_step % cfg.save_steps == 0:
                    self.save(cfg.output_dir)

        self.save(cfg.output_dir)
        result = {
            "output_dir": cfg.output_dir,
            "model": getattr(self.bundle, "name", ""),
            "quantization": getattr(cfg, "quantization", "none"),
            "optimizer_steps": opt_step,
            "n_executions": self.oracle.n_exec,
            "gen_tokens_total": gen_tokens_total,
            "elapsed_s": round(time.time() - t0, 1),
            **gpu_stats(),
            "history": self.history,
        }
        if steplog:
            steplog.write({"record": "summary", **{k: v for k, v in result.items()
                                                   if k != "history"}})
            steplog.close()
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
        return result

    def save(self, output_dir: str) -> None:
        from .sft import save_student

        save_student(self.bundle, output_dir)


def _mean(xs: Sequence[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def load_reference(bundle: ModelBundle):
    """The frozen SFT policy the KL term pulls toward.

    With LoRA we do not need a second copy of the weights: disabling the
    adapters recovers the base model, and the SFT checkpoint *is* the base +
    adapters, so we snapshot the adapter state instead.
    """
    import copy

    import torch

    model = bundle.model
    if hasattr(model, "disable_adapter"):
        return _AdapterReference(model)
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


class _AdapterReference:
    """Calls the base model with LoRA temporarily disabled."""

    def __init__(self, peft_model):
        self.peft_model = peft_model

    def __call__(self, **kw):
        import torch

        with torch.no_grad(), self.peft_model.disable_adapter():
            return self.peft_model(**kw)

    def eval(self):
        return self

    def parameters(self):
        return self.peft_model.parameters()
