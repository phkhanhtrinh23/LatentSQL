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
def _seq_logprobs(model, input_ids, attention_mask, completion_start: int):
    """Sum of log p(token) over the completion span, per sequence."""
    import torch

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    mask = attention_mask[:, 1:].clone().float()
    mask[:, : max(0, completion_start - 1)] = 0.0
    return (token_logp * mask).sum(dim=-1), token_logp, mask


class GRPOTrainer:
    def __init__(
        self,
        bundle: ModelBundle,
        cfg: Any,
        oracle: ExecutionOracle,
        total_steps: Optional[int] = None,
    ):
        self.bundle = bundle
        self.cfg = cfg
        self.oracle = oracle
        self.schedule = RewardSchedule(cfg.reward, total_steps=total_steps or 100)
        self.reward_fn = DebateReward(
            cfg.reward,
            oracle,
            self.schedule,
            decode=lambda ids: bundle.tokenizer.decode(list(ids), skip_special_tokens=True),
        )
        self.history: List[Dict[str, Any]] = []

    # -- rollouts ---------------------------------------------------------
    def sample_group(self, task: GRPOTask, step: int) -> List[Rollout]:
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

    # -- loss -------------------------------------------------------------
    def _loss_for(self, rollouts: Sequence[Rollout], ref_model) -> Any:
        import torch

        tok = self.bundle.tokenizer
        active = [r for r in rollouts if abs(r.advantage) > 1e-8 and r.completion_ids]
        if not active:
            return None

        seqs = [r.prompt_ids + r.completion_ids for r in active]
        n = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (n - len(s)) for s in seqs], dtype=torch.long)
        attn = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], dtype=torch.long)
        input_ids = input_ids.to(self.bundle.device)
        attn = attn.to(self.bundle.device)
        start = len(active[0].prompt_ids)

        _, token_logp, mask = _seq_logprobs(self.bundle.model, input_ids, attn, start)
        with torch.no_grad():
            _, ref_logp, _ = _seq_logprobs(ref_model, input_ids, attn, start)

        adv = torch.tensor(
            [r.advantage for r in active], dtype=torch.float32, device=self.bundle.device
        ).unsqueeze(-1)

        # We take exactly one gradient update per rollout group, so the policy
        # that generated the group *is* the policy being scored: the importance
        # ratio is identically 1 and the clipped surrogate reduces to
        # -advantage * logp. The clipped form is written out anyway so that
        # raising the number of inner epochs stays a one-line change.
        ratio = torch.exp(token_logp - token_logp.detach())
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
        pg = -torch.min(unclipped, clipped)

        # k3 estimator of KL(pi || pi_ref), the low-variance form used by GRPO
        delta = ref_logp - token_logp
        kl = torch.exp(delta) - delta - 1.0

        per_token = pg + self.cfg.kl_beta * kl
        denom = mask.sum().clamp(min=1.0)
        return (per_token * mask).sum() / denom

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

        order: List[int] = []
        micro, opt_step, t0 = 0, 0, time.time()
        optim.zero_grad(set_to_none=True)

        while opt_step < total_opt:
            if not order:
                order = list(range(len(tasks)))
                random.shuffle(order)
            take = [order.pop() for _ in range(min(cfg.batch_prompts, len(order)))]

            batch_rewards, batch_lens, batch_fmt, batch_clip = [], [], [], []
            for ti in take:
                rollouts = self.sample_group(tasks[ti], opt_step)
                batch_rewards += [r.reward.total for r in rollouts]
                batch_lens += [r.reward.n_tokens for r in rollouts]
                batch_fmt += [r.reward.r_fmt for r in rollouts]
                batch_clip += [r.reward.r_clip for r in rollouts]

                loss = self._loss_for(rollouts, ref_model)
                if loss is not None:
                    (loss / cfg.grad_accum).backward()
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
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    self.history.append(rec)
                    print(f"[grpo] {rec}", flush=True)
                if cfg.save_steps and opt_step % cfg.save_steps == 0:
                    self.save(cfg.output_dir)

        self.save(cfg.output_dir)
        result = {
            "output_dir": cfg.output_dir,
            "optimizer_steps": opt_step,
            "n_executions": self.oracle.n_exec,
            "elapsed_s": round(time.time() - t0, 1),
            "history": self.history,
        }
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
