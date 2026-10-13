"""Dr. GRPO on the latent student (Sec. 4.3, Figure 2-3).

For each training question:

1. run the J roles as latent steps (no gradient) and sample ``group_size``
   SQL completions after <sql> at temperature 1.0;
2. score each with ``s(Y) = R_atomic + 1[EX=1] R_spur`` (``rewards.py``);
3. advantage ``A_i = s_i - mean(s)`` (Dr. GRPO: no std normalisation);
4. maximise the clipped surrogate. The log-probability of a rollout is
   recomputed with the same forward pass as the curriculum (``stage_loss`` at
   stage J), so the gradient flows back through every latent step. As in Dr.
   GRPO, the token sum is divided by a constant (``group_size *
   max_new_tokens``) rather than by each rollout's length.

One policy update is taken per rollout group, so the importance ratio equals 1
at the update and the clipping range only matters if more inner epochs are
added. The latent rollouts carry no written reasoning, so no format or length
reward is used.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..latent.curriculum import stage_loss
from ..latent.reasoner import LatentReasoner
from ..prompts import extract_sql
from .rewards import LatentSQLReward, Reward, group_advantages


@dataclass
class Rollout:
    ids: List[int]
    text: str
    reward: Reward
    advantage: float = 0.0


class GRPOTrainer:
    def __init__(self, bundle: Any, reasoner: LatentReasoner, reward: LatentSQLReward, cfg: Any):
        self.bundle = bundle
        self.reasoner = reasoner
        self.reward = reward
        self.cfg = cfg
        self.history: List[Dict[str, Any]] = []

    def sample_group(self, task: Dict[str, Any]) -> List[Rollout]:
        model, tok = self.bundle.model, self.bundle.tokenizer
        was_training = model.training
        model.eval()
        self.reasoner.set_schema(task["db_path"], task.get("db_id"))
        past, logits, _ = self.reasoner.prefix(task["system"], task["prompt"])
        rollouts = []
        for _ in range(self.cfg.group_size):
            ids = self.reasoner.decode(_copy_cache(past), logits.clone(),
                                       self.cfg.max_new_tokens, self.cfg.temperature)
            text = tok.decode(ids, skip_special_tokens=True)
            rollouts.append(Rollout(ids=ids, text=text,
                                    reward=self.reward(extract_sql(text), task["db_path"],
                                                       task["gold_sql"])))
        if was_training:
            model.train()
        for r, a in zip(rollouts, group_advantages([r.reward.total for r in rollouts])):
            r.advantage = a
        return rollouts

    def group_loss(self, task: Dict[str, Any], rollouts: Sequence[Rollout], scale: float) -> float:
        """Backward the clipped surrogate of one group; returns the loss value."""
        import torch

        J = len(self.reasoner.roles)
        denom = float(self.cfg.group_size * self.cfg.max_new_tokens)
        total = 0.0
        for r in rollouts:
            if abs(r.advantage) < 1e-8 or not r.ids:
                continue
            ids = r.ids + ([self.bundle.tokenizer.eos_token_id]
                           if len(r.ids) < self.cfg.max_new_tokens else [])
            out = stage_loss(self.reasoner, task, J, ids)
            logp = -out["main"] * out["n_tokens"]            # sum of token log-probs
            ratio = torch.exp(logp - logp.detach())
            adv = torch.tensor(r.advantage, device=logp.device)
            surrogate = torch.min(ratio * adv, ratio.clamp(1 - self.cfg.clip_eps,
                                                           1 + self.cfg.clip_eps) * adv)
            loss = -surrogate / denom
            (loss * scale).backward()
            total += float(loss.detach())
            del out, logp, loss
        return total

    def train(self, tasks: Sequence[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
        import torch

        from ..utils.telemetry import RunLog, gpu_stats

        cfg = self.cfg
        rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        model = self.bundle.model
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=cfg.lr)
        n_batches = math.ceil(len(tasks) / cfg.batch_prompts)
        total_steps = max(1, int(n_batches * cfg.epochs / cfg.grad_accum))
        log = RunLog(os.path.join(out_dir, "rl_log.jsonl"))
        print(f"[grpo] {len(tasks)} questions | group {cfg.group_size} | "
              f"{total_steps} optimizer steps", flush=True)

        order: List[int] = []
        step, micro, t0 = 0, 0, time.time()
        stats: Dict[str, List[float]] = {}
        optim.zero_grad(set_to_none=True)
        while step < total_steps:
            if not order:
                order = list(range(len(tasks)))
                rng.shuffle(order)
            batch = [tasks[order.pop()] for _ in range(min(cfg.batch_prompts, len(order)))]
            for task in batch:
                rollouts = self.sample_group(task)
                self.group_loss(task, rollouts, scale=1.0 / (cfg.grad_accum * len(batch)))
                self.reward.oracle.release(task["db_path"], task["gold_sql"])
                rw = [r.reward for r in rollouts]
                for name, vals in (("reward", [x.total for x in rw]), ("ex", [x.ex for x in rw]),
                                   ("r_atomic", [x.r_atomic for x in rw]),
                                   ("r_spur", [x.r_spur for x in rw if x.ex]),
                                   ("live", [float(abs(r.advantage) > 1e-8) for r in rollouts]),
                                   ("tokens", [float(len(r.ids)) for r in rollouts]),
                                   ("counterexamples", [float(x.counterexample_found) for x in rw])):
                    stats.setdefault(name, []).extend(vals)
            micro += 1
            if micro % cfg.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.logging_steps == 0 or step == total_steps:
                entry = {"step": step, "of": total_steps, "elapsed_s": round(time.time() - t0, 1),
                         **{f"{k}_mean": round(sum(v) / len(v), 4) for k, v in stats.items() if v},
                         "n_executions": self.reward.oracle.n_exec, **gpu_stats()}
                stats = {}
                self.history.append(entry)
                log.write(entry)
                print(f"[grpo] {entry}", flush=True)
            if cfg.save_steps and step % cfg.save_steps == 0:
                self.save(os.path.join(out_dir, f"checkpoint_{step}"))
        final = os.path.join(out_dir, "final")
        self.save(final)
        log.close()
        result = {"optimizer_steps": step, "elapsed_s": round(time.time() - t0, 1),
                  "n_executions": self.reward.oracle.n_exec,
                  "counterexample_stats": (self.reward.oracle.pool.stats
                                           if self.reward.oracle.pool is not None else None)}
        with open(os.path.join(out_dir, "rl_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=4, default=str)
        return result

    def save(self, out_dir: str) -> None:
        from ..latent.student import save_student

        save_student(self.bundle, out_dir)
        self.reasoner.save(out_dir)


def _copy_cache(past: Any) -> Any:
    """Each rollout branches from the same latent prefix."""
    import copy

    return copy.deepcopy(past)
