#!/usr/bin/env python
"""End-to-end smoke test on a handful of examples.

    python scripts/smoke_test.py                 # everything that can run here
    python scripts/smoke_test.py --tier 1        # pipeline only, no torch
    python scripts/smoke_test.py --model <path>  # use a specific local model

Two tiers:

* **tier 1** - no torch needed. Builds the synthetic databases, runs the
  FusionSQL teacher and all four baselines on the mock backend, checks the
  trace round-trips, and checks every reward term and schedule.
* **tier 2** - needs torch. Loads a *tiny* causal LM, runs one SFT step, one
  GRPO step (with a real execution-based reward), one latent-thoughts
  generation, and one steering-vector extraction.

Tier 2 exercises code paths, not model quality: a 4-layer random-init model
cannot write SQL. What it proves is that the shapes, masks, caches, hooks and
schedules are correct before you spend a GPU-week on the real run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config

RESULTS: List[Dict[str, Any]] = []


def check(name: str):
    """Decorator turning a function into a reported pass/fail check."""

    def deco(fn):
        def wrapper(*a, **kw):
            t0 = time.time()
            try:
                detail = fn(*a, **kw)
                RESULTS.append(
                    {"name": name, "ok": True, "detail": detail or "",
                     "s": round(time.time() - t0, 2)}
                )
                print(f"  PASS  {name}  ({time.time() - t0:.2f}s)  {detail or ''}", flush=True)
                return True
            except Exception as exc:
                RESULTS.append(
                    {"name": name, "ok": False,
                     "detail": f"{type(exc).__name__}: {exc}",
                     "s": round(time.time() - t0, 2)}
                )
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                return False

        return wrapper

    return deco


def section(title: str) -> None:
    print(f"\n{'-' * 74}\n{title}\n{'-' * 74}", flush=True)


# ==========================================================================
# tier 1
# ==========================================================================
def tier1(cfg, work: str, n: int) -> Dict[str, Any]:
    from latentsql.data.synthetic import load_synthetic
    from latentsql.db.execute import execution_accuracy, execute_sql
    from latentsql.db.schema import load_schema, render_schema
    from latentsql.db.value_index import get_value_index
    from latentsql.distill.build_dataset import build_sft_records, run_teacher
    from latentsql.llm.mock import MockBackend
    from latentsql.mas.baselines import BASELINES
    from latentsql.mas.trace import DebateTrace, parse_trace, save_traces, load_traces

    state: Dict[str, Any] = {}

    section("tier 1 - data, database layer")

    @check("synthetic benchmark builds")
    def _data():
        exs = load_synthetic(os.path.join(work, "synthetic"))
        state["examples"] = exs
        assert len(exs) >= 20, f"only {len(exs)} examples"
        assert os.path.exists(exs[0].db_path)
        return f"{len(exs)} questions, 2 databases"

    _data()
    exs = state["examples"]

    @check("gold queries all execute")
    def _gold():
        bad = []
        for e in exs:
            r = execute_sql(e.db_path, e.gold_sql, timeout=10)
            if not r.ok:
                bad.append((e.qid, r.error))
        assert not bad, f"{len(bad)} gold queries fail: {bad[:3]}"
        return f"{len(exs)}/{len(exs)} ok"

    _gold()

    @check("execution accuracy metric is sound")
    def _ex():
        e = exs[0]
        ok, _, _ = execution_accuracy(e.db_path, e.gold_sql, e.gold_sql)
        assert ok, "gold vs gold should match"
        bad, _, _ = execution_accuracy(e.db_path, "SELECT 12345", e.gold_sql)
        assert not bad, "wrong query should not match"
        # order sensitivity only when the gold query is ordered
        from latentsql.db.execute import gold_is_ordered

        assert gold_is_ordered("SELECT a FROM t ORDER BY a")
        assert not gold_is_ordered("SELECT a FROM t WHERE b IN (SELECT c FROM u ORDER BY c)")
        return "match / mismatch / order-sensitivity all correct"

    _ex()

    @check("write statements are refused")
    def _ro():
        r = execute_sql(exs[0].db_path, "DROP TABLE stadium")
        assert not r.ok and "read-only" in r.error, r.error
        return "DROP rejected"

    _ro()

    @check("schema introspection + CR^p rendering")
    def _schema():
        s = load_schema(exs[0].db_path, exs[0].db_id)
        text = render_schema(s, style="code")
        assert len(s.tables) == 4, f"{len(s.tables)} tables"
        assert s.foreign_keys, "no foreign keys found"
        assert "CREATE TABLE" in text and "FOREIGN KEY" in text
        pruned = s.prune({"singer": ["name", "country"]})
        assert len(pruned.tables) == 1
        return f"{len(s.tables)} tables, {s.n_columns} cols, {len(s.foreign_keys)} FKs"

    _schema()

    @check("LSH value index retrieves cell values")
    def _values():
        idx = get_value_index(exs[0].db_path, exs[0].db_id, cache_dir=os.path.join(work, "cache"))
        hits = idx.search("France", top_k=5, min_similarity=0.6)
        assert hits, "no hits for 'France'"
        assert any(h.value == "France" for h in hits), [h.value for h in hits]
        typo = idx.search("Franc", top_k=5, min_similarity=0.6)
        assert typo, "typo probe returned nothing"
        return f"exact + fuzzy hits ({hits[0].as_text()})"

    _values()

    section("tier 1 - teacher multi-agent system")
    cfg.llm.backend = "mock"
    llm = MockBackend(cfg.llm).set_oracle(exs)
    sub = exs[:n]

    @check("FusionSQL teacher produces a labelled trace")
    def _mas():
        traces = run_teacher(
            sub, cfg, llm, example_pool=exs[n:],
            cache_dir=os.path.join(work, "cache"), progress=False,
        )
        state["traces"] = traces
        assert len(traces) == len(sub)
        assert all(t.consensus_sql for t in traces), "some traces have no final SQL"
        n_ok = sum(1 for t in traces if t.correct)
        assert n_ok > 0, "teacher got nothing right"
        agents = {tt.agent for t in traces for tt in t.turns}
        assert len(agents) == 3, agents
        rounds = max(t.n_rounds for t in traces)
        return f"EX={n_ok}/{len(traces)}, agents={sorted(agents)}, max_rounds={rounds}"

    _mas()

    @check("debate beats its own panellists")
    def _beats():
        traces = state["traces"]
        team = sum(1 for t in traces if t.correct)
        solo: Dict[str, int] = {}
        for t in traces:
            first = {}
            for turn in t.turns:
                first.setdefault(turn.agent, turn)
            for a, turn in first.items():
                solo[a] = solo.get(a, 0) + (1 if turn.correct else 0)
        best_solo = max(solo.values()) if solo else 0
        detail = f"debate={team}/{len(traces)} vs best single agent={best_solo}/{len(traces)}"
        assert team >= best_solo, detail
        return detail

    _beats()

    @check("trace renders, parses and round-trips through jsonl")
    def _trace():
        traces = state["traces"]
        text = traces[0].render()
        parsed = parse_trace(text)
        assert parsed["has_end"] and parsed["has_consensus"] and parsed["has_schema_link"]
        assert parsed["n_rounds"] == traces[0].n_rounds, (parsed["n_rounds"], traces[0].n_rounds)
        assert parsed["final_sql"], "no final SQL parsed back"
        assert len(parsed["agent_names"]) == 3, parsed["agent_names"]
        p = os.path.join(work, "traces.jsonl")
        save_traces(traces, p)
        back = load_traces(p)
        assert back[0].render() == text, "jsonl round-trip changed the trace"
        return f"{len(text)} chars, {parsed['n_rounds']} rounds, {len(parsed['agents'])} turns"

    _trace()

    @check("SFT corpus filtering")
    def _corpus():
        recs, stats = build_sft_records(state["traces"], cfg.trace)
        for r in recs:
            match = next((e for e in sub if e.qid == r.qid), None)
            if match:
                r.db_path = match.db_path
        state["records"] = recs
        assert recs, f"everything got filtered: {stats.summary()}"
        assert all(r.completion.rstrip().endswith("<|endofdebate|>") for r in recs)
        assert all(r.system and r.prompt for r in recs)
        return stats.summary()

    _corpus()

    section("tier 1 - published baselines")

    @check("all four baselines run")
    def _baselines():
        out = {}
        for name in ("single", "din_sql", "dail_sql", "chess", "chase_sql"):
            llm.reset_usage()
            ok = 0
            for e in sub:
                sql = BASELINES[name](llm, cfg.mas, e, pool=exs[n:])
                ok += 1 if execution_accuracy(e.db_path, sql, e.gold_sql)[0] else 0
            out[name] = (ok, llm.usage.total_tokens // max(1, len(sub)))
        state["baselines"] = out
        assert all(v[0] >= 0 for v in out.values())
        return " | ".join(f"{k}:{v[0]}/{len(sub)}@{v[1]}tok" for k, v in out.items())

    _baselines()

    section("tier 1 - IMAD reward function")

    @check("reward schedule anneals w_fmt and l")
    def _sched():
        from latentsql.distill.rewards import RewardSchedule

        sched = RewardSchedule(cfg.grpo.reward, total_steps=100)
        assert abs(sched.w_fmt(0) - cfg.grpo.reward.w_fmt_start) < 1e-6
        assert abs(sched.w_fmt(99) - cfg.grpo.reward.w_fmt_end) < 1e-6
        assert sched.length_limit(0) == cfg.grpo.reward.len_start
        assert sched.length_limit(99) == cfg.grpo.reward.len_end
        assert sched.length_limit(0) > sched.length_limit(50) > sched.length_limit(99)
        return (
            f"w_fmt {sched.w_fmt(0):.2f}->{sched.w_fmt(99):.2f}, "
            f"l {sched.length_limit(0)}->{sched.length_limit(99)}"
        )

    _sched()

    @check("R_fmt + R(y;l) behave as IMAD eq. 1-2")
    def _reward():
        from latentsql.distill.rewards import DebateReward, ExecutionOracle, RewardSchedule

        rec = state["records"][0]
        oracle = ExecutionOracle(timeout=10)
        sched = RewardSchedule(cfg.grpo.reward, total_steps=10)
        rf = DebateReward(cfg.grpo.reward, oracle, sched)

        full = rf(rec.completion, None, rec.db_path, rec.gold_sql, step=0)
        assert full.r_fmt == 1.0, f"complete trace scored R_fmt={full.r_fmt}"
        assert full.r_clip == 1.0, "correct trace should earn the correctness reward"

        naked = f"<|consensus|>\n```sql\n{rec.gold_sql}\n```\n<|/consensus|>\n<|endofdebate|>"
        short = rf(naked, None, rec.db_path, rec.gold_sql, step=9)
        assert short.r_clip == 1.0, "internalised answer must still earn R(y;l)"
        assert short.r_fmt < 1.0, "an answer with no debate tags must lose format reward"
        assert short.w_fmt < full.w_fmt, "w_fmt must have decayed"
        # ... and by the end of training, dropping the debate is worth *more*
        late_full = rf(rec.completion, None, rec.db_path, rec.gold_sql, step=9)
        wrong = rf("<|consensus|>```sql\nSELECT 999\n```<|/consensus|><|endofdebate|>",
                   None, rec.db_path, rec.gold_sql, step=9)
        assert wrong.r_clip == 0.0, "a wrong query must not earn the correctness reward"
        return (
            f"full={full.total:.3f} internalised={short.total:.3f} "
            f"late_full={late_full.total:.3f} wrong={wrong.total:.3f}"
        )

    _reward()

    @check("length clipping actually clips")
    def _clip():
        from latentsql.distill.rewards import DebateReward, ExecutionOracle, RewardSchedule

        rec = state["records"][0]
        oracle = ExecutionOracle(timeout=10)
        cfg2 = type(cfg.grpo.reward)(**{**cfg.grpo.reward.__dict__})
        cfg2.len_start = cfg2.len_end = 12       # far too short for a full trace
        rf = DebateReward(cfg2, oracle, RewardSchedule(cfg2, total_steps=2),
                          decode=lambda ids: "".join(chr(i) for i in ids))
        ids = list(rec.completion.encode("utf-8", "ignore"))[:4000]
        clipped = rf(rec.completion, ids, rec.db_path, rec.gold_sql, step=0)
        assert clipped.r_clip == 0.0, "a 12-token prefix cannot contain the answer"
        return "verbose trace earns 0 under a 12-token clip"

    _clip()

    from latentsql.distill.rewards import group_advantages

    @check("GRPO group advantages are centred within a group")
    def _adv2():
        a = group_advantages([1.0, 0.0, 0.0, 0.0])
        assert abs(sum(a)) < 1e-6, a
        assert a[0] > 0 > a[1], a
        assert group_advantages([0.5] * 4) == [0.0] * 4, "degenerate group must give 0"
        return f"advantages={[round(x, 3) for x in a]}"

    _adv2()

    return state


# ==========================================================================
# tier 2
# ==========================================================================
def tier2(cfg, work: str, state: Dict[str, Any], model_path: str) -> None:
    import torch

    from latentsql.distill.grpo import GRPOTrainer, load_reference, tasks_from_records
    from latentsql.distill.latent import LatentReasoner, build_aligner, latent_slot_curriculum
    from latentsql.distill.rewards import ExecutionOracle
    from latentsql.distill.sft import build_example, prepare_student, train_sft
    from latentsql.llm.hf_local import load_model
    from latentsql.steering.caa import (
        build_pairs, cosine_separation, extract_steering_vectors, get_layers, steer_layer,
    )

    records = state["records"]
    section(f"tier 2 - torch {torch.__version__} | model {model_path}")

    cfg.sft.model_name_or_path = model_path
    cfg.sft.output_dir = os.path.join(work, "sft")
    cfg.sft.epochs = 1.0
    cfg.sft.batch_size = 1
    cfg.sft.grad_accum = 2
    cfg.sft.max_seq_len = 1024
    cfg.sft.bf16 = False
    cfg.sft.gradient_checkpointing = False
    cfg.sft.logging_steps = 1
    cfg.sft.save_steps = 0
    cfg.sft.device = "cpu"

    @check("student loads with LoRA attached")
    def _load():
        bundle = prepare_student(cfg.sft, hub_dir=None)
        state["bundle"] = bundle
        n_train = sum(p.numel() for p in bundle.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in bundle.model.parameters())
        assert 0 < n_train < n_all, (n_train, n_all)
        return f"{n_all / 1e6:.1f}M params, {n_train / 1e3:.0f}K trainable"

    _load()
    bundle = state["bundle"]

    @check("prompt tokens are masked out of the SFT loss")
    def _mask():
        ex = build_example(bundle.tokenizer, records[0], cfg.sft.max_seq_len, True)
        assert ex is not None
        labels = ex["labels"]
        n_masked = sum(1 for x in labels if x == -100)
        assert 0 < n_masked < len(labels), (n_masked, len(labels))
        assert all(a == b for a, b in zip(ex["input_ids"][n_masked:], labels[n_masked:]))
        cons = build_example(bundle.tokenizer, records[0], cfg.sft.max_seq_len, False)
        assert cons is not None
        n_sup_full = len(labels) - n_masked
        n_sup_cons = sum(1 for x in cons["labels"] if x != -100)
        assert n_sup_cons < n_sup_full, (n_sup_cons, n_sup_full)
        return f"{n_masked} prompt tokens masked; full-trace supervises {n_sup_full} vs {n_sup_cons} for consensus-only"

    _mask()

    @check("SFT runs and the loss is finite")
    def _sft():
        res = train_sft(records[:4], cfg.sft, hub_dir=None, bundle=bundle)
        assert res["optimizer_steps"] >= 1
        loss = res["final_loss"]
        assert loss is not None and loss == loss and loss < 1e4, loss
        assert os.path.exists(os.path.join(cfg.sft.output_dir, "latentsql_student.json"))
        return f"{res['optimizer_steps']} steps, loss={loss:.3f}, {res['elapsed_s']}s"

    _sft()

    section("tier 2 - GRPO with the annealed reward")
    cfg.grpo.output_dir = os.path.join(work, "grpo")
    cfg.grpo.group_size = 2
    cfg.grpo.batch_prompts = 1
    cfg.grpo.grad_accum = 1
    cfg.grpo.epochs = 1.0
    cfg.grpo.max_new_tokens = 24
    cfg.grpo.logging_steps = 1
    cfg.grpo.save_steps = 0
    cfg.grpo.reward.len_start = 20
    cfg.grpo.reward.len_end = 8

    @check("GRPO takes a real optimizer step")
    def _grpo():
        tasks = tasks_from_records(records[:2])
        assert tasks, "no GRPO tasks built (missing db_path/gold_sql)"
        oracle = ExecutionOracle(timeout=10)
        trainer = GRPOTrainer(bundle, cfg.grpo, oracle)
        ref = load_reference(bundle)

        before = [p.detach().clone() for p in bundle.model.parameters() if p.requires_grad]
        res = trainer.train(tasks, ref)
        after = [p.detach() for p in bundle.model.parameters() if p.requires_grad]
        moved = any(not torch.allclose(a, b) for a, b in zip(before, after))

        assert res["optimizer_steps"] >= 1
        assert res["history"], "no GRPO log records"
        h = res["history"][-1]
        assert h["length_limit"] <= cfg.grpo.reward.len_start
        assert h["gen_tokens_mean"] > 0, "no tokens were sampled"
        # A random-init model emits no parseable SQL, so the reward legitimately
        # short-circuits before executing anything. Probe the oracle directly to
        # confirm the execution-based branch of the reward is wired up.
        t = tasks[0]
        assert oracle.is_correct(t.db_path, t.gold_sql, t.gold_sql), "oracle rejects gold vs gold"
        assert not oracle.is_correct(t.db_path, "SELECT 424242", t.gold_sql)
        assert oracle.n_exec > 0, "the oracle never executed any SQL"
        return (
            f"{res['optimizer_steps']} steps, params_changed={moved}, "
            f"{oracle.n_exec} SQL executions, l={h['length_limit']}"
        )

    _grpo()

    @check("GRPO updates parameters when the group has reward variance")
    def _grpo_grad():
        # The previous check exercises the loop, but a random-init model earns
        # reward 0 on every rollout, so the advantages are all 0 and nothing
        # backpropagates (correct, but it leaves the gradient path untested).
        # Force one rollout in each group to look good and verify the update.
        from latentsql.distill.rewards import RewardBreakdown

        tasks = tasks_from_records(records[:1])
        oracle = ExecutionOracle(timeout=10)
        trainer = GRPOTrainer(bundle, cfg.grpo, oracle)

        calls = {"n": 0}
        real_reward = trainer.reward_fn

        def fake_reward(text, ids, db, gold, step):
            rb = real_reward(text, ids, db, gold, step)
            calls["n"] += 1
            rb.total = 1.0 if calls["n"] % 2 else 0.0   # alternate within the group
            return rb

        trainer.reward_fn = fake_reward
        ref = load_reference(bundle)

        trainable = [p for p in bundle.model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        trainer.train(tasks, ref)
        moved = sum(
            1 for a, b in zip(before, trainable) if not torch.allclose(a, b.detach())
        )
        assert calls["n"] >= 2, calls
        assert moved > 0, "no trainable parameter moved despite non-zero advantages"
        return f"{moved}/{len(trainable)} LoRA tensors updated over {calls['n']} rollouts"

    _grpo_grad()

    section("tier 2 - LatentMAS latent thoughts")

    @check("input-output aligners map hidden states to embedding space")
    def _align():
        d = bundle.model.config.hidden_size
        h = torch.randn(1, 1, d, dtype=next(bundle.model.parameters()).dtype)
        emb = bundle.model.get_input_embeddings().weight
        out = {}
        for mode in ("soft_embed", "linear", "identity"):
            fn = build_aligner(bundle, mode=mode, top_k=8)
            e = fn(h)
            assert e.shape == h.shape, (mode, e.shape, h.shape)
            assert torch.isfinite(e).all(), f"{mode} produced non-finite values"
            out[mode] = float(e.float().norm())
        # the soft-embedding must land near the real embedding scale
        ref = float(emb.float().norm(dim=-1).mean())
        assert out["soft_embed"] < 10 * ref, (out["soft_embed"], ref)
        return " ".join(f"{k}:|e|={v:.2f}" for k, v in out.items()) + f" (emb ref {ref:.2f})"

    _align()

    @check("latent roles run and only the final query is decoded")
    def _latent():
        cfg.latent.latent_steps = 3
        cfg.latent.roles = ["divide_conquer", "query_plan", "consensus"]
        reasoner = LatentReasoner(bundle, cfg.latent)
        run = reasoner.generate(records[0].system, records[0].prompt, max_new_tokens=12)
        assert run.n_latent_steps == 3 * 2, run.n_latent_steps
        assert run.n_decoded_tokens <= 12
        return (
            f"{run.n_latent_steps} latent steps emitted 0 tokens; "
            f"{run.n_decoded_tokens} tokens decoded"
        )

    _latent()

    @check("latent working memory carries earlier roles forward")
    def _mem():
        cfg.latent.latent_steps = 3
        cfg.latent.roles = ["divide_conquer", "query_plan", "consensus"]
        cfg.latent.share_working_memory = True
        a = LatentReasoner(bundle, cfg.latent).generate(
            records[0].system, records[0].prompt, max_new_tokens=8
        )
        cfg.latent.share_working_memory = False
        b = LatentReasoner(bundle, cfg.latent).generate(
            records[0].system, records[0].prompt, max_new_tokens=8
        )
        cfg.latent.share_working_memory = True
        # with sharing on, the decoder attends to the prompt plus every role's
        # latent thoughts; with it off, only to the last role's block
        assert a.n_memory_positions > b.n_memory_positions, (
            a.n_memory_positions, b.n_memory_positions
        )
        assert a.n_latent_steps == b.n_latent_steps == 6
        return (
            f"shared memory = {a.n_memory_positions} positions vs "
            f"{b.n_memory_positions} when isolated (same {a.n_latent_steps} latent steps)"
        )

    _mem()

    @check("per-role latent checkpointing matches the uncheckpointed gradient exactly")
    def _ckpt_grad():
        """The fix for the OOM that a 96-step (2-round x 4-agent) latent
        debate causes when trained with GRPO: build_prefix(grad=True) holds
        every latent step's activations for backward simultaneously unless
        checkpointed. Checkpointing a KV-cache-carrying loop is only safe if
        repeated invocation is idempotent - DynamicCache.crop() back to the
        role's starting length makes it so, since it is a true no-op on the
        first (real) call and undoes exactly the second (backward-recompute)
        call's own append before it happens again. This test proves that
        empirically: same model, same seed, same rollout, compare LoRA
        gradients with and without the per-role checkpoint. They must match
        bit for bit, not just approximately - any mismatch means the
        recompute is seeing a different cache state than the original pass.
        """
        from latentsql.mas.trace import agent_tag as _tag, T_CONSENSUS_OPEN as _cons

        def run_once(checkpoint_flag: bool):
            bundle.model.zero_grad(set_to_none=True)
            tok = bundle.tokenizer
            chat = tok.apply_chat_template(
                [{"role": "system", "content": records[0].system},
                 {"role": "user", "content": records[0].prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tok(chat, return_tensors="pt", add_special_tokens=False)
            attn = enc["attention_mask"]
            embeds = bundle.model.get_input_embeddings()(enc["input_ids"])
            past = None
            reasoner = LatentReasoner(bundle, cfg.latent)
            with torch.enable_grad():
                for role in cfg.latent.roles:
                    tag = _cons if role == "consensus" else _tag(role) + "\n"
                    tag_e, past, attn = reasoner._append_text(tag, past, attn)
                    embeds = torch.cat([embeds, tag_e], dim=1) if past is None else tag_e
                    if role == "consensus":
                        break
                    embeds, past, attn, _, _aux = reasoner._latent_steps(
                        embeds, past, attn, cfg.latent.latent_steps,
                        checkpoint=checkpoint_flag,
                    )
                out = bundle.model(inputs_embeds=embeds, past_key_values=past,
                                   attention_mask=attn, use_cache=True)
            logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            loss = -logp[0, min(200, logp.shape[-1] - 1)]
            loss.backward()
            grads = {n: p.grad.clone() for n, p in bundle.model.named_parameters()
                    if p.requires_grad and p.grad is not None}
            return grads, float(loss.detach())

        # eval() disables LoRA dropout: this bundle was SFT-trained earlier in
        # this same tier-2 run (lora_dropout=0.05 by default), so two SEPARATE
        # stochastic forward passes would legitimately disagree regardless of
        # whether checkpointing is correct - the thing under test here is
        # checkpointing, not dropout, so the comparison needs it off.
        was_training = bundle.model.training
        bundle.model.eval()
        cfg.latent.latent_steps = 3
        cfg.latent.roles = ["divide_conquer", "query_plan", "consensus"]
        g_plain, l_plain = run_once(checkpoint_flag=False)
        g_ckpt, l_ckpt = run_once(checkpoint_flag=True)
        bundle.model.train(was_training)
        bundle.model.zero_grad(set_to_none=True)

        assert abs(l_plain - l_ckpt) < 1e-4, (l_plain, l_ckpt)
        assert set(g_plain) == set(g_ckpt) and len(g_plain) > 0
        worst = max((g_plain[n] - g_ckpt[n]).abs().max().item() for n in g_plain)
        assert worst < 1e-4, f"gradients diverge by {worst:.2e}"
        return f"{len(g_plain)} grad tensors match to {worst:.1e} (loss {l_plain:.4f})"

    _ckpt_grad()

    @check("latent-slot curriculum replaces agent blocks")
    def _curr():
        full = records[0].completion
        s1 = latent_slot_curriculum(full, ["divide_conquer"], stage=1, latent_steps=4)
        s3 = latent_slot_curriculum(full, [], stage=3, latent_steps=4)
        assert len(s1) < len(full), "stage 1 did not shorten the trace"
        assert len(s3) < len(s1), "stage 3 did not shorten further"
        assert s1.count("<|latent|>") == 4
        return f"{len(full)} -> {len(s1)} -> {len(s3)} chars"

    _curr()

    section("tier 2 - agent subspaces (CAA)")

    @check("contrastive pairs are built from the traces")
    def _pairs():
        pairs = build_pairs(state["traces"], "divide_conquer", max_pairs=8)
        assert pairs, "no contrastive pairs"
        p = pairs[0]
        assert p.positive != p.negative
        assert p.context.rstrip().endswith("<|agent divide_conquer|>")
        return f"{len(pairs)} pairs, context ends at the target agent tag"

    _pairs()

    @check("steering vectors extract and are separable")
    def _caa():
        cfg.steering.layer = -1
        cfg.steering.n_pairs = 6
        vecs = extract_steering_vectors(
            bundle, state["traces"], ["divide_conquer", "query_plan", "dail_icl"], cfg.steering
        )
        assert len(vecs.vectors) >= 2, list(vecs.vectors)
        for name, v in vecs.vectors.items():
            assert torch.isfinite(v).all(), name
            assert abs(float(v.norm()) - 1.0) < 1e-3, (name, float(v.norm()))
        cos = cosine_separation(vecs)
        assert cos, "no pairwise cosines"
        return f"{len(vecs.vectors)} unit vectors @ layer {vecs.layer}, cos={{" + \
               ", ".join(f"{k}:{v:+.2f}" for k, v in cos.items()) + "}"

    _caa()

    @check("steering hook changes generation")
    def _steer():
        from latentsql.llm.hf_local import render_chat

        tok = bundle.tokenizer
        text = render_chat(
            tok, [{"role": "user", "content": "SELECT"}], add_generation_prompt=True
        )
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        d = bundle.model.config.hidden_size
        v = torch.randn(d) * 50.0
        with torch.no_grad():
            plain = bundle.model.generate(
                **enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id
            )
            with steer_layer(bundle.model, -1, v, alpha=1.0):
                steered = bundle.model.generate(
                    **enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id
                )
        changed = not torch.equal(plain, steered)
        assert changed, "a large steering vector did not change greedy decoding"
        return f"{len(get_layers(bundle.model))} layers; steering altered the output"

    _steer()

    section("tier 2 - student evaluation path")

    @check("student evaluation produces scored predictions")
    def _eval():
        from latentsql.eval.metrics import build_report
        from latentsql.eval.run_eval import eval_student
        from latentsql.eval.metrics import score_predictions

        exs = state["examples"][:2]
        preds = eval_student(exs, bundle, max_new_tokens=16, mode="student", progress=False)
        score_predictions(preds, timeout=10)
        rep = build_report(preds, method="student", dataset="synthetic")
        assert rep.n == len(exs)
        assert all(p.completion_tokens > 0 for p in preds)
        return f"n={rep.n} EX={rep.ex * 100:.0f}% avg_tokens={rep.avg_total_tokens:.0f}"

    _eval()


# ==========================================================================
def get_tiny_model(work: str, explicit: Optional[str], hub: Optional[str]) -> Optional[str]:
    """Find a small local model, else download a tiny test model."""
    from latentsql.llm.hf_local import find_in_hub, scan_hub

    if explicit:
        return explicit
    if hub and os.path.isdir(hub):
        found = scan_hub(hub, max_params_b=2.0)
        if found:
            print(f"  using local model from hub: {found[0]['name']}")
            return found[0]["path"]
    # Windows without Developer Mode cannot create the symlinks the HF cache
    # uses by default (WinError 1314), and it is worse inside a synced folder.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    for name in ("hf-internal-testing/tiny-random-LlamaForCausalLM",
                 "HuggingFaceTB/SmolLM2-135M-Instruct"):
        hit = find_in_hub(name, os.path.join(work, "hub"))
        if hit:
            return hit
        try:
            from huggingface_hub import snapshot_download

            print(f"  downloading tiny test model {name} ...", flush=True)
            return snapshot_download(name, cache_dir=os.path.join(work, "hub"))
        except Exception as exc:
            print(f"  could not fetch {name}: {type(exc).__name__}: {exc}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tier", type=int, default=2, choices=(1, 2))
    ap.add_argument("--n", type=int, default=6, help="examples through the teacher")
    ap.add_argument("--model", default=None, help="model path for tier 2")
    ap.add_argument("--work", default=None, help="scratch directory")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = args.work or os.path.join(cfg.paths.output_dir, "_smoke")
    if os.path.isdir(work) and not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    cfg.paths.cache_dir = os.path.join(work, "cache")

    print("=" * 74)
    print("LatentSQL smoke test")
    print(f"  scratch: {work}")
    print(f"  teacher: mock backend (offline test double - not a language model)")
    print("=" * 74)

    t0 = time.time()
    state = tier1(cfg, work, args.n)

    tier2_skipped = ""
    if args.tier >= 2:
        try:
            import torch  # noqa: F401
        except ImportError:
            tier2_skipped = "torch is not installed (pip install -r requirements.txt)"
        else:
            model = get_tiny_model(work, args.model, cfg.paths.hf_hub)
            if model is None:
                tier2_skipped = (
                    "no model available - point --model at a local model, or set "
                    "paths.hf_hub, or enable Windows Developer Mode so the HF cache "
                    "can be written"
                )
            else:
                tier2(cfg, work, state, model)
    if tier2_skipped:
        print(f"\n  !! TIER 2 SKIPPED: {tier2_skipped}")

    section("summary")
    n_ok = sum(1 for r in RESULTS if r["ok"])
    for r in RESULTS:
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['name']}")
    print(f"\n{n_ok}/{len(RESULTS)} checks passed in {time.time() - t0:.1f}s")
    if tier2_skipped:
        # a green tier-1 run must never be mistaken for a full pass
        print(
            "\n  !! INCOMPLETE: only tier 1 ran. Training, latent reasoning and\n"
            f"  !! steering were NOT exercised. Reason: {tier2_skipped}"
        )
        return 2

    if "baselines" in state:
        print("\nMock-backend pipeline sanity (NOT results - the teacher is a stub):")
        n = args.n
        for name, (ok, tok) in state["baselines"].items():
            print(f"  {name:<12} {ok}/{n} correct, ~{tok} tokens/question")
        traces = state["traces"]
        print(f"  {'fusion_mas':<12} {sum(1 for t in traces if t.correct)}/{n} correct, "
              f"~{sum(t.teacher_tokens for t in traces) // max(1, n)} tokens/question")

    if not args.keep:
        print(f"\n(scratch kept at {work} - pass --keep to reuse, it is deleted on rerun)")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
