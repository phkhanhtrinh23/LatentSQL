#!/usr/bin/env python
"""Assemble one report from every log a run produced.

    python scripts/06_report.py --run outputs/pilot

Reads the teacher's per-question log, the SFT/GRPO step logs and every
evaluation directory under ``--run``, and writes ``report.md`` plus
``report.json``: accuracy per benchmark, token and latency savings of the
internalised student against both its own teacher and the four published
systems driven by the *same* 3B model, and the GPU/wall-clock cost of getting
there.

Everything printed here comes from the logs; nothing is recomputed or assumed,
so a stage that did not run shows as absent rather than as a zero.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: how each evaluated method is labelled in the report
METHOD_LABEL = {
    "single": "Single agent (zero-shot)",
    "din_sql": "DIN-SQL",
    "dail_sql": "DAIL-SQL",
    "chess": "CHESS",
    "chase_sql": "CHASE-SQL",
    "fusion_mas": "Explicit MAS (teacher panel)",
    "student": "LatentSQL student (ours)",
    "latent": "LatentSQL student + LatentMAS (ours)",
    "sft_trace": "SFT-only student (ablation)",
}
OURS = {"student", "latent", "sft_trace"}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _mean(xs: List[float]) -> float:
    return round(statistics.fmean(xs), 2) if xs else 0.0


def _pct(x: Optional[float]) -> str:
    return f"{x * 100:.2f}" if x is not None else "-"


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def teacher_section(run: str) -> Dict[str, Any]:
    samples = [r for r in read_jsonl(os.path.join(run, "traces", "teacher_samples.jsonl"))
               if r.get("record") == "sample"]
    header = next((r for r in read_jsonl(os.path.join(run, "traces", "teacher_samples.jsonl"))
                   if r.get("record") == "header"), {})
    summary = read_json(os.path.join(run, "traces", "summary.json")) or {}
    if not samples:
        return {}

    by_ds: Dict[str, Dict[str, Any]] = {}
    for r in samples:
        b = by_ds.setdefault(r.get("dataset", "?"),
                             {"n": 0, "correct": 0, "consensus": 0, "s": [], "tok": []})
        b["n"] += 1
        b["correct"] += 1 if r.get("correct") else 0
        b["consensus"] += 1 if r.get("consensus") else 0
        b["s"].append(r.get("latency_s", 0.0))
        b["tok"].append(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0))

    return {
        "model": header.get("model") or summary.get("model"),
        "quantization": header.get("quantization", "none"),
        "panel": header.get("panel") or summary.get("panel"),
        "rounds": header.get("rounds") or summary.get("rounds"),
        "n_questions": len(samples),
        "ex": round(sum(1 for r in samples if r.get("correct")) / len(samples), 4),
        "consensus_rate": round(
            sum(1 for r in samples if r.get("consensus")) / len(samples), 4),
        "kept": summary.get("filter", {}).get("kept"),
        "filter": summary.get("filter", {}),
        "mean_latency_s": _mean([r.get("latency_s", 0.0) for r in samples]),
        "mean_total_tokens": _mean([r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
                                    for r in samples]),
        "mean_completion_tokens": _mean([r.get("completion_tokens", 0) for r in samples]),
        "mean_llm_calls": _mean([r.get("llm_calls", 0) for r in samples]),
        "total_wall_s": round(sum(r.get("latency_s", 0.0) for r in samples), 1),
        "gpu_peak_gb": max([r.get("gpu_peak_gb", 0.0) for r in samples] or [0.0]),
        "gpu_name": header.get("gpu_name", ""),
        "by_dataset": {
            k: {"n": v["n"],
                "ex": round(v["correct"] / v["n"], 4),
                "consensus_rate": round(v["consensus"] / v["n"], 4),
                "mean_latency_s": _mean(v["s"]),
                "mean_total_tokens": _mean(v["tok"])}
            for k, v in sorted(by_ds.items())
        },
    }


def training_section(run: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for stage in ("sft", "grpo"):
        cands = [p for p in glob.glob(os.path.join(run, stage, "*_log.json"))]
        blob = read_json(cands[0]) if cands else None
        step_files = glob.glob(os.path.join(run, stage, "*_log_steps.jsonl"))
        steps = read_jsonl(step_files[0]) if step_files else []

        if blob is None:
            # A killed or timed-out stage never writes its summary json, but the
            # per-step log is flushed as it goes. Reconstruct from that rather
            # than reporting the stage as absent - a truncated run still has
            # results worth reading.
            step_recs = [r for r in steps if r.get("record") == "step"]
            if not step_recs:
                continue
            hdr = next((r for r in steps if r.get("record") == "header"), {})
            blob = {
                "model": hdr.get("model", ""),
                "quantization": hdr.get("quantization", ""),
                "optimizer_steps": step_recs[-1].get("step"),
                "elapsed_s": step_recs[-1].get("elapsed_s"),
                "gpu_peak_gb": max(r.get("gpu_peak_gb", 0) or 0 for r in step_recs),
                "gen_tokens_total": step_recs[-1].get("gen_tokens_total"),
                "n_executions": step_recs[-1].get("n_executions"),
                "history": step_recs,
                "truncated": True,
            }
        rec = {k: v for k, v in blob.items() if k != "history"}
        rec["n_logged_steps"] = len([s for s in steps if s.get("record") == "step"])
        hist = blob.get("history") or []
        if hist:
            rec["first"] = hist[0]
            rec["last"] = hist[-1]
            # summarise here: `history` is dropped from rec to keep the report
            # json small, so anything derived from it must be computed now
            if "adv_nonzero_frac" in hist[0]:
                live = [h for h in hist if (h.get("adv_nonzero_frac") or 0) > 0]
                rec["n_live_steps"] = len(live)
                rec["n_steps_logged"] = len(hist)
        out[stage] = rec
    return out


def eval_section(run: str) -> Dict[str, Any]:
    """Merge every eval directory under the run into one method table."""
    methods: Dict[str, Dict[str, Any]] = {}
    for reports_path in sorted(glob.glob(os.path.join(run, "eval*", "reports.json"))):
        eval_dir = os.path.dirname(reports_path)
        header = next((r for r in read_jsonl(os.path.join(eval_dir, "eval_samples.jsonl"))
                       if r.get("record") == "header"), {})
        for rep in read_json(reports_path) or []:
            m = rep.get("method")
            if not m:
                continue
            preds = read_jsonl(os.path.join(eval_dir, f"preds_{m}.jsonl"))
            rep = dict(rep)
            rep["eval_dir"] = os.path.basename(eval_dir)
            # which model actually drove this method - the whole point of the
            # 3B comparison is that it is the same backbone as the student
            rep["driver_model"] = (
                header.get("student_base") if m in OURS else header.get("teacher_model")
            )
            rep["mean_latency_s"] = _mean([p.get("latency_s", 0.0) for p in preds])
            rep["p50_latency_s"] = (
                round(statistics.median([p.get("latency_s", 0.0) for p in preds]), 2)
                if preds else 0.0
            )
            rep["gpu_peak_gb"] = (rep.get("resources") or {}).get("gpu_peak_gb")
            methods[m] = rep
    return methods


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def render(run: str, teacher: Dict[str, Any], training: Dict[str, Any],
           methods: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append
    A(f"# LatentSQL run report — `{run}`\n")

    # ---- headline comparison ------------------------------------------
    A("## 1. Accuracy — our internalised student vs. the four published systems\n")
    if methods:
        driver = next((r.get("driver_model") for m, r in methods.items()
                       if m not in OURS and r.get("driver_model")), None)
        ours_driver = next((r.get("driver_model") for m, r in methods.items()
                            if m in OURS and r.get("driver_model")), None)
        A(f"Baselines driven by `{driver}`; student is `{ours_driver}` + adapters. "
          "Same questions, same schema rendering, same executor, same scorer.\n")
        sources = sorted({k for r in methods.values() for k in (r.get("by_dataset") or {})})
        split = [s for s in sources if s in ("bird", "spider")] if len(sources) > 1 else []
        A("| Method | n | EX (%) |" + "".join(f" EX {s} (%) |" for s in split)
          + " Avg tokens | Output tokens | Latency (s) | GPU peak (GB) |")
        A("|---|---:|---:|" + "---:|" * len(split) + "---:|---:|---:|---:|")
        order = ["single", "din_sql", "dail_sql", "chess", "chase_sql",
                 "fusion_mas", "sft_trace", "student", "latent"]
        for m in [x for x in order if x in methods] + \
                 [x for x in methods if x not in order]:
            r = methods[m]
            cells = ""
            for s in split:
                b = (r.get("by_dataset") or {}).get(s)
                cells += f" {_pct(b['ex'])} |" if b and b.get("n") else " - |"
            label = METHOD_LABEL.get(m, m)
            if m in OURS:
                label = f"**{label}**"
            A(f"| {label} | {r.get('n', 0)} | {_pct(r.get('ex'))} |" + cells +
              f" {r.get('avg_total_tokens', 0):.1f} |"
              f" {r.get('avg_completion_tokens', 0):.1f} |"
              f" {r.get('mean_latency_s', 0):.2f} |"
              f" {r.get('gpu_peak_gb') if r.get('gpu_peak_gb') is not None else '-'} |")
        A("")
    else:
        A("_No evaluation results found._\n")

    # ---- savings -------------------------------------------------------
    A("## 2. Cost — what internalising the debate saves\n")
    ours = next((methods[m] for m in ("latent", "student") if m in methods), None)
    if ours:
        A("| Compared with | EX (%) | Avg tokens | Latency (s) | Student tokens | "
          "Student latency |")
        A("|---|---:|---:|---:|---:|---:|")
        for m, r in methods.items():
            if m in OURS:
                continue
            tok = r.get("avg_total_tokens") or 0
            lat = r.get("mean_latency_s") or 0
            otok = ours.get("avg_total_tokens") or 0
            olat = ours.get("mean_latency_s") or 0
            A(f"| {METHOD_LABEL.get(m, m)} | {_pct(r.get('ex'))} | {tok:.1f} | {lat:.2f} |"
              f" {(100.0 * otok / tok) if tok else 0:.1f}% |"
              f" {(100.0 * olat / lat) if lat else 0:.1f}% |")
        A("")
        if teacher:
            tt = teacher.get("mean_total_tokens") or 0
            ot = ours.get("avg_total_tokens") or 0
            A(f"Against the **7B teacher panel** that produced the training data "
              f"({teacher.get('mean_total_tokens')} tokens and "
              f"{teacher.get('mean_llm_calls')} LLM calls per question): the student "
              f"answers in {ot:.1f} tokens and **1** call — "
              f"{(100.0 * ot / tt) if tt else 0:.1f}% of the teacher's tokens.\n")
    else:
        A("_No student result to compare._\n")

    # ---- teacher -------------------------------------------------------
    A("## 3. Teacher run\n")
    if teacher:
        A(f"- Model: `{teacher.get('model')}` ({teacher.get('quantization')}) on "
          f"{teacher.get('gpu_name')}")
        A(f"- Panel: `{teacher.get('panel')}`, {teacher.get('rounds')} rounds")
        A(f"- {teacher['n_questions']} questions, EX **{_pct(teacher['ex'])}%**, "
          f"consensus {_pct(teacher['consensus_rate'])}%")
        A(f"- **{teacher.get('kept')} traces kept** after filtering — "
          f"`{teacher.get('filter')}`")
        A(f"- {teacher['mean_latency_s']}s, {teacher['mean_total_tokens']} tokens, "
          f"{teacher['mean_llm_calls']} calls per question; "
          f"{teacher['total_wall_s'] / 60:.1f} min total, "
          f"GPU peak {teacher['gpu_peak_gb']} GB\n")
        A("| Benchmark | n | EX (%) | Consensus (%) | Latency (s) | Tokens |")
        A("|---|---:|---:|---:|---:|---:|")
        for k, v in teacher["by_dataset"].items():
            A(f"| {k} | {v['n']} | {_pct(v['ex'])} | {_pct(v['consensus_rate'])} | "
              f"{v['mean_latency_s']} | {v['mean_total_tokens']} |")
        A("")
    else:
        A("_No teacher log found._\n")

    # ---- training ------------------------------------------------------
    A("## 4. Student training\n")
    if training:
        for stage, rec in training.items():
            A(f"**{stage.upper()}** — `{rec.get('model', '')}` "
              f"{rec.get('quantization', '')}")
            A(f"- {rec.get('optimizer_steps')} optimizer steps in "
              f"{rec.get('elapsed_s')}s, GPU peak {rec.get('gpu_peak_gb')} GB")
            if stage == "sft":
                A(f"- {rec.get('n_examples')} training traces, "
                  f"{rec.get('tokens_seen')} tokens seen, "
                  f"final loss {rec.get('final_loss')}")
            else:
                A(f"- {rec.get('gen_tokens_total')} tokens generated, "
                  f"{rec.get('n_executions')} SQL executions for the reward")
                if rec.get("n_steps_logged"):
                    nl, nt = rec["n_live_steps"], rec["n_steps_logged"]
                    A(f"- **{nl}/{nt} steps produced a gradient** "
                      f"({100 * nl / nt:.0f}%); a step with "
                      "`adv_nonzero_frac = 0` had every rollout score alike and "
                      "taught the model nothing")
                if rec.get("truncated"):
                    A("- ⚠ stage was cut short (timeout/kill); reconstructed "
                      "from the step log")
                if rec.get("first") and rec.get("last"):
                    a, b = rec["first"], rec["last"]
                    first_len = a.get("gen_tokens_mean") or 0
                    last_len = b.get("gen_tokens_mean") or 0
                    A(f"- reward {a.get('reward_mean')} -> {b.get('reward_mean')}")
                    # a *falling* generated length under a shrinking clip is
                    # what internalisation looks like; say which way it went
                    # rather than asserting the conclusion
                    direction = "shortened" if last_len < first_len else "did NOT shorten"
                    A(f"- generated length {first_len} -> {last_len} tokens "
                      f"({direction}); clip {a.get('length_limit')} -> "
                      f"{b.get('length_limit')}, w_fmt {a.get('w_fmt')} -> "
                      f"{b.get('w_fmt')}")
                    if rec.get("optimizer_steps", 0) < 50:
                        A(f"- **{rec.get('optimizer_steps')} optimizer steps is far "
                          "too few to internalise anything** - this run exercises "
                          "the mechanism, it does not train it")
            A("")
    else:
        A("_No training logs found._\n")

    A("## 5. Where the logs are\n")
    A(f"- `{run}/traces/teacher_samples.jsonl` — one record per teacher question")
    A(f"- `{run}/sft/sft_log_steps.jsonl`, `{run}/grpo/grpo_log_steps.jsonl` — "
      "per-step loss/reward, tokens, GPU")
    A(f"- `{run}/eval*/eval_samples.jsonl` — one record per evaluated question")
    A(f"- `{run}/eval*/preds_<method>.jsonl` — every predicted query")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/pilot")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    teacher = teacher_section(args.run)
    training = training_section(args.run)
    methods = eval_section(args.run)

    out_dir = args.out or args.run
    os.makedirs(out_dir, exist_ok=True)
    md = render(args.run, teacher, training, methods)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"teacher": teacher, "training": training, "eval": methods},
                  fh, indent=2, default=str)
    print(md)
    print(f"\nwrote {out_dir}/report.md and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
