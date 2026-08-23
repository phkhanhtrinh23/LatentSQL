"""Per-sample and per-step run logs: time, tokens, GPU memory.

Every stage of the pipeline writes one JSONL record per unit of work - one per
question for the teacher and the evaluator, one per optimizer step for SFT and
GRPO - so a run can be audited after the fact without re-running it. Nothing
here imports torch at module level, so the torch-free tier-1 path is unaffected.
"""
from __future__ import annotations

import json
import os
import platform
import time
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------
# GPU / process telemetry
# --------------------------------------------------------------------------
def gpu_stats(reset_peak: bool = False) -> Dict[str, Any]:
    """Current and peak CUDA memory in GB. Empty dict when there is no GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        dev = torch.cuda.current_device()
        out = {
            "gpu_name": torch.cuda.get_device_name(dev),
            "gpu_alloc_gb": round(torch.cuda.memory_allocated(dev) / 2**30, 3),
            "gpu_reserved_gb": round(torch.cuda.memory_reserved(dev) / 2**30, 3),
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3),
            "gpu_total_gb": round(
                torch.cuda.get_device_properties(dev).total_memory / 2**30, 3
            ),
        }
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(dev)
        return out
    except Exception:
        return {}


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_environment(**extra: Any) -> Dict[str, Any]:
    """A header record: what hardware and library versions produced a run."""
    env: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:
        pass
    env.update(gpu_stats())
    env.update(extra)
    return env


# --------------------------------------------------------------------------
# JSONL sink
# --------------------------------------------------------------------------
class RunLog:
    """Append-only JSONL log, flushed per record so a killed run keeps its data."""

    def __init__(self, path: str, header: Optional[Dict[str, Any]] = None):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._t0 = time.time()
        if header is not None:
            self.write({"record": "header", **header})

    def write(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record.setdefault("elapsed_s", round(time.time() - self._t0, 2))
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        return record

    def write_with_gpu(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record.update(gpu_stats())
        return self.write(record)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Stopwatch:
    """Wall-clock for one unit of work."""

    def __init__(self) -> None:
        self.t0 = time.time()

    def lap(self) -> float:
        dt = time.time() - self.t0
        self.t0 = time.time()
        return round(dt, 3)

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.t0, 3)
