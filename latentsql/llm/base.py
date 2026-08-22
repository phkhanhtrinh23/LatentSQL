"""Backend-agnostic chat interface used by every teacher agent.

The teacher MAS only ever calls :meth:`LLMBackend.ask`, so the same agent code
runs against a hosted API, a local vLLM server, a local HuggingFace model, or
the deterministic offline stub in ``latentsql.llm.mock``.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls


class LLMBackend(ABC):
    """Chat completion with per-call bookkeeping."""

    def __init__(self) -> None:
        self.usage = Usage()
        self.log: List[Dict[str, Any]] = []
        self.record_log = False

    @abstractmethod
    def chat(self, messages: Sequence[Dict[str, str]], **kw: Any) -> str:
        ...

    def ask(self, system: str, user: str, **kw: Any) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        out = self.chat(msgs, **kw)
        if self.record_log:
            self.log.append({"tag": kw.get("tag", ""), "system": system,
                             "user": user, "output": out})
        return out

    def reset_usage(self) -> None:
        self.usage = Usage()

    # rough fallback used when a backend does not report token counts
    @staticmethod
    def approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# output parsing helpers shared by every agent
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
_SQL_START = re.compile(r"(?is)\b(WITH|SELECT)\b")


def extract_code(text: str, langs: Sequence[str] = ("sql", "sqlite", "cte")) -> Optional[str]:
    """Pull a fenced code block out of an LLM response, preferring SQL fences."""
    if not text:
        return None
    blocks = [(m.group(1).lower().strip(), m.group(2).strip()) for m in _FENCE.finditer(text)]
    for lang, body in blocks:
        if lang in langs:
            return body
    for lang, body in blocks:
        if lang not in ("json", "python", "text"):
            return body
    return None


def extract_sql(text: Optional[str]) -> Optional[str]:
    """Best-effort SQL extraction: fenced block, else the first SELECT/WITH."""
    if not text:
        return None
    code = extract_code(text)
    if code:
        return _clean_sql(code)
    m = _SQL_START.search(text)
    if m:
        return _clean_sql(text[m.start():])
    return None


def _clean_sql(sql: str) -> str:
    sql = sql.strip()
    # drop a trailing prose paragraph that follows the statement terminator
    if ";" in sql:
        sql = sql.split(";")[0]
    sql = re.sub(r"(?m)^\s*--.*$", " ", sql)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def extract_list(text: Optional[str], max_items: int = 32) -> List[str]:
    """Parse a comma- or newline-separated list out of a short response."""
    if not text:
        return []
    text = re.sub(r"```[a-z]*\n?", " ", text)
    parts = re.split(r"[,\n;]+", text)
    out = []
    for p in parts:
        p = p.strip().strip("-*`'\"[] ")
        if p and len(p) < 120:
            out.append(p)
        if len(out) >= max_items:
            break
    return out


def build_backend(cfg: Any) -> LLMBackend:
    """Instantiate the backend named by ``cfg.backend`` (see ``LLMConfig``)."""
    from .mock import MockBackend

    name = (getattr(cfg, "backend", "mock") or "mock").lower()
    if name == "mock":
        return MockBackend(cfg)
    if name in ("openai", "vllm", "openai_compat"):
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(cfg)
    if name in ("hf", "transformers", "local"):
        from .hf_local import HFLocalBackend

        return HFLocalBackend(cfg)
    raise ValueError(f"unknown LLM backend: {name!r}")
