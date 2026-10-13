"""OpenAI-compatible chat backend (OpenAI, vLLM, SGLang, Ollama, ...).

Uses ``urllib`` so there is no hard dependency on the ``openai`` package; point
``base_url`` at any server that speaks ``/v1/chat/completions``. The paper's
teacher is GPT-5.4 through the OpenAI API.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Sequence

from .base import LLMBackend, Usage


class OpenAICompatBackend(LLMBackend):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(cfg.api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key and "api.openai.com" in self.base_url:
            raise RuntimeError(
                f"no API key in ${cfg.api_key_env}. Set it, or point "
                f"llm.base_url at a local vLLM server."
            )

    def chat(self, messages: Sequence[Dict[str, str]], **kw: Any) -> str:
        payload: Dict[str, Any] = {
            "model": kw.get("model", self.cfg.model),
            "messages": list(messages),
        }
        # GPT-5-family models accept only their default sampling parameters;
        # leave temperature / top_p unset (None) to use the provider default
        for key in ("temperature", "top_p"):
            value = kw.get(key, getattr(self.cfg, key, None))
            if value is not None:
                payload[key] = value
        max_tokens = kw.get("max_tokens", self.cfg.max_tokens)
        if max_tokens:
            # api.openai.com expects max_completion_tokens; most compatible servers max_tokens
            key = "max_completion_tokens" if "api.openai.com" in self.base_url else "max_tokens"
            payload[key] = int(max_tokens)
        if getattr(self.cfg, "seed", None) is not None:
            payload["seed"] = int(self.cfg.seed)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
            },
            method="POST",
        )

        last_err = ""
        for attempt in range(max(1, self.cfg.max_retries)):
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.request_timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"] or ""
                usage = body.get("usage") or {}
                self.usage.add(
                    Usage(
                        prompt_tokens=int(usage.get("prompt_tokens", 0))
                        or self.approx_tokens(json.dumps(list(messages))),
                        completion_tokens=int(usage.get("completion_tokens", 0))
                        or self.approx_tokens(text),
                        calls=1,
                    )
                )
                if self.record_log:
                    self.log.append({"tag": kw.get("tag", ""), "output": text})
                return text
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_err = f"HTTP {exc.code}: {body}"
                if exc.code in (400, 401, 403, 404):   # not transient
                    break
                time.sleep(min(2 ** attempt, 30))
            except (urllib.error.URLError, KeyError, TimeoutError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"chat completion failed after retries: {last_err}")
