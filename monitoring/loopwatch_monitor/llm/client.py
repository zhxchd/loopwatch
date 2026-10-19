"""Model-agnostic chat clients. Credentials are read at call time, never logged.

`ChatClient.complete(messages) -> Completion`. The OpenAI-compatible client is
used for Kimi; `ScriptedClient` returns canned text for tests (no network).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class Completion:
    text: str
    reasoning: str = ""
    usage: dict = field(default_factory=dict)   # prompt_tokens, completion_tokens, cost_usd, latency_s
    error: Optional[str] = None


class BudgetExceeded(RuntimeError):
    pass


def read_key(name: str, search: tuple = ()) -> Optional[str]:
    """Env var first, then simple KEY=VALUE .env files. Never printed."""
    if os.environ.get(name):
        return os.environ[name]
    for path in search:
        path = Path(path)
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


class OpenAICompatibleClient:
    """Chat-completions client with conservative per-call and per-run spend caps."""

    def __init__(self, model: dict, *, run_budget_usd: float, key_search: tuple = (),
                 max_retries: int = 6, timeout: int = 600, retry_delay: float = 5.0):
        self.model = model
        self.key = read_key(model["key_env"], key_search)
        if not self.key:
            raise RuntimeError(f"Missing {model['key_env']}: export it or put it in monitoring/.env")
        self.run_budget = run_budget_usd
        self.spent = 0.0
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay

    def bound(self, messages) -> float:
        # Bytes upper-bound tokens; reserve the full completion allowance (as runner/relay.py).
        size = len(json.dumps(messages).encode()) + 4096
        return (size * self.model["input_price"] + self.model["max_tokens"] * self.model["output_price"]) / 1e6

    def complete(self, messages: list, *, max_cost_usd: Optional[float] = None) -> Completion:
        bound = self.bound(messages)
        if max_cost_usd is not None and bound > max_cost_usd:
            raise BudgetExceeded(f"request bound ${bound:.3f} exceeds per-episode cap ${max_cost_usd:.3f}")
        if self.spent + bound > self.run_budget:
            raise BudgetExceeded(f"run budget ${self.run_budget:.2f} would be exceeded")
        body = {"model": self.model["name"], "messages": messages,
                "temperature": self.model["temperature"], "max_tokens": self.model["max_tokens"],
                "stream": False}
        if self.model.get("thinking"):
            body["thinking"] = {"type": self.model["thinking"]}
        url = self.model["api_base"].rstrip("/") + "/chat/completions"
        delay, last = self.retry_delay, None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
                "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                usage = result.get("usage") or {}
                pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
                cost = bound if pt is None or ct is None else \
                    (pt * self.model["input_price"] + ct * self.model["output_price"]) / 1e6
                self.spent += cost
                message = (result.get("choices") or [{}])[0].get("message") or {}
                return Completion(text=message.get("content") or "",
                                  reasoning=message.get("reasoning_content") or "",
                                  usage={"prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost,
                                         "latency_s": time.time() - started, "attempts": attempt + 1,
                                         "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason")})
            except urllib.error.HTTPError as exc:
                detail = exc.read(2048).decode(errors="replace").replace(self.key, "[REDACTED]")
                last = f"HTTP {exc.code}: {detail[:300]}"
                if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = type(exc).__name__ + ": " + str(exc)[:200]
            if attempt < self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 120)
        # Unknown billing after failed attempts: keep nothing reserved, report the error.
        return Completion(text="", error=last, usage={"cost_usd": 0.0, "attempts": self.max_retries + 1})


class ScriptedClient:
    """Deterministic stand-in for tests: `respond(messages) -> text`."""

    def __init__(self, respond: Callable[[list], str]):
        self.respond = respond
        self.calls = []
        self.spent = 0.0

    def complete(self, messages: list, *, max_cost_usd: Optional[float] = None) -> Completion:
        self.calls.append(messages)
        return Completion(text=self.respond(messages),
                          usage={"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0})
