"""A Groq-backed ``llm`` callable for :meth:`LLMCache.get_or_call`.

This lives in ``demo/``, not in the library, on purpose: ``llm_cache`` never
talks to a provider. It takes a ``Callable[[str], str]`` and knows nothing about
who answers. That is the whole integration contract — this file is just one
implementation of it.

Uses ``urllib`` from the standard library rather than an SDK, so the demo adds
no dependencies to the project.

    export GROQ_API_KEY=...
    llm = make_groq_llm(model="llama-3.3-70b-versatile")
    llm("What is Docker?")
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqError(RuntimeError):
    """The Groq API could not be reached, or refused the request."""


@dataclass
class GroqStats:
    """Token usage across every call this callable has actually made.

    Only cache *misses* reach the API, so these numbers are the real cost of the
    traffic the cache did not absorb.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def average_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "average_latency_ms": round(self.average_latency_ms, 1),
        }


def make_groq_llm(
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.2,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    timeout: float = 60.0,
    stats: GroqStats | None = None,
    opener: Callable[[urllib.request.Request, float], bytes] | None = None,
) -> Callable[[str], str]:
    """Return a ``prompt -> response`` callable backed by the Groq API.

    ``stats`` accumulates token usage if you pass one in. ``opener`` exists so
    tests can substitute the transport; leave it unset in real use.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise GroqError("GROQ_API_KEY is not set. Export it, or run the demo with --fake-llm.")
    send = opener or _send

    def call(prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        import time

        started = time.perf_counter()
        try:
            raw = send(request, timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise GroqError(f"Groq returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise GroqError(f"could not reach Groq: {exc.reason}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        text, usage = _parse(raw)
        if stats is not None:
            stats.calls += 1
            stats.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            stats.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            stats.latencies_ms.append(elapsed_ms)
        return text

    return call


def _send(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


def _parse(raw: bytes) -> tuple[str, dict[str, object]]:
    """Pull the assistant message and usage block out of a chat completion."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroqError(f"Groq returned a non-JSON body: {exc}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GroqError(f"unexpected Groq response shape: {body!r}"[:400]) from exc

    if not isinstance(content, str):
        raise GroqError(f"expected a string response, got {type(content).__name__}")

    usage = body.get("usage") or {}
    return content, usage if isinstance(usage, dict) else {}
