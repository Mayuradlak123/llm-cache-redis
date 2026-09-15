"""A Groq-backed ``llm`` callable for :meth:`LLMCache.get_or_call`.

This lives in ``demo/``, not in the library, on purpose: ``llm_cache`` never
talks to a provider. It takes a ``Callable[[str], str]`` and knows nothing about
who answers. That is the whole integration contract — this file is just one
implementation of it.

Uses ``urllib`` from the standard library rather than an SDK, so the demo adds
no dependencies to the project.

    export GROQ_API_KEY=...
    llm = make_groq_llm(model="openai/gpt-oss-20b")
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
MODELS_URL = "https://api.groq.com/openai/v1/models"

# Groq's published catalogue is NOT the same as the list any given key can call:
# a 404 "does not exist or you do not have access to it" means the model is real
# but your tier cannot reach it. The llama-* ids are the usual example — widely
# documented, absent from many keys. Run `python demo/server.py --list-models`
# to see yours, and set GROQ_MODEL in .env accordingly.
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Groq sits behind Cloudflare, which rejects urllib's default
# ``User-Agent: Python-urllib/3.x`` with "Error 1010: Access denied"
# (browser_signature_banned) before the request ever reaches the API. Any
# explicit User-Agent gets through, so identify ourselves honestly rather than
# impersonating a browser.
USER_AGENT = "llm-cache-demo/0.1.0 (+https://github.com/Mayuradlak123/llm-cache)"


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
    # Generous by default: reasoning models spend part of this budget on
    # `reasoning` before emitting any answer at all.
    max_tokens: int = 1024,
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
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )

        import time

        started = time.perf_counter()
        try:
            raw = send(request, timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise GroqError(_explain_http_error(exc.code, detail)) from exc
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


def list_models(
    api_key: str | None = None,
    *,
    timeout: float = 30.0,
    opener: Callable[[urllib.request.Request, float], bytes] | None = None,
) -> list[str]:
    """Return the model ids this API key can actually use, sorted.

    Groq's published model list is not the same as *your* list: a key on one
    tier gets a 404 for models another tier can call. This asks the API rather
    than guessing.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise GroqError("GROQ_API_KEY is not set, so the model list cannot be fetched.")

    request = urllib.request.Request(
        MODELS_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    send = opener or _send
    try:
        raw = send(request, timeout)
    except urllib.error.HTTPError as exc:
        raise GroqError(
            _explain_http_error(exc.code, exc.read().decode("utf-8", "replace"))
        ) from exc
    except urllib.error.URLError as exc:
        raise GroqError(f"could not reach Groq: {exc.reason}") from exc

    try:
        body = json.loads(raw.decode("utf-8"))
        entries = body["data"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GroqError(f"unexpected model list response: {exc}") from exc

    return sorted(str(item["id"]) for item in entries if isinstance(item, dict) and "id" in item)


def _explain_http_error(status: int, body: str) -> str:
    """Turn an HTTP failure into something you can act on.

    Groq's own errors arrive as JSON; Cloudflare's arrive as its own error
    document and mean the request never reached Groq at all.
    """
    if "error_code" in body and "1010" in body:
        return (
            "blocked by Cloudflare before reaching Groq (Error 1010, "
            "browser_signature_banned). This happens when the request has no "
            "User-Agent header — check that USER_AGENT is still being sent."
        )

    message = body.strip()[:400]
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            message = str(parsed["error"].get("message", message))
    except json.JSONDecodeError:
        pass

    hints = {
        401: "check GROQ_API_KEY in your .env",
        403: "the key may lack access to this model",
        404: (
            "the model does not exist, or your key's tier cannot reach it — "
            "run 'python demo/server.py --list-models' to see what yours can use"
        ),
        429: "rate limited; wait and retry, or lower your request rate",
    }
    hint = hints.get(status)
    return f"Groq returned HTTP {status}: {message}" + (f" ({hint})" if hint else "")


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
        choice = body["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GroqError(f"unexpected Groq response shape: {body!r}"[:400]) from exc

    if content is not None and not isinstance(content, str):
        raise GroqError(f"expected a string response, got {type(content).__name__}")

    # Reasoning models (openai/gpt-oss-*, qwen3) split their output into
    # `reasoning` and `content`. When max_tokens runs out mid-reasoning the
    # answer itself is never emitted and `content` is empty. Returning that
    # would let the cache store an empty answer permanently, so refuse it.
    if not (content or "").strip():
        finish = str(choice.get("finish_reason", "unknown"))
        reasoning = str(choice.get("message", {}).get("reasoning") or "")
        if finish == "length":
            raise GroqError(
                "the model used its whole token budget on reasoning and returned "
                "no answer. Raise max_tokens (reasoning models such as "
                "openai/gpt-oss-* need room for both) or pick a non-reasoning model."
            )
        detail = f" (reasoning: {reasoning[:120]}...)" if reasoning else ""
        raise GroqError(f"Groq returned an empty answer, finish_reason={finish}{detail}")

    usage = body.get("usage") or {}
    return content, usage if isinstance(usage, dict) else {}
