"""A tiny HTTP server that puts the Tailwind UI in front of ``LLMCache``.

Standard library only — no FastAPI, no Flask — because the demo should not add
dependencies to a library whose whole point is staying small.

    cp .env.example .env     # then set GROQ_API_KEY
    uv run python demo/server.py

Flags:

    --fake-llm      answer locally instead of calling Groq (no API key needed)
    --memory        skip Redis entirely and use the in-process store
    --list-models   print the model ids this API key can use, then exit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_settings  # noqa: E402
from groq_llm import GroqError, GroqStats, list_models, make_groq_llm  # noqa: E402
from memory_store import MemoryRedis  # noqa: E402

from llm_cache import LLMCache  # noqa: E402
from llm_cache.redis_store import RedisStore  # noqa: E402

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"


class Demo:
    """Holds the cache, the LLM callable, and the facts the UI wants to show."""

    def __init__(self, *, force_memory: bool = False, force_fake: bool = False) -> None:
        self.settings = load_settings()
        self.groq_stats = GroqStats()
        self.backend, store = self._build_store(force_memory)
        self.llm, self.llm_name, self.llm_error = self._build_llm(force_fake)
        self.cache = LLMCache(
            similarity_threshold=self.settings.similarity_threshold,
            ttl=self.settings.ttl,
            namespace=self.settings.namespace,
            store=store,
        )
        self.warmup_ms = self._warm_embeddings()

    def _warm_embeddings(self) -> float:
        """Load the embedding model now, not on the first user request.

        sentence-transformers loads lazily, which would otherwise put the whole
        model load (tens of seconds on a cold start) inside request #1 and make
        the UI look broken.
        """
        print("  loading embedding model...", flush=True)
        started = time.perf_counter()
        self.cache.embeddings.embed("warmup")
        elapsed = (time.perf_counter() - started) * 1000.0
        print(f"  ready in {elapsed / 1000:.1f}s", flush=True)
        return elapsed

    def _build_store(self, force_memory: bool) -> tuple[str, RedisStore]:
        """Use real Redis when it answers; otherwise fall back to memory."""
        if not force_memory:
            candidate = RedisStore(
                redis_url=self.settings.redis_url, namespace=self.settings.namespace
            )
            if candidate.ping():
                return f"redis ({self.settings.redis_url})", candidate
        return "in-memory (no Redis)", RedisStore(
            namespace=self.settings.namespace, client=MemoryRedis()
        )

    def _build_llm(self, force_fake: bool) -> tuple[Any, str, str | None]:
        if force_fake:
            return _fake_llm, "fake (local echo)", None
        try:
            llm = make_groq_llm(
                model=self.settings.groq_model,
                api_key=self.settings.groq_api_key,
                temperature=self.settings.temperature,
                stats=self.groq_stats,
            )
            self._check_model_access()
            return llm, f"groq ({self.settings.groq_model})", None
        except GroqError as exc:
            # Keep serving: the UI shows the reason and the page still works
            # for everything that does not need the API.
            return _fake_llm, "fake (local echo)", str(exc)

    def _check_model_access(self) -> None:
        """Fail at startup, not on the first question, if GROQ_MODEL is unusable.

        Groq's published model list is not the same as the one a given key can
        call, so this asks the API and names the alternatives instead of leaving
        you to discover a 404 mid-demo.
        """
        available = list_models(self.settings.groq_api_key)
        if self.settings.groq_model in available:
            return
        listing = "\n    ".join(available[:20]) or "(none returned)"
        raise GroqError(
            f"GROQ_MODEL={self.settings.groq_model!r} is not available to this API key.\n"
            f"  Models this key can use:\n    {listing}\n"
            f"  Set GROQ_MODEL in .env to one of those."
        )

    # -- request handling ---------------------------------------------------

    def ask(self, prompt: str) -> dict[str, Any]:
        """Run one prompt through the cache and report what happened."""
        if not prompt.strip():
            raise ValueError("prompt is empty")

        started = time.perf_counter()
        calls_before = self.groq_stats.calls
        result = self.cache.get_or_call(
            prompt=prompt,
            llm=self.llm,
            model=self.settings.groq_model,
            temperature=self.settings.temperature,
        )
        total_ms = (time.perf_counter() - started) * 1000.0

        return {
            "prompt": prompt,
            "response": result.response,
            "cache_hit": result.cache_hit,
            "similarity": result.similarity,
            "latency_ms": round(total_ms, 1),
            "cached_at": result.cached_at.isoformat() if result.cached_at else None,
            "llm_called": self.groq_stats.calls > calls_before,
            "threshold": self.settings.similarity_threshold,
        }

    def state(self) -> dict[str, Any]:
        return {
            "config": self.settings.describe(),
            "backend": self.backend,
            "llm": self.llm_name,
            "llm_error": self.llm_error,
            "stats": self.cache.stats(),
            "groq": self.groq_stats.as_dict(),
        }


def _fake_llm(prompt: str) -> str:
    """Stand-in used when no API key is configured, or with --fake-llm."""
    time.sleep(0.6)  # pretend to be a network call, so HIT vs MISS is visible
    return (
        f"(fake answer — no Groq API key configured)\n\n"
        f"You asked: {prompt}\n\n"
        f"With a real key this text would come from Groq. The caching "
        f"behaviour you see is identical either way."
    )


class Handler(BaseHTTPRequestHandler):
    demo: Demo

    def log_message(self, format: str, *args: Any) -> None:
        print(f"  {self.address_string()} {format % args}")

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        if self.path in ("/", "/index.html"):
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(self.demo.state())
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON body"}, status=400)
            return

        if self.path == "/api/ask":
            try:
                self._json(self.demo.ask(str(payload.get("prompt", ""))))
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
            except GroqError as exc:
                self._json({"error": str(exc)}, status=502)
        elif self.path == "/api/clear":
            self._json({"removed": self.demo.cache.clear()})
        elif self.path == "/api/reset-stats":
            self.demo.cache.reset_stats()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, status=404)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="llm-cache demo server")
    parser.add_argument("--fake-llm", action="store_true", help="do not call Groq")
    parser.add_argument("--memory", action="store_true", help="skip Redis, use memory")
    parser.add_argument(
        "--list-models", action="store_true", help="print usable model ids and exit"
    )
    args = parser.parse_args(argv)

    if args.list_models:
        settings = load_settings()
        try:
            models = list_models(settings.groq_api_key)
        except GroqError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Models available to this key ({len(models)}):")
        for model in models:
            marker = "  <- GROQ_MODEL" if model == settings.groq_model else ""
            print(f"  {model}{marker}")
        return 0

    demo = Demo(force_memory=args.memory, force_fake=args.fake_llm)
    Handler.demo = demo

    settings = demo.settings
    print("llm-cache demo")
    print(f"  cache store : {demo.backend}")
    print(f"  llm         : {demo.llm_name}")
    if demo.llm_error:
        print(f"                ({demo.llm_error})")
    print(f"  namespace   : {settings.namespace}")
    print(f"  threshold   : {settings.similarity_threshold}")
    print(f"  ttl         : {settings.ttl if settings.ttl is not None else 'no expiry'}")
    print(f"\n  open http://{settings.host}:{settings.port}\n")

    server = ThreadingHTTPServer((settings.host, settings.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
