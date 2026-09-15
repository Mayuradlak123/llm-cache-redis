"""The demo's Groq adapter and .env config.

The API is never actually called: the transport is substituted, so these run
offline and need no key.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from conftest import FakeRedis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from config import Settings, load_env, load_settings  # noqa: E402
from groq_llm import GroqError, GroqStats, make_groq_llm  # noqa: E402
from memory_store import MemoryRedis  # noqa: E402

from llm_cache import LLMCache  # noqa: E402
from llm_cache.redis_store import RedisStore  # noqa: E402


def reply(text: str, *, prompt_tokens: int = 11, completion_tokens: int = 7) -> bytes:
    """Build a Groq chat-completion body."""
    return json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    ).encode("utf-8")


# ------------------------------------------------------------------ adapter


def test_returns_the_assistant_message() -> None:
    llm = make_groq_llm(api_key="test-key", opener=lambda req, timeout: reply("Docker is..."))

    assert llm("What is Docker?") == "Docker is..."


def test_sends_a_well_formed_request() -> None:
    seen: dict[str, object] = {}

    def opener(request: urllib.request.Request, timeout: float) -> bytes:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[union-attr]
        return reply("ok")

    llm = make_groq_llm(
        api_key="secret", model="llama-3.1-8b-instant", temperature=0.7, opener=opener
    )
    llm("hello")

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["method"] == "POST"
    assert seen["auth"] == "Bearer secret"
    body = seen["body"]
    assert body["model"] == "llama-3.1-8b-instant"  # type: ignore[index]
    assert body["temperature"] == 0.7  # type: ignore[index]
    assert body["messages"] == [{"role": "user", "content": "hello"}]  # type: ignore[index]


def test_system_prompt_is_sent_first() -> None:
    seen: dict[str, object] = {}

    def opener(request: urllib.request.Request, timeout: float) -> bytes:
        seen["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[union-attr]
        return reply("ok")

    llm = make_groq_llm(api_key="k", system_prompt="Be terse.", opener=opener)
    llm("hi")

    messages = seen["body"]["messages"]  # type: ignore[index]
    assert messages[0] == {"role": "system", "content": "Be terse."}
    assert messages[1]["role"] == "user"


def test_missing_api_key_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(GroqError, match="GROQ_API_KEY"):
        make_groq_llm()


def test_http_error_is_wrapped() -> None:
    def opener(request: urllib.request.Request, timeout: float) -> bytes:
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            None,  # type: ignore[arg-type]
        )

    llm = make_groq_llm(api_key="bad", opener=opener)

    with pytest.raises(GroqError, match="HTTP 401"):
        llm("hi")


def test_network_error_is_wrapped() -> None:
    def opener(request: urllib.request.Request, timeout: float) -> bytes:
        raise urllib.error.URLError("name resolution failed")

    llm = make_groq_llm(api_key="k", opener=opener)

    with pytest.raises(GroqError, match="could not reach Groq"):
        llm("hi")


def test_malformed_response_is_wrapped() -> None:
    llm = make_groq_llm(api_key="k", opener=lambda req, timeout: b'{"choices": []}')

    with pytest.raises(GroqError, match="unexpected Groq response shape"):
        llm("hi")


def test_non_json_response_is_wrapped() -> None:
    llm = make_groq_llm(api_key="k", opener=lambda req, timeout: b"<html>502</html>")

    with pytest.raises(GroqError, match="non-JSON"):
        llm("hi")


def test_token_usage_accumulates() -> None:
    stats = GroqStats()
    llm = make_groq_llm(api_key="k", stats=stats, opener=lambda req, timeout: reply("answer"))

    llm("one")
    llm("two")

    assert stats.calls == 2
    assert stats.prompt_tokens == 22
    assert stats.completion_tokens == 14
    assert stats.total_tokens == 36
    assert stats.as_dict()["llm_calls"] == 2


# ------------------------------------------------- the point of the exercise


def test_the_cache_stops_repeat_prompts_reaching_groq(client: FakeRedis) -> None:
    """A semantically similar second prompt must not produce a second API call."""
    stats = GroqStats()
    llm = make_groq_llm(
        api_key="k",
        stats=stats,
        opener=lambda req, timeout: reply("Docker is a container runtime."),
    )
    cache = LLMCache(
        similarity_threshold=0.90,
        namespace="demo",
        embeddings=_TopicEmbeddings(),
        store=RedisStore(namespace="demo", client=client),
    )

    first = cache.get_or_call("What is Docker?", llm)
    second = cache.get_or_call("Can you explain Docker?", llm)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.response == first.response
    assert stats.calls == 1, "the second prompt must never reach the API"


class _TopicEmbeddings:
    """Same trick as the main suite: one axis per topic word."""

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "docker" in lowered else 0.0,
            1.0 if "coffee" in lowered else 0.0,
            1.0 if not any(w in lowered for w in ("docker", "coffee")) else 0.0,
        ]


# -------------------------------------------------------------------- config


def test_env_file_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("REDIS_URL", "SIMILARITY_THRESHOLD", "TTL", "NAMESPACE"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                'REDIS_URL="redis://example:6379"',
                "export SIMILARITY_THRESHOLD=0.85",
                "TTL=120",
                "NAMESPACE='my-app'",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env)

    assert settings.redis_url == "redis://example:6379"
    assert settings.similarity_threshold == 0.85
    assert settings.ttl == 120
    assert settings.namespace == "my-app"


def test_real_environment_wins_over_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("NAMESPACE=from-file", encoding="utf-8")
    monkeypatch.setenv("NAMESPACE", "from-shell")

    assert load_settings(env).namespace == "from-shell"


def test_defaults_apply_without_an_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("REDIS_URL", "SIMILARITY_THRESHOLD", "TTL", "NAMESPACE", "TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(tmp_path / "missing.env")

    assert settings.redis_url == "redis://localhost:6379"
    assert settings.similarity_threshold == 0.90
    assert settings.ttl == 3600
    assert settings.namespace == "example"


def test_ttl_none_disables_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTL", raising=False)
    env = tmp_path / ".env"
    env.write_text("TTL=none", encoding="utf-8")

    assert load_settings(env).ttl is None


def test_bad_threshold_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
    env = tmp_path / ".env"
    env.write_text("SIMILARITY_THRESHOLD=1.5", encoding="utf-8")

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        load_settings(env)


def test_bad_number_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
    env = tmp_path / ".env"
    env.write_text("SIMILARITY_THRESHOLD=high", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a number"):
        load_settings(env)


def test_load_env_reports_what_it_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOO", raising=False)
    env = tmp_path / ".env"
    env.write_text("FOO=bar\nnot-a-pair\n", encoding="utf-8")

    assert load_env(env) == {"FOO": "bar"}


def test_api_key_is_never_exposed_by_describe() -> None:
    described = Settings(groq_api_key="super-secret").describe()

    assert described["has_api_key"] is True
    assert "super-secret" not in json.dumps(described)


# -------------------------------------------------------------- memory store


def test_memory_store_backs_the_cache_without_redis() -> None:
    cache = LLMCache(
        similarity_threshold=0.90,
        namespace="demo",
        embeddings=_TopicEmbeddings(),
        store=RedisStore(namespace="demo", client=MemoryRedis()),
    )
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return "cached answer"

    assert cache.get_or_call("What is Docker?", llm).cache_hit is False
    assert cache.get_or_call("Explain Docker", llm).cache_hit is True
    assert len(calls) == 1


def test_memory_store_expires_entries() -> None:
    store = MemoryRedis()
    store.hset("k", {"a": "b"})
    store.expire("k", 0)

    assert store.hgetall("k") == {}
