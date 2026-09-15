"""Core hit/miss behaviour, cache identity and failure handling."""

from __future__ import annotations

from conftest import (
    BrokenRedis,
    CountingLLM,
    FailingEmbeddings,
    FakeRedis,
    build_cache,
)

from llm_cache import LLMCache


def test_first_request_is_a_miss(cache: LLMCache, llm: CountingLLM) -> None:
    result = cache.get_or_call("What is Docker?", llm)

    assert result.cache_hit is False
    assert result.similarity is None
    assert result.cached_at is None
    assert result.response == "answer: What is Docker?"
    assert llm.calls == 1


def test_second_exact_request_is_a_hit(cache: LLMCache, llm: CountingLLM) -> None:
    first = cache.get_or_call("What is Docker?", llm)
    second = cache.get_or_call("What is Docker?", llm)

    assert second.cache_hit is True
    assert second.response == first.response
    assert second.similarity == 1.0
    assert second.cached_at is not None


def test_semantically_similar_request_is_a_hit(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)
    result = cache.get_or_call("Can you explain Docker in simple words?", llm)

    assert result.cache_hit is True
    assert result.response == "answer: What is Docker?"


def test_unrelated_request_is_a_miss(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)
    result = cache.get_or_call("How do I brew good coffee?", llm)

    assert result.cache_hit is False
    assert llm.calls == 2


def test_llm_is_called_only_on_miss(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)
    cache.get_or_call("Explain Docker please", llm)
    cache.get_or_call("Tell me about python", llm)

    assert llm.calls == 2
    assert llm.prompts == ["What is Docker?", "Tell me about python"]


def test_llm_is_not_called_on_hit(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)
    before = llm.calls

    result = cache.get_or_call("Docker overview", llm)

    assert result.cache_hit is True
    assert llm.calls == before


def test_different_namespaces_do_not_collide(client: FakeRedis, llm: CountingLLM) -> None:
    app_a = build_cache(client, namespace="app-a")
    app_b = build_cache(client, namespace="app-b")

    app_a.get_or_call("What is Docker?", llm)
    result = app_b.get_or_call("What is Docker?", llm)

    assert result.cache_hit is False
    assert llm.calls == 2


def test_different_model_does_not_reuse_cache(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm, model="model-a")
    other = cache.get_or_call("What is Docker?", llm, model="model-b")
    same = cache.get_or_call("What is Docker?", llm, model="model-a")

    assert other.cache_hit is False
    assert same.cache_hit is True
    assert llm.calls == 2


def test_temperature_and_system_prompt_are_part_of_cache_identity(
    cache: LLMCache, llm: CountingLLM
) -> None:
    cache.get_or_call("What is Docker?", llm, model="m", temperature=0.0)
    hotter = cache.get_or_call("What is Docker?", llm, model="m", temperature=0.9)
    assert hotter.cache_hit is False

    cache.get_or_call("What is python?", llm, system_prompt="You are terse.")
    other_system = cache.get_or_call("What is python?", llm, system_prompt="You are verbose.")
    assert other_system.cache_hit is False


def test_redis_failure_still_calls_the_llm(llm: CountingLLM) -> None:
    cache = build_cache(BrokenRedis(), namespace="broken")

    first = cache.get_or_call("What is Docker?", llm)
    second = cache.get_or_call("What is Docker?", llm)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert first.response == "answer: What is Docker?"
    assert llm.calls == 2


def test_embedding_failure_still_calls_the_llm(client: FakeRedis, llm: CountingLLM) -> None:
    cache = build_cache(client, embeddings=FailingEmbeddings())

    result = cache.get_or_call("What is Docker?", llm)

    assert result.cache_hit is False
    assert result.response == "answer: What is Docker?"
    assert llm.calls == 1
    assert client.data == {}, "nothing should be cached when embedding fails"


def test_clear_removes_entries_in_the_namespace(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)

    assert cache.clear() == 1
    assert cache.get_or_call("What is Docker?", llm).cache_hit is False


def test_invalid_threshold_is_rejected(client: FakeRedis) -> None:
    import pytest

    with pytest.raises(ValueError):
        build_cache(client, similarity_threshold=1.5)


def test_latency_is_recorded(cache: LLMCache, llm: CountingLLM) -> None:
    result = cache.get_or_call("What is Docker?", llm)

    assert result.latency_ms >= 0.0
