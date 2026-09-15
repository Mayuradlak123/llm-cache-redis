"""Statistics tracking."""

from __future__ import annotations

import pytest
from conftest import CountingLLM

from llm_cache import CacheStats, LLMCache


def test_stats_start_empty(cache: LLMCache) -> None:
    assert cache.stats() == {
        "total_requests": 0,
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0,
        "llm_calls_avoided": 0,
    }


def test_hits_and_misses_are_counted(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)
    cache.get_or_call("Explain Docker", llm)
    cache.get_or_call("What is python?", llm)

    stats = cache.stats()

    assert stats["total_requests"] == 3
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["hit_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_llm_calls_avoided_matches_hits(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)
    cache.get_or_call("Explain Docker", llm)
    cache.get_or_call("Docker basics", llm)

    stats = cache.stats()

    assert stats["llm_calls_avoided"] == 2
    assert stats["llm_calls_avoided"] == stats["total_requests"] - llm.calls


def test_estimated_savings_uses_caller_supplied_price(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)
    cache.get_or_call("Explain Docker", llm)

    assert cache.stats(cost_per_call=0.002)["estimated_savings"] == pytest.approx(0.002)


def test_estimated_savings_is_absent_without_a_price(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)

    assert "estimated_savings" not in cache.stats()


def test_reset_stats_clears_counters(cache: LLMCache, llm: CountingLLM) -> None:
    cache.get_or_call("What is Docker?", llm)
    cache.reset_stats()

    assert cache.stats()["total_requests"] == 0


def test_hit_rate_of_empty_stats_is_zero() -> None:
    assert CacheStats().hit_rate == 0.0


def test_stats_object_arithmetic() -> None:
    stats = CacheStats()
    for _ in range(67):
        stats.record_hit()
    for _ in range(33):
        stats.record_miss()

    assert stats.as_dict() == {
        "total_requests": 100,
        "hits": 67,
        "misses": 33,
        "hit_rate": 0.67,
        "llm_calls_avoided": 67,
    }
