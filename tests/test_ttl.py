"""TTL: entries disappear once Redis expires them."""

from __future__ import annotations

from conftest import CountingLLM, FakeClock, FakeRedis, build_cache


def test_entry_survives_until_the_ttl_elapses(
    client: FakeRedis, clock: FakeClock, llm: CountingLLM
) -> None:
    cache = build_cache(client, ttl=60)
    cache.get_or_call("What is Docker?", llm)

    clock.advance(59)

    assert cache.get_or_call("What is Docker?", llm).cache_hit is True


def test_entry_expires_after_the_ttl(client: FakeRedis, clock: FakeClock, llm: CountingLLM) -> None:
    cache = build_cache(client, ttl=60)
    cache.get_or_call("What is Docker?", llm)

    clock.advance(61)
    result = cache.get_or_call("What is Docker?", llm)

    assert result.cache_hit is False
    assert llm.calls == 2


def test_per_request_ttl_overrides_the_instance_ttl(
    client: FakeRedis, clock: FakeClock, llm: CountingLLM
) -> None:
    cache = build_cache(client, ttl=3600)
    cache.get_or_call("What is Docker?", llm, ttl=10)

    clock.advance(11)

    assert cache.get_or_call("What is Docker?", llm).cache_hit is False


def test_ttl_none_means_no_expiry(client: FakeRedis, clock: FakeClock, llm: CountingLLM) -> None:
    cache = build_cache(client, ttl=None)
    cache.get_or_call("What is Docker?", llm)

    clock.advance(10_000_000)

    assert cache.get_or_call("What is Docker?", llm).cache_hit is True


def test_ttl_is_written_to_redis(client: FakeRedis, llm: CountingLLM) -> None:
    cache = build_cache(client, ttl=120)
    cache.get_or_call("What is Docker?", llm)

    assert len(client.expiry) == 1
    assert set(client.expiry) == set(client.data)
