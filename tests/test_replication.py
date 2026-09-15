"""Replica reads, WAIT write quorum, and Sentinel wiring."""

from __future__ import annotations

import logging

import pytest
from conftest import BrokenRedis, CountingLLM, FakeEmbeddings, FakeRedis, build_cache

from llm_cache import ReplicationConfig, ReplicationError, SentinelConfig
from llm_cache.connection import Endpoints, build_endpoints
from llm_cache.redis_store import RedisStore, RedisStoreError

# --------------------------------------------------------------------------- WAIT


def test_wait_is_not_issued_by_default(client: FakeRedis, llm: CountingLLM) -> None:
    cache = build_cache(client)
    cache.get_or_call("What is Docker?", llm)

    assert client.wait_calls == [], "WAIT should cost nothing when disabled"


def test_wait_is_issued_with_configured_values(client: FakeRedis, llm: CountingLLM) -> None:
    client.replicas = 2
    cache = build_cache(client, replication=ReplicationConfig(wait_replicas=2, wait_timeout_ms=250))

    cache.get_or_call("What is Docker?", llm)

    assert client.wait_calls == [(2, 250)]


def test_wait_is_issued_once_per_stored_entry(client: FakeRedis, llm: CountingLLM) -> None:
    client.replicas = 1
    cache = build_cache(client, replication=ReplicationConfig(wait_replicas=1))

    cache.get_or_call("What is Docker?", llm)
    cache.get_or_call("Explain Docker", llm)  # HIT: nothing written, so no WAIT
    cache.get_or_call("What is python?", llm)

    assert len(client.wait_calls) == 2


def test_full_acknowledgement_returns_the_replica_count(client: FakeRedis) -> None:
    client.replicas = 3
    client.acks = 3
    store = RedisStore(client=client, replication=ReplicationConfig(wait_replicas=2))

    assert store.wait_for_replicas() == 2


def test_under_replication_warns_but_still_caches(
    client: FakeRedis, llm: CountingLLM, caplog: pytest.LogCaptureFixture
) -> None:
    client.replicas = 2
    client.acks = 1  # only one replica keeps up
    cache = build_cache(client, replication=ReplicationConfig(wait_replicas=2))

    with caplog.at_level(logging.WARNING, logger="llm_cache"):
        result = cache.get_or_call("What is Docker?", llm)

    assert result.cache_hit is False
    assert "1 of 2" in caplog.text or "replicated to 1" in caplog.text
    # The entry is on the master regardless, so it still serves the next request.
    assert cache.get_or_call("Explain Docker", llm).cache_hit is True


def test_require_acks_raises_replication_error(client: FakeRedis) -> None:
    client.replicas = 2
    client.acks = 0
    store = RedisStore(
        client=client, replication=ReplicationConfig(wait_replicas=2, require_acks=True)
    )

    with pytest.raises(ReplicationError) as excinfo:
        store.wait_for_replicas()

    assert excinfo.value.acked == 0
    assert excinfo.value.required == 2
    assert isinstance(excinfo.value, RedisStoreError), "must stay catchable as a store error"


def test_require_acks_does_not_break_the_application(
    client: FakeRedis, llm: CountingLLM, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed write quorum must never deny the caller their LLM response."""
    client.replicas = 1
    client.acks = 0
    cache = build_cache(client, replication=ReplicationConfig(wait_replicas=1, require_acks=True))

    with caplog.at_level(logging.WARNING, logger="llm_cache"):
        result = cache.get_or_call("What is Docker?", llm)

    assert result.response == "answer: What is Docker?"
    assert "under-replicated" in caplog.text


def test_wait_failure_is_reported_as_a_store_error(client: FakeRedis) -> None:
    def boom(num_replicas: int, timeout: int) -> int:
        raise ConnectionError("connection reset")

    client.wait = boom  # type: ignore[method-assign]
    store = RedisStore(client=client, replication=ReplicationConfig(wait_replicas=1))

    with pytest.raises(RedisStoreError):
        store.wait_for_replicas()


def test_invalid_replication_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        ReplicationConfig(wait_replicas=-1)
    with pytest.raises(ValueError):
        ReplicationConfig(wait_timeout_ms=-1)


# --------------------------------------------------------------- replica reads


def test_reads_go_to_the_reader_and_writes_to_the_master(llm: CountingLLM) -> None:
    master = FakeRedis(role="master", replicas=1)
    replica = FakeRedis(role="slave")
    cache = build_cache(master, reader=replica)

    cache.get_or_call("What is Docker?", llm)

    assert master.data, "the entry must be written to the master"
    assert replica.data == {}, "nothing is ever written to the replica"


def test_a_lagging_replica_causes_a_miss_not_an_error(llm: CountingLLM) -> None:
    """Reading a stale replica is safe: the worst case is an extra LLM call."""
    master = FakeRedis(role="master", replicas=1)
    replica = FakeRedis(role="slave")  # never receives the write: maximally stale
    cache = build_cache(master, reader=replica)

    cache.get_or_call("What is Docker?", llm)
    second = cache.get_or_call("What is Docker?", llm)

    assert second.cache_hit is False
    assert second.response == "answer: What is Docker?"
    assert llm.calls == 2


def test_replica_read_failure_falls_back_to_the_llm(llm: CountingLLM) -> None:
    master = FakeRedis(role="master")
    cache = build_cache(master, reader=BrokenRedis())

    result = cache.get_or_call("What is Docker?", llm)

    assert result.cache_hit is False
    assert result.response == "answer: What is Docker?"


# ------------------------------------------------------------ replication info


def test_replication_status_reports_replicas_and_lag(client: FakeRedis) -> None:
    client.replicas = 2
    client.replica_lag = 48
    store = RedisStore(client=client, replication=ReplicationConfig(wait_replicas=1))

    status = store.replication_status()

    assert status["role"] == "master"
    assert status["connected_replicas"] == 2
    assert status["wait_replicas"] == 1
    assert [r["lag_bytes"] for r in status["replicas"]] == [48, 48]
    assert status["replicas"][0]["address"] == "10.0.0.2:6379"
    assert status["replicas"][0]["state"] == "online"


def test_replication_status_flags_a_replica_write_endpoint(client: FakeRedis) -> None:
    client.role = "slave"

    assert RedisStore(client=client).replication_status()["role"] == "slave"


def test_replication_status_survives_an_outage() -> None:
    cache = build_cache(BrokenRedis(), embeddings=FakeEmbeddings())

    status = cache.replication_status()

    assert status["available"] is False
    assert "error" in status


def test_reads_from_replica_flag(client: FakeRedis) -> None:
    assert RedisStore(client=client).replication_status()["reads_from_replica"] is False
    paired = RedisStore(client=client, reader=FakeRedis())
    assert paired.replication_status()["reads_from_replica"] is True


# ----------------------------------------------------------------- Sentinel


def test_sentinel_config_requires_an_address() -> None:
    with pytest.raises(ValueError):
        SentinelConfig(sentinels=[])


def test_build_endpoints_without_sentinel_shares_one_client() -> None:
    endpoints = build_endpoints(redis_url="redis://localhost:6379")

    assert endpoints.writer is endpoints.reader
    assert endpoints.reads_from_replica is False


def test_sentinel_endpoints_use_master_for_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    master, replica = FakeRedis(), FakeRedis()
    seen: dict[str, object] = {}

    class StubSentinel:
        def __init__(self, sentinels: list[tuple[str, int]], **kwargs: object) -> None:
            seen["sentinels"] = sentinels

        def master_for(self, service_name: str, **kwargs: object) -> FakeRedis:
            seen["master_service"] = service_name
            return master

        def slave_for(self, service_name: str, **kwargs: object) -> FakeRedis:
            seen["replica_service"] = service_name
            return replica

    monkeypatch.setattr("redis.sentinel.Sentinel", StubSentinel)

    endpoints = build_endpoints(
        sentinel=SentinelConfig(
            sentinels=[("s1", 26379), ("s2", 26379), ("s3", 26379)],
            service_name="llm-cache-master",
            read_from_replicas=True,
        )
    )

    assert endpoints.writer is master
    assert endpoints.reader is replica
    assert seen["master_service"] == "llm-cache-master"
    assert len(seen["sentinels"]) == 3  # type: ignore[arg-type]
    assert "sentinel[llm-cache-master]" in endpoints.description


def test_sentinel_without_replica_reads_uses_the_master_for_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = FakeRedis()

    class StubSentinel:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def master_for(self, service_name: str, **kwargs: object) -> FakeRedis:
            return master

        def slave_for(self, service_name: str, **kwargs: object) -> FakeRedis:
            raise AssertionError("slave_for must not be called when replica reads are off")

    monkeypatch.setattr("redis.sentinel.Sentinel", StubSentinel)

    endpoints = build_endpoints(sentinel=SentinelConfig(sentinels=[("s1", 26379)]))

    assert endpoints.writer is endpoints.reader is master


def test_endpoints_description_is_reported_by_the_store(client: FakeRedis) -> None:
    store = RedisStore(client=client)

    assert store.replication_status()["endpoint"] == "injected client"


def test_endpoints_dataclass_detects_split_clients() -> None:
    one = FakeRedis()
    assert Endpoints(writer=one, reader=one).reads_from_replica is False
    assert Endpoints(writer=one, reader=FakeRedis()).reads_from_replica is True
