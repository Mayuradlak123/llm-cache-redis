"""Shared fakes.

The test suite never touches a real Redis server or a real embedding model:
everything runs against in-process doubles so tests stay fast and offline.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from llm_cache import LLMCache, ReplicationConfig
from llm_cache.redis_store import RedisStore


class FakeClock:
    """A manually advanced clock, so TTL behaviour is testable without sleeping."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRedis:
    """The handful of Redis commands :class:`RedisStore` actually uses."""

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        *,
        role: str = "master",
        replicas: int = 0,
        acks: int | None = None,
        replica_lag: int = 0,
    ) -> None:
        self.clock = clock or FakeClock()
        self.data: dict[str, dict[str, Any]] = {}
        self.expiry: dict[str, float] = {}
        self.role = role
        self.replicas = replicas
        # How many replicas WAIT reports; defaults to all of them acknowledging.
        self.acks = replicas if acks is None else acks
        self.replica_lag = replica_lag
        self.wait_calls: list[tuple[int, int]] = []

    def _expired(self, key: str) -> bool:
        deadline = self.expiry.get(key)
        return deadline is not None and self.clock() >= deadline

    def _sweep(self) -> None:
        for key in [k for k in self.data if self._expired(k)]:
            del self.data[key]
            self.expiry.pop(key, None)

    def ping(self) -> bool:
        return True

    def hset(self, key: str, mapping: dict[str, Any]) -> int:
        self.data.setdefault(key, {}).update(mapping)
        return len(mapping)

    def expire(self, key: str, seconds: int) -> bool:
        if key not in self.data:
            return False
        self.expiry[key] = self.clock() + seconds
        return True

    def hgetall(self, key: str) -> dict[str, Any]:
        self._sweep()
        return dict(self.data.get(key, {}))

    def scan_iter(self, match: str = "*", count: int = 10) -> Iterator[str]:
        self._sweep()
        yield from [k for k in list(self.data) if fnmatch.fnmatch(k, match)]

    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if self.data.pop(key, None) is not None:
                removed += 1
            self.expiry.pop(key, None)
        return removed

    def wait(self, num_replicas: int, timeout: int) -> int:
        """Stand in for Redis WAIT: record the call, report the configured acks."""
        self.wait_calls.append((num_replicas, timeout))
        return min(self.acks, num_replicas)

    def info(self, section: str = "all") -> dict[str, Any]:
        offset = 1000
        data: dict[str, Any] = {
            "role": self.role,
            "master_repl_offset": offset,
            "connected_slaves": self.replicas,
        }
        for index in range(self.replicas):
            data[f"slave{index}"] = (
                f"ip=10.0.0.{index + 2},port=6379,state=online,"
                f"offset={offset - self.replica_lag},lag=0"
            )
        return data


class BrokenRedis:
    """A Redis client where every command fails, standing in for an outage."""

    def _boom(self, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("redis is down")

    ping = hset = expire = hgetall = delete = wait = info = _boom

    def scan_iter(self, *args: Any, **kwargs: Any) -> Iterator[str]:
        raise ConnectionError("redis is down")


TOPICS = ("docker", "python", "kubernetes", "coffee")


class FakeEmbeddings:
    """Deterministic topic embeddings: same topic ~1.0 similar, different topics ~0.0.

    Prompts are mapped onto one axis per topic, which is enough to exercise
    "semantically similar" versus "unrelated" without downloading a model.
    Unknown topics fall back to a per-text axis so they match only themselves.
    """

    def __init__(self, overrides: dict[str, list[float]] | None = None) -> None:
        self.overrides = overrides or {}
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self.overrides:
            return self.overrides[text]
        lowered = text.lower()
        vector = [0.0] * (len(TOPICS) + 1)
        for index, topic in enumerate(TOPICS):
            if topic in lowered:
                vector[index] = 1.0
        if not any(vector):
            vector[-1] = 1.0
        return vector


class FailingEmbeddings:
    """An embedding provider that always raises."""

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding model unavailable")


class CountingLLM:
    """Records every prompt it is asked to answer."""

    def __init__(self, prefix: str = "answer") -> None:
        self.prefix = prefix
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"{self.prefix}: {prompt}"


def build_cache(
    client: Any,
    *,
    namespace: str = "test",
    similarity_threshold: float = 0.90,
    ttl: int | None = 3600,
    embeddings: Any = None,
    reader: Any = None,
    replication: ReplicationConfig | None = None,
) -> LLMCache:
    """Build an :class:`LLMCache` wired to the given fake client."""
    store = RedisStore(namespace=namespace, client=client, reader=reader, replication=replication)
    return LLMCache(
        similarity_threshold=similarity_threshold,
        ttl=ttl,
        namespace=namespace,
        embeddings=embeddings or FakeEmbeddings(),
        store=store,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(clock: FakeClock) -> FakeRedis:
    return FakeRedis(clock)


@pytest.fixture
def llm() -> CountingLLM:
    return CountingLLM()


@pytest.fixture
def cache(client: FakeRedis) -> LLMCache:
    return build_cache(client)
