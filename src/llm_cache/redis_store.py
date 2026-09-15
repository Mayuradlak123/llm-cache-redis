"""Redis-backed storage for cache entries.

Phase 1 keeps this intentionally plain: one Redis hash per cache entry, with the
embedding stored as raw ``float32`` bytes, and a ``SCAN`` over the key namespace
to gather candidates for similarity search.

Why not Redis Stack vector search? It would require an index, a schema and a
Redis Stack image for a lookup that is a handful of dot products at MVP scale.
The store exposes a small surface (:meth:`add`, :meth:`candidates`,
:meth:`ping`, :meth:`clear`) so it can be replaced by a vector-search backed
implementation later without touching :mod:`llm_cache.cache`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import numpy as np
import redis

from .connection import Endpoints, ReplicationConfig, SentinelConfig, build_endpoints
from .models import CacheEntry

KEY_PREFIX = "llm-cache"

logger = logging.getLogger("llm_cache")


class RedisStoreError(RuntimeError):
    """Raised when Redis cannot serve a request. Always recoverable by the caller."""


class ReplicationError(RedisStoreError):
    """Raised when a write did not reach the configured number of replicas.

    A subclass of :class:`RedisStoreError`, so the cache layer's existing
    fallback catches it and the caller still gets their LLM response. The entry
    *is* on the master; only its durability under failover is in doubt.
    """

    def __init__(self, acked: int, required: int, timeout_ms: int) -> None:
        self.acked = acked
        self.required = required
        self.timeout_ms = timeout_ms
        super().__init__(
            f"write acknowledged by {acked} of {required} required replicas within {timeout_ms}ms"
        )


def scope_id(
    model: str | None,
    temperature: float | None,
    system_prompt: str | None,
) -> str:
    """Return a short, stable id for one generation configuration.

    Entries produced under different models, temperatures or system prompts land
    in different key scopes, so they can never be returned for one another.
    """
    system_hash = hash_system_prompt(system_prompt) or ""
    raw = f"{model or ''}|{temperature if temperature is not None else ''}|{system_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def hash_system_prompt(system_prompt: str | None) -> str | None:
    """Return a short hash of the system prompt, or ``None`` when absent."""
    if system_prompt is None:
        return None
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


def _encode(embedding: list[float]) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _decode(blob: bytes) -> list[float]:
    return [float(x) for x in np.frombuffer(blob, dtype=np.float32)]


class RedisStore:
    """Stores and retrieves :class:`CacheEntry` objects in Redis."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        namespace: str = "default",
        client: redis.Redis | None = None,
        *,
        reader: redis.Redis | None = None,
        sentinel: SentinelConfig | None = None,
        replication: ReplicationConfig | None = None,
    ) -> None:
        """Wire up the store.

        Pass ``client`` (and optionally ``reader``) to supply your own clients,
        ``sentinel`` to resolve the master through Sentinel, or neither to
        connect straight to ``redis_url``.
        """
        self.namespace = namespace
        self.replication = replication or ReplicationConfig()
        if client is not None:
            self._endpoints = Endpoints(
                writer=client,
                reader=reader if reader is not None else client,
                description="injected client",
            )
        else:
            self._endpoints = build_endpoints(redis_url=redis_url, sentinel=sentinel)

    @property
    def _client(self) -> redis.Redis:
        """The master. Every write goes here."""
        return self._endpoints.writer

    @property
    def _reader(self) -> redis.Redis:
        """The read client — a replica when one is configured, else the master."""
        return self._endpoints.reader

    def key(self, scope: str, entry_id: str) -> str:
        return f"{KEY_PREFIX}:{self.namespace}:{scope}:{entry_id}"

    def pattern(self, scope: str) -> str:
        return f"{KEY_PREFIX}:{self.namespace}:{scope}:*"

    def ping(self) -> bool:
        """Return ``True`` when Redis answers, ``False`` otherwise."""
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def add(self, entry: CacheEntry, scope: str) -> None:
        """Write one entry, applying ``entry.ttl`` as the Redis key TTL."""
        mapping: dict[str, str | bytes | float] = {
            "id": entry.id,
            "prompt": entry.prompt,
            "embedding": _encode(entry.embedding),
            "response": entry.response,
            "created_at": entry.created_at.isoformat(),
        }
        if entry.model is not None:
            mapping["model"] = entry.model
        if entry.temperature is not None:
            mapping["temperature"] = str(entry.temperature)
        if entry.system_prompt_hash is not None:
            mapping["system_prompt_hash"] = entry.system_prompt_hash
        if entry.ttl is not None:
            mapping["ttl"] = str(entry.ttl)

        key = self.key(scope, entry.id)
        try:
            self._client.hset(key, mapping=mapping)  # type: ignore[arg-type]
            if entry.ttl is not None and entry.ttl > 0:
                self._client.expire(key, entry.ttl)
        except Exception as exc:  # pragma: no cover - exercised via fake store
            raise RedisStoreError(f"failed to write cache entry: {exc}") from exc

        self.wait_for_replicas()

    def wait_for_replicas(self) -> int:
        """Block until ``wait_replicas`` replicas acknowledge, returning how many did.

        A no-op returning ``0`` when replication acknowledgement is disabled,
        which is the default. Under-replication is logged, and additionally
        raises :class:`ReplicationError` when ``require_acks`` is set.
        """
        config = self.replication
        if not config.enabled:
            return 0

        try:
            acked = int(self._client.wait(config.wait_replicas, config.wait_timeout_ms))
        except Exception as exc:
            raise RedisStoreError(f"WAIT failed: {exc}") from exc

        if acked < config.wait_replicas:
            if config.require_acks:
                raise ReplicationError(acked, config.wait_replicas, config.wait_timeout_ms)
            logger.warning(
                "cache entry replicated to %d of %d requested replicas within %dms",
                acked,
                config.wait_replicas,
                config.wait_timeout_ms,
            )
        return acked

    def candidates(self, scope: str) -> Iterator[CacheEntry]:
        """Yield every live entry in this namespace/scope.

        Expired keys are simply absent: Redis TTL does the cleanup for us.
        """
        try:
            reader = self._reader
            keys = list(reader.scan_iter(match=self.pattern(scope), count=200))
            for key in keys:
                raw = reader.hgetall(key)
                if raw:
                    yield _entry_from_hash(raw)
        except RedisStoreError:
            raise
        except Exception as exc:
            raise RedisStoreError(f"failed to read cache entries: {exc}") from exc

    def clear(self, scope: str | None = None) -> int:
        """Delete entries in this namespace (optionally a single scope). Returns the count."""
        pattern = self.pattern(scope) if scope else f"{KEY_PREFIX}:{self.namespace}:*"
        try:
            keys = list(self._client.scan_iter(match=pattern, count=200))
            if not keys:
                return 0
            self._client.delete(*keys)
            return len(keys)
        except Exception as exc:
            raise RedisStoreError(f"failed to clear cache entries: {exc}") from exc

    def replication_status(self) -> dict[str, Any]:
        """Summarise ``INFO replication`` on the node we write to.

        Reports the node's role, its connected replicas and each replica's
        acknowledged offset, plus ``lag_bytes`` — how far behind the master's
        write offset each replica is. A replica that is permanently non-zero is
        one that would lose data in a failover.
        """
        try:
            info = dict(self._client.info("replication"))
        except Exception as exc:
            raise RedisStoreError(f"failed to read replication info: {exc}") from exc

        role = _text(info.get("role", "unknown"))
        master_offset = int(info.get("master_repl_offset", 0) or 0)
        replicas: list[dict[str, Any]] = []
        for index in range(int(info.get("connected_slaves", 0) or 0)):
            raw = info.get(f"slave{index}")
            entry = _parse_replica(raw)
            if entry is None:
                continue
            entry["lag_bytes"] = max(master_offset - int(entry.get("offset", 0)), 0)
            replicas.append(entry)

        return {
            "endpoint": self._endpoints.description,
            "role": role,
            "master_repl_offset": master_offset,
            "connected_replicas": len(replicas),
            "replicas": replicas,
            "reads_from_replica": self._endpoints.reads_from_replica,
            "wait_replicas": self.replication.wait_replicas,
            "wait_timeout_ms": self.replication.wait_timeout_ms,
            "require_acks": self.replication.require_acks,
        }


def _parse_replica(raw: object) -> dict[str, Any] | None:
    """Parse one ``slaveN`` line from ``INFO replication``.

    redis-py hands these back either already parsed into a dict, or as the raw
    ``ip=...,port=...,state=...,offset=...,lag=...`` string, depending on the
    client version — so handle both.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        fields = {str(k): v for k, v in raw.items()}
    else:
        fields = {}
        for part in _text(raw).split(","):
            name, _, value = part.partition("=")
            if name:
                fields[name.strip()] = value.strip()
    return {
        "address": f"{fields.get('ip', '?')}:{fields.get('port', '?')}",
        "state": str(fields.get("state", "unknown")),
        "offset": int(fields.get("offset", 0) or 0),
    }


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _entry_from_hash(raw: dict[bytes | str, bytes | str]) -> CacheEntry:
    data = {_text(k): v for k, v in raw.items()}
    embedding_blob = data["embedding"]
    blob = embedding_blob if isinstance(embedding_blob, bytes) else embedding_blob.encode("latin-1")
    temperature = data.get("temperature")
    ttl = data.get("ttl")
    created_at = data.get("created_at")
    return CacheEntry(
        id=_text(data["id"]),
        prompt=_text(data["prompt"]),
        embedding=_decode(blob),
        response=_text(data["response"]),
        model=_text(data["model"]) if "model" in data else None,
        temperature=float(_text(temperature)) if temperature is not None else None,
        system_prompt_hash=(
            _text(data["system_prompt_hash"]) if "system_prompt_hash" in data else None
        ),
        created_at=(
            datetime.fromisoformat(_text(created_at))
            if created_at is not None
            else datetime.now(UTC)
        ),
        ttl=int(_text(ttl)) if ttl is not None else None,
    )
